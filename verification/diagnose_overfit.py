"""
实验2：过拟合测试 + 梯度流检查
  1) 评估 35000_final 基础checkpoint在清洗数据上的损失（判断ABC预训练是否也失败）
  2) 用20个样本做300步过拟合训练（全部参数可训练）——正常架构应把损失压到<2
     若损失几乎不动 → 梯度被阻断，需定位阻断点
  3) 单次backward打印各组件梯度范数
"""

import os
import sys
import json
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.inference_v4 import load_model

CKPT = "d:/AetherMind-Nano3/checkpoints_v4_fixed/v4_checkpoint_20000_phaseG_final.pt"
CKPT_BASE = "d:/AetherMind-Nano3/checkpoints_v4_fixed/v4_checkpoint_35000_final.pt"
DATA = "d:/AetherMind-Nano3/03_dialogue_clean/clean_part_000000.jsonl"


def load_samples(path, n):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            if "<|MOSS|>:" not in text or len(text) < 60:
                continue
            samples.append(text)
            if len(samples) >= n:
                break
    return samples


def encode(model, tokenizer, cfg, text):
    ids = tokenizer(text, return_tensors="pt")["input_ids"]
    if ids.shape[1] >= cfg.max_seq_len:
        ids = ids[:, :cfg.max_seq_len]
    else:
        ids = torch.cat([ids, torch.full((1, cfg.max_seq_len - ids.shape[1]),
                                          cfg.pad_token_id, dtype=ids.dtype)], dim=1)
    return ids


@torch.no_grad()
def eval_loss(model, tokenizer, cfg, samples, device):
    tot, cnt = 0.0, 0
    for text in samples:
        ids = encode(model, tokenizer, cfg, text).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(ids, task_id=0, t=0, phase="generate")
        ce = F.cross_entropy(out["logits"][0, :-1].float(), ids[0, 1:],
                              reduction="none", ignore_index=cfg.pad_token_id)
        mask = (ids[0, 1:] != cfg.pad_token_id)
        tot += ce[mask].sum().item()
        cnt += mask.sum().item()
    return tot / max(cnt, 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 实验1: 基础checkpoint损失 =====
    print("=" * 60)
    print("[实验1] 评估 35000_final (ABC预训练终点) 在清洗数据上的损失")
    model_b, tok_b, cfg_b, _, _ = load_model(CKPT_BASE, device="cuda", arch_mode="train")
    samples = load_samples(DATA, 16)
    loss_b = eval_loss(model_b, tok_b, cfg_b, samples, device)
    print(f"  >> 35000_final 损失 = {loss_b:.4f} nats")
    del model_b
    torch.cuda.empty_cache()

    # ===== 实验2: 过拟合测试 =====
    print("=" * 60)
    print("[实验2] 过拟合测试: 20样本 × 300步, 全参数可训练, lr=1e-4")
    model, tokenizer, cfg, _, _ = load_model(CKPT, device="cuda", arch_mode="train")
    model.train()

    batch_ids = torch.cat([encode(model, tokenizer, cfg, t) for t in samples[:20]]).to(device)
    print(f"  batch: {batch_ids.shape}")

    # 解冻全部参数
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))

    # ===== 实验3: 梯度范数（第一次backward）=====
    out = model(batch_ids[:1], task_id=0, t=0, phase="generate")
    ce = F.cross_entropy(out["logits"][0, :-1].float(), batch_ids[0, 1:],
                         reduction="none", ignore_index=cfg.pad_token_id)
    mask = (batch_ids[0, 1:] != cfg.pad_token_id)
    loss = ce[mask].mean()
    loss.backward()
    groups = {
        "token_emb/lm_head(绑定)": model.encoder.token_emb.weight,
        "encoder.attn[0].Wq": model.encoder.attn_layers[0].Wq.weight,
        "decoder_attns[0].Wq": model.decoder_attns[0].Wq.weight,
        "decoder_layers[0].W_g": model.decoder_layers[0].W_g.weight,
        "decoder_layers[-1].W_g": model.decoder_layers[-1].W_g.weight,
        "metacog.fuse_mlp.fc1": model.metacog.fuse_mlp.fc1.weight,
        "final_norm": model.final_norm.weight,
    }
    print("\n[实验3] 梯度范数:")
    for name, p in groups.items():
        g = p.grad
        print(f"  {name:28s}: grad_norm={'None' if g is None else f'{g.norm().item():.6e}'}")
    opt.zero_grad(set_to_none=True)

    # 训练循环
    for step in range(300):
        idx = step % batch_ids.shape[0]
        ids = batch_ids[idx:idx + 1]
        labels = ids.clone()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(ids, labels, task_id=0, t=float(step), phase="generate")
            loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 30 == 0 or step == 299:
            with torch.no_grad():
                print(f"  step {step:3d}: loss = {loss.item():.4f}")
        del out, loss

    final = eval_loss(model, tokenizer, cfg, samples[:20], device)
    print(f"\n  >> 过拟合后 20样本 损失 = {final:.4f} nats (正常应 < 2; 若仍 >5 = 梯度阻断)")


if __name__ == "__main__":
    main()
