"""
诊断：验证训练版注意力是否存在"未来信息泄漏"（双向注意力 + next-token loss 作弊）

原理：
  因果（正确）模型中，位置 i 的预测只依赖 token 0..i。
  把序列后半段（答案）打乱后，前半段（问题部分）位置上的损失不应有任何变化。
  若前半段损失随后半段内容变化而显著变化 → 注意力看见未来 → 训练/推理错位 → 生成乱码。

用法：
  C:\\Python312\\python.exe verification\\diagnose_causal_leak.py
"""

import os
import sys
import json
import random
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.inference_v4 import load_model

CKPT = "d:/AetherMind-Nano3/checkpoints_v4_fixed/v4_checkpoint_20000_phaseG_final.pt"
DATA = "d:/AetherMind-Nano3/03_dialogue_clean/clean_part_000000.jsonl"
N_SAMPLES = 8


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


@torch.no_grad()
def per_position_ce(model, cfg, input_ids, task_id=0, pad=False, autocast_bf16=False):
    """返回每个位置的 next-token CE 损失（位置 i 的损失 = 预测 token i+1）。
    pad=True 时补到 max_seq_len，模拟训练分布；autocast_bf16 模拟训练精度。"""
    pad_id = cfg.pad_token_id
    if pad and input_ids.shape[1] < cfg.max_seq_len:
        input_ids = torch.cat(
            [input_ids, torch.full((1, cfg.max_seq_len - input_ids.shape[1]), pad_id,
                                   dtype=input_ids.dtype, device=input_ids.device)], dim=1)
    if autocast_bf16 and input_ids.is_cuda:
        ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
    else:
        ctx = torch.amp.autocast("cpu", enabled=False)
    with ctx:
        out = model(input_ids, task_id=task_id, t=0, phase="generate")
    logits = out["logits"][0].float()  # (S, V)
    tgt = input_ids[0, 1:]
    ce = F.cross_entropy(logits[:-1], tgt, reduction="none",
                         ignore_index=pad_id)  # (S-1,)
    valid = (tgt != pad_id)
    return ce.cpu(), valid.cpu()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, cfg, dev, arch = load_model(CKPT, device="cuda", arch_mode="train")

    samples = load_samples(DATA, N_SAMPLES)
    print(f"\n[诊断] 载入 {len(samples)} 个样本\n" + "=" * 70)

    # ===== 测试1: 训练保真设置（pad到512 + 随机task_id + bf16）=====
    print("\n[测试1] 训练保真: pad=512, 随机task_id, bf16 —— 模型在自己训练数据上的真实损失")
    losses_f = []
    for si, text in enumerate(samples):
        ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        if ids.shape[1] < 30:
            continue
        ce, valid = per_position_ce(model, cfg, ids, task_id=random.randint(0, 15),
                                   pad=True, autocast_bf16=True)
        losses_f.append(ce[valid].mean().item())
        print(f"  样本{si}: 训练保真损失 = {losses_f[-1]:.4f} (S={ids.shape[1]})")
    print(f"  >> 平均: {sum(losses_f)/len(losses_f):.4f} nats (随机基线 ln(151680)≈11.93)")

    # ===== 测试2: task_id 敏感性 =====
    print("\n[测试2] task_id 敏感性 (pad=512, bf16)")
    text = samples[0]
    ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    for tid in [0, 3, 7, 15]:
        ce, valid = per_position_ce(model, cfg, ids, task_id=tid, pad=True, autocast_bf16=True)
        print(f"  task_id={tid:2d}: 损失 = {ce[valid].mean().item():.4f}")

    # ===== 测试3: 未padding短序列（推理时的真实输入分布）=====
    print("\n[测试3] 未padding短序列 (推理真实场景, task_id=0, fp32)")
    losses_s = []
    for si, text in enumerate(samples):
        ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        if ids.shape[1] < 30:
            continue
        ce, valid = per_position_ce(model, cfg, ids, task_id=0, pad=False)
        losses_s.append(ce[valid].mean().item())
    print(f"  >> 平均: {sum(losses_s)/len(losses_s):.4f} nats")

    # ===== 测试4: 未来信息泄漏（因果性检查）=====
    print("\n[测试4] 未来泄漏检查: 打乱答案后问题部分损失变化 (pad=512)")
    diffs = []
    for si, text in enumerate(samples):
        ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        if ids.shape[1] < 30:
            continue
        S_real = ids.shape[1]
        prefix_text = text.split("<|MOSS|>:")[0] + "<|MOSS|>:"
        n_prefix = tokenizer(prefix_text, return_tensors="pt")["input_ids"].shape[1]
        if n_prefix >= S_real - 5:
            continue
        pos = torch.zeros(cfg.max_seq_len - 1, dtype=torch.bool)
        pos[1:n_prefix - 1] = True  # 问题部分位置
        valid_pos = (torch.tensor([int(x != cfg.pad_token_id) for x in ids[0, 1:].tolist()]))
        pos = pos & valid_pos.bool()
        ce_true, _ = per_position_ce(model, cfg, ids, task_id=7, pad=True)
        ans = ids[0, n_prefix:].clone()
        if ans.numel() >= 2:
            perm = torch.randperm(ans.numel(), device=device)
            ids_shuf = torch.cat([ids[0, :n_prefix], ans[perm]], dim=0).unsqueeze(0)
            ce_shuf, _ = per_position_ce(model, cfg, ids_shuf, task_id=7, pad=True)
            d = (ce_shuf[pos].mean() - ce_true[pos].mean()).item()
            diffs.append(d)
            print(f"  样本{si}: 问题部分 Δ = {d:+.4f}")
    if diffs:
        avg = sum(diffs) / len(diffs)
        print(f"  >> 平均Δ = {avg:+.4f} nats" + ("  ★存在未来泄漏(无因果掩码)" if abs(avg) > 0.1 else "  无明显泄漏"))


if __name__ == "__main__":
    main()
