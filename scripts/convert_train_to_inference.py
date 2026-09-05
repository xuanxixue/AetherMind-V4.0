"""
训练权重 → 推理架构迁移脚本
================================
把训练版 checkpoint（绝对坐标 τ/C，O(S²)）转换为推理架构 checkpoint
（相对偏移核 τ_rel/consolidated_rel，O(1)），核心权重 W 直接复用。

原理（三方案结合）：
  - 核心权重 Wq/Wk/Wv/Wo/FFN/LN/embedding/lm_head 完全一致，直接复制；
  - τ/C 从 (h,S,S) 绝对坐标 对角平均压缩到 (h,2k-1) 相对核；
  - 输出 arch_mode='inference' 的 checkpoint，供 inference_v4.py 加载。

用法：
  python scripts/convert_train_to_inference.py --ckpt <训练checkpoint> [--rel_k 64] [--output <输出路径>]
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.aethermind4_config import AetherMind4Config
from src.model.aethermind4 import AetherMind4
from src.data.dataset import build_tokenizer


def abs_to_rel(tau_abs: torch.Tensor, rel_k: int) -> torch.Tensor:
    """把 (h,S,S) 绝对信息素/固化压缩为 (h,2k-1) 相对核（按相对距离对角平均）。"""
    if tau_abs.dim() != 3:
        raise ValueError(f"期望 (h,S,S) 3D 张量，得到 {tuple(tau_abs.shape)}")
    h, S, _ = tau_abs.shape
    # 相对桶索引：clip(i-j, -k+1, k-1) + k - 1 ∈ [0, 2k-2]
    rng = torch.arange(S)
    rel = (rng[:, None] - rng[None, :]).clamp(-rel_k + 1, rel_k - 1) + rel_k - 1
    rel = rel.long().reshape(-1)

    tau_rel = torch.zeros(h, 2 * rel_k - 1, dtype=tau_abs.dtype)
    for head in range(h):
        tau_rel[head].scatter_add_(0, rel, tau_abs[head].reshape(-1))

    cnt = torch.zeros(2 * rel_k - 1, dtype=torch.long)
    cnt.scatter_add_(0, rel, torch.ones(S * S, dtype=torch.long))
    cnt = cnt.clamp(min=1)
    return tau_rel / cnt.float()


def convert(train_ckpt_path: str, output_path: str, rel_k: int = 64):
    print(f"[Convert] 读取训练checkpoint: {train_ckpt_path}")
    ckpt = torch.load(train_ckpt_path, map_location="cpu", weights_only=False)
    train_state = ckpt.get("model_state", ckpt.get("model_state_dict", {}))
    model_cfg = ckpt.get("model_config", None)

    if model_cfg is None:
        model_cfg = AetherMind4Config()
        print("[Convert] 警告: checkpoint 无 model_config, 使用默认配置")

    # 去 DataParallel 前缀
    if any(k.startswith("module.") for k in train_state):
        train_state = {k[len("module."):]: v for k, v in train_state.items()}

    # 对齐 token id（与推理脚本一致）
    print("[Convert] 加载 tokenizer 对齐 token id...")
    tokenizer = build_tokenizer()
    model_cfg.pad_token_id = tokenizer.pad_token_id
    model_cfg.eos_token_id = tokenizer.eos_token_id
    model_cfg.bos_token_id = tokenizer.eos_token_id
    model_cfg.unk_token_id = tokenizer.pad_token_id

    # 构建推理架构模型（相对核 + 窗口 + 扩散）
    model_cfg.device = "cpu"
    print(f"[Convert] 构建推理架构模型 (arch_mode=inference, rel_k={rel_k})...")
    model = AetherMind4(model_cfg, arch_mode="inference")
    if rel_k != 64:
        # 仅当 rel_k 非默认时重建相对核 buffer（保持 register_buffer 注册）
        for attn in list(model.encoder.attn_layers) + list(model.decoder_attns):
            attn.rel_k = rel_k
            attn.register_buffer("tau_rel", torch.ones(attn.num_heads, 2 * rel_k - 1))
            attn.register_buffer("consolidated_rel", torch.zeros(attn.num_heads, 2 * rel_k - 1))

    # 映射 state_dict：核心权重复制，τ/C 压缩
    infer_state = {}
    n_tau = 0
    n_cons = 0
    for k, v in train_state.items():
        if k.endswith(".tau") and v.dim() == 3:
            infer_state[k[:-4] + ".tau_rel"] = abs_to_rel(v, rel_k)
            n_tau += 1
        elif k.endswith(".consolidated") and v.dim() == 3:
            infer_state[k[:-13] + ".consolidated_rel"] = abs_to_rel(v, rel_k)
            n_cons += 1
        else:
            infer_state[k] = v

    print(f"[Convert] τ 压缩: {n_tau} 层, C 压缩: {n_cons} 层")

    # 载入推理模型
    target = model.state_dict()
    loaded = 0
    skipped = 0
    for k, v in infer_state.items():
        if k in target and target[k].shape == v.shape:
            target[k].copy_(v)
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(target, strict=False)
    print(f"[Convert] 载入 {loaded} 个张量, 跳过 {skipped} 个")

    # 保存推理 checkpoint
    out = {
        "model_state": model.state_dict(),
        "model_config": model_cfg,
        "steps": ckpt.get("steps", 0),
        "phase": ckpt.get("phase", "A"),
        "evolution_stats": ckpt.get("evolution_stats", {}),
        "arch_mode": "inference",
        "rel_k": rel_k,
    }
    torch.save(out, output_path)
    print(f"[Convert] 完成! 推理权重已保存: {output_path}")
    print(f"[Convert] 参数量: {sum(p.numel() for p in model.parameters()):,}")


def main():
    p = argparse.ArgumentParser(description="训练checkpoint → 推理架构checkpoint")
    p.add_argument("--ckpt", type=str, default=None,
                   help="训练checkpoint路径（默认自动找最新 final）")
    p.add_argument("--ckpt_dir", type=str, default="d:/AetherMind-Nano3/checkpoints_v4_fixed",
                   help="checkpoint目录")
    p.add_argument("--rel_k", type=int, default=64, help="相对核桶半径")
    p.add_argument("--output", type=str, default=None, help="输出路径（默认 <ckpt>_inference.pt）")
    args = p.parse_args()

    ckpt_path = args.ckpt
    if ckpt_path is None:
        import glob
        finals = sorted(glob.glob(os.path.join(args.ckpt_dir, "v4_checkpoint_*_final.pt")),
                        key=os.path.getmtime)
        if finals:
            ckpt_path = finals[-1]
        else:
            ckpt_path = sorted(glob.glob(os.path.join(args.ckpt_dir, "v4_checkpoint_*.pt")),
                               key=os.path.getmtime)[-1]
    if not os.path.exists(ckpt_path):
        print(f"[Convert] 错误: checkpoint 不存在: {ckpt_path}")
        sys.exit(1)

    output = args.output
    if output is None:
        base = os.path.splitext(ckpt_path)[0]
        output = base + "_inference.pt"

    convert(ckpt_path, output, args.rel_k)


if __name__ == "__main__":
    main()