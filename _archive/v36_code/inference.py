import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
import torch
from pathlib import Path

from configs.aethermind36_config import AetherMind36Config
from src.model.aethermind36 import AetherMind36
from src.data.dataset import build_tokenizer, _wrap_tokenizer_call


def load_checkpoint(ckpt_path: str):
    print(f"[Load] 读取检查点: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    steps = ckpt.get("steps", 0)
    phase = ckpt.get("phase", "?")
    print(f"[Load] 训练步数: {steps}, 阶段: {phase}")
    if "model_config" in ckpt:
        model_cfg = ckpt["model_config"]
        print(f"[Load] vocab_size={model_cfg.vocab_size}, d_model={model_cfg.d_model}, "
              f"n_layers={model_cfg.n_layers}, max_seq_len={model_cfg.max_seq_len}")
    else:
        model_cfg = AetherMind36Config()
        print("[Load] 检查点中无 model_config, 使用默认")
    train_cfg = ckpt.get("train_config", None)
    return ckpt, model_cfg, train_cfg


def _try_hf_tokenizer(candidates):
    try:
        from transformers import AutoTokenizer
    except Exception:
        return None
    for cand in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(cand)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token or "<pad>"
            try:
                vs = tok.vocab_size
            except Exception:
                try:
                    vs = len(tok)
                except Exception:
                    vs = 0
            if vs >= 8000:
                return tok, vs
        except Exception:
            continue
    return None


def setup_tokenizer(model_cfg, train_cfg, ckpt_dir: str = ""):
    print("[Tokenizer] 构建分词器...")

    # 优先从 checkpoint 目录加载保存的分词器
    if ckpt_dir:
        tok_path = os.path.join(ckpt_dir, "tokenizer.json")
        if os.path.exists(tok_path):
            try:
                from tokenizers import Tokenizer
                tok = Tokenizer.from_file(tok_path)
                vs = tok.get_vocab_size()
                print(f"[Tokenizer] 从 checkpoint 目录加载: vocab_size={vs}")
                return _wrap_tokenizer_call(tok)
            except Exception as e:
                print(f"[Tokenizer] checkpoint 分词器加载失败: {e}")

    data_dir = None
    if train_cfg is not None and hasattr(train_cfg, "data_dir"):
        data_dir = train_cfg.data_dir

    candidates = [
        "bert-base-chinese",
        "hfl/chinese-roberta-wwm-ext",
        "hfl/chinese-bert-wwm-ext",
        "nghuyong/ernie-3.0-base-zh",
        "xlnet-base-cased",
        "uer/gpt2-chinese-cluecorpussmall",
    ]
    tokenizer = None
    vs = 0

    hf_res = _try_hf_tokenizer(candidates)
    if hf_res is not None:
        tokenizer, vs = hf_res
        print(f"[Tokenizer] 加载 HuggingFace 分词器成功: vocab_size={vs}")
        tokenizer = _wrap_tokenizer_call(tokenizer)

    if tokenizer is None:
        try:
            tokenizer = build_tokenizer(
                vocab_size=model_cfg.vocab_size,
                data_dir=data_dir,
            )
            try:
                vs = tokenizer.vocab_size
            except Exception:
                try:
                    vs = len(tokenizer)
                except Exception:
                    vs = model_cfg.vocab_size
        except Exception as e:
            print(f"[Tokenizer] build_tokenizer 失败: {e}")

    if vs < 2000:
        print(f"[Tokenizer] 警告: 当前词汇表大小 {vs} 过小，可能与训练不匹配，生成结果可能异常")
        print(f"[Tokenizer] 建议安装 transformers 并缓存 bert-base-chinese 等中文分词器")

    try:
        vs_final = tokenizer.vocab_size
    except Exception:
        try:
            vs_final = len(tokenizer)
        except Exception:
            vs_final = model_cfg.vocab_size
    print(f"[Tokenizer] 词汇表大小: {vs_final}")
    model_cfg.vocab_size = max(model_cfg.vocab_size, vs_final)
    return tokenizer


def build_model(model_cfg, ckpt_state):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg.device = device
    model_cfg.set_phase_C(progress=1.0)
    print(f"[Model] 构建设备: {device}")
    model = AetherMind36(model_cfg)
    state = ckpt_state
    if any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Model] 缺失参数: {len(missing)} (前5: {missing[:5]})")
    if unexpected:
        print(f"[Model] 多余参数: {len(unexpected)} (前5: {unexpected[:5]})")
    params = model.count_params()
    size_mb = params * 4 / 1024 / 1024
    print(f"[Model] 参数量: {params:,} ({size_mb:.1f} MB)")
    model = model.to(device)
    model.eval()
    return model, device


def encode_prompt(tokenizer, text: str, max_seq_len: int, bos_token_id: int, device: str):
    try:
        enc = tokenizer(text, truncation=True, max_length=max_seq_len, return_tensors="pt")
        ids = enc["input_ids"]
    except Exception:
        try:
            ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long)
        except Exception as e:
            print(f"[Error] 分词失败: {e}")
            ids = torch.tensor([[bos_token_id]], dtype=torch.long)
    if ids.dim() == 2 and ids.shape[0] > 1:
        ids = ids[:1, :]
    ids = ids[:, :max_seq_len]
    if ids.numel() == 0 or (ids.numel() == 1 and ids[0, 0].item() == 0):
        ids = torch.tensor([[bos_token_id]], dtype=torch.long)
    return ids.to(device)


def decode_tokens(tokenizer, token_ids: torch.Tensor, skip_special: bool = True) -> str:
    ids = token_ids.cpu().tolist()
    if isinstance(ids[0], list):
        ids = ids[0]
    try:
        if hasattr(tokenizer, "decode"):
            return tokenizer.decode(ids, skip_special_tokens=skip_special)
    except Exception:
        pass
    try:
        if hasattr(tokenizer, "id_to_token"):
            parts = []
            for i in ids:
                try:
                    t = tokenizer.id_to_token(i)
                except Exception:
                    t = ""
                if t and not (skip_special and t.startswith("<")):
                    parts.append(t)
            return "".join(parts)
    except Exception:
        pass
    return str(ids)


@torch.no_grad()
def generate_once(model, tokenizer, prompt: str, max_new: int,
                  temperature: float, top_k: int, top_p: int,
                  device: str, task_id: int = 0) -> tuple[str, float]:
    cfg = model.config
    input_ids = encode_prompt(tokenizer, prompt, cfg.max_seq_len, cfg.bos_token_id, device)
    prompt_len = input_ids.shape[1]
    t0 = time.time()
    out_ids = model.generate(
        input_ids,
        max_new_tokens=max_new,
        temperature=temperature if temperature > 0 else None,
        top_k=top_k,
        top_p=top_p,
        task_id=task_id,
    )
    elapsed = time.time() - t0
    new_ids = out_ids[:, prompt_len:]
    gen_tokens = new_ids.shape[1]
    new_text = decode_tokens(tokenizer, new_ids, skip_special=True)
    speed = gen_tokens / elapsed if elapsed > 0 else 0.0
    return new_text, speed


def interactive_mode(model, tokenizer, device, max_new, temperature, top_k, top_p):
    print()
    print("=" * 60)
    print("  AetherMind-Nano3 交互推理 (输入空行退出)")
    print("=" * 60)
    while True:
        try:
            prompt = input("\nPrompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[退出]")
            return
        if not prompt:
            return
        if prompt.startswith("/"):
            cmd = prompt[1:].strip().lower()
            if cmd == "exit" or cmd == "quit":
                return
            if cmd.startswith("temp"):
                try:
                    temperature = float(cmd.split()[1])
                    print(f"[设置] temperature={temperature}")
                except Exception:
                    print(f"当前 temperature={temperature}")
                continue
            if cmd.startswith("max"):
                try:
                    max_new = int(cmd.split()[1])
                    print(f"[设置] max_new_tokens={max_new}")
                except Exception:
                    print(f"当前 max_new_tokens={max_new}")
                continue
            if cmd.startswith("topk"):
                try:
                    top_k = int(cmd.split()[1])
                    print(f"[设置] top_k={top_k}")
                except Exception:
                    print(f"当前 top_k={top_k}")
                continue
            if cmd.startswith("topp"):
                try:
                    top_p = float(cmd.split()[1])
                    print(f"[设置] top_p={top_p}")
                except Exception:
                    print(f"当前 top_p={top_p}")
                continue
            print(f"未知命令: {cmd}. 支持 /temp /max /topk /topp /exit")
            continue
        print("=" * 50)
        try:
            text, speed = generate_once(model, tokenizer, prompt, max_new, temperature, top_k, top_p, device)
            print(f"Generated ({speed:.1f} tok/s):")
            print("-" * 40)
            print(text)
            print("-" * 40)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[生成失败] {e}")


def main():
    parser = argparse.ArgumentParser(description="AetherMind-Nano3 文本生成推理")
    parser.add_argument("--ckpt", type=str,
                        default="d:/AetherMind-Nano3/checkpoints/checkpoint_25000_final.pt",
                        help="检查点路径")
    parser.add_argument("--prompt", type=str, default=None,
                        help="单次生成的 prompt (不指定则进入交互模式)")
    parser.add_argument("--max-new", type=int, default=256, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度 (0=使用模型自适应温度)")
    parser.add_argument("--top-k", type=int, default=50, help="top-k 采样")
    parser.add_argument("--top-p", type=float, default=0.9, help="top-p (nucleus) 采样")
    parser.add_argument("--task-id", type=int, default=0, help="task_id (0-15)")
    parser.add_argument("--n-samples", type=int, default=1, help="单次 prompt 生成样本数")
    args = parser.parse_args()

    print("=" * 60)
    print("  AetherMind-Nano3 文本生成推理")
    print("=" * 60)

    if not os.path.exists(args.ckpt):
        print(f"[Error] 检查点不存在: {args.ckpt}")
        sys.exit(1)

    ckpt, model_cfg, train_cfg = load_checkpoint(args.ckpt)
    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
    tokenizer = setup_tokenizer(model_cfg, train_cfg, ckpt_dir)
    model, device = build_model(model_cfg, ckpt["model_state"])

    if args.prompt:
        print(f"\n[Prompt] {args.prompt}")
        for i in range(args.n_samples):
            print(f"\n--- Sample {i+1}/{args.n_samples} ---")
            try:
                text, speed = generate_once(
                    model, tokenizer, args.prompt, args.max_new,
                    args.temperature, args.top_k, args.top_p, device, args.task_id
                )
                print(f"Generated ({speed:.1f} tok/s):\n{text}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[生成失败] {e}")
    else:
        interactive_mode(model, tokenizer, device, args.max_new, args.temperature, args.top_k, args.top_p)


if __name__ == "__main__":
    main()
