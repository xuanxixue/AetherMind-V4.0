import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import math
import time
import json
import random
import argparse
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

try:
    from tqdm import tqdm as _tqdm_base
    def _make_tqdm(iterable=None, **kw):
        return _tqdm_base(iterable, dynamic_ncols=True, mininterval=0.3, file=sys.stderr, **kw)
    TQDM_AVAILABLE = True
except ImportError:
    def _make_tqdm(iterable=None, **kw):
        return iterable
    TQDM_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.aethermind36_config import AetherMind36Config, TrainingConfig
from src.model.aethermind36 import AetherMind36
from src.data.dataset import StreamingTextDataset, build_tokenizer


class Trainer:
    def __init__(self, model_config: AetherMind36Config, train_config: TrainingConfig):
        self.model_cfg = model_config
        self.train_cfg = train_config
        self.device = torch.device(model_config.device)
        self.steps = 0
        self.phase = "A"
        self.phase_progress = 0.0

    def setup(self):
        print("[Setup] 加载分词器...")
        sys.stdout.flush()
        self.tokenizer = build_tokenizer(
            vocab_size=self.model_cfg.vocab_size,
            data_dir=self.train_cfg.data_dir
        )
        try:
            vs = self.tokenizer.vocab_size
        except Exception:
            try:
                vs = self.tokenizer.vocab_size if hasattr(self.tokenizer, "vocab_size") else len(self.tokenizer)
            except Exception:
                vs = 50000
        self.model_cfg.vocab_size = max(self.model_cfg.vocab_size, vs)
        print(f"[Setup] 词汇表大小: {self.model_cfg.vocab_size}")

        # Flush tokenizer loading
        sys.stdout.flush()

        if str(self.model_cfg.device).startswith("cuda") and not torch.cuda.is_available():
            sys.stderr.write(f"[Warn] config.device={self.model_cfg.device} 但 torch.cuda.is_available()=False，降级到 CPU\n")
            self.model_cfg.device = "cpu"
        if torch.cuda.is_available():
            try:
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
        self.device = torch.device(self.model_cfg.device)
        dev_name = "CPU"
        dev_mem_gb = 0
        if self.device.type == "cuda":
            idx = 0 if self.model_cfg.device_ids is None or len(self.model_cfg.device_ids) == 0 else self.model_cfg.device_ids[0]
            dev_name = torch.cuda.get_device_name(idx)
            dev_mem_gb = round(torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3), 2)
        print(f"[Setup] 训练设备: {self.device} | {dev_name} | 显存 {dev_mem_gb} GB")

        print("[Setup] 构建模型...")
        sys.stdout.flush()
        self.model = AetherMind36(self.model_cfg).to(self.device)
        if self.device.type == "cuda" and self.model_cfg.device_ids and len(self.model_cfg.device_ids) > 1:
            try:
                self.model = torch.nn.DataParallel(self.model, device_ids=self.model_cfg.device_ids)
                print(f"[Setup] 启用 DataParallel GPU: {self.model_cfg.device_ids}")
            except Exception as e:
                sys.stderr.write(f"[Warn] DataParallel 启用失败，继续单卡训练: {e}\n")
        params = self.model.module.count_params() if hasattr(self.model, "module") else self.model.count_params()
        size_mb = params * 4 / 1024 / 1024
        print(f"[Setup] 参数量: {params:,} ({size_mb:.1f} MB)")

        print("[Setup] 构建数据集...")
        sys.stdout.flush()
        train_ds = StreamingTextDataset(
            self.train_cfg.data_dir,
            self.tokenizer,
            max_seq_len=self.model_cfg.max_seq_len,
            max_samples=self.train_cfg.max_train_samples,
            shuffle=True,
            split="train",
            split_ratio=self.train_cfg.train_ratio
        )
        print("[Setup] 训练数据集已构建（流式，按需加载）")
        sys.stdout.flush()
        eval_ds = StreamingTextDataset(
            self.train_cfg.data_dir,
            self.tokenizer,
            max_seq_len=self.model_cfg.max_seq_len,
            max_samples=self.train_cfg.max_eval_samples,
            shuffle=False,
            split="eval",
            split_ratio=self.train_cfg.train_ratio
        )
        print("[Setup] 验证数据集已构建")

        pin_mem = (self.device.type == "cuda")
        self.train_loader = DataLoader(
            train_ds, batch_size=self.train_cfg.batch_size,
            num_workers=0, pin_memory=pin_mem
        )
        self.eval_loader = DataLoader(
            eval_ds, batch_size=self.train_cfg.batch_size,
            num_workers=0, pin_memory=pin_mem
        )

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
            betas=(0.9, 0.95)
        )

        warmup = LinearLR(self.optimizer, start_factor=1e-4, end_factor=1.0, total_iters=self.train_cfg.warmup_steps)
        anneal_steps = self.train_cfg.total_steps - self.train_cfg.warmup_steps
        anneal = CosineAnnealingLR(self.optimizer, T_max=anneal_steps, eta_min=self.train_cfg.min_lr)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup, anneal], milestones=[self.train_cfg.warmup_steps])

        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))

        self.output_dir = self.train_cfg.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        print("[Setup] 完成")
        sys.stdout.flush()

    def _set_phase(self):
        steps = self.steps
        if steps < self.train_cfg.phase_A_steps:
            self.phase = "A"
            self.phase_progress = steps / max(1, self.train_cfg.phase_A_steps)
            self.model_cfg.set_phase_A()
        elif steps < self.train_cfg.phase_A_steps + self.train_cfg.phase_B_steps:
            self.phase = "B"
            self.phase_progress = (steps - self.train_cfg.phase_A_steps) / max(1, self.train_cfg.phase_B_steps)
            self.model_cfg.set_phase_B(self.phase_progress)
        else:
            self.phase = "C"
            base = self.train_cfg.phase_A_steps + self.train_cfg.phase_B_steps
            self.phase_progress = (steps - base) / max(1, self.train_cfg.phase_C_steps)
            self.model_cfg.set_phase_C(self.phase_progress)

    def _train_step(self, batch):
        input_ids = batch["input_ids"].to(self.device, non_blocking=(self.device.type == "cuda"))
        labels = batch["labels"].to(self.device, non_blocking=(self.device.type == "cuda"))

        with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda"), dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32):
            out = self.model(input_ids, labels, task_id=random.randint(0, 15), t=self.steps)
            loss = out["loss"] / self.train_cfg.gradient_accumulation_steps

        self.scaler.scale(loss).backward()
        scalar_only = {}
        for k, v in out.items():
            if k in {"logits", "probs", "Z_cog", "p_copy", "last_hidden"}:
                continue
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    scalar_only[k] = float(v.detach().cpu().item())
            elif isinstance(v, (int, float)):
                scalar_only[k] = float(v)
        return scalar_only

    def _step_optim(self):
        if self.steps % self.train_cfg.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        losses = []
        lm_losses = []
        count = 0
        for batch in self.eval_loader:
            if count >= 50:
                break
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            out = self.model(input_ids, labels)
            losses.append(out["loss"].item())
            lm_losses.append(out["loss_LM"].item())
            count += 1
            if count % 10 == 0 or count == 50:
                try:
                    total_batches = len(self.eval_loader)
                except (TypeError, AttributeError):
                    total_batches = 50
                print(f"  [Eval] {count}/{min(50, total_batches)} 批次完成", flush=True)
        self.model.train()
        return {
            "eval_loss": sum(losses) / max(1, len(losses)),
            "eval_lm_loss": sum(lm_losses) / max(1, len(lm_losses)),
        }

    def save(self, suffix: str = ""):
        path = os.path.join(self.output_dir, f"checkpoint_{self.steps}{suffix}.pt")
        torch.save({
            "steps": self.steps,
            "phase": self.phase,
            "phase_progress": self.phase_progress,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "model_config": self.model_cfg,
            "train_config": self.train_cfg,
        }, path)
        print(f"[Save] 保存到: {path}")

        # 同时保存分词器到 output_dir
        self._save_tokenizer()

    def _save_tokenizer(self):
        try:
            tok = self.tokenizer
            # 解包 _Wrapper 获取底层 tokenizer
            if hasattr(tok, 'inner'):
                tok = tok.inner
            tok_path = os.path.join(self.output_dir, "tokenizer.json")
            if hasattr(tok, 'save'):
                tok.save(tok_path)
                print(f"[Save] 分词器保存到: {tok_path}")
            elif hasattr(tok, 'save_pretrained'):
                tok.save_pretrained(self.output_dir)
                print(f"[Save] 分词器保存到: {self.output_dir}")
        except Exception as e:
            print(f"[Save] 分词器保存失败 (推理时可用 --data_dir 重建): {e}")

    def train(self):
        self.setup()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total = self.train_cfg.total_steps
        accum_stats = {}
        t_start = time.time()
        last_loss = None

        print(f"[Train] 开始训练, 总步数: {total}")
        sys.stdout.flush()

        epoch_done = False
        while self.steps < total:
            for batch in self.train_loader:
                if self.steps >= total:
                    epoch_done = True
                    break

                self._set_phase()
                stats = self._train_step(batch)
                for k, v in stats.items():
                    accum_stats[k] = accum_stats.get(k, 0.0) + v
                self.steps += 1
                self._step_optim()

                loss_val = stats.get("loss", stats.get("loss_LM", None))
                if loss_val is not None:
                    last_loss = loss_val
                lr = self.scheduler.get_last_lr()[0]

                if self.steps % self.train_cfg.log_interval == 0:
                    elapsed = time.time() - t_start
                    eta = (elapsed / self.steps) * (total - self.steps) if self.steps > 0 else 0
                    pct = self.steps / total * 100
                    n = min(self.train_cfg.log_interval, self.steps)
                    bar_w = 25
                    filled = int(bar_w * self.steps / total)
                    bar = "#" * filled + "-" * (bar_w - filled)
                    eta_str = f"{eta/60:.0f}min" if eta < 7200 else f"{eta/3600:.1f}h"
                    lr_str = f"{lr:.2e}"
                    loss_str = f"{last_loss:.4f}" if last_loss is not None else "---"
                    print(f"\r[{bar}] {pct:5.1f}% ({self.steps:>5}/{total}) | phase={self.phase} | loss={loss_str} | lr={lr_str} | ETA {eta_str:>5}", end="")
                    sys.stdout.flush()
                    accum_stats = {}

                if self.steps % self.train_cfg.eval_interval == 0:
                    print(f"[Eval] 评估中... step={self.steps}")
                    sys.stdout.flush()
                    eval_s = self.evaluate()
                    print("[Eval] step={} | {}".format(
                        self.steps,
                        " | ".join(f"{k}={v:.4f}" for k, v in eval_s.items())
                    ))
                    sys.stdout.flush()

                if self.steps % self.train_cfg.save_interval == 0:
                    self.save()

            if epoch_done:
                break

        elapsed_total = time.time() - t_start
        print(f"[Train] 训练完成! 总耗时: {elapsed_total/60:.1f}min ({elapsed_total:.1f}s)")
        self.save("_final")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="F:/数据集")
    parser.add_argument("--output_dir", type=str, default="d:/AetherMind-Nano3/checkpoints")
    parser.add_argument("--phase_A", type=int, default=50000)
    parser.add_argument("--phase_B", type=int, default=100000)
    parser.add_argument("--phase_C", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--max_seq", type=int, default=1024)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=200)
    args = parser.parse_args()

    model_cfg = AetherMind36Config(
        vocab_size=50000,
        d_model=args.d_model,
        d_ff=args.d_model * 4,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq,
    )
    train_cfg = TrainingConfig(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        phase_A_steps=args.phase_A,
        phase_B_steps=args.phase_B,
        phase_C_steps=args.phase_C,
        total_steps=args.phase_A + args.phase_B + args.phase_C,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
    )
    if args.log_interval is not None:
        train_cfg.log_interval = args.log_interval
    if args.eval_interval is not None:
        train_cfg.eval_interval = args.eval_interval
    if args.save_interval is not None:
        train_cfg.save_interval = args.save_interval
    if args.max_train_samples is not None:
        train_cfg.max_train_samples = args.max_train_samples
    if args.max_eval_samples is not None:
        train_cfg.max_eval_samples = args.max_eval_samples

    trainer = Trainer(model_cfg, train_cfg)
    trainer.train()


if __name__ == "__main__":
    main()
