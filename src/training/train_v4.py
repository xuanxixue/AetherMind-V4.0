"""
AetherMind V4 训练器 — 息壤·双权重演化认知体
=============================================
三阶段训练：
  Phase A (纯BP):  λ_phys=0, 信息素不沉积, 高温探索, 获得基础语言能力
  Phase B (混合):  λ渐入, 信息素开始沉积(loss改进量奖励), 物理层激活
  Phase C (演化):  全激活, 自由能下降奖励, 降温收敛, 定期LTP固化
  Phase D (固化):  冻结W, 纯前向演化+分位数固化 — 从final checkpoint
                   加载权重专训LTP固化, 无backward, 速度约为训练3倍

断点续训：自动扫描 output_dir 中最新 v4_checkpoint_*.pt 恢复；
         也可用 --resume <path> 指定；--fresh 强制从头开始。
         Phase D模式下优先恢复 *_phaseD.pt 进度, 否则从 --resume 的
         base checkpoint 载入权重, 步数归零。
"""

import os
import sys
import gc
import glob
import time
import math
import random
import argparse

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.aethermind4_config import AetherMind4Config, TrainingConfig
from src.model.aethermind4 import AetherMind4
from src.data.dataset import StreamingTextDataset, build_tokenizer
from src.utils.gpu_guard import pre_train_gpu_cleanup, emergency_free_gpu_memory, GPUMemoryWatchdog
from src.validation.hooks import ValidationHooks


class TrainerV4:
    def __init__(self, model_config, train_config, resume_path=None, fresh=False,
                 phase_d=False, phase_e=False, phase_f=False, phase_g=False):
        self.model_cfg = model_config
        self.train_cfg = train_config
        self.device = torch.device(model_config.device)
        self.steps = 0
        self.phase = "A"
        self.phase_progress = 0.0
        self.fresh = fresh
        self.resume_path = resume_path
        self.phase_d = phase_d
        self.phase_e = phase_e
        self.phase_f = phase_f
        self.phase_g = phase_g
        self.bad_batch_count = 0
        self.start_time = None

    # ------------------------------------------------------------
    # 数据自检：启动训练前抽样验证一个batch
    # ------------------------------------------------------------
    def _data_selfcheck(self, loader):
        print("[SelfCheck] 数据自检: 抽样1个batch验证形状...", flush=True)
        try:
            batch = next(iter(loader))
            ids = batch["input_ids"]
            lbl = batch["labels"]
            assert ids.dim() == 2, f"input_ids应为2D(B,S), 实际{ids.shape}"
            assert lbl.dim() == 2, f"labels应为2D(B,S), 实际{lbl.shape}"
            assert ids.shape == lbl.shape, f"ids/labels形状不一致: {ids.shape} vs {lbl.shape}"
            assert ids.shape[1] == self.model_cfg.max_seq_len, \
                f"seq_len={ids.shape[1]}, 期望{self.model_cfg.max_seq_len}"
            print(f"[SelfCheck] 通过: input_ids={tuple(ids.shape)}, labels={tuple(lbl.shape)}, "
                  f"vocab_range=[{ids.min().item()},{ids.max().item()}]", flush=True)
        except StopIteration:
            print("[SelfCheck] WARN: DataLoader为空, 继续但可能无数据", flush=True)
        except Exception as e:
            print(f"[SelfCheck] FAIL: {e}", flush=True)
            raise

    # ------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------
    def setup(self):
        # GPU显存守护：训练前自动清理占用GPU的非必要进程
        print("[Setup] GPU显存守护：训练前清理...", flush=True)
        pre_train_gpu_cleanup(auto_kill=True)
        print(flush=True)
        
        print("[Setup] 加载分词器...", flush=True)
        self.tokenizer = build_tokenizer(vocab_size=self.model_cfg.vocab_size,
                                         data_dir=self.train_cfg.data_dir)
        try:
            vs = self.tokenizer.vocab_size
        except Exception:
            vs = self.model_cfg.vocab_size
        # 使用tokenizer的实际词表大小（对齐到128倍数）
        vs_aligned = ((vs + 127) // 128) * 128
        self.model_cfg.vocab_size = max(self.model_cfg.vocab_size, vs_aligned)
        # 更新特殊token ID
        if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
            self.model_cfg.pad_token_id = self.tokenizer.pad_token_id
        if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
            self.model_cfg.eos_token_id = self.tokenizer.eos_token_id
            self.model_cfg.bos_token_id = self.tokenizer.eos_token_id
        print(f"[Setup] 词汇表大小: {self.model_cfg.vocab_size}", flush=True)

        # 注意: torch.cuda.is_available() 在 CUDA_VISIBLE_DEVICES="" 时仍返回True但device_count()==0
        # 必须同时检查device_count, 否则get_device_name(0)会抛 Invalid device id
        if str(self.model_cfg.device).startswith("cuda") and (not torch.cuda.is_available() or torch.cuda.device_count() == 0):
            self.model_cfg.device = "cpu"
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.device = torch.device(self.model_cfg.device)
        dev_name, dev_mem = "CPU", 0
        if self.device.type == "cuda":
            idx = 0
            dev_name = torch.cuda.get_device_name(idx)
            dev_mem = round(torch.cuda.get_device_properties(idx).total_memory / (1024**3), 1)
        print(f"[Setup] 设备: {self.device} | {dev_name} | {dev_mem} GB", flush=True)

        print("[Setup] 构建V4模型 (双权重演化架构)...", flush=True)
        self.model = AetherMind4(self.model_cfg).to(self.device)
        if self.device.type == "cuda" and self.model_cfg.device_ids and len(self.model_cfg.device_ids) > 1:
            try:
                self.model = nn.DataParallel(self.model, device_ids=self.model_cfg.device_ids)
            except Exception:
                pass
        params = self.model.module.count_params() if hasattr(self.model, "module") else self.model.count_params()
        print(f"[Setup] 参数量: {params:,} ({params*4/1024/1024:.1f} MB)", flush=True)

        print("[Setup] 构建数据集...", flush=True)
        # Phase G 使用清洗后的对话数据目录
        _data_dir = self.train_cfg.phaseG_data_dir if self.phase_g else self.train_cfg.data_dir
        train_ds = StreamingTextDataset(_data_dir, self.tokenizer,
                                        max_seq_len=self.model_cfg.max_seq_len,
                                        max_samples=self.train_cfg.max_train_samples,
                                        shuffle=True, split="train", split_ratio=self.train_cfg.train_ratio)
        eval_ds = StreamingTextDataset(_data_dir, self.tokenizer,
                                       max_seq_len=self.model_cfg.max_seq_len,
                                       max_samples=self.train_cfg.max_eval_samples,
                                       shuffle=False, split="eval", split_ratio=self.train_cfg.train_ratio)
        pin = self.device.type == "cuda"
        self.train_loader = DataLoader(train_ds, batch_size=self.train_cfg.batch_size,
                                       num_workers=0, pin_memory=pin, drop_last=True)
        self.eval_loader = DataLoader(eval_ds, batch_size=self.train_cfg.batch_size,
                                      num_workers=0, pin_memory=pin, drop_last=False)

        self._data_selfcheck(self.train_loader)

        self.optimizer = AdamW(self.model.parameters(), lr=self.train_cfg.learning_rate,
                               weight_decay=self.train_cfg.weight_decay, betas=(0.9, 0.95))
        wu = LinearLR(self.optimizer, 1e-4, 1.0, self.train_cfg.warmup_steps)
        ann = CosineAnnealingLR(self.optimizer, self.train_cfg.total_steps - self.train_cfg.warmup_steps,
                                self.train_cfg.min_lr)
        self.scheduler = SequentialLR(self.optimizer, [wu, ann], [self.train_cfg.warmup_steps])
        # bfloat16 不需要 GradScaler（动态范围足够，启用会导致scale无限增长+静默跳步）
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)

        os.makedirs(self.train_cfg.output_dir, exist_ok=True)

        # 断点续训：自动找最新或指定路径
        self._maybe_resume()

        # 验证钩子（三层验证结构）
        val_log_dir = os.path.join(self.train_cfg.output_dir, "validation_logs")
        self.val_hooks = ValidationHooks(
            log_dir=val_log_dir, model=self._get_model(),
            tokenizer=self.tokenizer, check_interval=500,
            migration_interval=5000, device=str(self.device),
        )
        print(f"[Setup] 验证钩子已加载: 日志->{val_log_dir}", flush=True)

        # 显存守护线程（训练进程内OOM预防）
        if self.device.type == "cuda":
            self.gpu_watchdog = GPUMemoryWatchdog(
                threshold=0.95, critical=0.98, interval=20
            )
            self.gpu_watchdog.start()
        else:
            self.gpu_watchdog = None

        print("[Setup] 完成\n", flush=True)

    def _get_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    # ------------------------------------------------------------
    # 断点续训
    # ------------------------------------------------------------
    def _find_latest_checkpoint(self, phase_d_only=False, exclude_phase_d=True,
                                phase_e_only=False, phase_f_only=False, phase_g_only=False):
        pattern = os.path.join(self.train_cfg.output_dir, "v4_checkpoint_*.pt")
        ckpts = glob.glob(pattern)
        if phase_d_only:
            ckpts = [p for p in ckpts if "_phaseD" in os.path.basename(p)]
        elif phase_e_only:
            ckpts = [p for p in ckpts if "_phaseE" in os.path.basename(p)]
        elif phase_f_only:
            ckpts = [p for p in ckpts if "_phaseF" in os.path.basename(p)]
        elif phase_g_only:
            ckpts = [p for p in ckpts if "_phaseG" in os.path.basename(p)]
        elif exclude_phase_d:
            ckpts = [p for p in ckpts if "_phaseD" not in os.path.basename(p)
                     and "_phaseE" not in os.path.basename(p)
                     and "_phaseF" not in os.path.basename(p)
                     and "_phaseG" not in os.path.basename(p)]
        if not ckpts:
            return None
        # 按步数排序（文件名含步数）
        def _step(p):
            try:
                name = os.path.basename(p)
                num = name.replace("v4_checkpoint_", "").replace(".pt", "").replace("_final", "")
                for tag in ("_phaseD", "_phaseE", "_phaseF", "_phaseG"):
                    num = num.replace(tag, "")
                return int(num) if num.isdigit() else -1
            except Exception:
                return -1
        ckpts.sort(key=_step)
        return ckpts[-1] if ckpts else None

    def _maybe_resume(self):
        if self.fresh:
            print("[Resume] --fresh 指定: 从头开始训练", flush=True)
            return
        # Phase F: 优先恢复自身进度(*_phaseF.pt), 否则载入Phase E产出
        if self.phase_f:
            f_path = self._find_latest_checkpoint(phase_f_only=True)
            if f_path is not None:
                print(f"[Phase F] 恢复对齐专训进度: {f_path}", flush=True)
                if self._load_checkpoint(f_path, reset_steps=False, load_optim=False):
                    return
            base = self.resume_path or self._find_latest_checkpoint(phase_e_only=True) \
                   or self._find_latest_checkpoint(exclude_phase_d=True)
            if base is None or not os.path.exists(base):
                print("[Phase F] ERROR: 未找到base checkpoint(先跑Phase E, 或用 --resume 指定), 退出", flush=True)
                raise FileNotFoundError("Phase F requires Phase E output (--resume)")
            print(f"[Phase F] 从base checkpoint载入权重(步数归零): {base}", flush=True)
            self._load_checkpoint(base, reset_steps=True, load_optim=False)
            return
        # Phase G: 优先恢复自身进度(*_phaseG.pt), 否则载入PhaseF/PhaseE/base权重
        if self.phase_g:
            g_path = self._find_latest_checkpoint(phase_g_only=True)
            if g_path is not None:
                print(f"[Phase G] 恢复对话SFT专训进度: {g_path}", flush=True)
                if self._load_checkpoint(g_path, reset_steps=False, load_optim=False):
                    return
            base = self.resume_path or self._find_latest_checkpoint(phase_f_only=True) \
                   or self._find_latest_checkpoint(phase_e_only=True) \
                   or self._find_latest_checkpoint(exclude_phase_d=True)
            if base is None or not os.path.exists(base):
                print("[Phase G] ERROR: 未找到base checkpoint(先跑PhaseF/PhaseE, 或用 --resume 指定), 退出", flush=True)
                raise FileNotFoundError("Phase G requires a base checkpoint (--resume)")
            print(f"[Phase G] 从base checkpoint载入权重(步数归零): {base}", flush=True)
            self._load_checkpoint(base, reset_steps=True, load_optim=False)
            return
        # Phase E: 优先恢复自身进度(*_phaseE.pt), 否则载入Phase D/base权重
        if self.phase_e:
            e_path = self._find_latest_checkpoint(phase_e_only=True)
            if e_path is not None:
                print(f"[Phase E] 恢复SFT专训进度: {e_path}", flush=True)
                if self._load_checkpoint(e_path, reset_steps=False, load_optim=False):
                    return
            base = self.resume_path or self._find_latest_checkpoint(phase_d_only=True) \
                   or self._find_latest_checkpoint(exclude_phase_d=True)
            if base is None or not os.path.exists(base):
                print("[Phase E] ERROR: 未找到base checkpoint(用 --resume 指定PhaseD/Final路径), 退出", flush=True)
                raise FileNotFoundError("Phase E requires a base checkpoint (--resume)")
            print(f"[Phase E] 从base checkpoint载入权重(步数归零): {base}", flush=True)
            self._load_checkpoint(base, reset_steps=True, load_optim=False)
            return
        # Phase D: 优先恢复自身进度(*_phaseD.pt), 否则载入base权重且步数归零
        if self.phase_d:
            d_path = self._find_latest_checkpoint(phase_d_only=True)
            if d_path is not None:
                path = d_path
                print(f"[Phase D] 恢固化专训进度: {path}", flush=True)
                if self._load_checkpoint(path, reset_steps=False):
                    return
            base = self.resume_path or self._find_latest_checkpoint(exclude_phase_d=True)
            if base is None or not os.path.exists(base):
                print("[Phase D] ERROR: 未找到base checkpoint(用 --resume 指定final路径), 退出", flush=True)
                raise FileNotFoundError("Phase D requires a base checkpoint (--resume)")
            print(f"[Phase D] 从base checkpoint载入权重(步数归零): {base}", flush=True)
            self._load_checkpoint(base, reset_steps=True, load_optim=False)
            return
        path = self.resume_path
        if path is None:
            path = self._find_latest_checkpoint()
        if path is None or not os.path.exists(path):
            print("[Resume] 未找到checkpoint, 从头开始训练", flush=True)
            return
        print(f"[Resume] 从checkpoint恢复: {path}", flush=True)
        if not self._load_checkpoint(path, reset_steps=False):
            print("[Resume] WARN: 恢复失败, 从头开始", flush=True)
            self.steps = 0
            self.phase = "A"

    def _load_checkpoint(self, path, reset_steps=False, load_optim=True):
        """加载checkpoint到self. 返回True/False"""
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            model = self._get_model()
            model.load_state_dict(ckpt["model_state"], strict=False)
            if load_optim:
                self.optimizer.load_state_dict(ckpt["optimizer_state"])
                self.scheduler.load_state_dict(ckpt["scheduler_state"])
                if "scaler_state" in ckpt and ckpt["scaler_state"] is not None:
                    self.scaler.load_state_dict(ckpt["scaler_state"])
            if reset_steps:
                self.steps = 0
                if self.phase_d:
                    self.phase = "D"
                elif self.phase_e:
                    self.phase = "E"
                elif self.phase_f:
                    self.phase = "F"
                elif self.phase_g:
                    self.phase = "G"
                else:
                    self.phase = "A"
            else:
                self.steps = int(ckpt.get("steps", 0))
                self.phase = ckpt.get("phase", "A")
            # 恢复evolver内部状态（Python属性不在state_dict里）
            evo = model.evolver
            if "evo_F_prev" in ckpt:
                with torch.no_grad():
                    evo._F_prev.fill_(float(ckpt["evo_F_prev"]))
            evo._F_initialized = bool(ckpt.get("evo_F_init", False))
            evo._loss_initialized = bool(ckpt.get("evo_loss_init", False))
            if "evo_loss_prev" in ckpt and ckpt["evo_loss_prev"] is not None:
                evo._loss_prev = torch.tensor(float(ckpt["evo_loss_prev"]), device=self.device)
            evo._global_step = self.steps
            print(f"[Resume] 恢复成功: step={self.steps}, phase={self.phase}", flush=True)
            return True
        except Exception as e:
            print(f"[Resume] WARN: 加载失败({e})", flush=True)
            return False

    # ------------------------------------------------------------
    # 阶段调度
    # ------------------------------------------------------------
    def _set_phase(self):
        # Phase G: 纯净对话SFT, 关闭演化/固化
        if self.phase_g:
            self.phase = "G"
            prog = min(1.0, self.steps / max(1, self.train_cfg.phase_G_steps))
            self.model_cfg.set_phase_G(prog)
            self._get_model().apply_pheromone_config()
            return
        # Phase E: 语言SFT, 关闭演化/固化
        if self.phase_e:
            self.phase = "E"
            prog = min(1.0, self.steps / max(1, self.train_cfg.phase_E_steps))
            self.model_cfg.set_phase_E(prog)
            self._get_model().apply_pheromone_config()
            return
        # Phase F: KG对齐, 独立循环(不走模型forward), 仅同步config
        if self.phase_f:
            self.phase = "F"
            self.model_cfg.set_phase_F(min(1.0, self.steps / max(1, self.train_cfg.phase_F_steps)))
            self._get_model().apply_pheromone_config()
            return
        # Phase D: 固化专训, 永远保持phase=D并应用set_phase_D(强沉积/低蒸发/分位数固化)
        if self.phase_d:
            self.phase = "D"
            prog = min(1.0, self.steps / max(1, self.train_cfg.phase_D_steps))
            self.model_cfg.set_phase_D(prog)
            self._get_model().apply_pheromone_config()
            return
        s = self.steps
        A = self.train_cfg.phase_A_steps
        B = self.train_cfg.phase_B_steps
        if s < A:
            self.phase, self.phase_progress = "A", s / max(1, A)
            self.model_cfg.set_phase_A()
        elif s < A + B:
            self.phase, self.phase_progress = "B", (s - A) / max(1, B)
            self.model_cfg.set_phase_B(self.phase_progress)
        else:
            base = A + B
            self.phase, self.phase_progress = "C", (s - base) / max(1, self.train_cfg.phase_C_steps)
            self.model_cfg.set_phase_C(self.phase_progress)
        # 把阶段config(pheromone deposit/rho/阈值)同步到注意力层, 否则不会生效
        self._get_model().apply_pheromone_config()

    # ------------------------------------------------------------
    # 单步训练
    # ------------------------------------------------------------
    def _train_step(self, batch):
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        model = self._get_model()

        try:
            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda"),
                                    dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32):
                out = model(input_ids, labels, task_id=random.randint(0, 15),
                            t=float(self.steps), phase=self.phase)
                loss = out["loss"] / self.train_cfg.gradient_accumulation_steps

            # NaN/Inf保护 + grad_fn防御（forward的常数替换会丢grad_fn, 必须拦截, 否则backward崩）
            if loss.grad_fn is None or not torch.isfinite(loss).all():
                self.bad_batch_count += 1
                del out, loss, input_ids, labels
                return None

            # === 关键显存优化：backward前提取所有标量，删除所有大张量 ===
            # out中logits(1,512,151680)等大张量在backward期间不再需要
            scalars = {}
            for k, v in out.items():
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    scalars[k] = float(v.detach().cpu().item())
                elif isinstance(v, (int, float)):
                    scalars[k] = float(v)
            # 立即释放所有forward输出大张量，只保留loss（计算图根节点）
            del out, input_ids, labels

            # backward（此时out/logits等大张量已释放，显存峰值最小）
            loss.backward()  # GradScaler已禁用(bfloat16), 直接反传
            del loss
            
            return scalars
            
        except torch.cuda.OutOfMemoryError:
            # OOM自动恢复：紧急释放显存，清空梯度，跳过该batch
            print(f'\n[OOM] step {self.steps} CUDA OOM! 紧急释放显存...', flush=True)
            if hasattr(self, 'optimizer'):
                self.optimizer.zero_grad(set_to_none=True)
            try:
                del out, loss, input_ids, labels
            except UnboundLocalError:
                pass
            # GPU守护：紧急释放显存（包括结束其他GPU进程）
            freed = emergency_free_gpu_memory(threshold_mb=500)
            print(f'[OOM] 紧急释放完成，释放了~{freed}MB显存，继续训练...', flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self.bad_batch_count += 1
            return None

    # ------------------------------------------------------------
    # 优化器步进
    # ------------------------------------------------------------
    def _step_optim(self):
        if self.steps % self.train_cfg.gradient_accumulation_steps == 0:
            try:
                self.scaler.unscale_(self.optimizer)
                # 诊断：记录裁剪前梯度范数（验证梯度是否流动）
                self._last_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            except torch.cuda.OutOfMemoryError:
                print(f'\n[OOM] optimizer.step() OOM, clearing cache...', flush=True)
                self.optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                return
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()  # 只在 optimizer.step() 后步进，修复 lr 调度 8x 过快问题

    # ------------------------------------------------------------
    # 评估
    # ------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self):
        model = self._get_model()
        was_training = model.training
        model.eval()
        
        # Bug1修复: 评估时临时切换到Phase A配置（关闭所有辅助损失）
        cfg = model.config
        saved_lambdas = {
            'lambda_phys': cfg.lambda_phys, 'lambda_PSR': cfg.lambda_PSR,
            'lambda_T': cfg.lambda_T, 'lambda_F': cfg.lambda_F,
            'lambda_copy_phys': cfg.lambda_copy_phys, 'lambda_phase': cfg.lambda_phase,
            'lambda_phi_decay': cfg.lambda_phi_decay, 'lambda_IB': cfg.lambda_IB,
            'lambda_D': cfg.lambda_D, 'lambda_aff': cfg.lambda_aff,
            'lambda_align': cfg.lambda_align,
        }
        saved_K = cfg.langevin_K
        saved_deposit = cfg.pheromone_deposit
        saved_rho = cfg.pheromone_rho
        cfg.set_phase_A()  # 切到纯LM模式
        
        losses, lm_losses = [], []
        try:
            for i, batch in enumerate(self.eval_loader):
                if i >= 50:
                    break
                ids = batch["input_ids"].to(self.device)
                lbl = batch["labels"].to(self.device)
                with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda"),
                                        dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32):
                    out = model(ids, lbl, phase="A")
                lv = float(out["loss"].item()) if torch.isfinite(out["loss"]).all() else None
                lm = float(out["loss_LM"].item()) if torch.isfinite(out["loss_LM"]).all() else None
                if lv is not None:
                    losses.append(lv)
                if lm is not None:
                    lm_losses.append(lm)
                if (i+1) % 10 == 0:
                    print(f"  [Eval] {i+1}/50 批次", flush=True)
        finally:
            # 恢复原配置
            for k, v in saved_lambdas.items():
                setattr(cfg, k, v)
            cfg.langevin_K = saved_K
            cfg.pheromone_deposit = saved_deposit
            cfg.pheromone_rho = saved_rho
        
        if was_training:
            model.train()
        if not losses:
            return {"eval_loss": float("inf"), "eval_lm_loss": float("inf")}
        return {"eval_loss": sum(losses)/len(losses), "eval_lm_loss": sum(lm_losses)/max(1,len(lm_losses))}

    # ------------------------------------------------------------
    # 信息素演化步进
    # ------------------------------------------------------------
    def _evolution_step(self, scalars):
        model = self._get_model()
        # Phase A/E/F/G: 不演化 (E/G=SFT专注CE, F=KG对齐专注关联矩阵)
        if self.phase in ("A", "E", "F", "G"):
            return
        # Phase B/C/D: 传入相应奖励信号（evolver内部用dF或dloss）
        loss_val = None
        free_energy_val = None
        if self.phase in ("C", "D") and "F" in scalars:
            free_energy_val = torch.tensor(scalars["F"], device=self.device)
        elif "loss" in scalars:
            loss_val = torch.tensor(scalars["loss"], device=self.device)
        model.evolution_step(self.steps, phase=self.phase, loss_val=loss_val,
                             free_energy=free_energy_val)

    # ------------------------------------------------------------
    # 保存checkpoint
    # ------------------------------------------------------------
    def save(self, suffix=""):
        model = self._get_model()
        path = os.path.join(self.train_cfg.output_dir, f"v4_checkpoint_{self.steps}{suffix}.pt")
        evo = model.evolver
        evo_stats = model.get_evolution_stats()
        torch.save({
            "steps": self.steps,
            "phase": self.phase,
            "phase_progress": self.phase_progress,
            "model_state": model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict() if self.device.type == "cuda" else None,
            "model_config": self.model_cfg,
            "train_config": self.train_cfg,
            "evolution_stats": evo_stats,
            "evo_F_prev": float(evo._F_prev.detach().cpu().item()),
            "evo_F_init": bool(evo._F_initialized),
            "evo_loss_init": bool(evo._loss_initialized),
            "evo_loss_prev": float(evo._loss_prev.detach().cpu().item())
                             if evo._loss_prev is not None and torch.is_tensor(evo._loss_prev) else
                             (float(evo._loss_prev) if evo._loss_prev is not None else None),
            "bad_batch_count": self.bad_batch_count,
        }, path)
        print(f"\n[Save] {path} | tau_conc={evo_stats.get('tau_concentration',0):.2f} "
              f"cons_mass={evo_stats.get('consolidation_mass',0):.1f}", flush=True)
        # 只保留最新5个checkpoint，防止磁盘空间不足导致写入失败
        self._prune_old_checkpoints(keep=5)

    def _prune_old_checkpoints(self, keep=5):
        """删除旧checkpoint，只保留最新的keep个（按文件修改时间排序）。

        注意：
        1. 不能按文件名里的步数数字排序，因为 Phase D/E/F/G 训练时步数会
           归零重置，D 轮的步数(0/50/100...)远小于 A-C 轮的步数(几千上万)，
           按步数排序会把当前轮次误判为"旧"而删掉自己。故改用文件修改时间判断新旧。
        2. 各阶段最终产物（*_final.pt，含 *_phaseD_final.pt 等双后缀）必须
           永久保留，不参与剪枝——这些是推理/下一阶段要用的最终权重。
        """
        try:
            import re
            out_dir = self.train_cfg.output_dir
            if not os.path.isdir(out_dir):
                return
            pts = []
            for fn in os.listdir(out_dir):
                m = re.match(r'v4_checkpoint_(\d+)(?:_\w+)?\.pt$', fn)
                if not m:
                    continue
                # final 文件（各阶段最终产物）永久保留，不参与剪枝
                if fn.endswith("_final.pt"):
                    continue
                fp = os.path.join(out_dir, fn)
                pts.append((os.path.getmtime(fp), fp))
            pts.sort(key=lambda x: x[0])  # 按修改时间升序，最旧的在前
            if len(pts) > keep:
                for _mtime, fp in pts[:-keep]:
                    try:
                        os.remove(fp)
                        print(f"[Prune] 删除旧checkpoint: {os.path.basename(fp)}", flush=True)
                    except OSError as e:
                        print(f"[Prune] 删除失败 {os.path.basename(fp)}: {e}", flush=True)
        except Exception as e:
            print(f"[Prune] 清理checkpoint异常: {e}", flush=True)

    # ------------------------------------------------------------
    # Phase D 单步（冻结W, 仅前向+演化+固化, 不反向传播）
    # ------------------------------------------------------------
    def _phaseD_step(self, batch):
        """与_train_step类似但完全no_grad：冻结W, 只为演化系统提供自由能奖励信号。
        labels=None 以跳过cross_entropy(省显存), F(自由能)与演化仍照常计算。"""
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        model = self._get_model()
        with torch.no_grad():
            out = model(input_ids, None, task_id=random.randint(0, 15),
                        t=float(self.steps), phase="D")
        scalars = {}
        for k, v in out.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                scalars[k] = float(v.detach().cpu().item())
            elif isinstance(v, (int, float)):
                scalars[k] = float(v)
        del out, input_ids
        return scalars

    # ------------------------------------------------------------
    # Phase D 固化专训主循环
    # ------------------------------------------------------------
    def _train_phaseD(self):
        self.model.eval()  # 冻结W, 不训练主干
        total = self.train_cfg.phase_D_steps
        self.start_time = time.time()
        print(f"\n[Phase D] 固化(LTP)专训 开始, 总步数: {total}")
        print(f"[Phase D] base权重已冻结, 仅演化τ信息素网络 + 分位数固化写入consolidated", flush=True)
        print(f"[Phase D] 从 {self.resume_path or 'auto(最新非D checkpoint)'} 加载", flush=True)

        done = False
        while self.steps < total:
            for batch in self.train_loader:
                if self.steps >= total:
                    done = True
                    break
                self._set_phase()  # phase=D + set_phase_D
                stats = self._phaseD_step(batch)
                if stats is None:
                    continue
                self.steps += 1
                # 演化：沉积τ + 定期分位数固化(consolidate_all内部按consolidate_interval触发)
                self._evolution_step(stats)

                if self.steps % self.train_cfg.log_interval == 0:
                    el = time.time() - self.start_time
                    sps = self.steps / el if el > 0 else 0
                    evo = self._get_model().get_evolution_stats()
                    print(f"[Phase D] {self.steps}/{total} "
                          f"tau={evo.get('tau_concentration', 0):.1f} "
                          f"cons={evo.get('consolidation_mass', 0):.1f} "
                          f"rounds={evo.get('consolidation_rounds', 0)} "
                          f"| {sps:.1f} step/s", flush=True)

                if self.steps % self.train_cfg.save_interval == 0:
                    self.save("_phaseD")

            if done:
                break

        el = time.time() - self.start_time
        evo = self._get_model().get_evolution_stats()
        print(f"\n[Phase D] 完成! 耗时 {el/60:.1f}min | "
              f"cons_mass={evo.get('consolidation_mass', 0):.1f} rounds={evo.get('consolidation_rounds', 0)}",
              flush=True)
        if hasattr(self, 'val_hooks'):
            _report = self.val_hooks.on_phase_end("D", self.steps, self._get_model(), self.tokenizer)
            print(f"[VAL] phaseD report: {_report.get('report_path', 'N/A')}", flush=True)
        self.save("_phaseD_final")

    # ------------------------------------------------------------
    # Phase E 单步（SFT: 标准CE + backward, 关闭演化）
    # ------------------------------------------------------------
    def _train_phaseE(self):
        """Phase E: 语言组织能力SFT。低lr解冻尾部(decoder后N层+final_norm+lm_head),
        冻结其余(embedding/encoder/physics/memory/metacog/dual_domain/前几层decoder)。
        数据 = MOSS + 沐雪(递归发现)。目标是"学会说话", 不接KG。"""
        total = self.train_cfg.phase_E_steps
        model = self._get_model()
        # 1) 冻结全部
        for p in model.parameters():
            p.requires_grad = False
        # 2) 解冻尾部
        unfreeze = getattr(self.train_cfg, "phaseE_unfreeze_layers", 2)
        n_dec = len(model.decoder_layers)
        for i, layer in enumerate(model.decoder_layers):
            if i >= n_dec - unfreeze:
                for p in layer.parameters():
                    p.requires_grad = True
        for p in model.final_norm.parameters():
            p.requires_grad = True
        for p in model.lm_head.parameters():
            p.requires_grad = True
        n_train = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"[Phase E] 解冻尾部{unfreeze}层decoder+final_norm+lm_head, 可训练模块数={n_train}", flush=True)

        # 3) 重建optimizer/scheduler/scaler（只含可训练参数）
        self.optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=self.train_cfg.phase_E_lr, weight_decay=self.train_cfg.weight_decay,
                               betas=(0.9, 0.95))
        self.scheduler = CosineAnnealingLR(self.optimizer, max(1, total), self.train_cfg.min_lr)
        # bfloat16 不需要 GradScaler（动态范围足够，启用会导致scale无限增长+静默跳步）
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self.start_time = time.time()

        print(f"\n[Phase E] 语言组织SFT 开始, 总步数: {total}, lr={self.train_cfg.phase_E_lr}", flush=True)
        print(f"[Phase E] 数据: {self.train_cfg.data_dir} (MOSS + Muice沐雪)", flush=True)
        print(f"[Phase E] base: {self.resume_path or 'auto(PhaseD优先, 否则Final)'}", flush=True)

        done = False
        while self.steps < total:
            for batch in self.train_loader:
                if self.steps >= total:
                    done = True
                    break
                self._set_phase()
                stats = self._train_step(batch)
                if stats is None:
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                self.steps += 1
                self._step_optim()

                if self.steps % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

                if self.steps % self.train_cfg.log_interval == 0:
                    el = time.time() - self.start_time
                    sps = self.steps / el if el > 0 else 0
                    lr = self.scheduler.get_last_lr()[0] if self.scheduler.get_last_lr() else self.train_cfg.phase_E_lr
                    loss = stats.get("loss", float("nan"))
                    print(f"[Phase E] {self.steps}/{total} loss={loss:.4f} lr={lr:.2e} "
                          f"| {sps:.1f} step/s bad={self.bad_batch_count}", flush=True)

                # 验证钩子
                if self.steps % 500 == 0 and hasattr(self, 'val_hooks'):
                    _vres = self.val_hooks.on_step(self.steps, loss, self._get_model(),
                                                   self.tokenizer, "E")
                    if "energy" in _vres and "dominance" in _vres["energy"]:
                        _er = _vres["energy"]
                        print(f"\n[VAL] physics={_er.get('var_physics_ratio',0):.3f} -> {_er['dominance']}", flush=True)

                if self.steps % self.train_cfg.save_interval == 0:
                    self.save("_phaseE")
            if done:
                break

        el = time.time() - self.start_time
        print(f"\n[Phase E] 完成! 耗时 {el/60:.1f}min | bad_batches={self.bad_batch_count}", flush=True)
        if hasattr(self, 'val_hooks'):
            _report = self.val_hooks.on_phase_end("E", self.steps, self._get_model(), self.tokenizer)
            print(f"[VAL] phaseE report: {_report.get('report_path', 'N/A')}", flush=True)
        self.save("_phaseE_final")

    # ------------------------------------------------------------
    # Phase G（纯净对话SFT: 清洗数据, 学到"正常对话", 关闭演化）
    # ------------------------------------------------------------
    def _train_phaseG(self):
        """Phase G: 用清洗后的 MOSS+沐雪数据做纯净对话 SFT。机制同 Phase E
        (低lr解冻尾部decoder+final_norm+lm_head), 差异在:
        - 数据目录 = phaseG_data_dir (已剥离 system prompt)
        - lr = phase_G_lr (可略高)
        - checkpoint 后缀 _phaseG, 与 Phase E 隔离"""
        total = self.train_cfg.phase_G_steps
        model = self._get_model()
        # 1) 冻结全部
        for p in model.parameters():
            p.requires_grad = False
        # 2) 解冻尾部
        unfreeze = getattr(self.train_cfg, "phaseE_unfreeze_layers", 2)
        n_dec = len(model.decoder_layers)
        for i, layer in enumerate(model.decoder_layers):
            if i >= n_dec - unfreeze:
                for p in layer.parameters():
                    p.requires_grad = True
        for p in model.final_norm.parameters():
            p.requires_grad = True
        for p in model.lm_head.parameters():
            p.requires_grad = True
        n_train = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"[Phase G] 解冻尾部{unfreeze}层decoder+final_norm+lm_head, 可训练模块数={n_train}", flush=True)

        # 3) 重建optimizer/scheduler/scaler
        self.optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=self.train_cfg.phase_G_lr, weight_decay=self.train_cfg.weight_decay,
                               betas=(0.9, 0.95))
        # Phase G 调度：warmup(10%步数) + cosine衰减
        g_warmup = max(1, total // 10)
        wu = LinearLR(self.optimizer, 1e-4, 1.0, g_warmup)
        ann = CosineAnnealingLR(self.optimizer, max(1, total - g_warmup), self.train_cfg.min_lr)
        self.scheduler = SequentialLR(self.optimizer, [wu, ann], [g_warmup])
        # bfloat16 不需要 GradScaler（动态范围足够，启用会导致scale无限增长+静默跳步）
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self.start_time = time.time()

        print(f"\n[Phase G] 纯净对话SFT 开始, 总步数: {total}, lr={self.train_cfg.phase_G_lr}", flush=True)
        print(f"[Phase G] 数据: {self.train_cfg.phaseG_data_dir} (已清洗 MOSS + Muice沐雪)", flush=True)
        print(f"[Phase G] base: {self.resume_path or 'auto(PhaseF优先, 否则PhaseE/Final)'}", flush=True)

        done = False
        while self.steps < total:
            for batch in self.train_loader:
                if self.steps >= total:
                    done = True
                    break
                self._set_phase()
                stats = self._train_step(batch)
                if stats is None:
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                self.steps += 1
                self._step_optim()

                if self.steps % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

                if self.steps % self.train_cfg.log_interval == 0:
                    el = time.time() - self.start_time
                    sps = self.steps / el if el > 0 else 0
                    lr = self.scheduler.get_last_lr()[0] if self.scheduler.get_last_lr() else self.train_cfg.phase_G_lr
                    loss = stats.get("loss", float("nan"))
                    print(f"[Phase G] {self.steps}/{total} loss={loss:.4f} lr={lr:.2e} "
                          f"| {sps:.1f} step/s bad={self.bad_batch_count}", flush=True)

                # 验证钩子
                if self.steps % 500 == 0 and hasattr(self, 'val_hooks'):
                    _vres = self.val_hooks.on_step(self.steps, loss, self._get_model(),
                                                   self.tokenizer, "G")
                    if "energy" in _vres and "dominance" in _vres["energy"]:
                        _er = _vres["energy"]
                        print(f"\n[VAL] physics={_er.get('var_physics_ratio',0):.3f} -> {_er['dominance']}", flush=True)

                if self.steps % self.train_cfg.save_interval == 0:
                    self.save("_phaseG")
            if done:
                break

        el = time.time() - self.start_time
        print(f"\n[Phase G] 完成! 耗时 {el/60:.1f}min | bad_batches={self.bad_batch_count}", flush=True)
        if hasattr(self, 'val_hooks'):
            _report = self.val_hooks.on_phase_end("G", self.steps, self._get_model(), self.tokenizer)
            print(f"[VAL] phaseG report: {_report.get('report_path', 'N/A')}", flush=True)
        self.save("_phaseG_final")

    # ------------------------------------------------------------
    # Phase F 单步（KG对齐: 冻结主干, 只训练 Entity->Atom 映射 + C/E 关联矩阵）
    # ------------------------------------------------------------
    def _train_phaseF(self):
        """Phase F: 知识图谱对齐。冻结100%主干(Qwen/PGTA/朗之万/GMM均值),
        只训练 logic_atoms.token_to_atom(实体->原子映射器) + logic_assoc_C/E(关联矩阵)。
        损失: L = -log sigmoid(z_h^T C z_t) - log sigmoid(-z_h^T C z_neg) - 0.3*log sigmoid(-z_h^T E z_t)
        知识只进逻辑域(L), 不干扰诗意域(P)。"""
        import json
        total = self.train_cfg.phase_F_steps
        model = self._get_model()
        dd = model.dual_domain

        # 1) 冻结全部
        for p in model.parameters():
            p.requires_grad = False
        # 2) 解冻: 实体->原子映射器 + 逻辑域C/E
        for p in dd.logic_atoms.token_to_atom.parameters():
            p.requires_grad = True
        dd.logic_assoc_C.requires_grad = True
        dd.logic_assoc_E.requires_grad = True
        n_train = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"[Phase F] 解冻 Entity->Atom映射器 + logic C/E 关联矩阵, 可训练模块数={n_train}", flush=True)

        # 3) 加载KG
        kg_path = getattr(self.train_cfg, "kg_path", "d:/AetherMind-Nano3/03_dialogue/knowledge_graph.json")
        with open(kg_path, encoding="utf-8") as f:
            kg = json.load(f)
        triples = kg["triples"]
        entities = kg["entities"]
        ent2idx = {e: i for i, e in enumerate(entities)}
        print(f"[Phase F] KG: {len(triples)} 三元组, {len(entities)} 实体 | {kg_path}", flush=True)

        # 4) 预缓存实体表示（冻结 embedding）
        emb = model.encoder.token_emb
        atom = dd.logic_atoms
        with torch.no_grad():
            ent_vec = torch.zeros(len(entities), self.model_cfg.d_model, device=self.device)
            for i, e in enumerate(entities):
                ids = self.tokenizer(e, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
                if ids.numel() == 0:
                    eid = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
                    ids = torch.tensor([[eid]], device=self.device)
                ent_vec[i] = emb(ids).squeeze(0).mean(0)  # (L,384)->(384,)
            ent_vec = ent_vec.detach()

        # 5) 重建optimizer
        self.optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=1e-4, weight_decay=0.0, betas=(0.9, 0.95))
        self.scheduler = CosineAnnealingLR(self.optimizer, max(1, total), 1e-6)
        self.model.eval()
        self.start_time = time.time()

        print(f"\n[Phase F] KG对齐训练 开始, 总步数: {total}", flush=True)
        print(f"[Phase F] base: {self.resume_path or 'auto(PhaseE优先, 否则PhaseD/Final)'}", flush=True)

        C = dd.logic_assoc_C
        E = dd.logic_assoc_E
        softmax = torch.softmax
        for step in range(total):
            h, r, t = random.choice(triples)
            hi, ti = ent2idx[h], ent2idx[t]
            neg = random.choice(entities)
            while neg in (h, t):
                neg = random.choice(entities)
            ni = ent2idx[neg]
            v_h, v_t, v_n = ent_vec[hi].unsqueeze(0), ent_vec[ti].unsqueeze(0), ent_vec[ni].unsqueeze(0)
            # 实体 -> 原子激活分布 w (1, n_atoms); C/E 是原子-原子关联矩阵
            w_h = softmax(atom.token_to_atom(v_h), dim=-1)
            w_t = softmax(atom.token_to_atom(v_t), dim=-1)
            w_n = softmax(atom.token_to_atom(v_n), dim=-1)
            s_pos = (w_h @ C @ w_t.t()).squeeze()
            s_neg = (w_h @ C @ w_n.t()).squeeze()
            s_rep = (w_h @ E @ w_t.t()).squeeze()
            loss = -torch.nn.functional.logsigmoid(s_pos) - torch.nn.functional.logsigmoid(-s_neg) \
                   - 0.3 * torch.nn.functional.logsigmoid(-s_rep)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.steps = step + 1

            if (step + 1) % max(1, total // 20) == 0 or step == total - 1:
                el = time.time() - self.start_time
                print(f"[Phase F] {step+1}/{total} loss={loss.item():.4f} "
                      f"s_pos={s_pos.item():.3f} s_neg={s_neg.item():.3f} "
                      f"| {total/el:.1f} step/s", flush=True)

        el = time.time() - self.start_time
        print(f"\n[Phase F] 完成! 耗时 {el/60:.1f}min", flush=True)
        self.save("_phaseF_final")

    # ------------------------------------------------------------
    # 主训练循环
    # ------------------------------------------------------------
    def train(self):
        self.setup()
        if self.phase_d:
            self._train_phaseD()
            return
        if self.phase_e:
            self._train_phaseE()
            return
        if self.phase_f:
            self._train_phaseF()
            return
        if self.phase_g:
            self._train_phaseG()
            return
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total = self.train_cfg.total_steps
        self.start_time = time.time()
        last_loss = None
        accum = {}
        accum_count = 0

        print(f"[Train] V4双权重演化训练 开始, 总步数: {total}")
        print(f"[Train] Phase A={self.train_cfg.phase_A_steps} | B={self.train_cfg.phase_B_steps} "
              f"| C={self.train_cfg.phase_C_steps}")
        print(f"[Train] 起始步数: {self.steps}", flush=True)

        done = False
        while self.steps < total:
            for batch in self.train_loader:
                if self.steps >= total:
                    done = True
                    break
                self._set_phase()
                stats = self._train_step(batch)
                if stats is None:
                    # 坏batch，跳过反传
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                for k, v in stats.items():
                    accum[k] = accum.get(k, 0.0) + v
                accum_count += 1
                self.steps += 1
                self._step_optim()

                # 信息素演化（Phase A跳过，B/C执行）
                self._evolution_step(stats)

                # 三层验证钩子（异常不中断训练）
                if self.steps % 500 == 0 and hasattr(self, 'val_hooks'):
                    try:
                        _loss = stats.get("loss", 0.0)
                        _vres = self.val_hooks.on_step(self.steps, _loss, self._get_model(),
                                                       self.tokenizer, self.phase)
                        if "numeric" in _vres and "tau_warning" in _vres["numeric"]:
                            print(f"\n[VAL] {_vres['numeric']['tau_warning']}", flush=True)
                        if "energy" in _vres and "dominance" in _vres["energy"]:
                            _er = _vres["energy"]
                            print(f"\n[VAL] 物理占比: physics={_er.get('var_physics_ratio',0):.3f} "
                                  f"qk={_er.get('var_qk_ratio',0):.3f} → {_er['dominance']}", flush=True)
                        if "migration" in _vres and "kl_warning" in _vres["migration"]:
                            print(f"\n[VAL] {_vres['migration']['kl_warning']}", flush=True)
                    except Exception as _ve:
                        print(f"\n[VAL] hook error (ignored): {_ve}", flush=True)

                if "loss" in stats:
                    last_loss = stats["loss"]
                lr = self.scheduler.get_last_lr()[0]
                
                # 每50步定期清理显存碎片
                if self.steps % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

                if self.steps % self.train_cfg.log_interval == 0 and accum_count > 0:
                    elapsed = time.time() - self.start_time
                    sps = self.steps / elapsed if elapsed > 0 else 0
                    eta = (total - self.steps) / sps if sps > 0 else 0
                    pct = self.steps / total * 100
                    filled = int(25 * self.steps / total)
                    bar = "#" * filled + "-" * (25 - filled)
                    eta_s = f"{eta/60:.0f}min" if eta < 7200 else f"{eta/3600:.1f}h"
                    evo_stats = self._get_model().get_evolution_stats()
                    tau_c = evo_stats.get("tau_concentration", 1.0)
                    cons = evo_stats.get("consolidation_mass", 0.0)
                    T_v = accum.get("T", 0.0) / accum_count
                    u_cog = accum.get("u_cog", 0.0) / accum_count
                    F_v = accum.get("F", 0.0) / accum_count
                    avg_loss = accum.get("loss", 0.0) / accum_count
                    bad = self.bad_batch_count
                    print(f"[{bar}] {pct:5.1f}% ({self.steps:>6}/{total}) ph={self.phase} "
                          f"loss={avg_loss:.4f} T={T_v:.3f} u={u_cog:.3f} F={F_v:+.3f} "
                          f"tau={tau_c:.1f} cons={cons:.0f} lr={lr:.2e} gnorm={getattr(self, '_last_grad_norm', 0):.2e} bad={bad} ETA {eta_s:>5}",
                          flush=True)
                    accum = {}
                    accum_count = 0

                if self.steps % self.train_cfg.eval_interval == 0:
                    print(f"\n[Eval] step={self.steps}", flush=True)
                    es = self.evaluate()
                    print("[Eval] " + " | ".join(f"{k}={v:.4f}" for k, v in es.items()), flush=True)

                if self.steps % self.train_cfg.save_interval == 0:
                    try:
                        self.save()
                    except Exception as _se:
                        print(f"\n[Save] checkpoint保存失败(忽略,继续训练): {_se}", flush=True)
                        self.optimizer.zero_grad(set_to_none=True)
                        gc.collect()
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()

            if done:
                break

        el = time.time() - self.start_time
        print(f"\n[Train] 完成! 耗时 {el/60:.1f}min ({el:.0f}s) | bad_batches={self.bad_batch_count}", flush=True)
        if hasattr(self, 'val_hooks'):
            _report = self.val_hooks.on_phase_end("ABC", self.steps, self._get_model(), self.tokenizer)
            print(f"[VAL] 阶段验收报告: {_report.get('report_path', 'N/A')}", flush=True)
        try:
            self.save("_final")
        except Exception as _se:
            print(f"\n[Save] final checkpoint保存失败: {_se}", flush=True)


# ============================================================
# CLI 入口
# ============================================================
def main():
    p = argparse.ArgumentParser(description="AetherMind V4 双权重演化训练")
    p.add_argument("--data_dir", default="d:/AetherMind-Nano3/03_dialogue")
    p.add_argument("--output_dir", default="d:/AetherMind-Nano3/checkpoints_v4")
    # 三阶段步数
    p.add_argument("--phase_A", type=int, default=5000)
    p.add_argument("--phase_B", type=int, default=10000)
    p.add_argument("--phase_C", type=int, default=10000)
    # 模型规模
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--n_heads", type=int, default=0, help="0=auto (d_model//64)")
    p.add_argument("--max_seq", type=int, default=512)
    # 日志/保存
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--eval_interval", type=int, default=2000)
    p.add_argument("--save_interval", type=int, default=2000)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=200)
    # 断点续训
    p.add_argument("--resume", type=str, default=None, help="指定checkpoint路径恢复")
    p.add_argument("--fresh", action="store_true", help="强制从头开始, 不自动恢复")
    # Phase D: 加载final checkpoint专训LTP固化(冻结W, 不反向传播)
    p.add_argument("--phase_d", action="store_true",
                   help="Phase D模式: 加载base checkpoint(默认final), 冻结W, 仅演化τ+分位数固化")
    p.add_argument("--phase_D", type=int, default=2000, help="Phase D 训练步数")
    # Phase E: 语言组织SFT (MOSS+沐雪)
    p.add_argument("--phase_e", action="store_true",
                   help="Phase E模式: 加载PhaseD/Final, 低lr解冻尾部, SFT学会说话(不接KG)")
    p.add_argument("--phase_E", type=int, default=3000, help="Phase E 训练步数")
    p.add_argument("--phaseE_lr", type=float, default=1e-5, help="Phase E 学习率(低lr)")
    p.add_argument("--unfreeze_layers", type=int, default=2, help="Phase E 解冻最后N层decoder")
    # Phase F: 知识图谱对齐
    p.add_argument("--phase_f", action="store_true",
                   help="Phase F模式: 加载PhaseE产出, 冻结主干, 训练Entity->Atom+C/E对齐KG")
    p.add_argument("--phase_F", type=int, default=2000, help="Phase F 训练步数")
    p.add_argument("--kg_path", type=str, default="d:/AetherMind-Nano3/03_dialogue/knowledge_graph.json",
                   help="知识图谱JSON路径(triples+entities)")
    # Phase G: 纯净对话SFT (清洗后 MOSS+沐雪)
    p.add_argument("--phase_g", action="store_true",
                   help="Phase G模式: 加载PhaseF/PhaseE/base, 用清洗数据做纯净对话SFT")
    p.add_argument("--phase_G", type=int, default=3000, help="Phase G 训练步数")
    p.add_argument("--phaseG_lr", type=float, default=3e-5, help="Phase G 学习率")
    p.add_argument("--phaseG_data_dir", type=str, default="d:/AetherMind-Nano3/03_dialogue_clean",
                   help="Phase G 清洗后对话数据目录")
    args = p.parse_args()

    n_heads = args.n_heads if args.n_heads > 0 else max(1, args.d_model // 64)

    # 先加载tokenizer获取正确的词表大小和特殊token ID
    print("[Setup] 加载tokenizer...", flush=True)
    tokenizer = build_tokenizer()
    vocab_size = tokenizer.vocab_size
    # 对齐到128的倍数（提高GPU利用率）
    vocab_size = ((vocab_size + 127) // 128) * 128
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id
    print(f"[Setup] tokenizer vocab_size={tokenizer.vocab_size}, 对齐后={vocab_size}, pad={pad_id}, eos={eos_id}", flush=True)

    mc = AetherMind4Config(
        vocab_size=vocab_size,
        d_model=args.d_model,
        d_ff=args.d_model * 4,
        n_layers=args.n_layers,
        n_heads=n_heads,
        max_seq_len=args.max_seq,
        dropout=0.1,
        pad_token_id=pad_id,
        bos_token_id=eos_id,  # Qwen没有bos，用eos
        eos_token_id=eos_id,
        unk_token_id=pad_id,
    )
    total_steps = args.phase_A + args.phase_B + args.phase_C
    tc = TrainingConfig(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        phase_A_steps=args.phase_A,
        phase_B_steps=args.phase_B,
        phase_C_steps=args.phase_C,
        total_steps=total_steps,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        warmup_steps=min(500, total_steps // 20),
        phase_D_steps=args.phase_D,
        phase_E_steps=args.phase_E,
        phase_E_lr=args.phaseE_lr,
        phaseE_unfreeze_layers=args.unfreeze_layers,
        phase_F_steps=args.phase_F,
        kg_path=args.kg_path,
        phase_G_steps=args.phase_G,
        phase_G_lr=args.phaseG_lr,
        phaseG_data_dir=args.phaseG_data_dir,
    )
    print("=" * 70)
    print("  AetherMind V4.0  息壤·双权重演化认知体  训练启动")
    print(f"  模型: d_model={args.d_model}, n_layers={args.n_layers}, n_heads={n_heads}, "
          f"d_ff={args.d_model*4}, max_seq={args.max_seq}")
    print(f"  训练: batch={args.batch_size}, grad_accum={args.grad_accum}, "
          f"lr={args.lr}, total_steps={total_steps}")
    print(f"  三阶段: A={args.phase_A} / B={args.phase_B} / C={args.phase_C}")
    print(f"  输出: {args.output_dir}")
    print(f"  数据: {args.data_dir}")
    print("=" * 70, flush=True)

    TrainerV4(mc, tc, resume_path=args.resume, fresh=args.fresh,
              phase_d=args.phase_d, phase_e=args.phase_e, phase_f=args.phase_f,
              phase_g=args.phase_g).train()


if __name__ == "__main__":
    main()
