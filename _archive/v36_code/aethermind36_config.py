import torch
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class AetherMind36Config:
    vocab_size: int = 50000
    d_model: int = 512
    d_ff: int = 2048
    n_layers: int = 8
    n_heads: int = 8
    max_seq_len: int = 1024
    dropout: float = 0.1

    n_atoms: int = 1024
    d_atom: int = 64
    n_gmm_components: int = 3

    n_anchor: int = 512
    d_anchor: int = 64

    d_state: int = 128
    d_counter: int = 16

    langevin_K: int = 3
    langevin_dt: float = 0.33
    langevin_T_enc: float = 0.05
    topk_neighbors: int = 16

    T0: float = 0.2
    kappa: float = 2.0
    T_min: float = 0.05
    T_max: float = 5.0

    lambda_phys: float = 0.0
    lambda_PSR: float = 0.0
    lambda_T: float = 0.0
    lambda_F: float = 0.0
    lambda_copy_phys: float = 0.0
    lambda_phase: float = 0.0
    lambda_phi_decay: float = 0.0

    lambda_IB: float = 0.0
    lambda_D: float = 0.0
    lambda_aff: float = 0.0
    lambda_align: float = 0.0
    lambda_shap: float = 0.0
    lambda_module: float = 0.0

    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = field(default_factory=lambda: torch.float32)
    device_ids: Optional[List[int]] = field(default_factory=lambda: list(range(torch.cuda.device_count())) if torch.cuda.is_available() else None)

    def set_phase_A(self):
        self.lambda_phys = 0.0
        self.lambda_PSR = 0.0
        self.lambda_T = 0.0
        self.lambda_F = 0.0
        self.lambda_copy_phys = 0.0
        self.lambda_phase = 0.0
        self.lambda_phi_decay = 0.0
        self.lambda_IB = 0.0
        self.lambda_D = 0.0
        self.lambda_aff = 0.0
        self.lambda_align = 0.0
        self.langevin_K = 0

    def set_phase_B(self, progress: float = 0.5):
        self.lambda_phys = 0.1 * progress
        self.lambda_PSR = 0.1 * progress
        self.lambda_T = 0.1 * progress
        self.lambda_F = 0.0
        self.lambda_copy_phys = 0.0
        self.lambda_phase = 0.15 * progress
        self.lambda_phi_decay = 0.0
        self.lambda_IB = 0.5 * progress
        self.lambda_D = 0.3 * progress
        self.lambda_aff = 0.2 * progress
        self.lambda_align = 0.3 * progress
        self.langevin_K = 2
        self.T0 = 0.7
        self.kappa = 0.5

    def set_phase_C(self, progress: float = 0.5):
        # 从 Phase B 末尾值平滑插值到 Phase C 目标值，避免参数跳变导致数值不稳定
        self.lambda_phys = 0.1 + (0.2 - 0.1) * progress
        self.lambda_PSR = 0.1 + (0.3 - 0.1) * progress
        self.lambda_T = 0.1 + (0.3 - 0.1) * progress
        self.lambda_F = 0.05 * progress
        self.lambda_copy_phys = 0.2 * progress
        self.lambda_phase = 0.15 + (0.3 - 0.15) * progress
        self.lambda_phi_decay = 0.01 * progress
        self.lambda_IB = 0.5 + (1.0 - 0.5) * progress
        self.lambda_D = 0.3 + (0.5 - 0.3) * progress
        self.lambda_aff = 0.2 + (0.3 - 0.2) * progress
        self.lambda_align = 0.3 + (0.5 - 0.3) * progress
        self.langevin_K = 2 + int(round((5 - 2) * progress))
        self.T0 = 0.7 + (0.2 - 0.7) * progress
        self.kappa = 0.5 + (2.0 - 0.5) * progress


@dataclass
class TrainingConfig:
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0

    phase_A_steps: int = 50000
    phase_B_steps: int = 100000
    phase_C_steps: int = 100000
    total_steps: int = 250000

    log_interval: int = 100
    eval_interval: int = 2000
    save_interval: int = 5000

    output_dir: str = "d:/AetherMind-Nano3/checkpoints"
    data_dir: str = "F:/数据集"

    train_ratio: float = 0.98
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = 2000
