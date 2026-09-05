import torch
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class AetherMind4Config:
    """AetherMind V4.0 — 双权重演化认知体配置 (修复版)
    使用Qwen2.5 tokenizer，修复所有已知Bug
    """
    # === 使用Qwen2.5-0.5B tokenizer ===
    vocab_size: int = 151936  # Qwen2.5词表大小 (对齐到128倍数)
    tokenizer_path: str = "models_local/Qwen/Qwen2___5-0___5B"
    
    d_model: int = 512
    d_ff: int = 2048
    n_layers: int = 8
    n_heads: int = 8
    max_seq_len: int = 1024
    dropout: float = 0.1

    # === 3.6 继承 ===
    n_atoms: int = 1024
    d_atom: int = 64
    n_gmm_components: int = 3
    n_anchor: int = 512
    d_anchor: int = 0  # 0 means auto-match d_atom
    d_state: int = 128
    d_counter: int = 16
    langevin_K: int = 3
    langevin_dt: float = 0.33
    langevin_T_enc: float = 0.05
    topk_neighbors: int = 16

    # === V4 新增：信息素路径网络 (PGTA) ===
    # 量级修复: 沉积量×S补偿后, 稳态τ≈deposit*gate*C/rho (C=注意力集中度, 均匀=1)
    # deposit/rho≈1.0-1.5 时τ均值回到设计量级1附近, 固化阈值1.2才有选择性
    pheromone_rho: float = 0.02         # 蒸发率
    pheromone_beta: float = 1.0         # 信息素敏感度
    pheromone_deposit: float = 0.04     # 沉积强度 (稳态τ_mean≈deposit*0.55/rho≈1.1)
    pheromone_tau_min: float = 1e-2     # τ下限
    pheromone_tau_max: float = 5.0      # τ上限
    pheromone_credit_mode: str = "soft_center"  # hard / soft / soft_center
    pheromone_whiten: bool = True       # 能量白化

    # === V4 新增：固化机制 (LTP 长时程增强) ===
    consolidate_threshold: float = 1.2  # τ超此阈值触发固化 (绝对模式)
    consolidate_lam: float = 0.15       # 固化写入强度
    consolidate_gamma: float = 0.4      # 固化后τ局部衰减
    consolidate_interval: int = 100     # 每N步尝试固化一次
    max_consolidations: int = 100000    # 固化预算上限(防膨胀; τ每层h*S*S≈1.5M元素)
    consolidate_top_frac: Optional[float] = None  # 分位数固化模式: 固化最强top比例(None=用绝对阈值)
    consolidate_warmup: int = 100       # 固化预热步数: τ未成型前不固化, 避免把噪声写入长期权重

    # === V4 新增：热力学注意力 ===
    init_temperature: float = 1.0       # 初始温度
    target_entropy_ratio: float = 0.3   # 目标熵比例 (目标H = ratio * log(N))
    lambda_H: float = 0.05              # 熵正则权重
    lambda_tau_reg: float = 0.01        # 信息素稀疏正则权重

    # === 温度控制 (继承+扩展) ===
    T0: float = 0.2
    kappa: float = 2.0
    T_min: float = 0.05
    T_max: float = 5.0

    # === 物理损失权重 ===
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

    # === 特殊token (Qwen2.5 tokenizer) ===
    # pad=eos=151643, bos不使用设为eos, unk=pad
    pad_token_id: int = 151643  # <|endoftext|>
    bos_token_id: int = 151643  # 没有bos，用eos代替
    eos_token_id: int = 151643  # <|endoftext|>
    unk_token_id: int = 151643

    # === 设备 ===
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = field(default_factory=lambda: torch.float32)
    device_ids: Optional[List[int]] = field(
        default_factory=lambda: list(range(torch.cuda.device_count())) if torch.cuda.is_available() else None
    )
    
    # === 显存优化 ===
    gradient_checkpointing: bool = True  # 梯度检查点，省显存~50%，速度慢~20%

    def set_phase_A(self):
        """Phase A: 纯结构预训练, 冻结所有物理/演化层"""
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
        self.pheromone_deposit = 0.0    # Phase A 不沉积信息素
        self.pheromone_rho = 0.0        # Phase A 不蒸发
        self.init_temperature = 1.0      # 高温探索
        self.T0 = 1.0

    def set_phase_B(self, progress: float = 0.5):
        """Phase B: 渐入物理层+信息素, 升温探索 (权重减半版)"""
        # Bug2修复: 辅助权重控制在总权重1.0以内
        self.lambda_phys = 0.05 * progress
        self.lambda_PSR = 0.08 * progress
        self.lambda_T = 0.08 * progress
        self.lambda_F = 0.0
        self.lambda_copy_phys = 0.05 * progress
        self.lambda_phase = 0.1 * progress
        self.lambda_phi_decay = 0.0
        self.lambda_IB = 0.3 * progress
        self.lambda_D = 0.15 * progress
        self.lambda_aff = 0.1 * progress
        self.lambda_align = 0.15 * progress
        self.langevin_K = 2
        self.pheromone_deposit = 0.03 * progress  # 渐入沉积 (新量纲: ×S补偿后)
        self.pheromone_rho = 0.02
        self.init_temperature = 0.7 + 0.3 * progress
        self.T0 = 0.7
        self.kappa = 0.5

    def set_phase_C(self, progress: float = 0.5):
        """Phase C: 全激活, 固化启动, 降温收敛 (Bug2修复: 权重减半)"""
        # Bug2修复: 辅助损失总权重从3.49降到约1.3
        self.lambda_phys = 0.1
        self.lambda_PSR = 0.15 * progress
        self.lambda_T = 0.15 * progress
        self.lambda_F = 0.03 * progress
        self.lambda_copy_phys = 0.1 * progress
        self.lambda_phase = 0.15
        self.lambda_phi_decay = 0.005 * progress
        self.lambda_IB = 0.5
        self.lambda_D = 0.25
        self.lambda_aff = 0.15
        self.lambda_align = 0.25
        self.langevin_K = 3
        self.pheromone_deposit = 0.04     # 新量纲: 稳态τ_mean≈0.04*0.55/0.02≈1.1
        self.pheromone_rho = 0.02
        self.consolidate_threshold = 1.2  # 绝对阈值模式(量级修复后有效)
        self.init_temperature = 0.2 + 0.3 * (1 - progress)  # 降温收敛
        self.T0 = 0.2
        self.kappa = 2.0

    def set_phase_D(self, progress: float = 0.5):
        """Phase D: LTP固化专训 — 冻结W, 纯前向演化+固化
        从Phase C的final checkpoint加载权重, 不做反向传播,
        高沉积低蒸发快速积累τ, 分位数固化最强路径写入consolidated。
        """
        # λ与Phase C一致(loss仅作统计与dF奖励信号, W已冻结不更新)
        self.lambda_phys = 0.1
        self.lambda_PSR = 0.15 * progress
        self.lambda_T = 0.15 * progress
        self.lambda_F = 0.03 * progress
        self.lambda_copy_phys = 0.1 * progress
        self.lambda_phase = 0.15
        self.lambda_phi_decay = 0.005 * progress
        self.lambda_IB = 0.5
        self.lambda_D = 0.25
        self.lambda_aff = 0.15
        self.lambda_align = 0.25
        self.langevin_K = 3
        # 强沉积: 稳态τ_mean≈0.06*0.55/0.01≈3.3, 最强路径冲顶tau_max=5
        self.pheromone_deposit = 0.06
        self.pheromone_rho = 0.01         # 低蒸发, 保留已沉积经验
        # 分位数固化: 每轮固化最强0.2%路径, 选择性不依赖绝对量级
        self.consolidate_top_frac = 0.002
        self.consolidate_lam = 0.2
        self.consolidate_gamma = 0.4
        self.consolidate_interval = 50    # 高频固化(2000步→40轮)
        self.max_consolidations = 100000
        self.init_temperature = 0.5
        self.T0 = 0.5
        self.kappa = 1.0

    def set_phase_E(self, progress: float = 0.5):
        """Phase E: 语言组织能力 SFT — 低lr解冻尾部微调, 关闭演化/固化干扰。
        只解冻最后N层decoder+final_norm+lm_head, 数据=MOSS+沐雪, 目标是"学会说话"。
        """
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
        self.langevin_K = 1
        self.pheromone_deposit = 0.0      # 关闭信息素沉积(不演化)
        self.pheromone_rho = 1.0          # 蒸发拉满, 清空τ干扰
        self.consolidate_top_frac = None  # 不固化
        self.consolidate_interval = 10**9
        self.init_temperature = 0.8
        self.T0 = 0.8
        self.kappa = 1.0

    def set_phase_G(self, progress: float = 0.5):
        """Phase G: 纯净对话 SFT — 用清洗后的 MOSS+沐雪数据, 目标是"正常对话"。
        关闭演化/固化/物理辅助损失(与Phase E一致), 温度适中(0.7)保证回复稳定连贯。
        """
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
        self.langevin_K = 1
        self.pheromone_deposit = 0.0      # 关闭信息素沉积(不演化)
        self.pheromone_rho = 1.0          # 蒸发拉满, 清空τ干扰
        self.consolidate_top_frac = None  # 不固化
        self.consolidate_interval = 10**9
        self.init_temperature = 0.7
        self.T0 = 0.7
        self.kappa = 1.0

    def set_phase_F(self, progress: float = 0.5):
        """Phase F: 知识图谱对齐 — 冻结主干, 只训练 Entity->Atom 映射器 + C/E 关联矩阵。
        知识只进逻辑域(Logic), 不干扰诗意域(P)。关闭演化/固化。
        """
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
        self.langevin_K = 1
        self.pheromone_deposit = 0.0
        self.pheromone_rho = 1.0
        self.consolidate_top_frac = None
        self.consolidate_interval = 10**9
        self.init_temperature = 0.7
        self.T0 = 0.7
        self.kappa = 1.0


@dataclass
class TrainingConfig:
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0

    # 三阶段训练 (与之前一致)
    phase_A_steps: int = 5000
    phase_B_steps: int = 10000
    phase_C_steps: int = 20000
    total_steps: int = 35000
    phase_D_steps: int = 2000          # Phase D (LTP固化专训) 步数
    phase_E_steps: int = 3000          # Phase E (语言组织SFT) 步数
    phase_E_lr: float = 1e-5           # Phase E 低学习率(只解冻尾部)
    phaseE_unfreeze_layers: int = 2    # Phase E 解冻最后N层decoder
    phase_F_steps: int = 2000          # Phase F (知识图谱对齐) 步数
    kg_path: str = "d:/AetherMind-Nano3/03_dialogue/knowledge_graph.json"  # KG文件
    phase_G_steps: int = 3000          # Phase G (纯净对话SFT) 步数
    phase_G_lr: float = 3e-5           # Phase G 学习率(清洗数据, 可略高于Phase E)
    phaseG_data_dir: str = "d:/AetherMind-Nano3/03_dialogue_clean"  # Phase G 清洗后数据目录

    log_interval: int = 10
    eval_interval: int = 2000
    save_interval: int = 2000

    output_dir: str = "d:/AetherMind-Nano3/checkpoints_v4_fixed"
    data_dir: str = "d:/AetherMind-Nano3/03_dialogue"

    train_ratio: float = 0.98
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = 2000
    
    # 奖励缩放 (Bug4修复: 放大奖励信号)
    reward_scale: float = 5.0
