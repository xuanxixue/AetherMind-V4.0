# AetherMind 3.6 架构报告

> 版本：3.6.1  
> 日期：2026-07-03  
> 基础模型：AetherMind 3.5.1 + 物理增强层（朗之万振荡器 + 热力学解码）

---

## 一、设计哲学

AetherMind 3.6 的核心命题是：**一个语言模型在被训练为「更准确」之前，必须先被训练为「更清醒」**。为此在 3.5.1 的 GMM 知识原子、双域分离、双记忆系统之上，引入了两个来自统计物理的增强层：

| 增强层 | 物理来源 | 功能 |
|--------|----------|------|
| 朗之万动力学相位松弛 | 统计力学 / 耦合振荡器 | 编码端跨域概念对齐 |
| 热力学自由能最小化 | 自由能原理 / 伊辛模型 | 解码端采样温度自适应 |

---

## 二、系统总览

```
                          ┌──────────────────────────────────────────┐
   input_ids ──────────► │  Encoder36 (Token Emb + Pos + IB + Shapley) │
                          └──────────────┬───────────────────────────┘
                                         │ x, z_IB_L, z_IB_P
                          ┌──────────────▼───────────────────────────┐
                          │         DualDomainSystem                  │
                          │  ┌─────────────┐  ┌─────────────┐        │
                          │  │ GMMAtom (L) │  │ GMMAtom (P) │        │
                          │  │ 逻辑域       │  │ 诗意域       │        │
                          │  └──────┬──────┘  └──────┬──────┘        │
                          │         │ 关联矩阵        │               │
                          │         ▼ (C_L,E_L)     ▼ (C_P,E_P)     │
                          └──────────────┬───────────────────────────┘
                                         │ z_L, z_P, mu, A, omega
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
          ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
          │ Langevin(L)  │    │ Langevin(P)  │    │ DualMemory   │
          │ 自由能最小化  │    │ 自由能最小化  │    │ 情景+结构     │
          └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
                 │ Z_phys_L         │ Z_phys_P          │ epi_mem
                 └────────┬─────────┘                   │
                          ▼                             ▼
                ┌────────────────────────────────────────────┐
                │          MetaCognitiveGate                   │
                │  TriPercept → α_L/α_P → u_cog → T           │
                │  SafetyFilter → Z_safe + safety_score       │
                │  FuseMLP → Z_cog                            │
                └──────────────────┬─────────────────────────┘
                                   │ Z_cog, T, θ
                          ┌────────▼──────────────────────────┐
                          │         GLUDecoder36               │
                          │  × N layers:                       │
                          │    GLU + TSR + PSR + PointerGate   │
                          │    + CounterSlot                   │
                          │    + Temperature-controlled Softmax│
                          └──────────────────┬─────────────────┘
                                             │ logits, probs
                                             ▼
                                         loss / generate
```

**模块清单**：

| 文件 | 类 | 职责 |
|------|-----|------|
| [encoder/ib_encoder.py](file:///d:/AetherMind-Nano3/src/model/encoder/ib_encoder.py) | `Encoder36`, `InformationBottleneck` | Token嵌入、位置编码、IB压缩、Shapley注意力、GMM判别器损失 |
| [domain/dual_domain.py](file:///d:/AetherMind-Nano3/src/model/domain/dual_domain.py) | `DualDomainSystem`, `GMMAtom` | 双域（逻辑/诗意）概率知识表示、关联矩阵、世界模型、锚点对齐 |
| [physics/langevin.py](file:///d:/AetherMind-Nano3/src/model/physics/langevin.py) | `LangevinOscillator` | 复振幅投影、朗之万K步迭代、自由能最小化 |
| [memory/dual_memory.py](file:///d:/AetherMind-Nano3/src/model/memory/dual_memory.py) | `DualMemorySystem`, `EpisodicMemory`, `StructuralMemory` | 情景记忆（环形缓冲）+ 结构记忆（技能键值） |
| [metacog/meta_gate.py](file:///d:/AetherMind-Nano3/src/model/metacog/meta_gate.py) | `MetaCognitiveGate`, `TriPercept`, `SafetyFilter` | 三感知器投票、温度-不确定性耦合、安全过滤 |
| [decoder/glu_decoder.py](file:///d:/AetherMind-Nano3/src/model/decoder/glu_decoder.py) | `GLUDecoder36`, `GLUBlock36`, `TSR`, `PSR`, `PointerGate`, `CounterSlot` | 线性复杂度解码器，含相位状态路由和温度控制采样 |
| [aethermind36.py](file:///d:/AetherMind-Nano3/src/model/aethermind36.py) | `AetherMind36` | 顶层模型，编排所有子模块、计算总损失、生成文本 |

---

## 三、核心模块详解

### 3.1 GMM 知识原子 (`GMMAtom`)

每个域（逻辑/诗意）拥有 **1024 个知识原子**，每个原子是一个 **3 分量高斯混合模型**：

- `mean_emb` `[1024, 3, 64]` — 各分量均值
- `logvar_emb` `[1024, 3, 64]` — 各分量对数方差（经 softplus 转为正）
- `mix_logits` `[1024, 3]` — 混合权重（经 softmax 归一化）

**激活过程** (见 [dual_domain.py#L66-L75](file:///d:/AetherMind-Nano3/src/model/domain/dual_domain.py#L66-L75))：

```
token → Linear(d_token → n_atoms) → softmax → atom_weight [B, S, N]
                                                     │
mu, sigma, pi → 加权均值 = Σ(π_k · μ_k)   [N, D]
                                                     │
z = einsum("bsn,nd→bsd", atom_weight, mean_atom)  [B, S, D]
```

原子同时具备 `mass`（惯性质量）和 `tau`（弛豫时间），用于朗之万动力学。

### 3.2 双域关联矩阵 (`DualDomainSystem`)

每个域有两条关联链：

- **C（竞争链）**：`torch.sigmoid(logic_assoc_C)` — 原子间竞争强度
- **E（排除链）**：`torch.sigmoid(logic_assoc_E)` — 互斥图结构
- **实际邻接矩阵**：`A = C × (1 - E)` — 软模块化网络

跨域对齐通过 **512 个共享锚点** (`anchor_emb`) 实现：逻辑域和诗意域的原子分别路由到同一锚点空间，用 MSE 损失拉近。

### 3.3 朗之万振荡器 (`LangevinOscillator`)

这是 3.6 相对于 3.5.1 的核心新增模块 (见 [langevin.py](file:///d:/AetherMind-Nano3/src/model/physics/langevin.py))。

**复振幅初始化** (见 [langevin.py#L21-L53](file:///d:/AetherMind-Nano3/src/model/physics/langevin.py#L21-L53))：

- 幅值 `r = ‖W_r·μ‖` — 从GMM均值投影
- 相位 `θ = θ_pos(from PE) + θ_u(from module_emb)` — 位置编码 + 模块嵌入

**自由能** (见 [langevin.py#L55-L73](file:///d:/AetherMind-Nano3/src/model/physics/langevin.py#L55-L73))：

```
F(r,θ) = ½k(r-r₀)²          ← 弹性能（约束幅值）
       - Σ Jᵢⱼ rᵢ rⱼ cos(θᵢ-θⱼ-φᵢⱼ)  ← 耦合能（相位同步）
       - T · S_phase(θ)      ← 熵项（温度驱动探索）
```

**K 步朗之万迭代** (见 [langevin.py#L132-L156](file:///d:/AetherMind-Nano3/src/model/physics/langevin.py#L132-L156))：

```
耦合项 = Σⱼ J_b[i,j] × r[j] × sin(θ[j] - θ[i] - φ)
θ ← θ + dt × (ω + coupling) + ξ_θ      (相位更新)
r ← r + dt × (-∂F/∂r) + ξ_r            (幅值更新，带梯度或无梯度)
```

最终输出 `Z_phys = [r·cos(θ), r·sin(θ)]` 经 MLP 投影到 `d_model` 维度。

### 3.4 元认知门控 (`MetaCognitiveGate`)

**三感知器投票** (见 [meta_gate.py#L8-L30](file:///d:/AetherMind-Nano3/src/model/metacog/meta_gate.py#L8-L30))：

- 3 个独立 MLP 各自输出 `[α_L, α_P]`（逻辑/诗意权重）
- 取 α_L 的**中位数**（抵抗单一感知器异常）
- **认知不确定性** `u_cog = Var(α_L) / 0.25`（三感知器分歧度）

**温度-不确定性耦合** (见 [meta_gate.py#L60-L62](file:///d:/AetherMind-Nano3/src/model/metacog/meta_gate.py#L60-L62))：

```
T = T₀ × (1 + κ × u_cog)
  受限: T_min = 0.05, T_max = 5.0
```

当三感知器分歧大时，自动升温 → 更保守/多样化的采样。

**安全过滤器** (`SafetyFilter`)：64 个危险锚点，计算余弦相似度，低于 0.3 阈值的特征被替换为均值骨架。

### 3.5 GLU 解码器 (`GLUDecoder36`)

每层包含 (见 [glu_decoder.py#L49-L137](file:///d:/AetherMind-Nano3/src/model/decoder/glu_decoder.py#L49-L137))：

| 子模块 | 功能 |
|--------|------|
| **GLU** (Gated Linear Unit) | 门控激活 `σ(u) ⊙ v` + 线性投影 |
| **TSR** (Token State Router) | 令牌状态记忆，γ-门控更新 `M = γ·M_prev + (1-γ)·x` |
| **PSR** (Phase State Router) | 相位状态记忆，`γ_phys = exp(-Δθ²/(2T))` 作为相位差的门控 |
| **层级注入** | `Z_cog` 拼位置嵌入 → `decomp_mlp` → scale/bias/alpha |
| **α_phys** | `sigmoid(cos(θ - θ_layer) / T)` — 相位谐振门控 |
| **PointerGate** | 复制概率 `p_copy = σ(W_copy·h + W_copyz·Z_cog)` |
| **CounterSlot** | GRU 计数器，输出经 tanh 缩放后加到 hidden state |

**最终采样** (见 [glu_decoder.py#L219-L228](file:///d:/AetherMind-Nano3/src/model/decoder/glu_decoder.py#L219-L228)):
- `probs = softmax(logits / T)` — T 由元认知门实时控制
- `p_copy` 与 `probs` 混合：`(1-p_copy)·probs + p_copy·copy_dist`
- copy_dist 使用半精度 fp16 计算以节省 4GB 显存

---

## 四、训练策略

### 4.1 三阶段渐进训练

| 阶段 | 步数 | λ_phys | λ_PSR | λ_T | λ_F | K | T₀ | 目标 |
|------|------|--------|-------|-----|-----|---|---|------|
| **A** | 5000 | 0 | 0 | 0 | 0 | 0 | 0.2 | 纯语言模型预训练，冻结所有物理层 |
| **B** | 10000 | 0→0.1 | 0→0.1 | 0→0.1 | 0 | 2 | 0.7 | 渐入物理层，升温采样，宽域探索 |
| **C** | 10000 | 0.2 | 0→0.3 | 0→0.3 | 0→0.05 | 5 | 0.2 | 全激活，自由能约束、相位同步、低温收敛 |

**关键设计**：Phase B 升温到 T=0.7（高于 A=0.2 和 C=0.2），让模型在物理层介入初期进行更广泛的相位空间探索，找到更好的自由能盆地，然后再在 Phase C 降温收敛。

### 4.2 损失函数

```
L_total = L_LM                                 ← 交叉熵（始终激活）
        + λ_IB × KL(q(z|x)‖p(z))              ← 信息瓶颈（B/C阶段）
        + λ_D   × BCE(D_score, 0.5)           ← GMM判别器均衡（B/C）
        + λ_aff × std(affect)                 ← 情感强度正则
        + λ_shap × H(attention)               ← Shapley注意力熵
        + λ_module × module_loss              ← 模块化结构损失
        + λ_align × anchor_MSE                ← 锚点对齐损失
        + λ_phys × F²_mean                    ← 自由能最小化
```

### 4.3 数据流

- **数据集**: `StreamingTextDataset` 流式加载 `F:/数据集` 下所有 `.txt`/`.jsonl`/`.db`
- **每文件限 5000 行**随机采样，避免单次加载过大
- **分词器**: BPE (tokenizers库)，vocab_size=50000，本地训练无网络依赖
- **DataLoader**: `num_workers=0`（兼容 Windows spawn），pin_memory + non_blocking GPU 传输
- **梯度累积**: batch_size=1 × grad_accum=8 = 等效 batch 8

---

## 五、关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `d_model` | 384 / 512 | 模型隐藏维度 |
| `n_layers` | 6 / 8 | GLU 解码器层数 |
| `n_heads` | 8 | 注意力头数（仅 Shapley 注意力使用） |
| `d_ff` | 2048 | FFN 中间维度 |
| `n_atoms` | 1024 | 每域知识原子数 |
| `d_atom` | 64 | 原子嵌入维度 |
| `n_gmm_components` | 3 | GMM 分量数 |
| `n_anchor` | 512 | 跨域共享锚点数 |
| `langevin_K` | 0→2→5 | 朗之万迭代步数（A→B→C） |
| `langevin_dt` | 0.33 | 朗之万时间步长 |
| `T₀` | 0.2→0.7→0.2 | 基础温度（A→B→C） |
| `κ` | 2.0→0.5→2.0 | 温度-不确定性耦合系数 |
| `max_seq_len` | 512 | 最大序列长度 |

---

## 六、硬件适配

- **目标 GPU**: NVIDIA GeForce RTX 3050 Laptop (4 GB VRAM)
- **CUDA**: 12.1 + PyTorch 2.5.1+cu121
- **混合精度**: bfloat16 autocast (encode/forward) + fp32 安全回退 (BCE loss)
- **多 GPU**: 支持 DataParallel (device_ids 可配置)
- **显存优化**: PYTORCH_CUDA_ALLOC_CONF、cudnn benchmark、copy_prob 半精度计算
- **模型参数量**: d=384→56.6M (215.7 MB) / d=512→99.9M (381.3 MB)

---

## 七、文件结构

```
AetherMind-Nano3/
├── configs/
│   └── aethermind36_config.py          # 模型+训练超参数
├── src/
│   ├── model/
│   │   ├── aethermind36.py             # 顶层模型
│   │   ├── encoder/ib_encoder.py       # IB编码器
│   │   ├── domain/dual_domain.py       # 双域GMM系统
│   │   ├── physics/langevin.py         # 朗之万振荡器
│   │   ├── memory/dual_memory.py       # 双记忆系统
│   │   ├── metacog/meta_gate.py        # 元认知门控
│   │   └── decoder/glu_decoder.py      # GLU解码器
│   ├── data/
│   │   └── dataset.py                  # 流式数据集
│   ├── training/
│   │   └── train.py                    # 训练主脚本
│   └── utils/
│       └── ops.py                      # MLP, safe_softmax, entropy 等工具
├── checkpoints/                        # 模型保存目录
├── docs/
├── train_gpu.cmd                       # 推荐训练脚本
├── train_gpu_smoke.cmd                 # 冒烟测试脚本
└── train_gpu_max.cmd                   # 大模型训练脚本
```
