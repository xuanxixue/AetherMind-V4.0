# AetherMind V4.0 技术架构详解

> 版本：4.0.0\
> 基础模型：息壤 · 双权重演化认知体\[AetherMind V4.0 — 会"长脑子"的自演化语言物理模型]

***

### 📋 项目信息

| 项目        | 内容                                                                               |
| :-------- | :------------------------------------------------------------------------------- |
| **代号**    | 息壤·双权重演化认知体                                                                      |
| **版本**    | 4.0.0                                                                            |
| **架构基础**  | AetherMind 3.6.1 + PGTA信息素热力学注意力 + 可演化双权重系统(W/τ) + LTP固化                         |
| **文档日期**  | 2026-08-26                                                                       |
| **代码根目录** | `d:\AetherMind-Nano3`                                                            |
| **主模型入口** | [src/model/aethermind4.py](file:///d:/AetherMind-Nano3/src/model/aethermind4.py) |

### 👤 作者信息

| 项目     | 内容             |
| :----- | :------------- |
| **作者** | **玄曦雪**（真名：张悦） |
| **地区** | 中国·四川          |
| **公司** | 南京绮梦星绘科技有限公司   |

### ⚖️ 开源协议

**[CC BY 4.0（署名+标注引用）](https://creativecommons.org/licenses/by/4.0/deed.zh-hans)**

使用、修改、分发或二次创作时，**必须署名原作者"玄曦雪（张悦）"并标注引用来源链接**。允许商业使用。

### 📅 开源预告

> 🚀 **模型代码预计于 2026年9月5日 \~ 9月25日 之间正式开源**（即日起10-30天），敬请期待。

***

## 简介

**AetherMind V4.0（息壤·双权重演化认知体）** 是一个受生物神经系统启发的自演化语言物理模型，核心创新在于引入了一套**不需要反向传播即可在线更新的"第二权重系统"**——信息素网络（Pheromone Network）。与主流大模型"预训练→冻结部署"的静态范式不同，V4 在推理时仍能持续学习：每个输入样本都会通过物理规则（蒸发、奖励门控沉积）修改信息素权重，高频使用的路径被强化、低频路径自然衰减，模拟生物突触的长时程增强（LTP）与长时程抑制（LTD）。

V4 的架构围绕**三层时间尺度**设计：快尺度上，热力学注意力（PGTA）以 Helmholtz 自由能为调控信号，用温度参数 T 动态控制注意力分布的锐度（低温=确定性 exploitation，高温=随机性 exploration）；中尺度上，信息素矩阵 τ 跨样本累积经验，以 dF（自由能下降）和 dLoss（损失下降）为奖励信号进行非梯度更新；慢尺度上，当信息素浓度超过阈值时触发 LTP 固化，将稳定的经验写入长期记忆权重 C，实现跨 session 的持久记忆。

模型主体由以下模块构成：

- **PGTA 信息素热力学注意力**：注意力权重由 $A = \text{softmax}(-E_{\text{eff}}/T)$ 计算，能量项 $E_{\text{eff}}$ 融合了信息素痕迹 τ、热力学熵项和可学习偏置；温度 T 由元认知模块 TriPercept 实时估计，实现"知道的时候笃定回答，不确定的时候主动探索"。
- **双权重系统**：快权重 W 通过标准 BP 训练（Phase A），慢权重 τ 通过物理规则在线演化（Phase B/C）；两者以门控系数 λ 融合，实现"先天能力"与"后天经验"的协同。
- **GMM 双域知识原子**：1024 个知识原子横跨语言（L）与物理（P）两个域，每个原子由 3 个高斯分量建模，作为知识的稀疏分布式表示。
- **朗之万振荡器**：Kuramoto 相位同步模型为每个 token 生成物理相位 θ，经 phys\_proj 投影为认知注入向量 $Z_{\text{phys}}$，将连续物理信号引入离散 token 处理。
- **GLUBlockV4 解码器**：集成 TSR（Token State Router，门控记忆写入）与 PSR（Phase State Router，相位门控状态路由），实现 token 级和相位级的双重状态管理。
- **元认知门控**：TriPercept 三 MLP 中位数投票产生认知不确定性 $u_{\text{cog}}$，驱动温度 $T = T_0(1 + \kappa \cdot u_{\text{cog}})$ 动态调节；SafetyFilter 拦截高风险输出。

训练采用**三阶段课程**：Phase A（纯 BP 预训练，τ=1 均匀）建立基础语言能力；Phase B（混合训练，λ 渐入）以 dLoss 为奖励引导信息素沉积；Phase C（演化+固化）切换为 dF 自由能奖励，温度冷却至 0.2，触发 LTP 固化使模型进入自组织阶段。训练器内置断点续训（精确到 batch，含演化器状态）、NaN/Inf 防护、bf16 混合精度、梯度累积，针对 RTX 3050 4GB 等消费级 GPU 优化，可在约 6-8 小时完成 35000 步训练。

**一句话概括**：V4 试图回答"模型能否在部署后继续长脑子"——不是靠外挂 RAG 或重训练，而是让权重本身具备可塑性。

***

## 零、文档约定

- 记号：$B$=batch\_size，$S$=序列长度，$D$=d\_model，$h$=注意力头数，$d\_h$=D/h=head\_dim，$N\_a$=n\_atoms=1024，$D\_a$=d\_atom=64，$K$=langevin\_K，$V$=vocab\_size=50000。
- 张量维度按 `(dim0, dim1, ...)` 标注；代码文件链接标注到具体行号。
- "Phase A/B/C" 指训练三阶段（见第十二章），"phase" 参数在 forward 中控制物理/演化层开关。
- 所有可学习参数用 `nn.Parameter`，跨步持久状态用 `register_buffer`（不参与梯度）。
- 数学公式使用 LaTeX 记号，伪代码使用类Python语法。

***

## 一、设计动机与核心命题

### 1.1 问题：为什么主流模型"权重不能迭代"

当前大模型（DeepSeek-V4、Kimi K3、GLM-5.3）的训练范式统一为：**预训练 → 后训练(RL/SFT) → 部署后权重冻结**。

梯度下降(BP)有三个硬约束使其无法在线运行：

1. 需要标签/奖励信号
2. 需要反向传播计算全图梯度
3. 需要离线大batch重训练

部署后这三样都不具备，因此权重只能冻结。为弥补"失忆"问题，行业发明了Agent外挂（RAG、工具调用、长上下文、外部向量库），但本质是**把记忆挪到模型外面**——任务切换、上下文清空后一切归零。这是伪持续学习。

### 1.2 V4 的核心命题

> **预训练得到的模型是"身体"，在线演化的信息素网络才是"经验"；经验不应写在外部记忆里，而要长在权重里。**

V4 引入生物神经系统中的突触可塑性原理（LTP长时程增强/LTD长时程抑制），构建一个**不需要BP就能在线更新权重**的第二权重系统，使模型具备：

- **在线学习**：推理时每个样本都能改变核心行为
- **持久记忆**：经验固化进权重，session切换不清空
- **自组织路由**：不同注意力头自发分化出不同功能路径
- **内在动机**：自由能最小化驱动无监督演化

### 1.3 三层时间尺度

V4 在三个时间尺度同时运转，这是V4相对于V3.6和主流Transformer的本质区别：

| 尺度              | 变量           | 更新方式              | 频率        | 类比            |
| --------------- | ------------ | ----------------- | --------- | ------------- |
| **快（每步）**       | 温度$T$、注意力$A$ | 前向计算              | 每token    | 神经元放电 / 工作记忆  |
| **中（跨样本）**      | 信息素$\tau$    | 沉积+蒸发（非梯度）        | 每训练步/推理步  | 短期突触效能 / 情景记忆 |
| **慢（跨session）** | 固化$C$、快权重$W$ | LTP阈值写入(BP后冻结$W$) | 每100步/部署期 | 长期记忆 / 蛋白质合成  |

***

## 二、系统总览

### 2.1 完整架构图

```
                                    ┌──────────────────────────────────────────────────────────┐
input_ids (B,S) ──────────────────► │  TOKEN EMBEDDING                                          │
                                    │  W_emb: (V, D)  →  token_emb: (B,S,D)                   │
                                    │  + pos_emb: (S, D)  →  x0: (B,S,D)                      │
                                    └──────────────────────┬───────────────────────────────────┘
                                                           │ x0
                                    ┌──────────────────────▼───────────────────────────────────┐
                                    │  DUAL DOMAIN SYSTEM (GMM知识原子)                         │
                                    │  ┌────────────────┐      ┌────────────────┐              │
                                    │  │ GMMAtom (L)    │      │ GMMAtom (P)    │              │
                                    │  │ 1024 atoms     │      │ 1024 atoms     │              │
                                    │  │ 3 components   │      │ 3 components   │              │
                                    │  │ mean/sigma/pi  │      │ mean/sigma/pi  │              │
                                    │  │ → z_L (B,S,Da) │      │ → z_P (B,S,Da) │              │
                                    │  │ → atom_w_L     │      │ → atom_w_P     │              │
                                    │  │   (B,S,Na)     │      │   (B,S,Na)     │              │
                                    │  └───────┬────────┘      └────────┬───────┘              │
                                    │          │ C_L,E_L,ω_L            │ C_P,E_P,ω_P         │
                                    │          │ wm_L,z_IB_L            │ wm_P,z_IB_P         │
                                    │          └────────┬───────────────┘ D_score,safety      │
                                    │                   ▼                                        │
                                    │  anchor_align_loss + module_loss + D_GAN_loss            │
                                    └──────┬─────────────────────────────┬─────────────────────┘
                                           │ z_IB_L,z_IB_P               │ z_L,z_P,atom_w
                                           │ (B,S,D)                     │ mu,ω,A,C,E
                                    ┌──────▼─────────────────────────────▼─────────────────────┐
                                    │  ENCODER V4 (PGTA×2 + IB)                                │
                                    │  x0 ──► LN ──► PGTA_attn0 ──► +res ──► FFN ──► +res      │
                                    │      ──► LN ──► PGTA_attn1 ──► +res ──► FFN ──► +res      │
                                    │      ──► IB_L(z_IB_L) ──► z_L_mu, z_L_logvar → z_L'      │
                                    │      ──► IB_P(z_IB_P) ──► z_P_mu, z_P_logvar → z_P'      │
                                    │      ──► loss_IB = KL_L + KL_P                            │
                                    │      ──► loss_H = λ_H·ReLU(H_enc - H_target)              │
                                    │  输出: x_enc (B,S,D)                                      │
                                    └──────────────────────┬───────────────────────────────────┘
                                                           │ x_enc, z_IB_L', z_IB_P'
                        ┌──────────────────────────────────┼──────────────────────────────────┐
                        ▼                                  ▼                                  ▼
            ┌──────────────────────┐          ┌──────────────────────┐           ┌──────────────────────┐
            │ LANGEVIN OSCILLATOR  │          │ DUAL MEMORY SYSTEM   │           │ (domain outputs      │
            │ 逻辑域(L)             │          │  ┌────────────────┐  │           │  routed to physics)  │
            │ 诗意域(P)             │          │  │ EpisodicMem L  │  │           │                      │
            │                      │          │  │ EpisodicMem P  │  │           │  mu_L/P: (Na,Da)     │
            │ r0 = ||W_r(mu)||     │          │  │ 4096 slots/域  │  │           │  ω: (Na,) eigenfreq  │
            │ θ0 = θ_pe + W_θ(μ,u) │          │  │ KV ring buffer │  │           │  A = C⊙(1-E):        │
            │                      │          │  │ time-decay     │  │           │    (Na,Na) coupling  │
            │ For k=1..K:          │          │  └────────────────┘  │           └──────────────────────┘
            │   coupling = ΣJ·r_j  │          │  ┌────────────────┐  │
            │     ·sin(θ_j-θ_i-φ)  │          │  │ StructMem L    │  │
            │   θ += dt(ω+coup)    │          │  │ StructMem P    │  │
            │     + √(2Tdt)·ξ      │          │  │ 1024 skills    │  │
            │   F = ½k(r-r₀)²      │          │  └────────────────┘  │
            │     - ΣJ·r_i·r_j     │          └──────────┬───────────┘
            │     ·cos(Δθ-φ) - T·S │                     │ epi_L, epi_P
            │   dr/dt = -∇_r F + ξ │                     │ (B,S,D)
            │   r = clamp(r,ε,∞)   │                     │
            │                      │                     │
            │ Output:              │                     │
            │   r: (B,Na)          │                     │
            │   θ: (B,Na)          │                     │
            │   Z_phys: (B,Na,D)   │                     │
            │   F, dF: scalar      │                     │
            └──────────┬───────────┘                     │
                       │ atom_w einsum mapping           │
                       │ Z_phys_tokens = Σ atom_w·Z_phys  │
                       │ θ_tokens = Σ atom_w·θ            │
                       ▼                                  ▼
            ┌─────────────────────────────────────────────────────────────────┐
            │ META COGNITIVE GATE                                            │
            │                                                                │
            │ TriPercept(3×MLP 投票):                                         │
            │   cat(z_L'.mean(S), z_P'.mean(S), task_emb) → 3×MLP → softmax  │
            │   α_L = median(α₁,α₂,α₃)                                       │
            │   u_cog = Var(α₁,α₂,α₃) / 0.25   (认知不确定性)                  │
            │                                                                │
            │ 物理融合: z_L = (1-λ)z_IB_L' + λ·Z_phys_L                      │
            │ SafetyFilter: 64 danger anchors, cosine_sim → skeleton         │
            │ FuseMLP: cat(Z_IB, epi_mem, Z_safe, task_exp) → Z_cog          │
            │ Temperature: T = T₀·(1+κ·u_cog), clamp[T_min,T_max]            │
            │                                                                │
            │ Output: Z_cog (B,S,D), T (scalar), u_cog (scalar),             │
            │         α_L/α_P (B,1,1), safety_score (B,S)                    │
            └────────────────────────────┬────────────────────────────────────┘
                                         │ Z_cog, T, θ_tokens
                                    ┌────▼────────────────────────────────────┐
                                    │ DECODER V4  (× n_layers=6/8 layers)      │
                                    │                                         │
                                    │ Each GLUBlockV4:                        │
                                    │  ┌─────────────────────────────────┐    │
                                    │  │ 1. Pre-LN → PGTA self-attention │    │
                                    │  │    (每层独立PGTA, 温度由meta_T)  │    │
                                    │  │    A = softmax(-E_eff/T)        │    │
                                    │  │    E_eff = E - βT·log τ - T·C  │    │
                                    │  └──────────────┬──────────────────┘    │
                                    │                 │ +residual             │
                                    │  ┌──────────────▼──────────────────┐    │
                                    │  │ 2. Pre-LN → GLU + TSR + PSR     │    │
                                    │  │   g = W_g·x → u,v = chunk(g,2) │    │
                                    │  │   TSR: γ=σ(Wx+Wz·Z_cog)         │    │
                                    │  │     M=γM+(1-γ)·Wx              │    │
                                    │  │   PSR: γ_p=exp(-Δθ²/(2T))      │    │
                                    │  │     M_p=γ_p M_p+(1-γ_p)Wx     │    │
                                    │  │   u = u + TSR_out + PSR_out    │    │
                                    │  └──────────────┬──────────────────┘    │
                                    │                 │                       │
                                    │  ┌──────────────▼──────────────────┐    │
                                    │  │ 3. Z_cog层级注入 + 物理门控α    │    │
                                    │  │   pel = pos_layer_emb[l]        │    │
                                    │  │   Z_l = MLP(cat(Z_cog,pel))     │    │
                                    │  │   α_t = σ(Wx + Wz·Z_l)         │    │
                                    │  │   α_p = σ(cos(θ-θ_l)/T)         │    │
                                    │  │   α = (1-λ_T)α_t + λ_T·α_p     │    │
                                    │  │   scale/bias = Z_l → W_s/W_b    │    │
                                    │  │   u = u·(1+scale)·α             │    │
                                    │  │       + bias·(1-α)              │    │
                                    │  │   out = σ(u)·v → W_proj         │    │
                                    │  └──────────────┬──────────────────┘    │
                                    │                 │ +residual             │
                                    └─────────────────┼───────────────────────┘
                                                      │ h: (B,S,D)
                                                      ▼
                                    ┌─────────────────────────────────────────┐
                                    │ FINAL NORM + COUNTER + POINTER          │
                                    │                                         │
                                    │ final_ln: LayerNorm(D)                  │
                                    │ CounterSlot: GRUCell(d_c, d_c)           │
                                    │   for t in range(S):                    │
                                    │     h_cnt = GRU(noise, h_cnt)           │
                                    │   h += tanh(W_cnt(h_cnt))*0.1           │
                                    │ PointerGate: p_copy = σ(MLP(h))  (B,S)  │
                                    └────────────────────┬────────────────────┘
                                                         │ h
                                                         ▼
                                    ┌─────────────────────────────────────────┐
                                    │ LM HEAD (weight-tied to token_emb)      │
                                    │ logits = h @ W_emb.T · 1/√D             │
                                    │ probs = safe_softmax(logits, T=T)       │
                                    │ Copy: p_copy · cum_onehot(S)             │
                                    │        + (1-p_copy) · probs             │
                                    └────────────────────┬────────────────────┘
                                                         │
                              ┌──────────────────────────┴─────────────────────┐
                              ▼                                                ▼
                   ┌─────────────────────┐                     ┌──────────────────────────┐
                   │ LOSS COMPUTATION    │                     │ EVOLVABLE WEIGHT SYSTEM   │
                   │ L_LM = CE(shift)    │                     │ (非梯度,每步后调用)        │
                   │ L_IB = KL_L + KL_P  │                     │                           │
                   │ L_D = BCE(D,0.5)    │                     │ Phase A: no-op            │
                   │ L_aff = λ·std(z_P)  │                     │ Phase B:                  │
                   │ L_H = λ·ReLU(H-H*)  │                     │   r = -loss               │
                   │ L_mod = module_loss │                     │   τ ← (1-ρ)τ + η·r·Ā     │
                   │ L_anc = anchor_loss │                     │ Phase C:                  │
                   │ L_phys = λ·F²       │                     │   r = -dF/(|dF|+1)        │
                   │                     │                     │   τ ← (1-ρ)τ + η·r·Ā     │
                   │ L_total = Σ λ_i L_i │                     │   every 100 steps:         │
                   │         + L_LM      │                     │     consolidate():         │
                   │                     │                     │     if τ>θ: C+=λ(τ-θ)     │
                   │                     │                     │            τ*=(1-γ)        │
                   └─────────────────────┘                     └──────────────────────────┘
```

### 2.2 模块清单与代码索引

| 编号  | 模块      | 文件                                                                                                    | 类名                              | 行数      | 版本    |
| --- | ------- | ----------------------------------------------------------------------------------------------------- | ------------------------------- | ------- | ----- |
| M1  | 主模型     | [aethermind4.py](file:///d:/AetherMind-Nano3/src/model/aethermind4.py)                                | `AetherMind4`                   | 263-515 | V4    |
| M2  | V4编码器   | [aethermind4.py#L34-L138](file:///d:/AetherMind-Nano3/src/model/aethermind4.py#L34-L138)              | `EncoderV4`                     | 34-138  | V4    |
| M3  | V4解码块   | [aethermind4.py#L144-L257](file:///d:/AetherMind-Nano3/src/model/aethermind4.py#L144-L257)            | `GLUBlockV4`                    | 144-257 | V4    |
| M4  | PGTA注意力 | [attention/pheromone\_thermo.py](file:///d:/AetherMind-Nano3/src/model/attention/pheromone_thermo.py) | `PheromoneThermoAttention`      | 20-218  | V4新增  |
| M5  | 演化控制器   | [evolution/evolvable\_weight.py](file:///d:/AetherMind-Nano3/src/model/evolution/evolvable_weight.py) | `EvolvableWeightSystem`         | 20-123  | V4新增  |
| M6  | 双域GMM   | [domain/dual\_domain.py](file:///d:/AetherMind-Nano3/src/model/domain/dual_domain.py)                 | `DualDomainSystem`/`GMMAtom`    | 9-196   | 3.6继承 |
| M7  | 朗之万     | [physics/langevin.py](file:///d:/AetherMind-Nano3/src/model/physics/langevin.py)                      | `LangevinOscillator`            | 9-220   | 3.6继承 |
| M8  | 双记忆     | [memory/dual\_memory.py](file:///d:/AetherMind-Nano3/src/model/memory/dual_memory.py)                 | `DualMemorySystem`              | 73-92   | 3.6继承 |
| M9  | 元认知门    | [metacog/meta\_gate.py](file:///d:/AetherMind-Nano3/src/model/metacog/meta_gate.py)                   | `MetaCognitiveGate`             | 49-99   | 3.6继承 |
| M10 | 工具函数    | [utils/ops.py](file:///d:/AetherMind-Nano3/src/utils/ops.py)                                          | `MLP`/`safe_softmax`/`stopgrad` | 1-96    | 3.6继承 |
| M11 | 配置      | [configs/aethermind4\_config.py](file:///d:/AetherMind-Nano3/configs/aethermind4_config.py)           | `AetherMind4Config`             | 7-170   | V4    |
| M12 | 训练器     | [training/train\_v4.py](file:///d:/AetherMind-Nano3/src/training/train_v4.py)                         | `TrainerV4`                     | 1-221   | V4    |
| M13 | 数据集     | [data/dataset.py](file:///d:/AetherMind-Nano3/src/data/dataset.py)                                    | `StreamingTextDataset`          | -       | 3.6继承 |

### 2.3 张量Shape总表（默认配置 d=512, h=8, dh=64, S=1024, Na=1024, Da=64, B=1）

| 张量              | Shape             | 类型          | 说明                      |
| --------------- | ----------------- | ----------- | ----------------------- |
| input\_ids      | (B,S)             | int64       | 输入token id              |
| token\_emb      | (B,S,D)           | fp32/bf16   | token嵌入                 |
| pos\_emb        | (B,S,D)           | fp32/bf16   | 可学习位置编码                 |
| atom\_w\_L/P    | (B,S,Na)          | fp32/bf16   | token→原子软分配权重           |
| z\_L/P          | (B,S,Da)          | fp32/bf16   | 原子空间token表征             |
| z\_IB\_L/P      | (B,S,D)           | fp32/bf16   | 经世界模型+投影的IB输入           |
| mu\_L/P         | (Na,n\_comp,Da)   | fp32        | GMM分量均值                 |
| omega\_L/P      | (Na,)             | fp32        | 本征频率 √σ̄                |
| A\_L/P (关联矩阵)   | (Na,Na)           | fp32        | 原子间耦合强度 C⊙(1-E)         |
| r               | (B,Na)            | fp32/bf16   | 朗之万振幅                   |
| θ               | (B,Na)            | fp32/bf16   | 朗之万相位                   |
| Z\_phys         | (B,Na,D)          | fp32/bf16   | 物理输出 (phys\_mlp投影后)     |
| Z\_phys\_tokens | (B,S,D)           | fp32/bf16   | atom\_w einsum映射到token级 |
| θ\_tokens       | (B,S)             | fp32/bf16   | 相位token级映射              |
| epi\_L/P        | (B,S,D)           | fp32/bf16   | 情景记忆检索结果                |
| Z\_cog          | (B,S,D)           | fp32/bf16   | 元认知融合表征                 |
| T               | scalar→(1,)       | fp32        | 元认知温度                   |
| Q/K/V in PGTA   | (B,h,S,dh)        | fp32/bf16   | Q/K/V投影分头后              |
| E (能量矩阵)        | (B,h,S,S)         | fp32/bf16   | -QK/√dh                 |
| τ (信息素)         | (h,S\_max,S\_max) | fp32 buffer | 跨步持久                    |
| C (固化)          | (h,S\_max,S\_max) | fp32 buffer | 永久偏置                    |
| A (注意力)         | (B,h,S,S)         | fp32/bf16   | softmax(-E\_eff/T)      |
| PGTA out        | (B,S,D)           | fp32/bf16   | Wo(A\@V)                |
| h (解码hidden)    | (B,S,D)           | fp32/bf16   | 解码器各层输出                 |
| TSR state M     | (B,S,Ds)          | fp32/bf16   | 跨层认知状态                  |
| PSR state M\_p  | (B,S,Ds)          | fp32/bf16   | 跨层物理状态                  |
| logits          | (B,S,V)           | fp32/bf16   | LM输出（权重绑定前）             |
| p\_copy         | (B,S)             | fp32/bf16   | 复制门概率                   |
| out\_prob       | (B,S,V)           | fp32/bf16   | 最终输出分布                  |

***

## 三、PGTA 信息素热力学注意力（V4核心创新）

### 3.1 从标准Softmax注意力到Boltzmann分布

**标准缩放点积注意力**（Vaswani 2017）：

$A\_{ij} = \text{softmax}\left(\frac{Q\_i K\_j^T}{\sqrt{d\_h}}\right), \quad O\_i = \sum\_j A\_{ij} V\_j$

本质是统计归一化——把相似度当对数几率，softmax只是概率归一化操作。温度隐式固定为1，没有物理自由度。

**V4重构为Boltzmann分布**：

将"相似度"定义为**能量** $E$（越相似能量越低），注意力权重定义为**正则系综中处于该状态的概率**：

$E^{(h)}\_{ij} = -\frac{Q^{(h)}\_i \cdot K^{(h)}\_j}{\sqrt{d\_h}}$

$A^{(h)}_{ij} = \frac{\exp(-E^{(h)}_{\text{eff},ij}/T)}{\sum\_k \exp(-E^{(h)}\_{\text{eff},ik}/T)}$

其中 $T$ 是**可学习温度**（log参数化 $T = \exp(\log T)$，保证正值）：

- $T \to 0$：注意力坍缩为one-hot（贪婪/确定性）
- $T = 1$：等价于标准softmax
- $T \to \infty$：注意力均匀分布（完全随机/最大熵探索）

这直接对应**统计物理中的模拟退火**：高温探索、低温收敛。

### 3.2 能量白化（统计辅助层）

为了让温度$T$的物理意义不依赖于数据尺度（否则不同层/不同训练阶段$E$的方差不同，$T=1$对应的"温度"不一致），对能量矩阵做运行统计白化：

$\mu\_E^{(t)} = m \cdot \mu\_E^{(t-1)} + (1-m) \cdot \text{mean}(E^{(t)})$
$\sigma\_E^{(t)} = m \cdot \sigma\_E^{(t-1)} + (1-m) \cdot \text{std}(E^{(t)})$
$\hat{E} = \frac{E - \mu\_E}{\sigma\_E + \epsilon}$

动量系数 $m = 0.99$（代码中 `stats_momentum`）。白化仅在训练时更新统计量，推理时使用固定统计量。

### 3.3 信息素路径调制（群体智能层）

信息素矩阵 $\tau^{(h)} \in \mathbb{R}^{S\_{\max} \times S\_{\max}}$ 是 `register_buffer`（非参数、不参与梯度），形状为 $(h, S\_{\max}, S\_{\max})$，跨forward持久保存。它模拟蚁群算法中的stigmergy（蚁迹通信）：**走过的路留下信息素，信息素浓的路更易被走**。

**有效能量**（融合结构能量和路径偏好）：

$\tilde{E}_{ij} = \hat{E}_{ij} - \beta T \log \tau\_{ij} - T \cdot C\_{ij}$

展开softmax后：

$A\_{ij} \propto \exp(-\hat{E}_{ij}/T) \cdot \tau_{ij}^{\beta} \cdot \exp(C\_{ij})$

这就是完整的三因子注意力公式：

| 因子      | 公式                 | 来源    | 更新方式          |
| ------- | ------------------ | ----- | ------------- |
| **结构项** | $\exp(-\hat{E}/T)$ | QK相似度 | BP学习W\_Q/W\_K |
| **路径项** | $\tau^\beta$       | 信息素缓冲 | 沉积+蒸发（非梯度）    |
| **固化项** | $\exp(C)$          | 永久偏置  | LTP阈值写入       |

- $\beta$：信息素敏感度（ACO中的$\alpha$参数），默认1.0
- $C$：固化权重偏置，见3.5节
- $\tau \geq \tau\_{\min} = 0.01$：防止log(0)数值错误

**信息论解释**：$\beta T \log \tau$ 项等价于给注意力分布加了一个由信息素诱导的先验 $P\_{\text{prior}} \propto \tau^\beta$，Boltzmann分布变为后验。

### 3.4 信息论量：熵、自由能、方差

每步注意力计算后，提取三个信息论/统计量作为训练信号和状态指标：

**注意力熵**（分布的聚焦程度）：

$H^{(h)}_i = -\sum\_j A_{ij} \log A\_{ij}, \quad \bar{H} = \text{mean}\_{i,h}(H^{(h)}\_i)$

**Helmholtz自由能**（物理目标量）：

$F = \langle \tilde{E} \rangle\_A - T \cdot \bar{H} = \sum\_{i,j} A\_{ij} \tilde{E}\_{ij} - T \bar{H}$

Boltzmann分布 $A \propto \exp(-\tilde{E}/T)$ 恰好是在固定$T$下\*\*最小化自由能$F$\*\*的分布——这不是巧合，而是统计物理的基本结论。因此注意力计算本身就在解一个变分自由能最小化问题。

**输出方差**（不确定性度量）：

$\text{Var}\_i = \mathbb{E}_A\[V^2] - (\mathbb{E}A\[V])^2 = \sum\_j A_{ij} V\_j^2 - \left(\sum\_j A{ij} V\_j\right)^2$

方差直接给出每个查询位置的"置信度"——方差大表示注意力分散、模型不确定。

### 3.5 信息素动力学：蒸发与沉积

每个训练/推理步结束后（forward返回后，由EvolvableWeightSystem统一调度），信息素矩阵按**非梯度物理规则**更新：

**蒸发**（用进废退/遗忘）：

$\tau \leftarrow (1 - \rho) \cdot \tau$

$\rho = 0.05$（Phase C降至0.03保护好路径）。所有边统一衰减，模拟蚁群中信息素的自然挥发。

**沉积**（奖励门控的路径强化）：

$\Delta\tau = \eta \cdot r \cdot \bar{A}, \quad \tau \leftarrow \tau + \Delta\tau, \quad \tau \leftarrow \text{clip}(\tau, \tau\_{\min}, \tau\_{\max})$

其中：

- $\eta = 0.05$：沉积强度
- $\bar{A} = \text{mean}\_B(A)$：本步batch平均注意力模式（"走了哪条路"）
- $r$：信用/奖励信号（见3.6节），决定"这条路好不好"
- 沉积对所有被注意力访问过的边(i,j)按 $A\_{ij}$ 比例增加信息素

**边界保护**：$\tau \in \[\tau\_{\min}, \tau\_{\max}] = \[0.01, 5.0]$，防止信息素发散到无穷或衰减到零。

### 3.6 信用信号设计（四种模式）

信用信号$r$决定沉积是正（强化好路径）还是负（削弱差路径）。V4实现了四种模式：

| 模式            | 公式                                                     | 适用阶段      | 特点             | <br />  | <br />    |
| ------------- | ------------------------------------------------------ | --------- | -------------- | :------ | :-------- |
| `hard`        | $r = \mathbb{1}\[\text{pred}=\text{target}] \in {0,1}$ | -         | 粗糙0/1，只知对错不知程度 | <br />  | <br />    |
| `soft`        | $r = P(\text{correct}) \in (0,1)$                      | -         | 连续但有偏（总是正）     | <br />  | <br />    |
| `soft_center` | $r = 2(P(\text{correct}) - 0.5) \in (-1,1)$            | Phase B默认 | 零中心化，对加错撤      | <br />  | <br />    |
| `free_energy` | \$r = -\text{clip}(dF/(                                | dF        | +1), -1, 1)\$  | Phase C | 最密信号：内在动机 |

**为什么Phase C用free\_energy**：参考 `pgtt_self_evolution.py` 的实验结论——粗糙0/1奖励会让蚁群锁定"够用但非因果"的退化解（7个种子中0-14%命中率锁定真路由），而自由能$dF = F\_t - F\_{t-1}$提供了连续的、密集的信用信号：

- $dF < 0$（自由能下降/模型更确定）→ 正奖励，当前路径被强化
- $dF > 0$（自由能升高/遇到意外）→ 负奖励，当前路径被削弱
- $|dF| \approx 0$（稳定态）→ 无奖励，信息素自然蒸发

这实现了**无监督内在动机**：模型主动强化能降低自身预测不确定性的注意力模式。

### 3.7 LTP固化偏置$C$

$C^{(h)} \in \mathbb{R}^{S\_{\max} \times S\_{\max}}$ 是第二个 `register_buffer`，初始化为零。它代表**已固化的永久权重偏置**——经验从短期信息素$\tau$写入长期权重$C$的"LTP长时程增强"过程（详见第四章）。

$C$在有效能量中以 $-T \cdot C$ 形式出现：$C\_{ij} > 0$ 的边(i,j)能量更低（被永久偏好）。展开后这等价于 $\exp(C\_{ij})$ 的乘性偏置。

### 3.8 温度控制机制

温度$T$有两种控制方式，V4使用元认知门控覆盖：

1. **可学习参数**：`log_temp`作为`nn.Parameter`，通过BP梯度更新（$\partial \mathcal{L}/\partial \log T$）
2. **外部覆盖**：`set_temperature(T_val)`由MetaCognitiveGate调用，根据认知不确定性$u\_{\text{cog}}$设定
   - $u\_{\text{cog}}$高（三感知器投票分歧大）→ 高温探索
   - $u\_{\text{cog}}$低（三感知器一致）→ 低温利用
   - 具体公式：$T = T\_0(1 + \kappa \cdot u\_{\text{cog}})$，见第八章

V4的forward中，元认知门输出的$T$通过`attn.set_temperature(T_val)`覆盖所有解码器PGTA层的温度，实现全局温度协调。编码器PGTA层使用自身可学习温度（因为编码器在元认知门之前运行）。

### 3.9 PGTA前向伪代码

```python
def PheromoneThermoAttention.forward(x, extra_mask=None, T_override=None, update_pheromone=True):
    B, S, D = x.shape
    h, dh = num_heads, D // num_heads

    # 1. Q/K/V投影
    Q = Wq(x).view(B,S,h,dh).transpose(1,2)   # (B,h,S,dh)
    K = Wk(x).view(B,S,h,dh).transpose(1,2)
    V = Wv(x).view(B,S,h,dh).transpose(1,2)

    # 2. 能量计算
    sim = einsum("bhid,bhjd->bhij", Q, K) / sqrt(dh)
    E = -sim                                    # (B,h,S,S)

    # 3. 能量白化（训练时更新运行统计）
    if whiten and training:
        em = momentum * energy_mean + (1-momentum) * E.mean()
        es = momentum * energy_std  + (1-momentum) * E.std(unbiased=False)
        energy_mean.copy_(em)
        energy_std.copy_(es)
        E = (E - energy_mean) / (energy_std + 1e-5)

    # 4. 温度
    if T_override is not None:
        T = tensor(T_override)
    else:
        T = exp(log_temp).clamp(1e-2, 1e2)

    # 5. 信息素+固化调制有效能量
    tau_slice = tau[:h, :S, :S].unsqueeze(0)      # (1,h,S,S)
    log_tau = log(tau_slice.clamp(min=tau_min))
    E_eff = E - beta * T * log_tau                # 信息素偏置

    cons_slice = consolidated[:h, :S, :S].unsqueeze(0)
    E_eff = E_eff - T * cons_slice                 # 固化偏置

    # 6. Causal/padding mask
    if extra_mask is not None:
        E_eff.masked_fill_(extra_mask, +inf)

    # 7. Boltzmann注意力
    A = softmax(-E_eff / T, dim=-1)               # (B,h,S,S)
    A = attn_drop(A)

    # 8. 加权求和输出
    out = einsum("bhij,bhjd->bhid", A, V)         # (B,h,S,dh)
    out = out.transpose(1,2).reshape(B,S,D)
    out = Wo(out)

    # 9. 信息论统计量
    entropy = -(A * (A+1e-8).log()).sum(-1).mean()
    free_energy = (A * E_eff.detach()).sum(-1).mean() - T.detach() * entropy.detach()
    out_sq = einsum("bhij,bhjd->bhid", A, V*V)
    variance = (out_sq - out.view(B,S,h,dh).transpose(1,2)**2).mean()
    tau_conc = tau[:h,:S,:S].max() / (tau[:h,:S,:S].mean() + 1e-9)

    # 10. 缓存供信息素沉积
    _last_A = A.detach()
    _last_E = E_eff.detach()

    return out, {entropy, free_energy, temperature:T, variance,
                 attention:A, tau_concentration:tau_conc}
```

**信息素步进伪代码**（在forward返回后由evolver调用）：

```python
def step_pheromone(reward=None):
    if _last_A is None: return
    A = _last_A.detach()                          # (B,h,S,S)
    B, h, S, _ = A.shape

    with torch.no_grad():
        # 蒸发
        tau[:, :S, :S].mul_(1 - rho)

        # 沉积
        if reward is not None:
            if reward.dim() == 0:
                delta = deposit * reward.item() * A.mean(0)[:, :S, :S]
            elif reward.dim() == 1:
                r = reward.view(-1,1,1,1)
                delta = deposit * (r * A).mean(0)[:, :S, :S]
            else:
                delta = deposit * reward.detach().mean(0)[:, :S, :S]
        else:
            delta = deposit * A.mean(0)[:, :S, :S]   # 无奖励按用量均匀沉积

        tau[:, :S, :S].add_(delta)
        tau.clamp_(tau_min, tau_max)
```

***

## 四、可演化双权重系统

### 4.1 双权重设计原理

V4的注意力层同时维护三类权重/状态，形成**快-中-慢**三级记忆系统：

| 名称                 | 存储类型              | 张量Shape             | 更新方式         | 生命周期                | 类比          |
| ------------------ | ----------------- | ------------------- | ------------ | ------------------- | ----------- |
| **快权重** **$W$**    | `nn.Parameter`    | Wq/Wk/Wv/Wo: (D,D)各 | 梯度下降(BP)     | 训练期可学，Phase C后冻结    | 语义记忆/大脑皮层   |
| **信息素** **$\tau$** | `register_buffer` | (h, S\_max, S\_max) | 沉积+蒸发（物理规则）  | 跨样本持久，session可reset | 工作记忆/短期突触效能 |
| **固化** **$C$**     | `register_buffer` | (h, S\_max, S\_max) | LTP阈值写入（非梯度） | 永久持久，跨session保留     | 长期程序记忆/LTP  |

关键区别：

- **W改变需要BP**：要标签、要反向传播、要大batch；部署后不可行
- **τ改变不需要梯度**：纯物理规则（乘+加+clamp），每步几十微秒；在线运行
- **C改变不需要重新训练**：阈值门控写入；写入后永久改变模型行为

这正是生物神经系统的三级记忆模型：

- W = 长期记忆中的**语义知识**（通过学习获得，相对稳定）
- τ = 短期/工作记忆（当前session的神经活动痕迹，会消退）
- C = 长期程序记忆（反复激活的突触被LTP巩固，永久保留）

### 4.2 LTP固化机制详解

长时程增强（LTP）是神经科学中记忆巩固的核心机制：高频刺激突触后，突触效能发生持久性增强。V4实现为阈值门控的稀疏写入：

**触发条件**：Phase C中，每 `consolidate_interval=100` 步执行一次：

$\text{mask} = (\tau > \theta\_c), \quad \theta\_c = 1.5$

**写入规则**：

$C\[\text{mask}] \mathrel{+}= \lambda \cdot (\tau\[\text{mask}] - \theta\_c), \quad \lambda = 0.1$

$\tau\[\text{mask}] \leftarrow (1 - \gamma) \cdot \tau\[\text{mask}], \quad \gamma = 0.5$

**机制解读**：

1. 只有信息素浓度持续**超过阈值**的路径（即被反复走过的稳定路径）才被固化
2. 超阈值的部分（$\tau - \theta\_c$，而非全部$\tau$）写入$C$——只有"超出基线的强信号"被巩固
3. 写入后$\tau$局部衰减50%——**短期记忆让位给长期权重**，释放工作记忆容量
4. $\lambda=0.1$较小——固化是保守的、渐进的，防止单次偶然路径被永久写入

**预算控制**：`max_consolidations=4096`——全局固化计数器超过上限后停止固化，防止权重无限膨胀。

**为什么固化有效**：$C$以$-T \cdot C$形式进入有效能量$\tilde{E}$，相当于在softmax中乘以$\exp(C)$。设固化后$C\_{ij} = 2.0$，则该边注意力权重乘$\exp(2.0) \approx 7.4$倍——路径偏好被永久放大。更重要的是，$\tau$被reset后$C$仍然存在，所以经验跨session保留。

MVP实验验证（来自 `evolvable_weight.py`）：

| 模型  | 训练后τ峰值    | 操作     | 清空τ后CLS指向        |
| --- | --------- | ------ | ---------------- |
| 有固化 | 3.95 @正确列 | 固化→清τ  | **正确列(0.804)** ✓ |
| 无固化 | 3.95 @正确列 | 不固化→清τ | 随机列(0.095) ✗     |

### 4.3 EvolvableWeightSystem控制器

`EvolvableWeightSystem`是统一管理所有PGTA层信息素生命周期的控制器：

**注册**：模型初始化时，所有PGTA层（编码器2层+解码器n\_layers层）通过`register_attention()`注册到evolver。

**每步调度**：`evolver.forward(step, free_energy, phase, loss_val)` 在训练步的optimizer.step()之后调用：

```python
def forward(step, free_energy=None, phase="A", loss_val=None):
    if phase == "A":
        return  # Phase A纯BP，不演化

    # 计算奖励
    if phase == "C" and free_energy is not None:
        reward = compute_free_energy_reward(free_energy, F_prev)
        # r = -clip((F - F_prev) / (|F - F_prev| + 1), -1, 1)
    elif loss_val is not None:
        reward = clip(-loss_val.detach(), -1, 1)   # Phase B用loss
    else:
        reward = None

    # 所有PGTA层信息素更新
    for attn in _attention_layers:
        attn.step_pheromone(reward)

    # 定期固化（仅Phase C）
    if phase == "C" and step % consolidate_interval == 0 and step > 0:
        for attn in _attention_layers:
            attn.consolidate(threshold, lam, gamma, max_cons)
        _consolidation_count += 1
```

**统计量**：

- `tau_concentration` = mean(max(τ)/mean(τ))：信息素集中度，越高表示路径越分化（初始=1.0）
- `consolidation_mass` = Σ|C|：固化总质量，表示已巩固的经验量（初始=0）

***

## 五、GMM双域知识原子（继承3.6）

### 5.1 设计原理

AetherMind从3.5起就采用**双域分离**：语言理解被分解为**逻辑域**（事实/推理/语法，对应左脑）和**诗意域**（情感/隐喻/风格，对应右脑），每个域有独立的GMM知识原子系统。

### 5.2 GMMAtom结构

每个域维护 $N\_a=1024$ 个知识原子，每个原子是一个**3分量高斯混合模型**（GMM）：

| 参数              | Shape         | 说明                 |
| --------------- | ------------- | ------------------ |
| mean\_emb       | (Na, K, Da)   | K=3个分量的均值向量        |
| logvar\_emb     | (Na, K, Da)   | 对数方差（经softplus→σ²） |
| mix\_logits     | (Na, K)       | 混合权重π（经softmax）    |
| mass            | (Na,)         | 惯性质量（朗之万用）         |
| tau             | (Na,)         | 弛豫时间（朗之万用）         |
| token\_to\_atom | Linear(D, Na) | token→原子的投影（激活用）   |
| atom\_to\_token | Linear(Da, D) | 原子→token空间投影       |

其中 $K = n\_{\text{gmm\_components}} = 3$，$D\_a = d\_{\text{atom}} = 64$。

**混合参数提取**：
$\mu = \text{mean\_emb}, \quad \sigma^2 = \text{softplus}(\text{logvar\_emb}) + 10^{-6}, \quad \pi = \text{softmax}(\text{mix\_logits})$

### 5.3 激活过程

```python
def activate(x):  # x: (B,S,D)
    atom_logits = token_to_atom(x)              # (B,S,Na)
    atom_weight = softmax(atom_logits, dim=-1)  # (B,S,Na) 软分配

    mu, sigma, pi = get_mixture_params()
    mean_atom = (pi.unsqueeze(-1) * mu).sum(dim=1)   # (Na, Da) 加权均值

    z = einsum("bsn,nd->bsd", atom_weight, mean_atom)  # (B,S,Da)
    return atom_weight, z
```

**解读**：每个token通过softmax获得对1024个原子的软分配权重 $w \in \Delta^{N\_a-1}$（概率单纯形），然后以这些权重加权求和各原子的均值，得到token在原子空间的表示 $z \in \mathbb{R}^{D\_a}$。

这是一种**软路由的MoE**：1024个"专家原子"被每个token按概率组合激活。

### 5.4 关联矩阵与世界模型

每个域维护一个**可学习的关联矩阵对** (C, E)，形状均为 (Na, Na)：

- $C\_{ij} = \text{sigmoid}(\text{logic\_assoc\_C}\_{ij})$：原子$i$和$j$之间的耦合强度
- $E\_{ij} = \text{sigmoid}(\text{logic\_assoc\_E}\_{ij})$：存在边（1=不存在，0=存在）
- 有效耦合：$A = C \odot (1-E)$，即"有边且有强度"

世界模型MLP `logic_world_model: MLP(D, 2D, D)` 将原子→token投影后的表征再做变换，预测 token 级表征：$z\_{\text{IB}} = \text{atom\_to\_token}(z) + \text{world\_model}(\cdot)$。

诗意域判别器 `poetic_discriminator: MLP(D, D, 1)` 输出 D\_score ∈ (0,1)，用于对抗训练防止诗意域坍缩——通过BCE(D\_score, 0.5)逼迫判别器无法区分真实/生成的诗意表征（类似GAN的思想，但更温和）。

### 5.5 锚点对齐损失

```
n_anchor=512个锚点(anchor_emb: (n_anc, Da))
每个域有独立router: Linear(Da, n_anc)
anc = argmax(router(z_flat))       # 硬路由到最近锚点
z_anc = stopgrad(anchor_emb[anc])  # 锚点向量（stopgrad防塌缩）
loss_anc = λ_align · [MSE(z_L_flat, z_L_anc) + MSE(z_P_flat, z_P_anc)]
```

锚点对齐使原子空间的分布均匀填充 $n\_{\text{anchor}}=512$ 个原型位置，防止所有原子坍缩到一个簇。

### 5.6 模块度损失

关联矩阵A上施加模块度约束：使有边的原子对($A\_{ij}\approx 1$)嵌入距离近，无边的原子对($A\_{ij}\approx 0$)嵌入距离至少为$\delta=1$：

$\mathcal{L}_{\text{mod}} = \sum_{i,j} A\_{ij} |u\_i - u\_j|^2 + \sum\_{i,j} (1-A\_{ij}) \cdot \text{ReLU}(\delta - |u\_i - u\_j|)^2$

其中$u$ = module\_emb（原子的模块嵌入，Na×Da）。

***

## 六、朗之万振荡器（继承3.6）

### 6.1 物理背景

朗之万方程描述了在热噪声环境中受势场力和耦合力的粒子运动。V4将每个知识原子建模为一个**耦合相位振荡器**（类似Kuramoto模型），原子振幅$r$和相位$\theta$在K步朗之万迭代中收敛到低自由能状态，实现编码端的跨域概念对齐和相位同步。

### 6.2 复振幅初始化

每个原子$n$的初始振幅和相位由GMM均值、位置编码、模块嵌入决定：

**振幅初始**：
$r\_{0,n} = |W\_r(\mu\_n)|\_2 + \epsilon$
其中 $W\_r: \mathbb{R}^{D\_a} \to \mathbb{R}^{D\_a}$ 是线性投影，取L2范数得到振幅。$r\_0$ 同时作为$r$的稳态目标（胡克定律平衡位置）。

**相位初始**：

- 位置编码贡献：$\theta\_{\text{pe}} = \text{atan2}(\overline{\sin\text{PE}}, \overline{\cos\text{PE}})$（位置编码的平均相位）
- 模块嵌入贡献：$\theta\_u = W\_\theta(\text{cat}(\mu\_n, u\_n))$（通过可学习MLP）
- 初始相位：$\theta\_n = \theta\_{\text{pe},n} + \theta\_{u,n}$

**本征频率**：
$\omega\_n = W\_\omega(\mu\_n), \quad \text{其中 } W\_\omega: \mathbb{R}^{D\_a} \to \mathbb{R}$
也可以从GMM方差推导：$\omega\_n = \sqrt{\bar{\sigma}\_n^2 + \epsilon}$。

### 6.3 自由能函数

系统总自由能：

$F = \underbrace{\frac{1}{2}k\sum\_n (r\_n - r\_{0,n})^2}_{F\_1:\text{胡克势能}} - \underbrace{\sum_{i,j} J\_{ij} r\_i r\_j \cos(\theta\_i - \theta\_j - \phi\_{ij})}_{F\_2:\text{耦合势能}} - \underbrace{T \cdot S_{\text{phase}}}\_{F\_3:\text{熵项}}$

- $k=1$：胡克常数，将振幅拉回平衡位置$r\_0$
- $J\_{ij} = A\_{ij}$（关联矩阵，top-k稀疏化后）：耦合强度
- $\phi\_{ij}$：相位偏移（逻辑域$\phi=0$，诗意域$\phi = \theta\_i - \theta\_j$为Hebb式学习）
- $S\_{\text{phase}} = -\sum\_n p\_n \log p\_n$，$p\_n = \text{softmax}(\cos(\theta\_n - \bar{\theta}))$：相位分布的熵

### 6.4 K步朗之万迭代

对于每个batch样本（batch维度独立循环），执行K步迭代：

```python
for step in range(K):
    # 相位更新
    r_i, r_j = r.unsqueeze(1), r.unsqueeze(0)      # (N,N)
    th_i, th_j = theta.unsqueeze(1), theta.unsqueeze(0)
    sin_dth = sin(th_j - th_i - phi)               # 相位差驱动项
    coupling = (J * r_j * sin_dth).sum(dim=1)      # (N,) 来自邻居的耦合

    xi_theta = randn(N) * sqrt(2 * T_enc * dt)     # 热噪声
    theta = theta + dt * (omega + coupling) + xi_theta

    # 振幅更新（沿自由能梯度下降）
    F = free_energy(r, r0, theta, J, phi, T_enc)
    if requires_grad:
        r_grad = -grad(F, r, create_graph=True)[0] # 解析梯度
    else:
        r_grad = -2 * (r - r0)                     # 胡克力 fallback
    xi_r = randn(N) * sqrt(2 * T_enc * dt)
    r = r + dt * r_grad + xi_r
    r = clamp(r, min=1e-3)                         # 防坍缩
```

参数：

- $K$：迭代步数（Phase A=0/关闭，Phase B=2，Phase C=5）
- $dt = 0.33$：时间步长
- $T\_{\text{enc}} = 0.05$：编码温度（热噪声强度）
- top-k稀疏化：每个原子只保留耦合最强的topk=16个邻居

**输出**：

- $r \in \mathbb{R}^{B \times N\_a}$：收敛后振幅
- $\theta \in \mathbb{R}^{B \times N\_a}$：收敛后相位
- $Z\_{\text{phys}}$：复振幅嵌入 $z = r\cos\theta \oplus r\sin\theta$ 经MLP投影到D维
- $F, dF$：自由能标量及其变化量（用于Phase C奖励）

***

## 七、双记忆系统（继承3.6）

### 7.1 情景记忆 EpisodicMemory

**环形KV缓冲**，容量4096 slots，按域（逻辑/诗意）独立：

| Buffer       | Shape       | 说明           |
| ------------ | ----------- | ------------ |
| slot\_k      | (4096, D/2) | 键向量          |
| slot\_v      | (4096, D)   | 值向量          |
| slot\_t      | (4096,)     | 写入时间戳        |
| slot\_domain | (4096,)     | 域标签(0=L,1=P) |
| slot\_ptr    | scalar      | 环形写指针        |

**检索**（attention-based）：
$\text{score} = \frac{Q K\_{\text{slot}}^T}{\sqrt{d\_k}} \cdot \mathbb{1}_{\text{domain}} - 10^9 \cdot (1 - \mathbb{1}_{\text{domain}})$
$\text{decay} = \exp(-\lambda\_t \cdot (t - t\_{\text{slot}}))$
$A = \text{softmax}(\text{score}) \odot \text{decay}$
$\text{mem} = A \cdot V\_{\text{slot}}$

时间衰减 $\lambda\_t = 0.01$ 使旧记忆自然消退，domain mask确保逻辑域只检索逻辑记忆。

**写入**：每步将当前x的KV投影reshape为tokens写入环形缓冲，指针取模前进。

### 7.2 结构记忆 StructuralMemory

**技能键值表**，1024个可学习技能槽：

- skill\_key: Embedding(1024, Da) — 技能键
- skill\_val: Embedding(1024, 2Da) — 技能值
- logit\_bias: (1024,) — 偏置

检索方式类似attention：$A = \text{softmax}(q K^T/\sqrt{D\_a} + b)$，输出 $A \cdot V$。结构记忆在BP中学习，代表可复用的"技能"。

***

## 八、元认知门控（继承3.6）

### 8.1 TriPercept三感知器投票

三个独立MLP（结构相同、参数不共享）对融合后的域特征做分类，投票决定逻辑域/诗意域的权重：

```python
x = cat(z_L.mean(dim=1), z_P.mean(dim=1), task_emb)  # (B, 3D)
outs = [softmax(MLP_i(x), dim=-1) for i in range(3)]   # each: (B,2)
alphas_L = stack([out[:,0] for out in outs])           # (3,B)
alpha_L = median(alphas_L, dim=0)[0]                   # (B,) 中位数投票
alpha_P = 1 - alpha_L
u_cog = var(alphas_L, dim=0) / 0.25                   # (B,) 归一化方差
```

- **中位数投票**比均值更鲁棒（抗极端值）
- **$u\_{\text{cog}}$**（认知不确定性）= 三个感知器输出的方差 / 0.25（0.25是Bernoulli分布最大方差）
  - $u\_{\text{cog}} \approx 0$：三个感知器一致→高置信
  - $u\_{\text{cog}} \approx 1$：三个感知器完全分歧→不确定

### 8.2 温度-不确定性耦合

$T = T\_0 \cdot (1 + \kappa \cdot u\_{\text{cog}})$
$T = \text{clip}(T, T\_{\min}, T\_{\max})$

| Phase | T0  | κ   | T范围       | 含义          |
| ----- | --- | --- | --------- | ----------- |
| A     | 1.0 | -   | 1.0       | 高温探索        |
| B     | 0.7 | 0.5 | 0.7\~1.05 | 中温          |
| C     | 0.2 | 2.0 | 0.2\~0.6  | 低温收敛但不确定时升温 |

安全模式：当 $u\_{\text{cog}} > 0.7$ 时强制 $\alpha\_L = \alpha\_P = 0.5$（等高混合），防止在高不确定下偏听某一域。

### 8.3 SafetyFilter

64个danger anchor向量：与输入pattern做cosine相似度，最小相似度→safety\_score。score<0.3的维度被替换为pattern均值（去激活危险模式）。

### 8.4 融合输出

$Z\_{\text{IB}} = \alpha\_L \cdot \[(1-\lambda\_p) z\_L' + \lambda\_p Z\_{\text{phys},L}] + \alpha\_P \cdot \[(1-\lambda\_p) z\_P' + \lambda\_p Z\_{\text{phys},P}]$
$Z\_{\text{cog}} = \text{MLP}_{\text{fuse}}(\text{cat}(Z_{\text{IB}}, \text{epi\_mem}, Z\_{\text{safe}}, \text{task\_exp}))$

$\lambda\_p = \lambda\_{\text{phys}}$ 控制IB表征和物理表征的融合比例（Phase C=0.2）。

***

## 九、V4编码器 EncoderV4

### 9.1 结构

```
Input: input_ids (B,S)
  → token_emb: Embedding(V, D)              (B,S,D)
  → pos_emb:   Embedding(S_max, D)          (B,S,D)
  → x = LN(drop(tok + pos))                (B,S,D)
  → PGTA Block × 2:
      Pre-LN → PGTA self-attention → residual
      → Pre-LN → FFN(Linear→GELU→Linear→Dropout) → residual
  → Information Bottleneck (域独立):
      z_L' = IB_L(z_IB_L): mu_L, logvar_L → sample → dec → out + KL_L
      z_P' = IB_P(z_IB_P): mu_P, logvar_P → sample → dec → out + KL_P
  → Auxiliary losses:
      loss_D (BCE D_score vs 0.5)
      loss_aff (λ·std(aff_score))
      loss_H (λ_H·ReLU(H_enc - H_target))
Output: {x_enc, z_IB_L', z_IB_P', loss_IB, loss_D, loss_aff, loss_H, attn_*}
```

### 9.2 信息瓶颈(IB)

每个域一个VAE：

$\mu = W\_{\mu}(x), \quad \log\sigma^2 = W\_{\text{lv}}(x)$
$z = \mu + \epsilon \cdot \sigma, \quad \epsilon \sim \mathcal{N}(0, I) \quad \text{（重参数化）}$
$\hat{x} = \text{MLP}_{\text{dec}}(z)$
$\mathcal{L}_{\text{KL}} = -\frac{1}{2}\sum\_j (1 + \log\sigma\_j^2 - \mu\_j^2 - \sigma\_j^2)$

IB迫使编码器将输入压缩为紧凑的潜表征，过滤噪声。KL权重λ\_IB在Phase B/C渐入。

### 9.3 注意事项

编码器PGTA的温度使用自身可学习log\_temp（不被元认知覆盖），因为编码器在元认知门之前运行。编码熵损失目标为 $H^\* = 0.3 \cdot \log S$。

***

## 十、V4解码器 GLUBlockV4

### 10.1 完整层结构

每个解码层 $l \in {0,...,n\_{\text{layers}}-1}$ 包含：

```
Input: x (B,S,D), Z_cog (B,S,D), T (scalar), θ_tokens (B,S), s_TSR, s_PSR
  → 1. Pre-LN → PGTA self-attention (独立PGTA，温度T)
     → residual: x = x + attn_out
  → 2. Pre-LN → GLU
       g = W_g(x_norm),  u, v = chunk(g, 2, dim=-1)  # 各 (B,S,d_ff)
  → 3. TSR (Token State Router)
       γ = σ(W_gx·x_norm + mean(W_gz·Z_cog, dim=1, keepdim=True))
       M_TSR = γ·s_TSR + (1-γ)·(W_M·x_norm)
       tsr_out = M_TSR @ W_uM.T
       u = u + tsr_out_padded
  → 4. PSR (Phase State Router)
       Δθ = θ[:,1:] - θ[:,:-1], pad(Δθ,0→1)
       γ_phys = exp(-Δθ²/(2T))
       M_PSR = γ_phys·s_PSR + (1-γ_phys)·(W_Mp·x_norm)
       psr_out = M_PSR @ W_uMp.T
       u = u + λ_PSR · psr_out_padded
  → 5. Z_cog层级注入+物理门控
       pel = pos_layer_emb[l]  (64,)
       Z_l = MLP(cat(Z_cog, pel.expand(B,S,-1)))
       α_t = σ(W_ax·x_norm + mean(W_az·Z_l))
       if θ is not None:
           cos_Δ = cos(θ_tokens.mean(-1,keepdim=True) - θ_layer[l].mean())
           α_p = σ(cos_Δ / T)
           α = (1-λ_T)α_t + λ_T·α_p
       else:
           α = α_t
       scale = mean(W_scale·Z_l + b_scale)
       bias  = mean(W_bias·Z_l + b_bias)
       u = u * (1+scale)*α + bias*(1-α)
  → 6. GLU输出
       out = (σ(u) * v) @ W_proj.T
       x = x + out  (residual)
Output: x (B,S,D), {TSR:M_TSR, PSR:M_PSR}
```

### 10.2 TSR/PSR状态路由

TSR是GRU风格的门控状态更新，Z\_cog调制门控值γ——认知状态影响每层的内部记忆。PSR使用相位差Δθ作为门控信号：相位差小（物理同步）时保留物理状态，相位差大时重置。两者状态跨层传递。

### 10.3 物理门控α

$\alpha\_t$是数据驱动的门（sigmoid输出），$\alpha\_p$是物理驱动的门（基于token相位与层偏好相位$\theta\_l$的一致性）。$\lambda\_T$控制物理门控权重（Phase C渐入到0.3）。

最终$u$被scale/bias调制后通过GLU门(sigmoid-gated)：$GLU(u,v) = \sigma(u) \odot v$，比标准ReLU更灵活（门控信息流）。

***

## 十一、AetherMind4主模型

### 11.1 初始化要点

```python
def __init__(config):
    # Auto-match d_anchor to d_atom if 0
    if config.d_anchor <= 0:
        config.d_anchor = config.d_atom

    # 核心模块 (继承3.6)
    encoder = EncoderV4(config)
    dual_domain = DualDomainSystem(config)
    physics = LangevinOscillator(config)
    memory = DualMemorySystem(config)
    metacog = MetaCognitiveGate(config)

    # V4: n_layers个独立PGTA + GLUBlockV4
    decoder_attns = ModuleList([PheromoneThermoAttention(...) for _ in range(n_layers)])
    decoder_layers = ModuleList([GLUBlockV4(config, i, decoder_attns[i]) for i in range(n_layers)])

    # 输出头
    pointer = MLP(D, D, 1)         # 复制门
    counter_rnn = GRUCell(Dc, Dc)  # 计数器
    counter_proj = Linear(Dc, 1)
    lm_head = Linear(D, V, bias=False)

    # V4: 演化控制器
    evolver = EvolvableWeightSystem(config)
    for attn in encoder.attn_layers + decoder_attns:
        evolver.register_attention(attn)

    # 权重绑定：lm_head与token_emb共享
    lm_head.weight = encoder.token_emb.weight
```

### 11.2 Forward 12步完整数据流

```
步骤1: token_emb = encoder.token_emb(input_ids)       # (B,S,D)
步骤2: domain_out = dual_domain(token_emb)             # → z_L/P, atom_w, D_score等
步骤3: enc_out = encoder(input_ids, domain_out)        # → x_enc, z_IB_L', z_IB_P', aux losses
步骤4: phys_out = physics({**domain_out, "z_IB_L/P": enc_out["z_IB_L/P"]}, pos_emb)
       # → Z_phys_L/P, r, θ, F, dF for both domains
步骤5: mem_out = memory(x_enc, domain_out, t)
       epi_mem = mem_out["epi_L"] + mem_out["epi_P"]   # (B,S,D)
步骤6: Z_phys_L_tokens = einsum("bsn,bnd->bsd", atom_w_L, phys_out["L"]["Z_phys"])
       Z_phys_P_tokens = einsum("bsn,bnd->bsd", atom_w_P, phys_out["P"]["Z_phys"])
       # 物理特征从原子空间映射到token空间，shape对齐
步骤7: meta_out = metacog(z_IB_L', z_IB_P', Z_phys_tokens, epi_mem, task_id)
       # → Z_cog, T, u_cog, α, safety_score
步骤8: θ_L_tokens = (atom_w_L * th_L.unsqueeze(1)).sum(-1)  # (B,S)
       θ_P_tokens = (atom_w_P * th_P.unsqueeze(1)).sum(-1)
       θ_tokens = θ_L_tokens + θ_P_tokens
步骤9: T_val = meta_out["T"].mean().item()
       for attn in decoder_attns: attn.set_temperature(T_val)
       h = x_enc
       for layer in decoder_layers:
           h, state = layer(h, Z_cog, meta_out["T"], θ_tokens, s_TSR, s_PSR)
           s_TSR, s_PSR = state["TSR"], state["PSR"]
       h = final_norm(h)
步骤10: CounterSlot: GRUCell per token position (噪声输入, 保持计数能力)
       h = h + counter_proj(cnt_stack).tanh() * 0.1
步骤11: p_copy = sigmoid(pointer(h)).squeeze(-1)       # (B,S)
       logits = h @ lm_head.weight.T * D^{-0.5}        # (B,S,V) 权重绑定
       probs = safe_softmax(logits, T=T_val)
       # 复制机制: 混合softmax分布和输入序列累积分布
       copy_prob = cumsum(one_hot(input_ids)) / arange(1,S+1)
       probs = (1-p_copy_gate)*probs + p_copy_gate*copy_prob
步骤12: Loss assembly:
       L_LM = CE(shift_logits, shift_labels, ignore_index=pad)
       L_total = L_LM + λ_IB·L_IB + λ_D·L_D + λ_aff·L_aff + λ_H·L_H
              + λ_mod·L_mod + λ_anc·L_anc + λ_phys·L_phys
```

### 11.3 关键shape对齐处理

当原子空间$Z\_{\text{phys}}$映射到token空间时，shape可能不匹配，代码中做了防御性处理：

- S维度pad（当物理输出长度≠序列长度时）
- D维度：当$Z\_{\text{phys}}$最后一维≠D时，动态创建Linear投影（注意：这个动态创建不推荐用于正式训练，已通过phys\_mlp保证正确维度）

θ映射到token级时：θ原子形状为$(B, N\_a)$，通过 `(atom_w * θ_atoms.unsqueeze(1)).sum(-1)` 加权求和得到$(B,S)$。

### 11.4 演化步进接口

```python
def evolution_step(step, phase="A", loss_val=None):
    F_val = self._last_F if hasattr(self, '_last_F') else None
    evolver(step, free_energy=F_val, phase=phase, loss_val=loss_val)
```

注意：当前实现中`_last_F`需要从forward中保存（在forward返回dict中包含"F"key），trainer在每步后调用evolution\_step。

### 11.5 生成推理

generate()方法：Phase C模式自回归生成，top-k + top-p采样：

1. 取最后max\_seq\_len个token作为当前输入
2. forward得到最后位置logits
3. temperature来自meta\_out（可外部覆盖）
4. top-k=50筛选→top-p=0.9核采样→multinomial采样
5. 拼接生成token，遇eos停止
6. 生成过程中PGTA的信息素持续沉积（但无外部奖励时按attention用量均匀沉积）

***

## 十二、训练系统

### 12.1 三阶段训练配方

| 维度                    | Phase A（0\~50k步） | Phase B（50k\~150k步） | Phase C（150k\~250k步） |
| --------------------- | ---------------- | ------------------- | -------------------- |
| **目标**                | 结构预训练            | 混合学习                | 演化收敛+固化              |
| **W学习**               | 全量BP             | 弹性BP                | 主要靠演化（BP减弱）          |
| **τ沉积η**              | 0.0（关闭）          | 0.05×progress（渐入）   | 0.05（全量）             |
| **τ蒸发ρ**              | -                | 0.05                | 0.03（保护好路）           |
| **langevin\_K**       | 0（关闭）            | 2                   | 5                    |
| **λ\_IB**             | 0                | 0.5×progress        | 1.0                  |
| **λ\_D**              | 0                | 0.3×progress        | 0.5                  |
| **λ\_phys**           | 0                | 0.1×progress        | 0.2                  |
| **λ\_PSR**            | 0                | 0.1×progress        | 0.3×progress         |
| **λ\_T**              | 0                | 0.1×progress        | 0.3×progress         |
| **λ\_align**          | 0                | 0.3×progress        | 0.5                  |
| **T0**                | 1.0              | 0.7                 | 0.2                  |
| **κ**                 | -                | 0.5                 | 2.0                  |
| **init\_temperature** | 1.0              | 0.7+0.3p            | 0.5→0.2退火            |
| **信用信号r**             | -                | -loss（粗）            | -dF（密）               |
| **固化**                | 关闭               | 关闭                  | 每100步执行              |
| **类比**                | 婴儿学语言            | 少年学技能               | 专家巩固经验               |

progress ∈ \[0,1]是Phase内的线性进度。

### 12.2 优化器与混合精度

```python
optimizer = AdamW(params, lr=3e-4, betas=(0.9,0.95), weight_decay=0.01)
scheduler = SequentialLR(
    [LinearLR(warmup=1000, 1e-4→1.0),
     CosineAnnealingLR(T_max=249000, eta_min=1e-5)],
    milestones=[1000]
)
scaler = GradScaler("cuda")  # bf16混合精度
```

训练技巧：

- CUDA: cudnn.benchmark=True, TF32=True
- DataParallel多GPU支持
- pin\_memory + non\_blocking=True
- grad\_accumulation=8（RTX 3050 4GB上batch=1等效batch=8）
- grad\_clip\_max\_norm=1.0

### 12.3 训练循环伪代码

```python
for each epoch:
    for batch in train_loader:
        steps += 1
        set_phase()  # 根据steps切换A/B/C，更新所有λ参数

        # Forward+Backward
        with autocast(cuda, dtype=bfloat16):
            out = model(input_ids, labels, task_id, t=steps, phase=phase)
            loss = out["loss"] / grad_accum
        scaler.scale(loss).backward()

        # 梯度累积后更新
        if steps % grad_accum == 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        # V4核心：信息素演化步进
        loss_val = out["loss"].detach()
        if hasattr(model, 'module'):
            model.module.evolution_step(steps, phase, loss_val)
        else:
            model.evolution_step(steps, phase, loss_val)

        # 日志/评估/保存
        if steps % log_interval == 0: print_progress()
        if steps % eval_interval == 0: evaluate()
        if steps % save_interval == 0: save_checkpoint()
```

### 12.4 数据管道

`StreamingTextDataset` 流式读取 `F:/数据集` 目录下的JSONL文件：

- 每文件最多5000行（防OOM）
- 无文件大小上限（12GB的moss文件正常读取）
- 递归提取嵌套JSON中的字符串字段
- HF\_HUB\_OFFLINE=1（离线模式，不访问HuggingFace）
- BPE tokenizer离线训练/加载
- train/eval按98%/2%分割

***

## 十三、完整超参数参考

### 13.1 模型结构参数

| 参数                 | 默认值   | RTX 3050配置 | 说明          |
| ------------------ | ----- | ---------- | ----------- |
| vocab\_size        | 50000 | 50000      | 词汇表大小       |
| d\_model           | 512   | 384        | 模型隐层维度      |
| d\_ff              | 2048  | 1536       | 前馈网络维度(4×D) |
| n\_layers          | 8     | 6          | 解码器层数       |
| n\_heads           | 8     | 6          | 注意力头数       |
| max\_seq\_len      | 1024  | 512        | 最大序列长度      |
| dropout            | 0.1   | 0.1        | Dropout率    |
| n\_atoms           | 1024  | 1024       | GMM原子数      |
| d\_atom            | 64    | 64         | 原子维度        |
| n\_gmm\_components | 3     | 3          | 每原子GMM分量数   |
| n\_anchor          | 512   | 512        | 锚点数         |
| d\_state           | 128   | 128        | TSR/PSR状态维度 |
| d\_counter         | 16    | 16         | 计数器GRU维度    |
| pad/bos/eos        | 0/1/2 | 0/1/2      | 特殊token ID  |

### 13.2 PGTA参数

| 参数                      | 值                             | 说明        |
| ----------------------- | ----------------------------- | --------- |
| pheromone\_rho          | 0.05 (C:0.03)                 | 蒸发率       |
| pheromone\_beta         | 1.0                           | 信息素敏感度    |
| pheromone\_deposit      | 0→0.05                        | 沉积强度（随阶段） |
| pheromone\_tau\_min     | 0.01                          | τ下界       |
| pheromone\_tau\_max     | 5.0                           | τ上界       |
| pheromone\_credit\_mode | "soft\_center"→"free\_energy" | 信用模式(B→C) |
| pheromone\_whiten       | True                          | 能量白化      |
| init\_temperature       | 1.0→0.2                       | 初始温度（退火）  |
| target\_entropy\_ratio  | 0.3                           | 目标熵比例     |
| lambda\_H               | 0.05                          | 熵正则权重     |

### 13.3 LTP固化参数

| 参数                     | 值    | 说明      |
| ---------------------- | ---- | ------- |
| consolidate\_threshold | 1.5  | 固化触发阈值  |
| consolidate\_lam       | 0.1  | 固化写入强度λ |
| consolidate\_gamma     | 0.5  | 固化后τ衰减γ |
| consolidate\_interval  | 100  | 固化间隔步数  |
| max\_consolidations    | 4096 | 固化预算上限  |

### 13.4 训练参数

| 参数                     | 值          | 说明     |
| ---------------------- | ---------- | ------ |
| batch\_size            | 8 (3050:1) | 批大小    |
| gradient\_accumulation | 4 (3050:8) | 梯度累积   |
| learning\_rate         | 3e-4       | 峰值学习率  |
| min\_lr                | 1e-5       | 最小学习率  |
| weight\_decay          | 0.01       | 权重衰减   |
| warmup\_steps          | 1000       | LR预热步数 |
| max\_grad\_norm        | 1.0        | 梯度裁剪   |
| phase\_A\_steps        | 50,000     | 结构预训练  |
| phase\_B\_steps        | 100,000    | 混合学习   |
| phase\_C\_steps        | 100,000    | 演化收敛   |
| total\_steps           | 250,000    | 总步数    |
| log\_interval          | 10         | 日志间隔   |
| eval\_interval         | 2000       | 评估间隔   |
| save\_interval         | 5000       | 保存间隔   |

***

## 十四、VRAM与计算预算

### 14.1 显存分解

以d=384, h=6, n\_layers=6, S=512, B=1, bf16为例：

| 组件                   | 计算                    | 大小          |
| -------------------- | --------------------- | ----------- |
| 模型参数                 | \~56M × 2B (bf16)     | \~112MB     |
| 梯度(fp32)             | \~56M × 4B            | \~224MB     |
| AdamW状态(m+v, fp32)   | 2×56M×4B              | \~448MB     |
| 激活值(bf16)            | \~B×S×D×n\_layers×3.5 | \~80MB      |
| PGTA τ (8层×h×S×S×4B) | 8×6×512×512×4B        | \~48MB      |
| PGTA C (同τ)          | 同上                    | \~48MB      |
| 情景记忆buffer(2域)       | 2×4096×(D/2+D)×4B     | \~14MB      |
| CUDA上下文+碎片           | -                     | \~200MB     |
| **总计**               | <br />                | **\~1.1GB** |

完整版d=512, n=8, S=1024：

- τ/C: (2+8)×8×1024×1024×4B×2 = \~640MB
- 总VRAM约\~3.5GB（勉强可在4GB上跑，需更小batch）

### 14.2 计算复杂度

| 操作              | 复杂度                         | 备注                     |
| --------------- | --------------------------- | ---------------------- |
| QKV投影           | $O(BSD^2)$                  | 标准Transformer          |
| 注意力矩阵乘          | $O(BS^2D)$                  | 标准self-attention       |
| 信息素更新           | $O(hS^2)$                   | 每步仅add+mul+clamp，极快    |
| 固化              | $O(hS^2)$                   | 每100步一次                |
| 朗之万迭代           | $O(K \cdot N\_a^2 \cdot B)$ | K=5步, Na=1024, B=1时可接受 |
| 总FLOPs per step | \~$2BSD^2 + 2BS^2D$         | 接近标准Transformer        |

**关键**：信息素更新是纯buffer操作（无矩阵乘、无梯度），额外开销<5%。

***

## 十五、损失函数完整推导

总损失：

$\mathcal{L} = \mathcal{L}_{\text{LM}} + \lambda_{\text{IB}}\mathcal{L}_{\text{IB}} + \lambda\_D\mathcal{L}D + \lambda_{\text{aff}}\mathcal{L}{\text{aff}} + \lambda\_H\mathcal{L}_H + \lambda_{\text{mod}}\mathcal{L}_{\text{mod}} + \lambda_{\text{anc}}\mathcal{L}_{\text{anc}} + \lambda_{\text{phys}}\mathcal{L}\_{\text{phys}}$

| 损失                           | 公式                                                                  | 作用         | λ（Phase C）     |
| ---------------------------- | ------------------------------------------------------------------- | ---------- | -------------- |
| $\mathcal{L}\_{\text{LM}}$   | $\text{CE}(\text{logits}\[:,:-1,:], \text{labels}\[:,1:])$          | 核心语言建模     | 1.0（固定）        |
| $\mathcal{L}\_{\text{IB}}$   | $\sum \text{KL}\[\mathcal{N}(\mu,\sigma^2)\|\mathcal{N}(0,I)]$      | 信息瓶颈压缩     | 1.0            |
| $\mathcal{L}\_D$             | $\text{BCE}(\text{D\_score}, 0.5)$                                  | 防域坍缩(GAN式) | 0.5            |
| $\mathcal{L}\_{\text{aff}}$  | $\text{std}(\text{affect\_score})$                                  | 情感多样性      | 0.3            |
| $\mathcal{L}\_H$             | $\text{ReLU}(H - H^\*)$                                             | 注意力熵防塌缩    | 0.05           |
| $\mathcal{L}\_{\text{mod}}$  | 模块度约束（见5.6）                                                         | 关联矩阵拓扑     | λ\_module(0→0) |
| $\mathcal{L}\_{\text{anc}}$  | $\text{MSE}(z, \text{anchor})\_L + \text{MSE}(z, \text{anchor})\_P$ | 锚点均匀填充     | 0.5            |
| $\mathcal{L}\_{\text{phys}}$ | $F\_L^2 + F\_P^2$                                                   | 物理自由能最小化   | 0.2            |

注意：BCE损失在代码中用`torch.amp.autocast("cuda", enabled=False)`包装并`.float()`，防止bf16精度下BCE的数值不稳定（"unsafe to autocast"错误）。

***

## 十六、与V3.6/主流模型对比

| 维度      | 标准Transformer      | AetherMind 3.6   | AetherMind V4.0                     |
| ------- | ------------------ | ---------------- | ----------------------------------- |
| 注意力机制   | Scaled Dot-Product | ShapleyAttention | **PGTA（Boltzmann+信息素）**             |
| 权重系统    | 单一BP权重             | 单一BP权重           | **双权重W/τ+C（LTP固化）**                 |
| 时间尺度    | 单步前向               | 单步+物理K步迭代        | **三层（快T/中τ/慢C）**                    |
| 温度控制    | 无/常数               | 元认知T             | **可学习logT+元认知覆盖+退火**                |
| 熵正则     | 无                  | Shapley值约束       | **Boltzmann熵+自由能**                  |
| 物理层     | 无                  | 朗之万振荡器           | 朗之万+自由能奖励信号                         |
| 记忆系统    | KV Cache           | 情景+结构双记忆         | 双记忆+**权重内C（程序记忆）**                  |
| 训练阶段    | 单阶段                | 单阶段（渐进λ）         | **三阶段A/B/C**                        |
| 部署行为    | 静态推理               | 静态推理             | **在线Agent持续演化**                     |
| 信用信号    | 梯度loss             | 梯度loss           | **soft\_center→free\_energy（内在动机）** |
| 路径记忆    | 无（每步重算）            | 无                | **τ跨步持久+C永久固化**                     |
| 参数规模比   | 1.0×               | \~1.1×（+物理层）     | **\~1.15×（+PGTA buffer，参数增加<5%）**   |
| VRAM开销比 | 1.0×               | \~1.3×           | **\~1.6×（τ/C buffer）**              |
| 神经科学类比  | -                  | 预测编码+自由能         | **快/慢权重+LTP+STDP**                  |
| 群体智能    | -                  | -                | **蚁群stigmergy路径**                   |

***

## 十七、已知局限与未来方向

### 17.1 当前局限

1. **信息素O(S²)存储**：长序列S=4096时τ大小达8×4096×4096×4B=512MB/层，需要低秩分解或稀疏注意力。设计文档中预留了DeepSeek CSA/HCA稀疏方向。
2. **信用信号密度**：Phase C的free\_energy奖励虽比0/1对错密，但仍不如"互信息/因果影响"密集。`pgtt_self_evolution.py`实验表明粗信用信号会锁定退化解。
3. **固化与BP的协同**：当前C是纯加性偏置，未考虑C与W的协同梯度。未来可能让C通过直通估计器(STE)获得梯度。
4. **动态Linear投影**：代码中shape对齐时动态创建Linear层（`aethermind4.py#L361`），正式训练时应去掉或用预定义投影。
5. **F\_prev追踪**：当前`_last_F`保存机制需要完善，确保dF计算使用正确的前一步自由能。

### 17.2 未来方向

1. **τ低秩分解**：$\tau = UV^T$，U,V∈ℝ^{S×r}，存储从O(S²)降到O(Sr)
2. **固化hash索引**：借鉴DeepSeek-V4 hash路由，高频固化路径直接查表
3. **MoE集成**：τ当expert路由先验，固化=把高频专家选择写进hash表
4. **SNN脉冲版**：E=脉冲势，τ=突触效能，固化=STDP
5. **4bit量化**：借鉴Kimi K3 QAT，让τ/C可量化存储
6. **Phase D部署演化**：完整的推理时在线演化+周期性固化+LoRA导出

***

## 十八、文件结构与代码索引

```
d:\AetherMind-Nano3\
├── configs/
│   ├── aethermind36_config.py                  # 3.6配置（保留）
│   └── aethermind4_config.py                   # V4配置 [NEW] L7-L170
├── src/
│   ├── model/
│   │   ├── aethermind36.py                     # 3.6主模型（保留）
│   │   ├── aethermind4.py                      # V4主模型 [NEW] L263-515
│   │   │   ├── EncoderV4                       # [NEW] L34-138
│   │   │   └── GLUBlockV4                      # [NEW] L144-257
│   │   ├── attention/
│   │   │   └── pheromone_thermo.py             # PGTA [NEW] L20-218
│   │   ├── evolution/
│   │   │   └── evolvable_weight.py             # 演化控制器 [NEW] L20-123
│   │   ├── domain/
│   │   │   └── dual_domain.py                  # GMM双域（3.6）L9-196
│   │   ├── physics/
│   │   │   └── langevin.py                     # 朗之万（3.6）L9-220
│   │   ├── memory/
│   │   │   └── dual_memory.py                  # 双记忆（3.6）L8-92
│   │   ├── metacog/
│   │   │   └── meta_gate.py                    # 元认知门（3.6）L49-99
│   │   └── decoder/
│   │       └── glu_decoder.py                  # 3.6 GLU（V4用GLUBlockV4）
│   ├── training/
│   │   ├── train.py                            # 3.6训练器
│   │   └── train_v4.py                         # V4训练器 [NEW]
│   ├── data/
│   │   └── dataset.py                          # 流式数据集（3.6）
│   └── utils/
│       └── ops.py                              # 工具函数（3.6）
├── docs/
│   ├── AetherMind36_架构报告.md                # 3.6文档
│   └── AetherMind4_架构报告.md                 # 本文档 [NEW]
├── test_v4_smoke.py                            # V4冒烟测试
├── 1212/                                       # V4设计灵感来源
│   ├── 信息素路径网络+热力学注意力+Transformer设计.md
│   ├── 可迭代权重的进化模型架构设计.md
│   ├── 物理驱动注意力机制方案.md
│   ├── pheromone_thermo_transformer.py         # PGTT参考实现
│   ├── pgtt_self_evolution.py                  # 自演化实验
│   ├── evolvable_weight.py                     # LTP固化MVP
│   └── thermo_info_attention.py               # 热力学注意力参考
└── train_gpu.cmd / train_v4_smoke.cmd          # 训练脚本
```

***

## 十九、验证状态

### 19.1 冒烟测试通过

```
Model params: 9,223,140 (d=128, n_layers=2, n_heads=4, batch=2, seq=64)
Phase A: loss=10.9589  Backward OK
Phase B: loss=11.0780  Evolution step OK
Phase C: loss=11.3571  Generate OK: torch.Size([1, 15])
ALL PASSED
```

### 19.2 已验证项

- [x] PGTA前向/反向传播正常
- [x] 信息素步进（蒸发+沉积）正常
- [x] LTP固化机制（代码路径正确，待长训练验证）
- [x] 三阶段配置切换正常
- [x] 权重绑定（lm\_head ↔ token\_emb）正常
- [x] 自回归generate正常
- [x] 双域GMM激活正常
- [x] 朗之万振荡器K步迭代正常
- [x] 元认知门控温度计算正常
- [x] 双记忆检索/存储正常
- [ ] 完整三阶段长训练（待执行）
- [ ] 自由能奖励信号dF计算精度（待验证）
- [ ] 固化后效果量化（待Phase C跑满100步后检验）
- [ ] 4GB VRAM上d=384配置训练稳定性

***

## 二十、2026-08-27 实战调整记录

> **说明**：本章为V4.0首次实战训练（d=384, L=6, h=6, seq=512, RTX 3050 4GB）后进行的调整记录。原有章节内容不删除、不改动，所有修改在此集中标注。每项调整包含：问题现象、根因分析、修改位置、具体改法、修改前后对比。

### 20.1 分词器替换：自定义BPE → Qwen2.5-0.5B

**问题现象**：推理时中文全部输出为`<unk>`，英文输出碎片化，自定义BPE分词器无法正确编码中文文本。

**根因分析**：原V4使用的自定义BPE分词器词表仅50000，且训练语料中中文覆盖不足，导致大量中文字符被映射为unk token。

**修改位置**：

- `configs/aethermind4_config.py`：`vocab_size`、`tokenizer_path`、`pad/eos/unk_token_id`
- `src/data/dataset.py`：`build_tokenizer()`函数完全重写

**具体改法**：

| 参数               | 修改前      | 修改后                                  |
| :--------------- | :------- | :----------------------------------- |
| `vocab_size`     | 50000    | 151936（对齐到128倍数后151680）              |
| `tokenizer_path` | 无（内置BPE） | `models_local/Qwen/Qwen2___5-0___5B` |
| `pad_token_id`   | 0        | 151643                               |
| `eos_token_id`   | 1        | 151643                               |
| `unk_token_id`   | 2        | 151643                               |

`dataset.py`中`build_tokenizer()`改为使用`transformers.AutoTokenizer.from_pretrained()`加载Qwen2.5-0.5B的tokenizer，并包装为统一接口（`encode()`/`decode()`/`vocab_size`属性）。数据加载流程中所有`self.tokenizer(...)`调用保持不变，通过包装层兼容。

**影响范围**：Embedding层和LM Head权重矩阵从`(50000, 384)`变为`(151680, 384)`，参数量从约19M增至约58M（总参数量101M），但模型维度d=384/L=6/h=6/seq=512均未改变。

***

### 20.2 Bug1修复：评估阶段辅助损失未关闭

**问题现象**：训练日志显示eval loss异常偏高，且eval时信息素仍在演化（tau值变化），评估结果不可比。

**根因分析**：`train_v4.py`的`evaluate()`方法直接调用`model.eval()`，但未切换config的phase参数。模型在eval模式下仍然加载了Phase B/C的辅助损失权重（lambda\_phys、lambda\_PSR等），且信息素沉积/蒸发仍在进行，导致评估损失包含了物理损失等非LM项。

**修改位置**：`src/training/train_v4.py` → `evaluate()`方法

**具体改法**：评估前保存当前所有lambda权重、langevin\_K、pheromone参数，调用`cfg.set_phase_A()`切换到纯LM模式（所有辅助lambda=0, K=0, deposit=0, rho=0），评估完成后恢复原始参数：

```python
# 评估前保存
saved_lambdas = {k: getattr(cfg, k) for k in [...]}
saved_K = cfg.langevin_K
saved_deposit = cfg.pheromone_deposit
saved_rho = cfg.pheromone_rho
cfg.set_phase_A()  # 切到纯LM模式

# ... 执行评估 ...

# 评估后恢复
for k, v in saved_lambdas.items():
    setattr(cfg, k, v)
cfg.langevin_K = saved_K
cfg.pheromone_deposit = saved_deposit
cfg.pheromone_rho = saved_rho
```

***

### 20.3 Bug2修复：Phase C辅助损失权重过大

**问题现象**：进入Phase C后loss不降反升，训练曲线出现明显跳变，辅助损失主导了总损失。

**根因分析**：原始Phase C配置中各辅助损失权重之和约3.49，而主LM损失权重仅1.0。物理损失、PSR损失、IB损失等辅助项梯度量级较大，导致模型优先优化辅助目标而忽视语言建模。

**修改位置**：`configs/aethermind4_config.py` → `set_phase_B()`、`set_phase_C()`

**具体改法**：所有辅助lambda权重减半，并乘以progress渐入系数：

| 参数                 | Phase B修改前 | Phase B修改后     | Phase C修改前 | Phase C修改后 |
| :----------------- | :--------- | :------------- | :--------- | :--------- |
| `lambda_phys`      | 0.1        | 0.05×prog      | 0.2        | 0.1        |
| `lambda_PSR`       | 0.15       | 0.08×prog      | 0.3        | 0.15×prog  |
| `lambda_T`         | 0.15       | 0.08×prog      | 0.3        | 0.15×prog  |
| `lambda_F`         | 0.0        | 0.0            | 0.1        | 0.03×prog  |
| `lambda_copy_phys` | 0.1        | 0.05×prog      | 0.2        | 0.1×prog   |
| `lambda_phase`     | 0.2        | 0.1×prog       | 0.3        | 0.15       |
| `lambda_phi_decay` | 0.0        | 0.0            | 0.01       | 0.005×prog |
| `lambda_IB`        | 0.5        | 0.3×prog       | 1.0        | 0.5        |
| `lambda_D`         | 0.3        | 0.15×prog      | 0.5        | 0.25       |
| `lambda_aff`       | 0.2        | 0.1×prog       | 0.3        | 0.15       |
| `lambda_align`     | 0.3        | 0.15×prog      | 0.5        | 0.25       |
| **辅助权重总和**         | **\~2.0**  | **\~0.6\~1.2** | **\~3.49** | **\~1.3**  |

***

### 20.4 Bug3修复：可学习温度被强制覆盖

**问题现象**：训练日志中`T`始终为固定值（如1.0或0.2），可学习温度参数`log_temp`没有梯度变化，元认知门控的温度调节失效。

**根因分析**：`aethermind4.py`的forward中调用了`self.set_temperature(T0)`直接覆写`self.log_temp.data`，将元认知模块计算的温度硬写入可学习参数，导致`log_temp`的梯度被覆盖、无法通过BP学习。

**修改位置**：`src/model/aethermind4.py` → forward方法、PGTA attention调用

**具体改法**：

1. 删除所有`self.set_temperature(...)`调用
2. PGTA attention的forward新增`temperature_override`参数，前向计算时通过该参数传入温度，不修改`self.log_temp`本身
3. 训练时`temperature_override=None`，attention内部使用可学习的`self.log_temp`；推理/生成时传入元认知温度`T_tensor`
4. Decoder层同理，温度通过参数逐层传递，不再覆写模块属性

```python
# 修改前（错误）：
self.set_temperature(T0)  # 直接覆写log_temp.data
attn_out = self.attn(x)

# 修改后（正确）：
attn_out, _ = self.attn(norm1(x), temperature_override=T)
# attention内部: T_actual = override if override is not None else torch.exp(log_temp)
```

***

### 20.5 Bug4修复：信息素演化未激活

**问题现象**：训练日志中`tau=1.0`（初始值）始终不变，`cons=0`（固化次数为0），信息素网络完全没有沉积。

**根因分析**：四个因素叠加导致信息素无法有效沉积：

1. 蒸发率`rho=0.05`过高，每步蒸发5%，沉积量`deposit=0.05`不足以补偿蒸发
2. 固化阈值`consolidate_threshold=2.0`过高，tau从1.0起步根本达不到
3. 奖励信号未缩放，dLoss/dF量级很小（\~0.01），沉积量≈reward×deposit≈0.0005/步
4. Phase B的deposit初始值为0，需要progress渐入，但progress增长缓慢

**修改位置**：`configs/aethermind4_config.py`

**具体改法**：

| 参数                           | 修改前       | 修改后       | 说明                  |
| :--------------------------- | :-------- | :-------- | :------------------ |
| `pheromone_rho`              | 0.05      | 0.02      | 降低蒸发率，沉积更容易保留       |
| `pheromone_deposit`（Phase C） | 0.05      | 0.12      | 增大沉积强度              |
| `pheromone_deposit`（Phase B） | 0.05×prog | 0.08×prog | 增大渐入沉积              |
| `consolidate_threshold`      | 2.0       | 1.2       | 降低固化门槛，tau达到1.2即可触发 |
| `reward_scale`               | 1.0       | 5.0       | 放大dLoss/dF奖励信号      |

修改后训练日志验证：step 12000时`tau=73.1`，信息素已有效沉积并持续增长。

***

### 20.6 显存优化：Gradient Checkpointing

**问题现象**：d=384配置在RTX 3050 4GB上，forward+backward峰值显存超过4GB，CUDA OOM。

**修改位置**：

- `configs/aethermind4_config.py`：新增`gradient_checkpointing: bool = True`
- `src/model/aethermind4.py`：`EncoderV4.forward()`和`AetherMind4.forward()`中的decoder层循环

**具体改法**：使用`torch.utils.checkpoint.checkpoint()`包装encoder和decoder的每一层：

```python
from torch.utils.checkpoint import checkpoint as torch_checkpoint

use_ckpt = cfg.gradient_checkpointing and self.training

# Encoder层
if use_ckpt:
    x, stats = torch_checkpoint(
        _enc_layer_block, x, attn, norm, ffn, ffn_norm,
        use_reentrant=False
    )
else:
    x, stats = _enc_layer_block(x, attn, norm, ffn, ffn_norm)

# Decoder层同理
```

**效果**：激活值显存节省约50%（\~1.0GB），代价是训练速度降低约20%（前向需重算）。推理时不启用（`self.training=False`），无速度影响。

***

### 20.7 显存优化：训练时跳过Softmax和Pointer Copy

**问题现象**：即使启用gradient checkpointing，LM Head的softmax和pointer copy仍产生大张量导致OOM。

**根因分析**：

1. `out_prob = softmax(logits/T)`产生`(1, 512, 151680)`的float张量，约310MB
2. Pointer copy的`F.one_hot(input_ids, num_classes=151680)`产生`(1, 512, 151680)`的float张量，约310MB
3. 这两个张量在训练时完全不需要——训练用`F.cross_entropy(logits, labels)`直接计算损失，不需要prob

**修改位置**：`src/model/aethermind4.py` → LM Head部分

**具体改法**：

```python
# 训练时(labels is not None)跳过softmax/prob计算
out_prob = None
if labels is None:
    # 仅推理/生成时计算prob
    out_prob = safe_softmax(logits.float(), T=T_safe).to(h.dtype)
    # Pointer copy仅在词表小时启用
    use_pointer = (cfg.vocab_size <= 65536) and ...
    if use_pointer:
        ...
```

Pointer copy机制（复制输入token的概率分布）在大词表下需要创建`(B, S, V)`的one-hot张量，当V=151680时约310MB。增加词表大小判断：`vocab_size <= 65536`时才启用，大词表下直接禁用。该机制对大词表收益极小（复制151680个token中已出现的512个），但显存代价巨大。

**效果**：训练时节省约600MB显存（softmax 310MB + one-hot 310MB），推理时仍保留完整功能。

***

### 20.8 显存优化：Backward前及时释放张量

**修改位置**：`src/training/train_v4.py` → `_train_step()`

**具体改法**：forward完成后、backward之前，立即提取所有标量统计量到CPU，然后删除out字典中的所有大张量：

```python
# 提取标量
scalars = {}
for k, v in out.items():
    if isinstance(v, torch.Tensor) and v.numel() == 1:
        scalars[k] = float(v.detach().cpu().item())

# 立即释放forward输出大张量
del out, input_ids, labels

# backward（此时out/logits等大张量已释放）
self.scaler.scale(loss).backward()
del loss
```

LM Head内部也在计算完cross\_entropy后立即`del shift_logits, shift_labels`。

此外，每50步自动执行`gc.collect()` + `torch.cuda.empty_cache()`清理显存碎片。

***

### 20.9 显存优化：Logits Clamp防溢出

**修改位置**：`src/model/aethermind4.py` → LM Head

**具体改法**：在计算softmax/cross\_entropy前，对logits进行clamp：

```python
logits = h @ self.lm_head.weight.T * self.token_emb_scale
logits = logits.clamp(-1e4, 1e4)  # 防止softmax溢出
```

bf16混合精度下logits可能出现极端值，导致softmax返回NaN/Inf。clamp到\[-1e4, 1e4]范围内，softmax后仍有足够区分度。

***

### 20.10 新增：GPU显存守护系统

**问题现象**：训练过程中电脑卡顿（网络问题导致其他软件云端压力转到本地），其他进程抢占GPU显存，导致CUDA OOM崩溃。

**修改位置**：

- 新增文件：`src/utils/gpu_guard.py`
- `src/training/train_v4.py`：启动时和OOM时调用

**具体改法**：

**(1) 训练前清理** `pre_train_gpu_cleanup(auto_kill=True)`：

- 通过`nvidia-smi --query-compute-apps`查询所有占用GPU显存的进程
- 白名单保护：系统进程（System/svchost/dwm）、Windows桌面（explorer）、NVIDIA驱动、Python训练进程自身、终端（cmd/powershell）、IDE（trae/code/cursor）
- 自动结束所有非保护进程（如浏览器硬件加速、其他CUDA程序等）
- 清理后报告释放的显存

**(2) OOM紧急响应** `emergency_free_gpu_memory(threshold_mb=500)`：

- 先执行`gc.collect()` + `torch.cuda.empty_cache()`清理PyTorch内部缓存
- 如果可用显存仍低于阈值，扫描并杀掉当前占用GPU的非保护进程
- 返回释放的显存MB数

**(3) OOM自动恢复**（`_train_step`和`_step_optim`中）：

```python
try:
    out = model(...)
    loss = out["loss"] / grad_accum
    self.scaler.scale(loss).backward()
except torch.cuda.OutOfMemoryError:
    self.optimizer.zero_grad(set_to_none=True)
    del out, loss, input_ids, labels
    freed = emergency_free_gpu_memory(threshold_mb=500)
    gc.collect()
    torch.cuda.empty_cache()
    self.bad_batch_count += 1
    return None  # 跳过该batch，继续训练
```

***

### 20.11 新增：崩溃自动重启

**问题现象**：部分CUDA OOM发生在kernel级别，Python的try-except无法捕获（直接触发`c10::Error`导致进程abort），训练完全退出。

**修改位置**：`train_v4.cmd`

**具体改法**：CMD脚本增加`:restart`循环：

```batch
:restart
python src\training\train_v4.py --fresh ... (其他参数)

if %ERRORLEVEL% EQU 0 (
    echo TRAINING COMPLETED SUCCESSFULLY!
    goto :end
)

echo TRAINING CRASHED (exit code %ERRORLEVEL%)
echo Waiting 15 seconds for GPU memory to free...
timeout /t 15 /nobreak >nul
echo Restarting...
goto :restart

:end
pause
```

- 正常退出（exit code 0）：结束
- 异常退出（非0 exit code）：等待15秒让显存自然释放，自动重启
- 重启后`train_v4.py`自动从最新checkpoint resume（除非指定`--fresh`）
- 每次重启都会重新执行`pre_train_gpu_cleanup()`清理GPU

***

### 20.12 CMD脚本编码修复

**问题现象**：CMD脚本中中文输出乱码，且`set`命令中的中文和特殊字符被CMD解释为命令，导致`'MERS_OFFLINE' 不是内部或外部命令`等错误。

**修改位置**：`train_v4.cmd`、`run_inference.cmd`

**具体改法**：

1. 文件开头添加`chcp 65001 >nul 2>&1`切换到UTF-8代码页
2. 所有`echo`输出改为纯英文，避免中文编码问题
3. 添加`setlocal`/`endlocal`隔离环境变量
4. 添加`set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128`减少显存碎片
5. 添加`set HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`避免联网请求

***

### 20.13 调整汇总表

|   编号  | 类别               | 修改文件                                | 模型大小影响             | 状态  |
| :---: | :--------------- | :---------------------------------- | :----------------- | :-- |
|  20.1 | 分词器替换            | `configs/`, `dataset.py`            | Embedding增大（d/L不变） | 已完成 |
|  20.2 | Bug1评估           | `train_v4.py`                       | 无                  | 已完成 |
|  20.3 | Bug2损失权重         | `aethermind4_config.py`             | 无                  | 已完成 |
|  20.4 | Bug3温度           | `aethermind4.py`                    | 无                  | 已完成 |
|  20.5 | Bug4信息素          | `aethermind4_config.py`             | 无                  | 已完成 |
|  20.6 | 显存：checkpointing | `configs/`, `aethermind4.py`        | 无（省\~1GB）          | 已完成 |
|  20.7 | 显存：跳过softmax     | `aethermind4.py`                    | 无（省\~600MB）        | 已完成 |
|  20.8 | 显存：释放张量          | `train_v4.py`                       | 无（省\~200MB）        | 已完成 |
|  20.9 | 显存：logits clamp  | `aethermind4.py`                    | 无                  | 已完成 |
| 20.10 | GPU守护系统          | `gpu_guard.py`（新增）, `train_v4.py`   | 无                  | 已完成 |
| 20.11 | 崩溃自动重启           | `train_v4.cmd`                      | 无                  | 已完成 |
| 20.12 | CMD编码修复          | `train_v4.cmd`, `run_inference.cmd` | 无                  | 已完成 |

**训练验证结果**（step 12000, Phase B）：

- 峰值显存：\~2.79GB / 4.0GB（余量1.21GB）
- Loss：5.59（从初始12.14下降）
- 温度T：0.705（可学习温度正常变化）
- 信息素tau：73.1（有效沉积）
- bad=0（无NaN/Inf/OOM导致的坏batch）

***

# 21. 二次修改：Phase D（LTP固化）修复 + Phase E（语言SFT）+ Phase F（知识图谱对齐）

## 21.1 Phase D 固化空转修复 + 独立固化专训

**问题现象**：Phase C 训练完成（35000步，loss=4.1956），但 `[Save]` 显示 `cons_mass=0.0`，固化从未写入 `consolidated` 长期权重；推理输出为"婴儿胡话"（未对齐状态）。

**根因**（加载 checkpoint 实测定位）：

1. `v4_checkpoint_35000_final.pt` 中 `tau` 全部塌缩到地板值 `0.010`（`tau_max` 仅 0.045），而固化用**绝对阈值** `threshold=1.2`（`mask = tau > 1.2`）→ τ 永远够不到 → `mask` 恒空 → 200 次固化全空转。
2. **隐藏根因**：注意力层在 `__init__` 时**缓存**了 `deposit/rho/beta/tau_min/tau_max`，后续 `config.set_phase_C/D()` 只改 config 对象，**不同步到注意力层** → Phase C/D 设定的"强沉积/低蒸发"从未生效，这是 τ 塌缩的真凶。

**修改位置**：

- `src/model/aethermind4.py`：新增 `apply_pheromone_config()`，把 config 的 pheromone 参数显式推送到各注意力层；`train_v4._set_phase` 每次切阶段后调用（**核心修复**）。
- `src/model/attention/pheromone_thermo.py`：`consolidate()` 增加兜底——绝对阈值无人达标时自动退化为分位数（top_frac=0.002）固化，绝不再空转。
- `src/model/evolution/evolvable_weight.py`：固化加 `consolidate_warmup`，τ 未成型前不固化，避免把噪声写入长期权重。
- `src/training/train_v4.py`：修复 `_evolution_step`（Phase D 也传自由能奖励，原仅 C 传）；新增独立 Phase D 分支 `_train_phaseD()`（冻结 W、仅前向 no_grad + 演化 + 固化、不反向传播、存 `*_phaseD.pt`）；`_set_phase` 增加 phase_d 分支；CLI 加 `--phase_d / --phase_D`。
- `configs/aethermind4_config.py`：`set_phase_D`（deposit=0.06 强沉积 / rho=0.01 低蒸发 / consolidate_top_frac=0.002 分位数固化 / interval=50 高频固化 / warmup=100）；`phase_D_steps`。
- 新增 `train_v4_phaseD.cmd`：双击即加载 `v4_checkpoint_35000_final.pt` 跑固化专训。

**验证结果**：

- CPU 冒烟（warmup=0, 80步）：`cons_mass` 0 → 127,274；`attn.deposit=0.06/rho=0.01`（同步生效）。
- GPU 正式跑完（2000步）：`cons_mass=792,944`，`rounds=39`（39轮固化），`tau_conc=42.1`，consolidated 非零元素 325 万。

## 21.2 train_v4_phaseD.cmd 编码修复（UTF-8 中文导致双击启动推理）

**问题现象**：用户双击 `train_v4_phaseD.cmd` 却弹出**推理界面**（"AetherMind V4 Inference (Fixed)"），且带乱码参数"加载checkpoint查看固化效果"；实际运行复现。

**根因**：`.cmd` 文件为 **UTF-8 编码含中文**，而 cmd.exe 用 **GBK(936) 代码页解析批处理文件**（`chcp 65001` 只影响显示、不影响解析）。UTF-8 中文字节错乱 → echo 行被拆成乱码命令执行（stderr 报"XX 不是内部或外部命令"），且"之后用 run_inference 加载即可"被错乱解析成**执行 run_inference.cmd** → 启动推理。

**修改位置**：`train_v4_phaseD.cmd` 重写为**纯 ASCII（英文提示）**。

**验证**：`open(p,'rb').read().decode('ascii')` 通过；实际运行正确显示 "PHASE D - LTP Consolidation Training" 并跑通固化。

**长期约定**：本项目所有 `.cmd/.bat` 必须保持纯 ASCII，禁止中文和特殊符号（τ、→、— 等）。

## 21.3 推理脚本适配（真实固化质量显示 + 阶段化配置）

**问题现象**：推理加载 Phase D 产物后仍显示 `cons_mass=0.00`，无法验证固化效果。

**根因**：`inference_v4.py` 的 `load_model` 用 `model.get_evolution_stats()` 读取演化统计——这是加载后的**实时默认值**（Python 属性未随 checkpoint 恢复），恒为 0；而真实固化质量存在 `ckpt["evolution_stats"]` 里未被读取。

**修改位置**：`scripts/inference_v4.py`

1. 演化统计改为**优先读 `ckpt["evolution_stats"]`**（训练时写入的真实值：tau_conc/cons_mass/rounds），并提示"已固化(LTP)"或"未固化"。
2. 模型配置按 checkpoint 阶段选择：`set_phase_D/E/F`（温度等推理参数与训练阶段一致），不再是固定 `set_phase_C`。

**验证**：`py_compile` 通过；加载 `v4_checkpoint_2000_phaseD_final.pt` 将显示 `cons_mass=792944, 固化轮数=39`。

## 21.4 沐雪数据集（Muice-Dataset）接入

**内容**：ModelScope `Moemuu/Muice-Dataset` 下载（9 文件 <1MB），含 train/test/Customized 子集，格式 `{"system", "conversation": [{"human", "assistant"}, ...]}` 多轮对话。

**转换**：新增 `scripts/convert_muice.py` → `03_dialogue/muice-jsonl/muice.jsonl`（**3737 条对话**），对话拼接为与推理一致的模板：

```
<|Human|>: xxx<eoh>
<|MOSS|>: yyy<eom>
```

**注意**：modelscope 命令行不在 PATH，须用完整路径 `C:\Users\玄曦雪\AppData\Roaming\Python\Python314\Scripts\modelscope.exe`；`python -m modelscope` 无 `__main__` 不可用。

## 21.5 知识图谱搭建

**内容**：新增 `scripts/build_knowledge_graph.py`，从 MOSS + 沐雪对话数据规则抽取三元组，输出 `03_dialogue/knowledge_graph.json`。

- 关系类型：`HAS_ATTR / BELONGS_TO / LOCATED_IN / ALSO_KNOWN_AS / MADE_OF / COMES_FROM / CONTAINS / RELATED_TO`
- 实体过滤：长度 2~10、排除虚词/代词前缀后缀（BAD_PREFIX/BAD_SUFFIX）、排除含标点碎片
- 结果：**8011 三元组 / 9948 实体**（745KB）

## 21.6 Phase E 语言组织 SFT（后训练，先学会说话）

**定位**：模型已有"思考路径"（固化完成），但语言组织能力不足（输出碎片化）。Phase E 用 MOSS + 沐雪对话数据做低学习率 SFT，**暂不接知识图谱**（避免 KG 强信号冲垮刚形成的注意力分布）。

**修改位置**：`src/training/train_v4.py` + `configs/aethermind4_config.py`

1. `TrainerV4` 增加 `phase_e` 模式；`_set_phase` 加 E 分支（`set_phase_E`：deposit=0/rho=1 关闭演化固化，专注 CE loss）；`_evolution_step` 在 E/F 跳过演化。
2. `_train_phaseE()`：冻结全部 → 解冻**最后 2 层 decoder + final_norm + lm_head**（89 个模块）→ 重建 AdamW（**lr=1e-5**）+ CosineAnnealingLR → 标准 CE 训练。
3. `_maybe_resume`：Phase E 优先恢复 `*_phaseE.pt`，否则自动取最新 `*_phaseD.pt`（无 D 则 Final）。
4. CLI：`--phase_e / --phase_E(默认3000) / --phaseE_lr(1e-5) / --unfreeze_layers(2)`。
5. 新增 `train_v4_phaseE.cmd`（纯 ASCII）：数据 = `03_dialogue`（MOSS + muice-jsonl 递归发现）。

**验证**：CPU 冒烟 6 步，loss 4.27 → 3.75（下降），`*_phaseE_final.pt` 正常保存。

## 21.7 Phase F 知识图谱对齐

**定位**：Phase E 之后接入 KG（外部符号知识）。冻结 100% 主干，只训练"实体→原子映射器"和"逻辑域 C/E 关联矩阵"——知识通过 GMM 原子空间注入，不暴力覆盖权重。

**修改位置**：`src/training/train_v4.py` + `configs/aethermind4_config.py`

1. `TrainerV4` 增加 `phase_f` 模式；`set_phase_F`（关闭演化/固化）。
2. `_train_phaseF()`：
   - 冻结全部参数 → 只解冻 `dual_domain.logic_atoms.token_to_atom`（实体→原子映射器）+ `logic_assoc_C/E`（原子关联矩阵，4 个模块）。
   - 实体表示：tokenizer 编码实体名 → 冻结 embedding 均值（预缓存，不重复前向）。
   - 损失：`L = -log σ(w_hᵀ·C·w_t) - log σ(-w_hᵀ·C·w_neg) - 0.3·log σ(-w_hᵀ·E·w_t)`（w=实体原子激活分布；C=耦合矩阵、E=排斥矩阵；负采样随机实体）。
   - 知识只进**逻辑域(L)**，不干扰诗意域(P)的创作表达。
3. `_maybe_resume`：优先恢复 `*_phaseF.pt`，否则自动取最新 `*_phaseE.pt`。
4. CLI：`--phase_f / --phase_F(默认2000) / --kg_path`。
5. 新增 `train_v4_phaseF.cmd`（纯 ASCII）。

**验证**：CPU 冒烟 10 步跑通（KG 8011 三元组加载、loss 正常），`*_phaseF_final.pt` 正常保存。

## 21.8 二次修改汇总表

|   编号  | 类别                   | 修改文件                                             | 模型大小影响       | 状态  |
| :---: | :------------------- | :---------------------------------------------- | :---------- | :-- |
| 21.1 | Phase D 固化修复+专训      | `aethermind4.py`, `pheromone_thermo.py`, `evolvable_weight.py`, `train_v4.py`, `configs/`, `train_v4_phaseD.cmd`(新增) | 无（新增长期权重）    | 已完成 |
| 21.2 | Phase D 脚本编码修复        | `train_v4_phaseD.cmd`                             | 无           | 已完成 |
| 21.3 | 推理脚本适配              | `inference_v4.py`                                 | 无           | 已完成 |
| 21.4 | 沐雪数据集               | `convert_muice.py`(新增), `03_dialogue/muice-jsonl/muice.jsonl` | 无（数据层）      | 已完成 |
| 21.5 | 知识图谱                | `build_knowledge_graph.py`(新增), `knowledge_graph.json` | 无（数据层）      | 已完成 |
| 21.6 | Phase E 语言SFT         | `train_v4.py`, `configs/`, `train_v4_phaseE.cmd`(新增)    | 无（复用主干）      | 已完成 |
| 21.7 | Phase F 知识图谱对齐        | `train_v4.py`, `configs/`, `train_v4_phaseF.cmd`(新增)    | 无（4模块微调）     | 已完成 |

**Phase D 正式验证结果**（GPU, 2000步）：

- `cons_mass`：0 → **792,944**
- 固化轮数：**39 轮**（预热100步后每50步一次）
- τ 浓度：42.1（高度分化）
- consolidated 非零元素：3,250,982

**训练总流程**：Phase A/B/C（语言流利度+物理探索）→ Phase D（固化，✅已跑完）→ Phase E（语言组织 SFT）→ Phase F（知识图谱对齐）→ Phase G（纯净对话 SFT，✅已跑通冒烟）→ 推理验证。对应启动脚本：`train_v4.cmd` → `train_v4_phaseD.cmd` → `train_v4_phaseE.cmd` → `train_v4_phaseF.cmd` → `train_v4_phaseG.cmd` → `run_inference.cmd`。

## 21.9 对话数据清洗（剥离 system prompt）

**问题现象**：Phase E/F 训练后推理输出中夹带大段英文系统提示词复读（`Image edition: disabled.`、`Text-to-speech: disabled.`、`Fudan University ... MOSS`、`It should avoid giving subjective opinions ...`），模型把数据集自带的 system prompt 当成了正文在背诵，中文对话被污染。

**根因分析**：`StreamingTextDataset._extract_text()` 从任意 JSON 结构中**递归收集所有字符串值**再拼接。MOSS 数据每个样本含 `meta_instruction`（英文系统提示）+ `chat.turn_N` 里的 `Inner Thoughts / Commands / Tool Responses / MOSS / Human` 全部字段；沐雪 muice-dataset 含 `system` 字段。这些非对话字段被无差别拼入训练文本，是"复读英文指令"污染的根源。

**修改位置**：新增 `scripts/clean_dialogue_data.py`

**具体改法**：

1. 自动识别三种输入格式：
   - MOSS：`{"meta_instruction", "chat": {"turn_N": {"Human", "MOSS", "Inner Thoughts", ...}}}`
   - 沐雪 muice-dataset：`{"system", "conversation": [{"human", "assistant"}, ...]}`
   - 沐雪 muice-jsonl：`{"text": "<|Human|>: ...<eoh>\n<|MOSS|>: ...<eom>"}`
2. 剥离字段：`meta_instruction`、`system`、`Inner Thoughts`、`Commands`、`Tool Responses`
3. 只保留真实对话对，统一输出模板：

```
<|Human|>: {用户}<eoh>
<|MOSS|>: {助手}<eom>
```

4. 过滤空/None 回复、长度异常（human>2000 或 assistant>4000 丢弃）
5. **分片输出**（每片 ≤4000 条，默认 `shard_size=4000`），因为训练端 `StreamingTextDataset` 每个 jsonl 最多读 5000 行——分片保证清洗数据全部被读到

**验证结果**：清洗产出 **237,707 条干净对话，60 个分片** → `d:/AetherMind-Nano3/03_dialogue_clean/`（MOSS 16.9 万 + 沐雪 6.8 万，system prompt 全部剥离）。

## 21.10 Phase G 纯净对话 SFT（正常对话）

**定位**：Phase F 之后，用**清洗后的数据**重新做对话 SFT，修复 system prompt 污染。目标是让模型"正常对话"，不再复读英文指令。

**修改位置**：`src/training/train_v4.py` + `configs/aethermind4_config.py` + 新增 `train_v4_phaseG.cmd`

1. `AetherMind4Config.set_phase_G(progress)`：关闭演化/固化/物理辅助损失（同 Phase E），`init_temperature=T0=0.7` 保证回复稳定连贯。
2. `TrainingConfig` 新增：`phase_G_steps=3000`、`phase_G_lr=3e-5`、`phaseG_data_dir="d:/AetherMind-Nano3/03_dialogue_clean"`。
3. `TrainerV4` 增加 `phase_g` 模式；`setup()` 中 Phase G 数据集改用 `phaseG_data_dir`。
4. `_train_phaseG()`：复用 Phase E 机制（冻结全部 → 解冻**尾部 3 层 decoder + final_norm + lm_head** → 重建 AdamW(lr=3e-5) + CosineAnnealingLR → 标准 CE）；checkpoint 后缀 `_phaseG`；`_evolution_step` 在 `phase in ("A","E","F","G")` 时跳过演化。
5. `_maybe_resume`：Phase G 优先恢复 `*_phaseG.pt`，否则自动取 `phaseF` → `phaseE` → 非 D/E/F/G 的 base，步数归零。
6. `_find_latest_checkpoint` 增加 `phase_g_only` 分支，`_step` 排序支持 `_phaseG` 标签。
7. CLI：`--phase_g / --phase_G(默认3000) / --phaseG_lr(3e-5) / --phaseG_data_dir`。
8. 新增 `train_v4_phaseG.cmd`（纯 ASCII）。

**验证结果**（GPU 冒烟 2 步）：成功载入 `v4_checkpoint_2000_phaseF_final.pt` base；清洗数据 58 train + 2 eval 分片；`loss≈7.5~7.8`（vocab=151680，远低于随机基线 ln(151680)≈11.93）；`bad=0`；`*_phaseG_final.pt` 正常保存。

## 21.11 三次修改汇总表（Phase G 相关）

|   编号  | 类别             | 修改文件                                          | 模型大小影响 | 状态  |
| :---: | :------------- | :-------------------------------------------- | :------ | :-- |
| 21.9  | 对话数据清洗         | `clean_dialogue_data.py`(新增)                  | 无（数据层）  | 已完成 |
| 21.10 | Phase G 纯净对话SFT  | `train_v4.py`, `aethermind4_config.py`, `train_v4_phaseG.cmd`(新增) | 无（复用主干） | 已完成 |

**Phase G 关键参数**：3000 步（≈24000 样本，覆盖清洗数据约 10%）/ lr=3e-5 / 解冻尾部 3 层 / batch=1+grad_accum=8 / 关闭演化与固化。数据量充足时可将 `--phase_G` 提升至 6000 步以吃满数据。

