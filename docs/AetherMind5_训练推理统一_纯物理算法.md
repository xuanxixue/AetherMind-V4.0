# AetherMind V4 · 训练与推理的统一：纯物理算法的神经网络（整合终版 V4.3）

> **主题**：把「训练架构（绝对 τ/C + BPTT）」与「推理架构（相对核 + 在线演化）」合并为**同一个算法**，并以 physics-first 的四条硬判据（J1~J4）为标准，诊断 PGTA 的物理一致性，给出方程级改造规格与量化验收标准。
>
> **状态**：架构统一设计文档 · physics-first 批判版 · 含 V5 目标规格与 Go/No-Go 验收门槛
> **版本历程**：V4.0（三个等价性）→ V4.1（引入四判据）→ V4.2（平衡/非平衡坐标拆分 + 方程规格 + 止损线）→ **V4.3（整合终版：补入骨架替换总表与组件重新分配表）**
> **前置文档**：[架构报告（实战调整后）](AetherMind4_架构报告_实战调整后的.md)、[三方案结合技术讲解](AetherMind4_三方案结合_技术讲解报告.md)、[推理在线演化物理方案](AetherMind4_推理在线演化_上下文显存_物理方案.md)

---

## 摘要（结论先行，分两层）

本文回答两个不同层级的问题，**切不可混为一谈**：

**第一层（上半篇，已解决）：训练与推理能否统一为同一算法？**
能。三个恒等式（非近似）把 Transformer 骨架钉死在物理坐标上——softmax 注意力 ≡ 现代 Hopfield 单步能量检索，τ/C 偏置 ≡ 能量景观势场项，BPTT ≡ 平衡传播 EP 的 β→0 极限。合并后训练/推理是同一条弛豫方程的两个 β 位。

**第二层（下半篇，真正的门槛）：这套方案是不是 physics-first？**
还不是。**PGTA 目前本质仍是"统计系综 + 物理外壳"。** 三个硬伤：

1. **变量无量纲约束**——温度 T 被设为 `exp(log_temp)` 的可学习参数，但物理系统中 T 由涨落-耗散定理锁定：⟨ξ²⟩ = 2γT，T 不是自由参数；
2. **平衡坐标与非平衡驱动坐标混淆**——朗之万方程把相位 θ（受本征频率 ω 驱动，本质是非平衡主动驱动）与振幅 r（有势能景观，可定义温度）写进同一个框架，导致 FDT 在两处都不成立，且熵产生无法定义；
3. **学习信号来自统计损失而非物理变分原理**——权重更新仍是 W ← W − η∇L 的改名，不是从自由能/Onsager/最小作用量的一阶变分推导出的局部 Hebbian 规则。

**核心判断一句话：physics-first 的真正门槛不是"用了哪些物理公式"，而是"哪些物理定律被写进了架构的结构里，使得违反它们在数学上不可能"。** 改造路径从「EqProp + Modern Hopfield」这个最易落地的切口开始，本文给出方程级规格（第 5.A 节）、伪代码（第 5.3 节）与量化验收门槛（第 7 节）。

---

## 零、判据：什么是"物理为核心"

"物理启发"（physics-inspired）和"物理一致"（physics-consistent）是两件完全不同的事。一个系统要被称为 physics-first，必须**同时满足**以下四条，缺一条就仍是统计模型披了物理皮：

| 判据 | 含义 | 反例（PGTA 现状） |
|---|---|---|
| **J1 量纲 + 守恒律** | E、T、θ、熵产生 σ 必须满足守恒或耗散不等式（dE/dt≤0、dS/dt≥0、Onsager 倒易关系 L_ij=L_ji）；每一项的量纲可验证 | T 设为可学习参数，等价于承认系统不是热力学系统 |
| **J2 变分推导的学习规则** | ΔW 来自自由能最小化/Onsager 变分/最小作用量/涨落定理，而非 W←W−η∇L 改名 | 现为 BPTT 全局链式求导 |
| **J3 推理是物理过程非优化过程** | 前向 = 能量下降到平衡态（Hopfield 收敛、EqProp 自由相、相位锁定、相变），而非 y=f_W(x) 一次矩阵乘 | 注意力离散 softmax，非不动点弛豫 |
| **J4 架构具几何/耗散结构** | 动力学 ODE/SDE 具有 Hamiltonian / port-Hamiltonian / metriplectic 结构，能量守恒与第二定律在**结构层面**成立 | 手写动力学，无结构保证 |

**写完任何"物理模块"后的自检清单**（逐项勾选，缺一即回炉）：

- [ ] 守恒量（能量/熵/磁化）在数值积分中真守恒？（RK4 vs 辛积分器对比能量漂移）
- [ ] 涨落-耗散定理在稳态成立？（对平衡坐标测 ⟨v²⟩ 是否等于 kT/m，误差 < 5%）
- [ ] 学习规则能否从某变分原理（Onsager/自由能/最小作用量）一阶变分**推出**，而非被设计出来？
- [ ] 去掉所有统计损失后，网络仍能靠纯物理动力学完成推理？
- [ ] 若硬件化为真实物理基质（光学/电子/机械），方程是否仍自洽？

---

## 一、现状诊断：为什么现在有两套算法

### 1.1 训练架构（`train_v4.py` + `pheromone_thermo.py`）

- **注意力**：softmax 全注意力，`A = softmax(−E_eff/T)`，E_eff = E − βT·log τ − T·C，O(S²)
- **状态**：τ/C 用绝对坐标 `(h, S_max, S_max)`
- **学习**：BPTT 更新 W；信息素沉积/蒸发是"物理规则"，但只是**挂在**训练循环旁，与 W 更新**不是同一条动力学**

### 1.2 推理架构（`inference_v4.py` + `pheromone_thermo_inference.py`）

- **注意力**：窗口掩码 / 可选低秩线性，相对核 `tau_rel`/`consolidated_rel` `(h, 2k−1)`
- **学习**：在线 Hebbian（注意力聚合 + 扩散演化），无反向传播
- 靠 `convert_train_to_inference.py` 搬运权重

### 1.3 根因：世界观分裂，且是深层分裂

转换脚本的存在证明"训练 ≠ 推理"。但更重要的是：**即使拆掉这层分裂，PGTA 也还没到 physics-first**——它只是把统计模型用物理语言重新描述，没有把物理定律写进结构。以下诊断给出比"漏阻尼"更精确的物理定位。

---

## 二、三个精确等价性（骨架层面的成果）

> **诚实定位前置**：以下三条是"数学等价性"，是 physics-first 的**必要前提而非充分条件**。它们解决"骨架能否物理替换"，不解决"替换后是否真物理"。

### 2.1 等价一：softmax 注意力 ≡ 现代 Hopfield 单步能量检索

Ramsauer et al. (ICLR 2021) 证明：连续模式 Hopfield 网络用 LSE 能量

$$E(\xi) = -\frac{1}{\beta}\log\sum_i \exp(\beta\, x_i^\top \xi) + \tfrac12 \xi^\top \xi + C$$

其更新一步数学上恒等于 softmax 注意力。拼接模式法取 $x_i=[k_i;v_i]$、$\xi_0=[q;0]$，取出 value 分量即得 $\text{Attention}(q,k,v)$，误差 0。**注意力 = 一次能量驱动的记忆检索，向吸引子弛豫一步。**

### 2.2 等价二：τ/C 偏置 ≡ 能量景观的势场项

$\tilde{E}_{ij}=\hat{E}_{ij}-\beta T\log\tau_{ij}-T\,C_{ij}$ 中取 $\beta \psi_i=\log \tau_i+C_i$，τ/C 就是能量景观的**势阱深度调制**，逐模式偏置只进 softmax 指数项，误差 0。**经验不改 W 也能通过改地形偏转检索**——这是 physical insight，但注意：此处 T 依旧是被"赐予"的，不是系统内生的。

### 2.3 等价三：反向传播 = 平衡传播 EP 的 β→0 极限

EP 用两相位（自由相 ξ⁰ / 钳制相 ξ^β）的局部差分

$$\Delta W = -\frac{\eta}{\beta}\left(\partial_W F\big|_{\xi^\beta} - \partial_W F\big|_{\xi^0}\right)$$

当 β→0 严格趋向 BP 梯度。**反向传播只是"对比两个平衡态"在弱扰动极限下的影子。**

三条合起来的结论：训练与推理**可以被**同一个能量最小化范式收编。但它们**不保证** J1~J4 成立。

### 2.4 骨架替换总表（Transformer → 物理对应物）

在三条等价性的基础上，把 Transformer 的骨架逐模块映射到物理对象，并**区分"数学可替换"与"替换后是否物理成立"**——后者必须再过 J1~J4：

| Transformer 组件 | 物理对应物 | 数学可替换 | 替换后的形式 | 物理成立（J1~J4）的额外条件 |
|---|---|---|---|---|
| 缩放点积注意力 | 现代 Hopfield 单步检索 | ✅ 严格 | LSE 能量 + CC 一步 | 能量对 ξ 下有界（否则不收敛） |
| softmax 归一化 | 玻尔兹曼分布（正则系综） | ✅ 同义 | softmax(−E/T) | **T 必须由 FDT 锁定，不能是可学习参数**（否则违反 J1） |
| 多头结构 | 多组独立存储模式 | ✅ 同义 | 逐头独立 Hopfield | 各头能量独立，无明显额外约束 |
| τ/C 偏置 | 能量景观势场项 | ✅ 已是 | βψ_i = logτ_i + C_i | τ 须成反应-扩散场（补 ∇²τ），否则非物理 |
| 反向传播 | 平衡传播 EP | ✅ 极限等价 | 两相位差分规则 | 只对**平衡层**成立；非平衡层须改用 σ |
| 逐层前向 | 状态弛豫到平衡 | ✅ 可替换 | K 步能量梯度下降 | 需 FDT 在稳态成立（milestone M2） |
| FFN / 归一化 | 对偶 Hopfield 键值存储 / 能量白化 | ◐ 可改可留 | 通常保留为可学习参数 | 属"相互作用"定义，保留不违反任何判据 |
| 位置编码 | 相对位置偏置（平移对称） | ◐ 方案 A 已替换 | 相对核 τ[i−j] | 平移不变性已满足 |
| SGD / Adam | 目标函数梯度下降 | ✓ 保留 | 仍作用于能量梯度 | 诚实标注为"外部协议"，不伪装成物理量 |

> **郑重声明（防止同行误读）**："纯物理算法"处理的是三样东西——**反向传播图、绝对坐标、训练/推理分叉**，而非"消灭矩阵"。线性变换在能量函数里是"相互作用"的定义方式，与"弹簧""耦合"一样是物理对象本身，不是非物理残留。**真正决定是否 physics-first 的，是 T 是否 FDT 锁定、阻尼是否配平、学习是否来自变分原理——而非有没有矩阵乘法。**

---

## 三、PGTA 逐模块物理一致性诊断（照妖镜）

| PGTA 模块 | 物理一致性诊断 | 判据 |
|---|---|---|
| 玻尔兹曼注意力 + 可学习 T | ❌ 标准 EBM 语言；真实 T 由 FDT 绑定，不是 free parameter | 违反 J1 |
| dθ=(ω+coup)dt+ξ，ξ·√(2T·dt) | ❌ **且问题比"漏阻尼"更深**：θ 受本征频率 ω 驱动，属于非平衡主动驱动，**详细平衡被打破，T 在这个自由度上根本不可定义**。详见 3.1 | 违反 J1/J4 |
| 胡克势 + 耦合势 | ⚠️ 是势能项，但构成 Hamiltonian 须配正则动量与辛积分器，否则不守恒 | 部分 J4 |
| 信息素 dτ/dt=−ρτ+source | ⚠️ 一阶反应-衰减，本质是带 leak 的 Hebbian 可塑性，非反应-扩散（缺 ∇²τ） | 部分 J3 |
| GMM / VAE / KL / 重参数化 | ❌ 纯统计，无物理内容 | 违反 J1~J4 |
| 梯度下降 + 余弦退火 | ❌ 纯统计优化 | 违反 J2 |

### 3.1 精确诊断：平衡坐标 vs 非平衡驱动坐标

这是上一版"漏阻尼"诊断的深化。朗之万方程的 FDT 成立有一个前提：**系统接近平衡态，漂移项是某个势函数的负梯度**。对照 PGTA 的两个自由度：

- **振幅 r**：`dr = −∇F·dt + ξ·√(2T·dt)`。漂移项 −∇F 是势的梯度，**这是合法的过阻尼朗之万方程**（γ=1 已隐含在 −∇F 的归一里），FDT 原则上可成立。问题只在数值实现（是否用 Milstein、步长是否满足稳定性条件）。
- **相位 θ**：`dθ = (ω + coup)dt + ξ·√(2T·dt)`。漂移项 ω+coup **不是任何势的梯度**（ω 是非保守驱动，coup 是旋转对称的交互项）。这是一个**非平衡驱动系统**（类似活性物质/约瑟夫森结阵列），稳态分布**不是** Boltzmann 分布，**T 无法通过 FDT 定义**，但系统有严格定义的**熵产生 σ > 0**。

**结论**：PGTA 把这两类自由度塞进同一个"温度 T"参数下，是物理上的范畴错误。正确做法是把系统拆成两层：

1. **平衡层**（r、ξ、Hopfield 状态）：有势、有 T、有 FDT、可用 EqProp；
2. **驱动层**（θ、信息素 τ）：非平衡、无 T、只有熵产生 σ 与 Onsager 系数，用它的 σ 作为驱动层的学习信号。

这恰好与"双权重快慢分离"的直觉吻合：**平衡层 = 快变量（推理计算），驱动层 = 慢变量（可塑性/记忆）**——这不是巧合，是热力学结构本来就支持这种分离。

**诊断总结**：PGTA 的物理成分是**碎片化**的——蚁群 + Kuramoto + 朗之万 + GMM + VAE + 可学习温度 + 玻尔兹曼 + 胡克，且每块都缺"守恒在结构层面强制成立"。做减法，见第六节。

---

## 四、文献里真实存在的 physics-first 路线

按"哪条物理原理被**结构性强制**"组织，而非"用没用物理名词"：

| 路线 | 核心物理原理 | 学习/推理 | 局部性 | 代表工作 | 成熟度 | 与 PGTA 契合点 |
|---|---|---|---|---|---|---|
| **Hamiltonian / Lagrangian NN** | 辛几何、能量守恒、最小作用量 | 学 H(q,p)/L(q,q̇)，正则方程给动力学 | 否(需BP) | Greydanus et al., NeurIPS 2019；Cranmer et al., LNN 2020；Chen et al., PRE 2022 | 研究活跃 | 把胡克势+耦合势强进 H(q,p)，自动获得能量守恒 |
| **port-Hamiltonian / metriplectic NN** | 能量+熵双结构，显式分离保守流/耗散流，第二定律结构性成立 | 学 H 与 R，保证 dH/dt≤0、dS/dt≥0 | 否 | Desai et al., PRE 2021；Hernández et al., 2023；Stable PH-NN（NeurIPS 2025 前后） | 正在兴起 | 与"高低温区分+分布平滑"最契合：T 作为耗散结构参数自然出现 |
| **Equilibrium Propagation / CHL** | Helmholtz 自由能、Gibbs-Boltzmann 系综 | 两相自由能差 → 局部 Hebbian ΔW∝(s⁺s⁺−s⁻s⁻)；2025 年工作已扩展到有限 nudge | **是** | Scellier & Bengio 2017；Laborieux et al. 2021；arXiv:2511.22024；CHL 综述 | 理论成熟，硬件实验已做 | 替换梯度下降：τ 和 E 都做两相自由能差信号 |
| **Predictive Coding / FEP** | Friston 自由能 = 意外 + 熵；分层生成模型 | 只传预测误差，权重严格局部；与 BP 特定条件下等价 | **是** | Whittington & Bogacz 2017；Friston 2010 | 双线成熟 | 替换元认知门：内在动机由分层预测误差给出 |
| **Modern Hopfield / Dense AM** | 能量景观、相变、统计力学关联函数 | 推理 = 能量下降至吸引子；高阶 F(z)=zᵃ 提容量 | **是**（局部） | Ramsauer et al., ICLR 2021；Krotov & Hopfield 系列 | 极成熟 | 替换玻尔兹曼注意力：T 由 FDT 决定 |
| **Physical Neural Networks（光/声/模拟）** | Landauer 极限、可逆计算、信息-能量等价 | 权重即物理参数（折射率/电阻），in-situ 训练 | 硬件上是 | Hughes et al., Optica 2018；Pai et al., Science 2023；McMahon 组声学 PNN；MIT DPNN | 实验阶段 | 终点形态：信息素/能量/温度映射到真实物理量 |
| **Landauer / 热力学成本** | Landauer 原理 kT·ln2、熵产生下界 | 把能耗作为约束/目标 | — | Tkachenko et al., 2025；Bormashenko 2019 | 理论框架已立 | 给 physics-first 提供根本动机 |
| **非平衡统计力学 + DL** | 涨落定理、熵产生 σ、Jarzynski 等式 | σ 作为学习信号/正则；扩散模型 = 非平衡热力学的统计实现 | 部分 | Entropy Production Localization 2025；NEQ Thermo NN 2025 | 前沿探索 | 与"高低温+信息素"最自然融合，直接对应本文 5.A 的驱动层 |
| **Kuramoto / 振荡器 NN** | 相位同步、序参量 | 推理 = 相位锁定；学 ωᵢ 或 J_ij | 是 | ONN 文献；储层计算综述 | 成熟 | 已有相位同步，但须把 J 也作为动力学变量 |

---

## 五、统一算法（上半篇成果，框架性保留）

### 5.1 统一能量函数 + 统一弛豫方程

$$F(x,\xi,W,\tau,C) = \underbrace{-{\textstyle\sum_i}\log\cosh(W^{(in)}x)}_{输入耦合} + \underbrace{F_{attn}(\xi)}_{Hopfield 记忆} + \underbrace{F_{phere}(\tau,C)}_{势场} + \underbrace{\tfrac12\|\xi\|^2}_{正则}$$

```
状态 ξ ← 从 x 初始化
for t in 1..K:
    ξ ← ξ − ε·∇_ξ F        # 弛豫到平衡（自由相）
```

**β 开关**：推理 = 自由相（nudge=0）；训练 = 钳制相（输出端加 −β·∇C 力）。

### 5.2 该层结论的诚实定位

统一算法消除了**反向传播图、绝对坐标、训练/推理分叉**三样东西，但**尚未消除"统计损失 + 可学习温度 + 平衡/非平衡坐标混淆"**这三个 physics-first 硬伤。以下给出方程级规格。

### 5.3 统一算法伪代码（两相 + 双层动力学）

```python
# ============ 平衡层：快变量（推理计算所在） ============
# 状态: ξ (Hopfield 状态, 带势, 合法朗之万)
# 动力学: 过阻尼朗之万, FDT 在此层成立
def relax(xi, tau, C, W, T_eff, nudge, beta, steps=K, dt=1e-2):
    for t in range(steps):
        drift = -grad_F(xi, tau, C, W, nudge)      # 漂移 = -∇F (平衡层关键性质)
        noise  = sqrt(2 * T_eff * dt) * randn_like(xi)  # FDT: <ξ²> = 2·T·dt
        xi = xi + drift * dt + noise               # Milstein/Euler-Maruyama
    return xi

# ============ 驱动层：慢变量（可塑性/记忆所在） ============
# 状态: θ (相位, 非平衡驱动), τ (信息素, 反应-扩散)
# 此层无 T, 只有熵产生 σ；σ 是驱动层的学习信号
def drive(theta, tau, r_bar, J, phi, rho, D_tau, eta, sigma_signal, dt):
    # 相位: 非平衡驱动 + 同步耦合 (承认非平衡, 不再伪装 T)
    dtheta = (omega + kuramoto_coupling(theta, J, phi)) * dt
    # 信息素: 补上扩散项, 才是真正的反应-扩散
    dtau = (D_tau * laplacian(tau) - rho * tau
            + eta * r_bar * sigma_signal) * dt   # 沉积强度由 σ 门控
    return theta + dtheta, tau + dtau

# ============ 训练: EqProp 两相 (替换 BPTT) ============
def train_step(x, target, W, tau, C, T_eff, beta, eta_w):
    xi_free = relax(init(x), tau, C, W, T_eff, nudge=0, beta=0)          # 自由相
    xi_clamp = relax(init(x), tau, C, W, T_eff, nudge=-beta*(target-xi_out), beta=beta)  # 钳制相
    # 严格局部 Hebbian, 可证 = ΔF/ΔW (Scellier & Bengio 2017)
    dW = (eta_w / beta) * (corr(xi_clamp) - corr(xi_free))
    # 驱动层用熵产生 σ 而非 dF 作门控 (第二定律非负性)
    sigma = entropy_production(xi_free, xi_clamp)
    _, tau_new = drive(theta, tau, r_bar, ..., sigma_signal=gate(sigma))
    return W + dW, tau_new
```

**关键变化对照表（旧 V4 → 新 V5）**：

| 旧（V4 当前） | 新（V5 目标） | 物理依据 |
|---|---|---|
| T = exp(log_temp) 可学习 | T_eff 只作用于平衡层，由噪声幅度与阻尼经 FDT 锁定；训练中"退火" = 降温调度（外部协议，诚实标注为协议而非物理量） | FDT |
| dθ=(ω+coup)dt+ξ·√(2Tdt) | 相位承认非平衡驱动，去掉伪装的 T；用 σ 作信号 | 非平衡热力学 |
| dτ/dt = −ρτ + source | ∂τ/∂t = D_τ∇²τ − ρτ + η·r·Ā·gate(σ) | 反应-扩散 |
| r = gate(−dF) 作信用 | σ = Σ(流量)×(热力学力) 作信用，非负性由第二定律保证 | 熵产生 |
| BPTT | EqProp 两相局部 Hebbian | 变分原理 |

### 5.A V5 目标架构的方程级规格

**双层动力学完整方程组**（这是 V5 应当实现的全部方程，不多不少）：

**平衡层**（N 个神经元，状态 ξ∈ℝᴺ）：

$$d\xi_i = -\frac{\partial F}{\partial \xi_i}dt + \sqrt{2T}\,dB_i, \qquad F = E_{Hopfield}(\xi;W) + E_{phere}(\xi;\tau,C)$$

**驱动层**（M 个振荡器 + 信息素场）：

$$d\theta_i = \Big[\omega_i + \sum_j J_{ij}\sin(\theta_j-\theta_i-\phi_{ij})\Big]dt \quad \text{(非平衡，无 T)}$$

$$\partial_t\tau = D_\tau\nabla^2\tau - \rho\tau + \eta\,\bar{r}\,\text{gate}(\sigma)$$

**结构约束**（写代码前先验证这三条在离散化后仍成立）：

1. F 对 ξ 是严格下有界的（否则朗之万不收敛）；
2. Onsager 矩阵对称：若驱动层未来加入互扩散（τ 梯度驱动 θ 漂移），必须有对应的倒易项（θ 同步驱动 τ 沉积），且 L_ij = L_ji；
3. J_ij 反对称 + R_ij 半正定（若采用 port-Hamiltonian 形式）。

### 5.B 现有组件在 V5 双层框架中的重新分配

V4 现有一堆物理组件，按 3.1 的双层诊断**重新归类**——不是把它们塞进一个温度，而是明确每一个该进平衡层还是驱动层：

| V4 现有组件 | V5 双层归属 | 重新诊断 | 需补的一致化 |
|---|---|---|---|
| 朗之万振幅 r | **平衡层**（快变量） | ✅ 合法过阻尼朗之万，FDT 可成立 | 数值实现用 Milstein / 受控步长 |
| 朗之万相位 θ | **驱动层**（慢变量） | ⚠️ 非平衡主动驱动，T 不可定义 | 去掉伪装的 T，改用 σ 作信号 |
| 信息素 τ | **驱动层**（慢变量） | ⚠️ 缺扩散项 ∇²τ | 补成真正反应-扩散场 |
| 固化 C | 平衡层势阱 | ✅ 势场偏置，天然兼容 EP | 无 |
| 自由能 F | 平衡层 Lyapunov | ✅ 已有 | 让它成为 EqProp 的能量函数本身 |
| 三层时间尺度 W/τ/C | 快/慢分离 | ✅ 与热力学快慢分离自然吻合 | 对应平衡层(快)/驱动层(慢) |
| GMM / VAE / KL | **剥离** | ❌ 纯统计 | 移出 physics-first 叙事，降级为工程组件 |

---

## 六、改造建议：三条主轴，优先 C → A → B

### 主轴 A：非平衡统计力学路线（最贴现有直觉，承接 5.A 的驱动层）

- 信息素补扩散项：∂τ/∂t = D_τ∇²τ − ρτ + η·r·Ā
- 相位承认非平衡驱动，不再伪装温度；熵产生 σ 替换"自由能下降信用"
- 温度不再可学习：平衡层的 T 由 FDT 从噪声幅度反推；要"高低温区分"，让 D_τ 和噪声幅度随空间位置变化，T(x) 由局域涨落**反推**
- 守恒检验：先查 Onsager 倒易关系 L_ij = L_ji

### 主轴 B：port-Hamiltonian 路线（最严谨、离现状最远）

$$\frac{dx}{dt} = \big[J(x) - R(x)\big]\frac{\partial H}{\partial x} + g(x)u$$

- J 反对称（能量守恒），R 半正定（dH/dt≤0），H 为学习的能量
- 信息素 = R 中的慢变量，温度 = R 的本征值，相位同步 = J 的非对角项
- **收益：第二定律、能量守恒、稳定性在结构层面自动成立，不需要"设计"**

### 主轴 C：学习规则替换路线（最易落地，首步执行）

1. PGTA 写成有能量函数 E(s;W) 的递归网络（Hopfield 形式）
2. 自由相：s 演化到平衡 s⁻
3. 弱微扰相：输出端加 β(target−s_out) nudge，演化到 s⁺
4. ΔW_ij ∝ (1/β)[s_i⁺s_j⁺ − s_i⁻s_j⁻]，严格局部，可证 = 自由能差梯度
5. 信息素 τ 作为慢变量（绝热近似），形成快慢双时间尺度

> 这条路线的物理意义：**学习过程本身就是物理系统的响应函数，不需要外部统计损失。**

---

## 七、可运行 MVP 与量化验收门槛（Go/No-Go）

| 里程碑 | 内容 | 验收门槛（量化） | 判据对应 |
|---|---|---|---|
| **M1** | 64 神经元 Modern Hopfield，验证能量单调下降与 attention 等价 | (a) 能量每步 ΔE≤0，漂移 < 1e-6/千步；(b) attention 输出与 LSE 能量梯度下降不动点的 L2 误差 < 1e-5；(c) 收敛步数 vs 理论 O(log(1/ε)) 吻合 | J3 |
| **M2** | 加入过阻尼朗之万噪声，验证 FDT | 稳态 ⟨ξ²⟩/(2T·dt) ∈ [0.95, 1.05]（10⁵ 步平均） | J1 |
| **M3** | MNIST 上 EqProp 替换 BP | (a) β→0 极限下 EqProp 梯度与 BP 梯度余弦相似度 > 0.99；(b) 测试准确率差距 < 1.5 个点 | J2 |
| **M4** | 加入信息素作慢变量（含扩散项） | (a) τ 场稳态满足 ∂τ/∂t=0 残差 < 1e-4；(b) LTP 阈值门控触发点与自由能景观分岔点重合 | J1/J3 |
| **M5** | 驱动层熵产生 σ 作为信用信号 | σ ≥ 0 在全部训练步成立（离散化后允许 < 1e-6 数值负值） | J1 |
| **M6** | （可选）port-Hamiltonian 重构 | 能量漂移在辛积分器下 < 1e-8/千步；dS/dt ≥ 0 结构性成立 | J4 |

M1~M3 全部通过后，才进入 M4~M6。任何一步不过，回到该步修方程，不跳步。

---

## 八、必须警惕的陷阱

1. **把"用物理名词命名统计操作"误认为 physics-first**：softmax(−E/T)+可学习 log_temp ≠ 热力学系统；热力学要求 T 与涨落幅度经 FDT 绑定。
2. **朗之万方程漏阻尼项**：过阻尼形式（−∇F 漂移 + √(2T) 噪声）是合法的，但前提是漂移必须是势的梯度；若漂移含非保守项（如 ω），必须改判为非平衡系统，放弃 T，改用 σ。
3. **把模拟退火当作"物理核心"**：退火是借物理隐喻的优化协议；physics-first 要求退火由系统能量预算驱动（绝热冷却、Landauer 擦除），或至少诚实标注为"外部协议"。
4. **混合不兼容的物理框架**：Kuramoto（非平衡振荡）、GMM（贝叶斯推断）、反应-扩散（开放系统耗散）三者各有自己的守恒律，强行堆叠会让每个框架的守恒律同时失效。先单框架跑通。
5. **数值方法不配物理**：平衡层 SDE 用 Milstein 或步长受控的 Euler-Maruyama；若引入 Hamiltonian 结构必须用辛积分器（Verlet/隐式中点），RK4 会造成系统性能量漂移，让"守恒验证"变成假阳性。
6. **把"非平衡 = 更高级"当作免罪牌**：非平衡系统确实更有意思（σ 可作学习信号），但它的代价是**失去 Boltzmann 分布、失去 EqProp 的直接适用性**。V5 的双层设计之所以拆成平衡层+驱动层，正是为了让 EqProp 只在平衡层用、σ 只在驱动层用——若有人建议"全部非平衡化"，必须先回答：EqProp 的两相自由能差在非平衡态上如何定义？答不上来就不许动。

---

## 九、预期的负结果与止损线

诚实的研究文档应有止损预案：

- **若 M3 失败**（EqProp 在 MNIST 上与 BP 差距 > 5 个点）：说明能量函数设计有问题或 β 调度不当，回到 M1 检查能量景观；不引入"混合 BP"打补丁——那等于回到统计路线，物理外壳无意义。
- **若 M4 失败**（LTP 门控与分岔点不重合）：说明信息素动力学不是自由能景观的慢变量，"双权重快慢分离"的物理根基不成立；此时应承认信息素只是启发式记忆机制，把它从 physics-first 叙事中剥离，降级为工程组件。
- **若 σ 信号在训练中长时间为 0 或饱和**：说明驱动层与平衡层耦合过弱/过强，调 Onsager 系数而非放宽 σ≥0 检验。
- **总体止损线**：若 M1~M3 在 4~6 周内无法全部通过，说明"物理-first"在当前算力和代码基建下不可行，应退回"统计为核心 + 物理约束正则化"的务实路线（这是学界目前的主流位置，并不丢人）——但必须**诚实标注**，不再宣称 physics-first。

---

## 十、关键文献清单

**理论基础**
- Hopfield, *Neural networks and physical systems with emergent collective computational abilities*, PNAS 1982
- Scellier & Bengio, *Equilibrium Propagation*, Front. Comput. Neurosci. 2017
- Friston, *The free-energy principle*, Nat. Rev. Neurosci. 2010

**Hamiltonian / 辛结构路线**
- Greydanus, Dzamba & Yosinski, *Hamiltonian Neural Networks*, NeurIPS 2019
- Cranmer et al., *Lagrangian Neural Networks*, 2020
- Desai et al., *Port-Hamiltonian Neural Networks*, Phys. Rev. E 2021
- Hernández et al., *Port-metriplectic Neural Networks*, 2023

**局部学习 / EqProp / CHL**
- Laborieux et al., *Scaling Equilibrium Propagation to Deep ConvNets*, 2021
- *Equilibrium Propagation Without Limits*, arXiv:2511.22024, 2025
- *Contrastive Hebbian Learning* 综述；*Hebbian Descent*, Neural Computation 2024
- Predictive Coding benchmark（Whittington & Bogacz 系列）

**Modern Hopfield / Energy-based**
- Ramsauer et al., *Hopfield Networks is All You Need*, ICLR 2021（配套开源 `hopfield-layers`）
- Krotov & Hopfield, *Dense Associative Memory* 系列
- *Non-linear Attention via Modern Hopfield*（2025）

**物理神经网络硬件**
- Hughes et al., *Training of photonic neural networks through in situ backpropagation*, Optica 2018
- Pai et al., *Experimentally realized in situ backpropagation*, Science 2023
- McMahon 组声学 PNN；MIT *Deep Physical Neural Networks*

**热力学 / 非平衡**
- Tkachenko et al., *Thermodynamic bounds on energy use in DNNs*, 2025
- *Non-equilibrium thermodynamic framework for neural networks*, 2025
- *Entropy Production Localization*, 2025
- Öttinger, *Beyond Equilibrium Thermodynamics*（Onsager 变分原理的标准教材）

---

## 结语

上一版曾断言"三者收拢后 V4 就是纯物理算法的神经网络"——这个断言过早了。V4.1 修订承认了"站在门口"；V4.2 把门槛具体化；本终版（V4.3）在前两者之上完成了两件收尾：

1. **范畴修正落地**：PGTA 的问题不只是"漏阻尼"，而是**把平衡自由度与非平衡驱动自由度塞进同一个温度下**。V5 的答案是双层拆分——平衡层用朗之万 + EqProp + T，驱动层用熵产生 σ——这与"双权重快慢分离"的直觉在热力学结构上严丝合缝（第 3.1、5.B 节）。
2. **规格 + 验收闭环**：第 2.4 节给 Transformer 逐模块替换总表（数学可替换 ≠ 物理成立，后者须过 J1~J4）；第 5.A 节给 V5 应实现的全部方程与三条结构约束；第 7 节把 J1~J4 翻译成六个里程碑的量化 Go/No-Go；第 9 节给止损预案，防止"物理叙事"在没有实验支撑时继续膨胀。

从泥沙到水晶的路，第一步是承认自己在泥沙里；第二步是**用可以测量的指标证明每一锹都挖在正确的方向上**。V5 的全部内容，就是这六把锹——而本文，是这六把锹的图纸。