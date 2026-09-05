# AetherMind V5 技术文档

> **版本**：5.0.0
> **代号**：physics-first 双层动力学认知体
> **文档日期**：2026-08-29
> **代码根目录**：`D:\AetherMind-Nano3\V5\`
> **作者**：玄曦雪（张悦）

---

## 一、设计哲学与定位

### 1.1 什么是 physics-first

V5 不是"用物理名词命名统计操作"，而是**把物理定律写进结构里，使违反它们在数学上不可能**。判定一个系统是否 physics-first，需同时满足四条硬判据（J1~J4）：

| 判据 | 含义 | V5 实现方式 |
|------|------|-------------|
| **J1 量纲+守恒律** | E、T、τ、熵产生σ必须满足守恒或耗散不等式；每一项的量纲可验证 | 平衡层过阻尼朗之万满足 FDT；驱动层σ=Σθ̇²≥0由结构保证 |
| **J2 变分推导的学习规则** | ΔW来自自由能最小化/Onsager变分/最小作用量，而非 W←W-η∂L/∂W 改名 | EqProp两相局部Hebbian，β→0严格趋向BP梯度 |
| **J3 推理是物理过程非优化过程** | 前向=能量下降到平衡态（Hopfield检索、EqProp自由相、相位锁定），而非 y=f_W(x) 一次矩阵乘 | 推理=沿能量梯度弛豫到不动点 |
| **J4 结构共几何/耗散结构** | 动力学具有 Hamiltonian/port-Hamiltonian/metriplectic 结构 | 平衡层=梯度流（势系统）；驱动层=非平衡主动驱动（无势） |

### 1.2 V5 与 V4 的本质区别

V4 是"统计系综+物理外壳"——温度T是可学习参数、相位θ伪装热噪声、信息素缺扩散项、学习用BPTT。V5 把这四处全部修正：

| 维度 | V4 | V5 |
|------|----|----|
| 温度 T | `log_temp` 可学习参数 | 仅平衡层用，由FDT锁定（噪声幅度=√(2Tdt)），非自由参数 |
| 相位 θ | `dθ=(ω+coup)dt+ξ√(2Tdt)` 伪装热噪声 | 承认非平衡主动驱动，去掉伪装的T，Kuramoto方程无噪声 |
| 信息素 τ | `dτ/dt=-ρτ+source`（缺扩散） | 补扩散项 `∂τ/∂t=D_τ∇²τ-ρτ+ηr̄gate(σ)` |
| 学习规则 | BPTT全局链式求导 | EqProp两相局部Hebbian，β→0严格等价BP |
| 信用信号 | `r=gate(-dF)`（统计损失改名） | 熵产生σ=Σθ̇²≥0，非负性由第二定律结构性保证 |
| GMM/VAE/KL | 双域知识原子等统计模块 | 剥离，不进physics-first叙事 |

### 1.3 原创贡献定位

M1~M3 是**复现**（入场券，非贡献）：
- M1 复现 Ramsauer et al. (2021) 的 attention≡Hopfield 等价性
- M2 验证自己实现的 SDE 满足 FDT
- M3 复现 Scellier & Bengio (2017) 的 EqProp

**真正的原创主张在 M4/M5**：
1. **M4**：信息素τ（含扩散项）是平衡层自由能景观的**慢变量**，LTP阈值门控对应自由能景观的**分岔点**
2. **M5**：驱动层的熵产生σ可作为**无监督信用信号**替代统计损失，σ≥0由第二定律结构性保证

---

## 二、完整数学架构

V5 是**双层动力学系统**：平衡层（快变量，有势，FDT成立）+ 驱动层（慢变量，非平衡主动驱动，无势）。

### 2.1 平衡层（快变量）

**状态**：ξ ∈ ℝᴺ（N个神经元的膜电位）

**能量函数**（现代 Hopfield LSE 型，对ξ严格下有界）：

$$E(\xi) = -\frac{1}{\beta}\log\sum_i \exp(\beta \, x_i^\top \xi) + \frac{1}{2}\xi^\top \xi - (\tau + C) \cdot \xi$$

其中：
- X ∈ ℝ^{P×N} 为存储模式矩阵（按行存储）
- β 为逆温度（越大检索越尖锐）
- τ_field 为信息素投影（慢变量，调制势阱深度）
- C_field 为固化偏置（LTP长期记忆）

**梯度**（闭式，无需自动微分）：

$$\nabla_\xi E = -\text{softmax}(\beta X\xi)^\top X + \xi - (\tau + C)$$

**动力学**（过阻尼朗之万，FDT在此层成立）：

$$d\xi = -\nabla_\xi F \, dt + \sqrt{2T \, dt} \cdot dB, \quad dB \sim \mathcal{N}(0, I)$$

**FDT 检验**（Ornstein-Uhlenbeck 型，过阻尼坐标）：
对二次势 F(ξ)=½λ‖ξ-ξ*‖²，稳态满足 ⟨‖ξ-ξ*‖²⟩ = N·T/λ。
注意：这不是朴素 ⟨ξ²⟩/(2T·dt)，而是位置方差对势曲率λ的反比关系。

### 2.2 驱动层（慢变量）

**相位动力学**（Kuramoto 振荡器，非平衡主动驱动，无温度T）：

$$\frac{d\theta_i}{dt} = \omega_i + \sum_j J_{ij} \sin(\theta_j - \theta_i - \phi_{ij})$$

其中：
- ω_i 为固有频率
- J_ij 为耦合强度
- φ_ij 为相位滞后（φ≠0时系统非互易，可产生净能量流）

**信息素反应-扩散**（V4缺失的∇²τ项在此补全）：

$$\frac{\partial \tau}{\partial t} = D_\tau \nabla^2 \tau - \rho \tau + \eta \, \bar{r} \, \text{gate}(\sigma)$$

其中：
- D_τ 为扩散系数
- ρ 为蒸发率（衰减）
- η 为沉积强度
- r̄ 为平均反应速率（来自平衡层）
- gate(σ) = σ/(σ+scale) 为饱和门控，σ为熵产生

**熵产生**（Onsager形式：σ=Σ通量×热力学力）：

驱动层相位方程是过阻尼且无势的（漂移项非任何势的梯度），故通量=θ̇，热力学力=漂移项（两者相等），于是：

$$\sigma = \sum_i \dot{\theta}_i^2 \geq 0$$

恒定非负，由第二定律结构性保证。离散化后允许 <1e-6 的数值负值按0计。

### 2.3 快慢耦合

驱动层信息素场τ通过**谱投影**回写到平衡层势场：

```
τ_field = project(τ, N)  # M≥N时取前N分量；M<N时周期延拓
```

平衡层势场由驱动层信息素场的（周期性）谱投影得到，体现"τ调制势阱深度"的快慢耦合关系。

**绝热近似适用条件**：τ的时间常数 ρ⁻¹ ≫ K·ε（K为平衡层弛豫步数，ε为弛豫步长），否则快慢耦合会破坏EqProp的两相假设。

### 2.4 学习规则：平衡传播 EqProp

**能量**（逐层连续Hopfield，s_0=x输入钳制）：

$$E = \sum_l \left[ \frac{1}{2}\|s_l\|^2 - b_l \cdot \rho(s_l) - \rho(s_{l-1})^\top W_l \rho(s_l) \right]$$

**两相动力学**：
- **自由相**（β=0）：系统弛豫到能量极小点 ξ⁰
- **钳制相**（β>0）：输出端加 β·∂C/∂s 的nudge，弛豫到 ξ^β

$$\Delta W_l = \frac{\eta}{\beta} \left[ \rho(s_l^+) \rho(s_{l-1}^+)^\top - \rho(s_l^-) \rho(s_{l-1}^-)^\top \right]$$

β→0极限下，该规则严格趋向自由相代价C关于W的梯度（隐式函数定理），即EqProp≡BP。

---

## 三、模块详解

### 3.1 energy.py — 平衡层能量

**现代 Hopfield LSE 能量与梯度**，对应里程碑 M1。

| 函数 | 作用 |
|------|------|
| `energy(xi, X, beta)` | 计算 LSE 能量 E(ξ) = -(1/β)logsumexp(βXξ) + ½‖ξ‖² |
| `grad(xi, X, beta)` | ∇E = -softmax(βXξ)ᵀX + ξ（闭式，无自动微分） |
| `retrieve(query, X, beta)` | 一步Hopfield检索 = softmax注意力输出（value即模式本身） |
| `relax_fixed_point(xi0, X, beta, lr, steps)` | 沿能量梯度弛豫到不动点（用于M1(b)对拍注意力） |

**关键性质**：softmax注意力 ≡ 现代Hopfield单步能量检索，误差为0（M1(b)验证 L2<1e-5）。

### 3.2 equilibrium.py — 平衡层动力学

**过阻尼朗之万 + FDT**，对应里程碑 M2。

| 函数 | 作用 |
|------|------|
| `euler_maruyama_step(xi, force, T, dt)` | Euler-Maruyama单步：ξ←ξ+force·dt+√(2Tdt)·noise |
| `relax(xi0, grad_fn, T, dt, steps, nudge_fn)` | 弛豫K步到（准）平衡态；nudge_fn=None为自由相，否则钳制相 |
| `harmonic_energy(xi, xi_star, lam)` | 二次势 F=½λ‖ξ-ξ*‖²（M2 FDT检验专用） |
| `harmonic_grad(xi, xi_star, lam)` | 二次势梯度 |

**FDT检验**：对二次势，稳态满足 ⟨δξ²⟩=T/λ ∈ [0.95, 1.05]（M2门槛）。

### 3.3 eqprop.py — 学习规则

**平衡传播 EqProp**，替换BPTT，对应里程碑 M3。

`EqPropNet` 类：两层连续Hopfield网络（输入→隐藏→输出），sigmoid激活ρ(x)=σ(x)。

| 方法 | 作用 |
|------|------|
| `relax(x, y, beta, lr, T_steps)` | 弛豫到平衡态；y=None自由相，y给定钳制相 |
| `update(x, y, beta, lr, T_steps)` | 一次EqProp两相更新，返回ΔW1/Δb1/ΔW2/Δb2 |
| `predict(x)` | 前向预测=ρ(s2) |
| `grad_check(x, y, beta, lr, T_steps)` | EqProp更新方向 vs 真实梯度的余弦相似度（M3门槛>0.99） |

**EqProp≡BP验证**：β=0.1时，W权重余弦>0.9999，b偏置余弦>0.9999。

### 3.4 entropy.py — 熵产生

**驱动层熵产生σ（非负信用信号）**，对应里程碑 M5。

| 函数 | 作用 |
|------|------|
| `phase_velocity(theta, omega, J, phi)` | θ̇_i = ω_i + Σ_j J_ij sin(θ_j-θ_i-φ_ij) |
| `entropy_production(theta_dot)` | σ = Σ_i θ̇_i² ≥ 0 |
| `gate(sigma, scale)` | 饱和门控 gate(σ)=σ/(σ+scale) ∈ [0,1) |

σ的离散定义采用验收标准候选B（耗散率形式）：驱动层做功 W_drive=∮J·dθ 的耗散率。

### 3.5 driving.py — 驱动层

**Kuramoto相位振荡器 + 信息素反应-扩散场**，对应里程碑 M4。

`DrivingLayer` 类：

| 方法 | 作用 |
|------|------|
| `step_phase(dt)` | 推进相位一步，返回θ̇供σ计算 |
| `step_pheromone(dt, r_bar, sigma)` | 推进信息素一步（反应-扩散+σ门控沉积） |
| `step(dt, r_bar)` | 完整慢变量推进：先相位后信息素，返回(σ, τ) |

**信息素稳态检验**（M4）：恒定源下弛豫到稳态，max|∂τ/∂t|<1e-4。

### 3.6 model.py — 顶层双层模型

`AetherMindV5` 类：平衡层（Modern Hopfield）+ 驱动层（Kuramoto/信息素）的双层动力学。

| 方法 | 作用 |
|------|------|
| `infer(query, dt, steps, noise)` | 查询→沿能量弛豫到吸引子（过阻尼朗之万，物理推理非矩阵乘） |
| `drive_step(dt)` | 推进驱动层一步，信息素场投影回平衡层势场（τ→τ_field） |
| `consolidate(threshold, lam, gamma)` | LTP固化：超阈值信息素写入长期偏置C，局部衰减τ |

### 3.7 milestones.py — M1~M5验收

可运行的Go/No-Go验收脚本：

```
C:\Python312\python.exe V5\milestones.py --all
```

| 里程碑 | 内容 | 门槛 |
|--------|------|------|
| M1 | 64神经元现代Hopfield | 能量单调下降、attention≡不动点L2<1e-5、收敛O(log(1/ε)) |
| M2 | 过阻尼朗之万FDT | ⟨δξ²⟩=T/λ ∈ [0.95,1.05] |
| M3 | EqProp替换BP | β→0极限梯度余弦相似度>0.99 |
| M4 | 信息素反应-扩散 | 稳态∂τ/∂t=0残差<1e-4 |
| M5 | 驱动层熵产生σ | σ≥0全部步成立 |

---

## 四、MNIST 训练流程（train_mnist.py）

### 4.1 架构

基于V5 EqProp内核的批处理扩展，两层连续Hopfield网络：

```
输入(784) → 隐藏层(hidden) → 输出(10)
```

**关键设计决策**：

| 设计选择 | 原因 |
|----------|------|
| 输入层恒等激活 ρ(x)=x | MNIST像素已在[0,1]，再过sigmoid会压缩动态范围到[0.5,0.73] |
| Leaky hard sigmoid 隐藏/输出激活 | 纯sigmoid导数≤0.25，Hopfield不动点s*被压缩在0附近，输出永远在0.5附近；hard sigmoid导数=1（活跃区），允许状态到达0/1做置信预测；leak=0.05防止完全死区 |
| 钳制相热启动 | 从自由相状态开始钳制相弛豫（标准EqProp做法），减少T_clamp步数 |
| 自由相T_free > 钳制相T_clamp | 自由相需从随机初值收敛；钳制相从自由相热启动，只需少量步 |

### 4.2 批处理 EqProp 更新

```python
# 自由相（较长弛豫）
s1_free, s2_free = relax(x, None, T_steps=T_free)
# 钳制相（热启动，较短弛豫）
s1_clamp, s2_clamp = relax(x, y, beta=beta, T_steps=T_clamp,
                            s1_init=s1_free, s2_init=s2_free)
# 局部Hebbian更新
dW1 = (1/β) * mean_batch[ρ(s1⁺)ρ(x)ᵀ - ρ(s1⁻)ρ(x)ᵀ]
dW2 = (1/β) * mean_batch[ρ(s2⁺)ρ(s1⁺)ᵀ - ρ(s2⁻)ρ(s1⁻)ᵀ]
W1 += lr_w * dW1; W2 += lr_w * dW2
```

### 4.3 驱动层σ门控（可选）

开启 `--driving` 后，每个batch：
1. 推进Kuramoto相位一步
2. 计算熵产生 σ=Σθ̇²
3. gate(σ)=σ/(σ+scale) ∈ [0,1)
4. 有效学习率 eff_lr = lr_w * (1 + α * gate(σ))

σ越大（系统越远离平衡），学习率越高——非平衡热力学作为学习信号的实例化。

### 4.4 运行指令

```powershell
# GPU标准训练
C:\Python312\python.exe V5\train_mnist.py --epochs 15 --hidden 500 --batch_size 64 --cuda

# 开启驱动层σ门控
C:\Python312\python.exe V5\train_mnist.py --epochs 15 --hidden 500 --driving --cuda

# CPU快速验证
C:\Python312\python.exe V5\train_mnist.py --epochs 3 --hidden 256 --no_cuda
```

### 4.5 超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 5 | 训练轮数 |
| `--hidden` | 500 | 隐藏层神经元数 |
| `--batch_size` | 32 | 批大小 |
| `--beta` | 0.5 | EqProp钳制强度 |
| `--T_free` | 30 | 自由相弛豫步数 |
| `--T_clamp` | 15 | 钳制相弛豫步数（热启动） |
| `--lr_w` | 0.1 | 权重学习率 |
| `--driving` | 关 | 开启驱动层σ门控 |
| `--sigma_scale` | 1.0 | gate(σ)饱和尺度 |
| `--sigma_alpha` | 0.5 | 学习率调制强度 |

### 4.6 训练收敛性

实测（GPU，hidden=500，batch=64，lr_w=0.01）：
- Epoch 1：test_acc=21.5%，cost=0.77
- Epoch 2：test_acc=34.6%，cost=0.46
- Epoch 3：test_acc=37.8%，cost=0.42（峰值）
- 后续：lr_w=0.01偏高导致震荡，建议降至0.003并增加T_free/T_clamp

---

## 五、与 V4 的组件归属映射

| V4 组件 | V5 去向 | 说明 |
|---------|---------|------|
| PGTA注意力 | 平衡层 Modern Hopfield | softmax注意力≡Hopfield单步检索（等价一） |
| τ/C信息素/固化 | 平衡层势场偏置 + 驱动层反应-扩散 | τ/C≡能量景观势场项（等价二）；补全扩散项 |
| BPTT训练 | EqProp两相局部Hebbian | BPTT≡EqProp β→0极限（等价三） |
| 在线演化（推理时学习） | 驱动层σ门控 + LTP固化 | 推理时驱动层持续演化，信息素超阈值写入C |
| 训练/推理架构转换 | 消除 | 同一套方程，β=0推理/β>0训练，无需convert脚本 |
| GMM/VAE/双域知识原子 | 剥离 | 不进physics-first叙事，降级为工程组件 |
| Agent记忆/RAG/KG | 未接入 | V5是最小自洽内核，上层应用待构建 |

---

## 六、已知限制与未来方向

### 6.1 当前限制

1. **MNIST准确率待提升**：当前配置下约38%，需进一步调参（降低lr_w、增加弛豫步、学习率衰减）
2. **仅两层网络**：EqPropNet为两层MLP，未扩展到多层或注意力结构
3. **驱动层未端到端验证**：σ门控学习率已实现，但M5主张"σ信号驱动训练不劣于统计信用"需对照实验验证
4. **M4分岔实验未做**："信息素是自由能景观慢变量"的原创主张需分岔点重合实验验证
5. **无port-Hamiltonian结构**（M6可选）：J4要求的共几何/耗散结构当前仅在平衡层（梯度流）成立，驱动层为非平衡主动驱动

### 6.2 未来方向

| 方向 | 内容 |
|------|------|
| M3b | MNIST完整训练，准确率门槛（标准MLP 2层应>97%） |
| M4实验 | 信息素慢变量+LTP分岔点重合验证 |
| M5实验 | σ信用信号 vs 统计损失的对照实验 |
| M6 | port-Hamiltonian重构（J反对称+R半正定） |
| 多层扩展 | EqProp扩展到深层网络（逐层两相或全局两相） |
| 注意力EqProp | Modern Hopfield检索层 + EqProp训练的统一 |
| 语言模型 | 接入tokenizer/embedding，从MNIST分类到序列生成 |

---

## 七、文件清单

```
V5/
├── energy.py        平衡层能量：现代Hopfield LSE能量与梯度（M1）
├── equilibrium.py   平衡层：过阻尼朗之万+FDT（M2）
├── eqprop.py        学习规则：平衡传播EqProp（M3）
├── entropy.py       驱动层熵产生σ（非负信用信号，M5）
├── driving.py       驱动层：Kuramoto相位+信息素反应-扩散（M4）
├── model.py         AetherMindV5顶层双层模型（5.A整合）
├── train_mnist.py   MNIST完整训练流程（批处理EqProp+驱动层σ门控）
├── milestones.py    M1~M5量化验收（Go/No-Go）
└── README.md        V5/V4区分说明
```

所有模块仅依赖 `torch` 与同级模块，无任何 V4 `src/` 依赖。V4代码完全未动。

---

## 八、运行验证

```powershell
# M1~M5 全部验收
C:\Python312\python.exe V5\milestones.py --all

# MNIST 训练（GPU）
C:\Python312\python.exe V5\train_mnist.py --epochs 15 --hidden 500 --batch_size 64 --cuda

# 顶层模型冒烟
C:\Python312\python.exe -c "import sys; sys.path.insert(0,'V5'); from model import AetherMindV5; import torch; X=torch.randn(16,64); X=X/X.norm(dim=1,keepdim=True); m=AetherMindV5(patterns=X); out=m.infer(X[0]+0.1*torch.randn(64)); print('infer OK, shape=', out.shape)"
```
