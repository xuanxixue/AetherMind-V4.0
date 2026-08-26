# AetherMind V4.0 Technical Architecture In-Depth Guide

> Codename: **Xirang - Dual-Weight Evolutionary Cognitive System**
> Version: 4.0.0
> Date: 2026-08-25
> Foundation: AetherMind 3.6.1 + Pheromone-Guided Thermodynamic Attention (PGTA) + Evolvable Dual-Weight System (W/τ) + LTP Consolidation
> Code Root: `d:\AetherMind-Nano3`
> Main Model Entry: [src/model/aethermind4.py](file:///d:/AetherMind-Nano3/src/model/aethermind4.py)

---

### Author Information

| Project | Content |

|:-----|:-----|

| **Author** | **Xuan Xixue** (Real Name: Zhang Yue) |

| **Region** | Sichuan, China |

| **Company** | Nanjing Qimeng Xinghui Technology Co., Ltd. |

### Open Source License

[**CC BY 4.0 (Attribution + Citation)**](https://creativecommons.org/licenses/by/4.0/deed.zh-hans)

When using, modifying, distributing, or creating derivative works, **you must credit the original author "Xuan Xixue (Zhang Yue)" and provide a link to the original source.** Commercial use is permitted.

### Open Source Announcement

> **The model code is expected to be officially open-sourced between September 5th and September 25th, 2026** (10-30 days from now). Please stay tuned.

---


## 0. Document Conventions

- Notation: $B$ = batch_size, $S$ = sequence length, $D$ = d_model, $h$ = number of attention heads, $d_h$ = D/h = head_dim, $N_a$ = n_atoms = 1024, $D_a$ = d_atom = 64, $K$ = langevin_K, $V$ = vocab_size = 50000.
- Tensor dimensions are annotated as `(dim0, dim1, ...)`; code file links point to specific line numbers.
- "Phase A/B/C" refers to the three training phases (see Chapter 12); the "phase" parameter in forward controls physics/evolution layer switches.
- All learnable parameters use `nn.Parameter`; cross-step persistent states use `register_buffer` (no gradient participation).
- Mathematical formulas use LaTeX notation; pseudocode uses Python-like syntax.

---

## 1. Design Motivation and Core Proposition

### 1.1 The Problem: Why Mainstream Models Cannot Iterate Weights

Current large models (DeepSeek-V4, Kimi K3, GLM-5.3) follow a unified training paradigm: **pre-training → post-training (RL/SFT) → frozen weights after deployment**.

Gradient descent (BP) has three hard constraints that make it impossible to run online:

1. Requires labels/reward signals
2. Requires backpropagation to compute full-graph gradients
3. Requires offline large-batch retraining

None of these are available after deployment, so weights can only be frozen. To compensate for "amnesia," the industry has invented agent add-ons (RAG, tool calling, long context, external vector databases), but the essence is **moving memory outside the model** — after task switching and context clearing, everything resets to zero. This is pseudo-continuous learning.

### 1.2 V4's Core Proposition

> **The model obtained from pre-training is the "body"; the online-evolving pheromone network is the "experience." Experience should not be stored in external memory but grown into the weights.**

V4 introduces synaptic plasticity principles from biological neural systems (LTP long-term potentiation / LTD long-term depression) to build a **second weight system that can update weights online without BP**, enabling the model to possess:

- **Online Learning**: Every sample during inference can alter core behavior
- **Persistent Memory**: Experience is consolidated into weights; session switches do not clear it
- **Self-Organizing Routing**: Different attention heads spontaneously differentiate into different functional pathways
- **Intrinsic Motivation**: Free energy minimization drives unsupervised evolution

### 1.3 Three Temporal Scales

V4 operates simultaneously across three temporal scales — this is the essential difference between V4 and V3.6 as well as mainstream Transformers:

| Scale               | Variables           | Update Method                       | Frequency          | Analogy                      |
| -------------------- | ------------------- | ----------------------------------- | ------------------ | ---------------------------- |
| **Fast (per-step)**  | Temperature $T$, Attention $A$ | Forward computation           | Per token          | Neuronal firing / Working memory |
| **Medium (cross-sample)** | Pheromone $\tau$    | Deposition + Evaporation (non-gradient) | Per training/inference step | Short-term synaptic efficacy / Episodic memory |
| **Slow (cross-session)** | Consolidation $C$, Fast weights $W$ | LTP threshold write (BP then freeze $W$) | Every 100 steps / Deployment period | Long-term memory / Protein synthesis |

---

## 2. System Overview

### 2.1 Complete Architecture Diagram

```
                                    ┌──────────────────────────────────────────────────────────┐
input_ids (B,S) ──────────────────► │  TOKEN EMBEDDING                                          │
                                    │  W_emb: (V, D)  →  token_emb: (B,S,D)                   │
                                    │  + pos_emb: (S, D)  →  x0: (B,S,D)                      │
                                    └──────────────────────┬───────────────────────────────────┘
                                                           │ x0
                                    ┌──────────────────────▼───────────────────────────────────┐
                                    │  DUAL DOMAIN SYSTEM (GMM Knowledge Atoms)                 │
                                    │  ┌────────────────┐      ┌────────────────┐              │
                                    │  │ GMMAtom (L)    │      │ GMMAtom (P)    │              │
                                    │  │ 1024 atoms     │      │ 1024 atoms     │              │
                                    │  │ 3 components   │      │ 3 components   │              │
                                    │  │ mean/sigma/pi  │      │ mean/sigma/pi  │              │
                                    │  │ → z_L (B,S,Da) │      │ → z_P (B,S,Da) │              │
                                    │  └───────┬────────┘      └───────┬────────┘              │
                                    │          │                      │                       │
                                    │     atom_w_L/P (B,S,Na)                                │
                                    │     A (Na,Na) coupling matrix                            │
                                    │     D_score (B,1) poetic discriminator                  │
                                    └──────────┬───────────────────┬───────────────────────────┘
                                               │                   │
                                    ┌──────────▼───────────────────▼───────────────────────────┐
                                    │  ENCODER V4                                              │
                                    │  PGTA Block × 2:                                         │
                                    │    Pre-LN → PGTA Self-Attention → Residual               │
                                    │    → Pre-LN → FFN(GELU) → Residual                       │
                                    │  Information Bottleneck (per-domain VAE):                 │
                                    │    z_IB_L', z_IB_P'                                     │
                                    │  Aux losses: L_IB, L_D, L_aff, L_H                       │
                                    └──────────┬───────────────────────────────────────────────┘
                                               │ x_enc (B,S,D)
                                    ┌──────────▼───────────────────────────────────────────────┐
                                    │  LANGEVIN OSCILLATOR (K steps)                           │
                                    │  Phase: θ (B,Na), Amplitude: r (B,Na)                    │
                                    │  Free energy F_L, F_P; dF for reward signal              │
                                    │  → Z_phys (B,Na,D) → Z_phys_tokens (B,S,D)             │
                                    └──────────┬───────────────────────────────────────────────┘
                                               │ Z_phys_tokens, r, θ
                                    ┌──────────▼───────────────────────────────────────────────┐
                                    │  DUAL MEMORY + META-COGNITIVE GATE                       │
                                    │  Episodic: ring KV buffer (4096 slots)                   │
                                    │  Structural: learnable skill KV (1024 slots)            │
                                    │  TriPercept: 3-MLP voting → α_L/α_P, u_cog              │
                                    │  SafetyFilter: 64 danger anchors                         │
                                    │  → Z_cog (B,S,D), T (scalar)                            │
                                    └──────────┬───────────────────────────────────────────────┘
                                               │ Z_cog, T, θ_tokens
                                    ┌──────────▼───────────────────────────────────────────────┐
                                    │  DECODER: GLUBlockV4 × n_layers                         │
                                    │  Per layer:                                               │
                                    │    1. PGTA Self-Attention (T set by metacog)             │
                                    │    2. GLU FFN                                             │
                                    │    3. TSR (cognitive state routing)                       │
                                    │    4. PSR (phase state routing)                           │
                                    │    5. Z_cog injection + physics gating (α_t, α_p)        │
                                    │  TSR/PSR state passed cross-layer                        │
                                    └──────────┬───────────────────────────────────────────────┘
                                               │ h (B,S,D)
                                    ┌──────────▼───────────────────────────────────────────────┐
                                    │  OUTPUT HEAD                                              │
                                    │  CounterSlot: GRUCell per position                       │
                                    │  logits = h @ W.T × D^{-0.5}  (weight-tied)              │
                                    │  probs = safe_softmax(logits, T=T)                       │
                                    │  Copy: p_copy · cum_onehot(S)                            │
                                    │        + (1-p_copy) · probs                              │
                                    └────────────────────┬────────────────────────────────────┘
                                                         │
                              ┌──────────────────────────┴─────────────────────┐
                              ▼                                                ▼
                   ┌─────────────────────┐                     ┌──────────────────────────┐
                   │ LOSS COMPUTATION    │                     │ EVOLVABLE WEIGHT SYSTEM   │
                   │ L_LM = CE(shift)    │                     │ (non-gradient, called post-step) │
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

### 2.2 Module Inventory and Code Index

| #   | Module        | File                                                                                                    | Class Name                       | Lines       | Version |
| --- | ------------- | ------------------------------------------------------------------------------------------------------- | -------------------------------- | ----------- | ------- |
| M1  | Main Model    | [aethermind4.py](file:///d:/AetherMind-Nano3/src/model/aethermind4.py)                                | `AetherMind4`                    | 263-515     | V4      |
| M2  | V4 Encoder    | [aethermind4.py#L34-L138](file:///d:/AetherMind-Nano3/src/model/aethermind4.py#L34-L138)              | `EncoderV4`                      | 34-138      | V4      |
| M3  | V4 Decoder Block | [aethermind4.py#L144-L257](file:///d:/AetherMind-Nano3/src/model/aethermind4.py#L144-L257)            | `GLUBlockV4`                     | 144-257     | V4      |
| M4  | PGTA Attention | [attention/pheromone_thermo.py](file:///d:/AetherMind-Nano3/src/model/attention/pheromone_thermo.py) | `PheromoneThermoAttention`       | 20-218      | V4 New  |
| M5  | Evolution Controller | [evolution/evolvable_weight.py](file:///d:/AetherMind-Nano3/src/model/evolution/evolvable_weight.py) | `EvolvableWeightSystem`          | 20-123      | V4 New  |
| M6  | Dual-Domain GMM | [domain/dual_domain.py](file:///d:/AetherMind-Nano3/src/model/domain/dual_domain.py)                 | `DualDomainSystem`/`GMMAtom`     | 9-196       | 3.6 Inherited |
| M7  | Langevin       | [physics/langevin.py](file:///d:/AetherMind-Nano3/src/model/physics/langevin.py)                      | `LangevinOscillator`             | 9-220       | 3.6 Inherited |
| M8  | Dual Memory    | [memory/dual_memory.py](file:///d:/AetherMind-Nano3/src/model/memory/dual_memory.py)                 | `DualMemorySystem`               | 73-92       | 3.6 Inherited |
| M9  | Meta-Cognitive Gate | [metacog/meta_gate.py](file:///d:/AetherMind-Nano3/src/model/metacog/meta_gate.py)                   | `MetaCognitiveGate`              | 49-99       | 3.6 Inherited |
| M10 | Utility Functions | [utils/ops.py](file:///d:/AetherMind-Nano3/src/utils/ops.py)                                          | `MLP`/`safe_softmax`/`stopgrad`  | 1-96        | 3.6 Inherited |
| M11 | Configuration  | [configs/aethermind4_config.py](file:///d:/AetherMind-Nano3/configs/aethermind4_config.py)           | `AetherMind4Config`              | 7-170       | V4      |
| M12 | Trainer        | [training/train_v4.py](file:///d:/AetherMind-Nano3/src/training/train_v4.py)                         | `TrainerV4`                      | 1-221       | V4      |
| M13 | Dataset        | [data/dataset.py](file:///d:/AetherMind-Nano3/src/data/dataset.py)                                    | `StreamingTextDataset`           | -           | 3.6 Inherited |

### 2.3 Tensor Shape Reference Table (Default Config: d=512, h=8, dh=64, S=1024, Na=1024, Da=64, B=1)

| Tensor              | Shape             | Type        | Description                                |
| ------------------- | ----------------- | ----------- | ------------------------------------------ |
| input_ids           | (B,S)             | int64       | Input token IDs                            |
| token_emb           | (B,S,D)           | fp32/bf16   | Token embeddings                           |
| pos_emb             | (B,S,D)           | fp32/bf16   | Learnable positional embeddings            |
| atom_w_L/P          | (B,S,Na)          | fp32/bf16   | Token-to-atom soft assignment weights     |
| z_L/P               | (B,S,Da)          | fp32/bf16   | Atom-space token representations           |
| z_IB_L/P            | (B,S,D)           | fp32/bf16   | IB inputs after world model + projection   |
| mu_L/P              | (Na,n_comp,Da)    | fp32        | GMM component means                        |
| omega_L/P           | (Na,)             | fp32        | Natural frequency √σ̄                        |
| A_L/P (Association Matrix) | (Na,Na)           | fp32        | Inter-atom coupling strength C⊙(1-E)      |
| r                   | (B,Na)            | fp32/bf16   | Langevin amplitude                         |
| θ                   | (B,Na)            | fp32/bf16   | Langevin phase                             |
| Z_phys              | (B,Na,D)          | fp32/bf16   | Physics output (after phys_mlp projection) |
| Z_phys_tokens       | (B,S,D)           | fp32/bf16   | atom_w einsum mapped to token level        |
| θ_tokens            | (B,S)             | fp32/bf16   | Phase token-level mapping                  |
| epi_L/P             | (B,S,D)           | fp32/bf16   | Episodic memory retrieval results          |
| Z_cog               | (B,S,D)           | fp32/bf16   | Meta-cognitive fused representation       |
| T                   | scalar→(1,)       | fp32        | Meta-cognitive temperature                  |
| Q/K/V in PGTA       | (B,h,S,dh)        | fp32/bf16   | Q/K/V after projection and head split      |
| E (Energy Matrix)   | (B,h,S,S)         | fp32/bf16   | -QK/√dh                                    |
| τ (Pheromone)       | (h,S_max,S_max)   | fp32 buffer | Cross-step persistent                      |
| C (Consolidation)   | (h,S_max,S_max)   | fp32 buffer | Permanent bias                             |
| A (Attention)       | (B,h,S,S)         | fp32/bf16   | softmax(-E_eff/T)                          |
| PGTA out            | (B,S,D)           | fp32/bf16   | Wo(A@V)                                    |
| h (Decoder hidden)  | (B,S,D)           | fp32/bf16   | Decoder layer output                       |
| TSR state M         | (B,S,Ds)          | fp32/bf16   | Cross-layer cognitive state                |
| PSR state M_p       | (B,S,Ds)          | fp32/bf16   | Cross-layer physics state                  |
| logits              | (B,S,V)           | fp32/bf16   | LM output (before weight tying)            |
| p_copy              | (B,S)             | fp32/bf16   | Copy gate probability                      |
| out_prob            | (B,S,V)           | fp32/bf16   | Final output distribution                   |

---

## 3. PGTA Pheromone-Guided Thermodynamic Attention (V4 Core Innovation)

### 3.1 From Standard Softmax Attention to Boltzmann Distribution

**Standard Scaled Dot-Product Attention** (Vaswani 2017):

$$A_{ij} = \text{softmax}\left(\frac{Q_i K_j^T}{\sqrt{d_h}}\right), \quad O_i = \sum_j A_{ij} V_j$$

Its essence is statistical normalization — treating similarity as log-odds, with softmax merely being a probability normalization operation. The temperature is implicitly fixed at 1, with no physical degrees of freedom.

**V4 reformulates this as a Boltzmann distribution**:

"Similarity" is redefined as **energy** $E$ (lower energy = higher similarity), and attention weights are defined as the **probability of being in that state in a canonical ensemble**:

$$E^{(h)}_{ij} = -\frac{Q^{(h)}_i \cdot K^{(h)}_j}{\sqrt{d_h}}$$

$$A^{(h)}_{ij} = \frac{\exp(-E^{(h)}_{\text{eff},ij}/T)}{\sum_k \exp(-E^{(h)}_{\text{eff},ik}/T)}$$

Where $T$ is the **learnable temperature** (log-parameterized $T = \exp(\log T)$, ensuring positivity):

- $T \to 0$: Attention collapses to one-hot (greedy/deterministic)
- $T = 1$: Equivalent to standard softmax
- $T \to \infty$: Attention is uniformly distributed (fully random / maximum entropy exploration)

This directly corresponds to **simulated annealing in statistical physics**: high temperature for exploration, low temperature for convergence.

### 3.2 Energy Whitening (Statistical Auxiliary Layer)

To ensure that the physical meaning of temperature $T$ does not depend on data scale (otherwise $E$'s variance differs across layers/training stages, making the "temperature" at $T=1$ inconsistent), running-statistic whitening is applied to the energy matrix:

$$\mu_E^{(t)} = m \cdot \mu_E^{(t-1)} + (1-m) \cdot \text{mean}(E^{(t)})$$
$$\sigma_E^{(t)} = m \cdot \sigma_E^{(t-1)} + (1-m) \cdot \text{std}(E^{(t)})$$
$$\hat{E} = \frac{E - \mu_E}{\sigma_E + \epsilon}$$

Momentum coefficient $m = 0.99$ (coded as `stats_momentum`). Whitening only updates statistics during training; inference uses fixed statistics.

### 3.3 Pheromone Path Modulation (Swarm Intelligence Layer)

The pheromone matrix $\tau^{(h)} \in \mathbb{R}^{S_{\max} \times S_{\max}}$ is a `register_buffer` (non-parameter, no gradient participation), with shape $(h, S_{\max}, S_{\max})$, and persists across forward passes. It simulates stigmergy in ant colony optimization: **traversed paths leave pheromones; paths with high pheromone concentrations are more likely to be traversed.**

**Effective Energy** (fusing structural energy and path preference):

$$\tilde{E}_{ij} = \hat{E}_{ij} - \beta T \log \tau_{ij} - T \cdot C_{ij}$$

Expanding the softmax:

$$A_{ij} \propto \exp(-\hat{E}_{ij}/T) \cdot \tau_{ij}^{\beta} \cdot \exp(C_{ij})$$

This is the complete three-factor attention formula:

| Factor         | Formula            | Source           | Update Method             |
| -------------- | ------------------ | ---------------- | ------------------------- |
| **Structural** | $\exp(-\hat{E}/T)$ | QK similarity     | BP learning W_Q/W_K      |
| **Path**       | $\tau^\beta$       | Pheromone buffer  | Deposition + Evaporation (non-gradient) |
| **Consolidated** | $\exp(C)$        | Permanent bias    | LTP threshold write       |

- $\beta$: Pheromone sensitivity (the $\alpha$ parameter in ACO), default 1.0
- $C$: Consolidated weight bias, see Section 3.5
- $\tau \geq \tau_{\min} = 0.01$: Prevents log(0) numerical errors

**Information-Theoretic Interpretation**: The $\beta T \log \tau$ term is equivalent to adding a pheromone-induced prior $P_{\text{prior}} \propto \tau^\beta$ to the attention distribution, turning the Boltzmann distribution into a posterior.

### 3.4 Information-Theoretic Quantities: Entropy, Free Energy, Variance

After each step of attention computation, three information-theoretic/statistical quantities are extracted as training signals and state indicators:

**Attention Entropy** (degree of distribution focus):

$$H^{(h)}_i = -\sum_j A_{ij} \log A_{ij}, \quad \bar{H} = \text{mean}_{i,h}(H^{(h)}_i)$$

**Helmholtz Free Energy** (physical target quantity):

$$F = \langle \tilde{E} \rangle_A - T \cdot \bar{H} = \sum_{i,j} A_{ij} \tilde{E}_{ij} - T \bar{H}$$

The Boltzmann distribution $A \propto \exp(-\tilde{E}/T)$ is exactly the distribution that **minimizes free energy $F$** at fixed $T$ — this is not a coincidence but a fundamental result of statistical physics. Therefore, the attention computation itself is solving a variational free energy minimization problem.

**Output Variance** (uncertainty measure):

$$\text{Var}_i = \mathbb{E}_A[V^2] - (\mathbb{E}_A[V])^2 = \sum_j A_{ij} V_j^2 - \left(\sum_j A_{ij} V_j\right)^2$$

Variance directly gives the "confidence" at each query position — high variance indicates scattered attention and model uncertainty.

### 3.5 Pheromone Dynamics: Evaporation and Deposition

After each training/inference step (after forward returns, dispatched uniformly by EvolvableWeightSystem), the pheromone matrix is updated according to **non-gradient physical rules**:

**Evaporation** (use-it-or-lose-it / forgetting):

$$\tau \leftarrow (1 - \rho) \cdot \tau$$

$\rho = 0.05$ (reduced to 0.03 in Phase C to protect established paths). All edges decay uniformly, simulating natural pheromone evaporation in ant colonies.

**Deposition** (reward-gated path reinforcement):

$$\Delta\tau = \eta \cdot r \cdot \bar{A}, \quad \tau \leftarrow \tau + \Delta\tau, \quad \tau \leftarrow \text{clip}(\tau, \tau_{\min}, \tau_{\max})$$

Where:

- $\eta = 0.05$: Deposition strength
- $\bar{A} = \text{mean}_B(A)$: Batch-averaged attention pattern for this step ("which path was taken")
- $r$: Credit/reward signal (see Section 3.6), determining "how good this path is"
- Deposition increases pheromone proportionally to $A_{ij}$ for all edges (i,j) visited by attention

**Boundary Protection**: $\tau \in [\tau_{\min}, \tau_{\max}] = [0.01, 5.0]$, preventing pheromones from diverging to infinity or decaying to zero.

### 3.6 Credit Signal Design (Four Modes)

The credit signal $r$ determines whether deposition is positive (reinforcing good paths) or negative (weakening bad paths). V4 implements four modes:

| Mode            | Formula                                                    | Applicable Phase | Characteristics                                  |
| --------------- | ---------------------------------------------------------- | ---------------- | ------------------------------------------------ |
| `hard`          | $r = \mathbb{1}[\text{pred}=\text{target}] \in \{0,1\}$     | -                | Coarse 0/1, only knows correct/incorrect, not degree |
| `soft`          | $r = P(\text{correct}) \in (0,1)$                          | -                | Continuous but biased (always positive)             |
| `soft_center`   | $r = 2(P(\text{correct}) - 0.5) \in (-1,1)$                | Phase B default   | Zero-centered, reinforces correct and weakens incorrect |
| `free_energy`   | $r = -\text{clip}(dF/(|dF|+1), -1, 1)$                      | Phase C          | Densest signal: intrinsic motivation                |

**Why Phase C Uses free_energy**: Referencing experimental conclusions from `pgtt_self_evolution.py` — coarse 0/1 rewards cause the ant colony to lock into "good enough but non-causal" degenerate solutions (0-14% hit rate of true routing locked in across 7 seeds), while free energy $dF = F_t - F_{t-1}$ provides a continuous, dense credit signal:

- $dF < 0$ (free energy decreasing / model becoming more certain) → Positive reward, current path is reinforced
- $dF > 0$ (free energy increasing / encountering unexpected input) → Negative reward, current path is weakened
- $|dF| \approx 0$ (steady state) → No reward, pheromones naturally evaporate

This achieves **unsupervised intrinsic motivation**: the model actively reinforces attention patterns that reduce its own prediction uncertainty.

### 3.7 LTP Consolidation Bias $C$

$C^{(h)} \in \mathbb{R}^{S_{\max} \times S_{\max}}$ is a second `register_buffer`, initialized to zero. It represents **consolidated permanent weight bias** — the "LTP long-term potentiation" process where experience is written from short-term pheromone $\tau$ into long-term weights $C$ (see Chapter 4 for details).

$C$ appears in the effective energy as $-T \cdot C$: edges with $C_{ij} > 0$ have lower energy (permanently preferred). When expanded, this is equivalent to a multiplicative bias of $\exp(C_{ij})$.

### 3.8 Temperature Control Mechanism

Temperature $T$ has two control methods; V4 uses meta-cognitive gating override:

1. **Learnable Parameter**: `log_temp` as an `nn.Parameter`, updated via BP gradients ($\partial \mathcal{L}/\partial \log T$)
2. **External Override**: `set_temperature(T_val)` called by MetaCognitiveGate, based on cognitive uncertainty $u_{\text{cog}}$
   - High $u_{\text{cog}}$ (high disagreement among three perceptual voters) → High temperature for exploration
   - Low $u_{\text{cog}}$ (three perceptual voters agree) → Low temperature for exploitation
   - Specific formula: $T = T_0(1 + \kappa \cdot u_{\text{cog}})$, see Chapter 8

In V4's forward pass, the temperature $T$ output by the meta-cognitive gate overrides all decoder PGTA layer temperatures via `attn.set_temperature(T_val)`, achieving global temperature coordination. Encoder PGTA layers use their own learnable temperature (because the encoder runs before the meta-cognitive gate).

### 3.9 PGTA Forward Pseudocode

```python
def PheromoneThermoAttention.forward(x, extra_mask=None, T_override=None, update_pheromone=True):
    B, S, D = x.shape
    h, dh = num_heads, D // num_heads

    # 1. Q/K/V projection
    Q = Wq(x).view(B,S,h,dh).transpose(1,2)   # (B,h,S,dh)
    K = Wk(x).view(B,S,h,dh).transpose(1,2)
    V = Wv(x).view(B,S,h,dh).transpose(1,2)

    # 2. Energy computation
    sim = einsum("bhid,bhjd->bhij", Q, K) / sqrt(dh)
    E = -sim                                    # (B,h,S,S)

    # 3. Energy whitening (update running stats during training)
    if whiten and training:
        em = momentum * energy_mean + (1-momentum) * E.mean()
        es = momentum * energy_std  + (1-momentum) * E.std(unbiased=False)
        energy_mean.copy_(em)
        energy_std.copy_(es)
        E = (E - energy_mean) / (energy_std + 1e-5)

    # 4. Temperature
    if T_override is not None:
        T = tensor(T_override)
    else:
        T = exp(log_temp).clamp(1e-2, 1e2)

    # 5. Pheromone + consolidation modulate effective energy
    tau_slice = tau[:h, :S, :S].unsqueeze(0)      # (1,h,S,S)
    log_tau = log(tau_slice.clamp(min=tau_min))
    E_eff = E - beta * T * log_tau                # Pheromone bias

    cons_slice = consolidated[:h, :S, :S].unsqueeze(0)
    E_eff = E_eff - T * cons_slice                 # Consolidation bias

    # 6. Causal/padding mask
    if extra_mask is not None:
        E_eff.masked_fill_(extra_mask, +inf)

    # 7. Boltzmann attention
    A = softmax(-E_eff / T, dim=-1)               # (B,h,S,S)
    A = attn_drop(A)

    # 8. Weighted sum output
    out = einsum("bhij,bhjd->bhid", A, V)         # (B,h,S,dh)
    out = out.transpose(1,2).reshape(B,S,D)
    out = Wo(out)

    # 9. Information-theoretic statistics
    entropy = -(A * (A+1e-8).log()).sum(-1).mean()
    free_energy = (A * E_eff.detach()).sum(-1).mean() - T.detach() * entropy.detach()
    out_sq = einsum("bhij,bhjd->bhid", A, V*V)
    variance = (out_sq - out.view(B,S,h,dh).transpose(1,2)**2).mean()
    tau_conc = tau[:h,:S,:S].max() / (tau[:h,:S,:S].mean() + 1e-9)

    # 10. Cache for pheromone deposition
    _last_A = A.detach()
    _last_E = E_eff.detach()

    return out, {entropy, free_energy, temperature:T, variance,
                 attention:A, tau_concentration:tau_conc}
```

**Pheromone Step Pseudocode** (called by evolver after forward returns):

```python
def step_pheromone(reward=None):
    if _last_A is None: return
    A = _last_A.detach()                          # (B,h,S,S)
    B, h, S, _ = A.shape

    with torch.no_grad():
        # Evaporation
        tau[:, :S, :S].mul_(1 - rho)

        # Deposition
        if reward is not None:
            if reward.dim() == 0:
                delta = deposit * reward.item() * A.mean(0)[:, :S, :S]
            elif reward.dim() == 1:
                r = reward.view(-1,1,1,1)
                delta = deposit * (r * A).mean(0)[:, :S, :S]
            else:
                delta = deposit * reward.detach().mean(0)[:, :S, :S]
        else:
            delta = deposit * A.mean(0)[:, :S, :S]   # No reward: uniform deposition by usage

        tau[:, :S, :S].add_(delta)
        tau.clamp_(tau_min, tau_max)
```

---

## 4. Evolvable Dual-Weight System

### 4.1 Dual-Weight Design Principles

V4's attention layers simultaneously maintain three types of weights/states, forming a **fast-medium-slow** three-level memory system:

| Name                | Storage Type       | Tensor Shape             | Update Method                  | Lifespan                          | Analogy                    |
| ------------------- | ------------------ | ------------------------ | ------------------------------ | --------------------------------- | -------------------------- |
| **Fast Weight $W$**     | `nn.Parameter`    | Wq/Wk/Wv/Wo: (D,D) each  | Gradient Descent (BP)           | Trainable during training, frozen after Phase C | Semantic memory / Cerebral cortex |
| **Pheromone $\tau$**    | `register_buffer` | (h, S_max, S_max)        | Deposition + Evaporation (physical rules) | Persistent across samples, can be reset per session | Working memory / Short-term synaptic efficacy |
| **Consolidation $C$**   | `register_buffer` | (h, S_max, S_max)        | LTP threshold write (non-gradient) | Permanently persistent, retained across sessions | Long-term procedural memory / LTP |

Key distinctions:

- **W changes require BP**: Needs labels, backpropagation, large batches; infeasible after deployment
- **τ changes require no gradients**: Pure physical rules (multiply + add + clamp), tens of microseconds per step; runs online
- **C changes require no retraining**: Threshold-gated writes; permanently alters model behavior after writing

This directly mirrors the three-level memory model of biological neural systems:

- W = **Semantic knowledge** in long-term memory (acquired through learning, relatively stable)
- τ = Short-term / working memory (neural activity traces from the current session, decays over time)
- C = Long-term procedural memory (synapses that are repeatedly activated are consolidated via LTP, permanently retained)

### 4.2 LTP Consolidation Mechanism In Detail

Long-Term Potentiation (LTP) is the core mechanism of memory consolidation in neuroscience: after high-frequency stimulation of a synapse, synaptic efficacy undergoes persistent enhancement. V4 implements this as threshold-gated sparse writing:

**Trigger Condition**: In Phase C, executed every `consolidate_interval=100` steps:

$$\text{mask} = (\tau > \theta_c), \quad \theta_c = 1.5$$

**Write Rules**:

$$C[\text{mask}] \mathrel{+}= \lambda \cdot (\tau[\text{mask}] - \theta_c), \quad \lambda = 0.1$$

$$\tau[\text{mask}] \leftarrow (1 - \gamma) \cdot \tau[\text{mask}], \quad \gamma = 0.5$$

**Mechanism Interpretation**:

1. Only paths where pheromone concentration consistently **exceeds the threshold** (i.e., stable paths that have been repeatedly traversed) are consolidated
2. The excess-above-threshold portion ($\tau - \theta_c$, not the full $\tau$) is written to $C$ — only "strong signals above baseline" are consolidated
3. After writing, $\tau$ locally decays by 50% — **short-term memory yields to long-term weights**, releasing working memory capacity
4. $\lambda=0.1$ is small — consolidation is conservative and gradual, preventing single-occasional paths from being permanently written

**Budget Control**: `max_consolidations=4096` — global consolidation counter stops further consolidation after exceeding the upper limit, preventing unbounded weight growth.

**Why Consolidation Works**: $C$ enters the effective energy $\tilde{E}$ as $-T \cdot C$, equivalent to multiplying by $\exp(C)$ in the softmax. If after consolidation $C_{ij} = 2.0$, that edge's attention weight is multiplied by $\exp(2.0) \approx 7.4$x — path preference is permanently amplified. More importantly, $C$ persists even after $\tau$ is reset, so experience is retained across sessions.

MVP Experimental Validation (from `evolvable_weight.py`):

| Model         | Post-Training τ Peak | Operation          | CLS Focus After Clearing τ      |
| ------------- | -------------------- | ------------------ | ------------------------------ |
| With consolidation | 3.95 @ correct column | Consolidate → Clear τ | **Correct column (0.804)** ✓  |
| Without consolidation | 3.95 @ correct column | No consolidate → Clear τ | Random column (0.095) ✗     |

### 4.3 EvolvableWeightSystem Controller

`EvolvableWeightSystem` is the controller that uniformly manages the pheromone lifecycle of all PGTA layers:

**Registration**: During model initialization, all PGTA layers (2 encoder layers + n_layers decoder layers) register with the evolver via `register_attention()`.

**Per-Step Dispatch**: `evolver.forward(step, free_energy, phase, loss_val)` is called after `optimizer.step()` in each training step:

```python
def forward(step, free_energy=None, phase="A", loss_val=None):
    if phase == "A":
        return  # Phase A: pure BP, no evolution

    # Compute reward
    if phase == "C" and free_energy is not None:
        reward = compute_free_energy_reward(free_energy, F_prev)
        # r = -clip((F - F_prev) / (|F - F_prev| + 1), -1, 1)
    elif loss_val is not None:
        reward = clip(-loss_val.detach(), -1, 1)   # Phase B uses loss
    else:
        reward = None

    # Pheromone update for all PGTA layers
    for attn in _attention_layers:
        attn.step_pheromone(reward)

    # Periodic consolidation (Phase C only)
    if phase == "C" and step % consolidate_interval == 0 and step > 0:
        for attn in _attention_layers:
            attn.consolidate(threshold, lam, gamma, max_cons)
        _consolidation_count += 1
```

**Statistics**:

- `tau_concentration` = mean(max(τ)/mean(τ)): Pheromone concentration, higher means more differentiated paths (initial = 1.0)
- `consolidation_mass` = Σ|C|: Total consolidation mass, representing the amount of consolidated experience (initial = 0)

---

## 5. GMM Dual-Domain Knowledge Atoms (Inherited from 3.6)

### 5.1 Design Principles

Since version 3.5, AetherMind has adopted **dual-domain separation**: language understanding is decomposed into a **Logic Domain** (facts/reasoning/grammar, corresponding to the left brain) and a **Poetic Domain** (emotion/metaphor/style, corresponding to the right brain), with each domain having an independent GMM knowledge atom system.

### 5.2 GMMAtom Structure

Each domain maintains $N_a=1024$ knowledge atoms, where each atom is a **3-component Gaussian Mixture Model (GMM)**:

| Parameter        | Shape         | Description                              |
| --------------- | ------------- | ---------------------------------------- |
| mean_emb        | (Na, K, Da)   | Mean vectors of K=3 components           |
| logvar_emb      | (Na, K, Da)   | Log-variance (via softplus → σ²)         |
| mix_logits      | (Na, K)       | Mixture weights π (via softmax)          |
| mass            | (Na,)         | Inertial mass (for Langevin)             |
| tau             | (Na,)         | Relaxation time (for Langevin)           |
| token_to_atom   | Linear(D, Na) | Token-to-atom projection (for activation) |
| atom_to_token   | Linear(Da, D) | Atom-to-token space projection           |

Where $K = n_{\text{gmm_components}} = 3$, $D_a = d_{\text{atom}} = 64$.

**Mixture Parameter Extraction**:
$$\mu = \text{mean\_emb}, \quad \sigma^2 = \text{softplus}(\text{logvar\_emb}) + 10^{-6}, \quad \pi = \text{softmax}(\text{mix\_logits})$$

### 5.3 Activation Process

```python
def activate(x):  # x: (B,S,D)
    atom_logits = token_to_atom(x)              # (B,S,Na)
    atom_weight = softmax(atom_logits, dim=-1)  # (B,S,Na) soft assignment

    mu, sigma, pi = get_mixture_params()
    mean_atom = (pi.unsqueeze(-1) * mu).sum(dim=1)   # (Na, Da) weighted mean

    z = einsum("bsn,nd->bsd", atom_weight, mean_atom)  # (B,S,Da)
    return atom_weight, z
```

**Interpretation**: Each token obtains soft assignment weights over 1024 atoms via softmax, yielding $w \in \Delta^{N_a-1}$ (a probability simplex). These weights are then used to compute a weighted sum of each atom's mean, producing the token's representation in atom space $z \in \mathbb{R}^{D_a}$.

This is a **soft-routed MoE**: 1024 "expert atoms" are activated per token by probabilistic combination.

### 5.4 Association Matrix and World Model

Each domain maintains a **learnable association matrix pair** (C, E), both of shape (Na, Na):

- $C_{ij} = \text{sigmoid}(\text{logic\_assoc\_C}_{ij})$: Coupling strength between atoms $i$ and $j$
- $E_{ij} = \text{sigmoid}(\text{logic\_assoc\_E}_{ij})$: Edge existence (1 = does not exist, 0 = exists)
- Effective coupling: $A = C \odot (1-E)$, i.e., "has edge and has strength"

The world model MLP `logic_world_model: MLP(D, 2D, D)` applies an additional transformation to the atom-to-token projected representation to predict token-level representations: $z_{\text{IB}} = \text{atom\_to\_token}(z) + \text{world\_model}(\cdot)$.

The poetic domain discriminator `poetic_discriminator: MLP(D, D, 1)` outputs $D_{\text{score}} \in (0,1)$, used for adversarial training to prevent poetic domain collapse — BCE($D_{\text{score}}$, 0.5) forces the discriminator to be unable to distinguish real from generated poetic representations (similar to GAN philosophy, but milder).

### 5.5 Anchor Alignment Loss

```
n_anchor=512 anchor points (anchor_emb: (n_anc, Da))
Each domain has an independent router: Linear(Da, n_anc)
anc = argmax(router(z_flat))       # Hard route to nearest anchor
z_anc = stopgrad(anchor_emb[anc])  # Anchor vector (stopgrad prevents collapse)
loss_anc = λ_align · [MSE(z_L_flat, z_L_anc) + MSE(z_P_flat, z_P_anc)]
```

Anchor alignment causes the atom-space distribution to uniformly fill $n_{\text{anchor}}=512$ prototype positions, preventing all atoms from collapsing into a single cluster.

### 5.6 Modularity Loss

A modularity constraint is applied on the association matrix $A$: atoms with edges ($A_{ij}\approx 1$) should have close embedding distances, while atoms without edges ($A_{ij}\approx 0$) should have embedding distances of at least $\delta=1$:

$$\mathcal{L}_{\text{mod}} = \sum_{i,j} A_{ij} |u_i - u_j|^2 + \sum_{i,j} (1-A_{ij}) \cdot \text{ReLU}(\delta - |u_i - u_j|)^2$$

Where $u$ = module_emb (atom module embeddings, Na×Da).

---

## 6. Langevin Oscillator (Inherited from 3.6)

### 6.1 Physics Background

The Langevin equation describes the motion of particles under potential field forces and coupling forces in a thermal noise environment. V4 models each knowledge atom as a **coupled phase oscillator** (similar to the Kuramoto model), where atom amplitude $r$ and phase $\theta$ converge to a low free energy state over K steps of Langevin iteration, achieving cross-domain concept alignment and phase synchronization in the encoder.

### 6.2 Complex Amplitude Initialization

The initial amplitude and phase for each atom $n$ are determined by the GMM mean, positional encoding, and module embedding:

**Amplitude Initialization**:
$$r_{0,n} = |W_r(\mu_n)|_2 + \epsilon$$
Where $W_r: \mathbb{R}^{D_a} \to \mathbb{R}^{D_a}$ is a linear projection, taking the L2 norm to obtain the amplitude. $r_0$ simultaneously serves as the steady-state target for $r$ (Hooke's law equilibrium position).

**Phase Initialization**:

- Positional encoding contribution: $\theta_{\text{pe}} = \text{atan2}(\overline{\sin\text{PE}}, \overline{\cos\text{PE}})$ (average phase of positional encoding)
- Module embedding contribution: $\theta_u = W_\theta(\text{cat}(\mu_n, u_n))$ (via learnable MLP)
- Initial phase: $\theta_n = \theta_{\text{pe},n} + \theta_{u,n}$

**Natural Frequency**:
$$\omega_n = W_\omega(\mu_n), \quad \text{where } W_\omega: \mathbb{R}^{D_a} \to \mathbb{R}$$
Alternatively derived from GMM variance: $\omega_n = \sqrt{\bar{\sigma}_n^2 + \epsilon}$.

### 6.3 Free Energy Function

System total free energy:

$$F = \underbrace{\frac{1}{2}k\sum_n (r_n - r_{0,n})^2}_{F_1:\text{Hooke's Potential}} - \underbrace{\sum_{i,j} J_{ij} r_i r_j \cos(\theta_i - \theta_j - \phi_{ij})}_{F_2:\text{Coupling Potential}} - \underbrace{T \cdot S_{\text{phase}}}_{F_3:\text{Entropy Term}}$$

- $k=1$: Hooke's constant, pulling amplitude back to equilibrium position $r_0$
- $J_{ij} = A_{ij}$ (association matrix, after top-k sparsification): Coupling strength
- $\phi_{ij}$: Phase offset (logic domain $\phi=0$, poetic domain $\phi = \theta_i - \theta_j$ for Hebbian learning)
- $S_{\text{phase}} = -\sum_n p_n \log p_n$, $p_n = \text{softmax}(\cos(\theta_n - \bar{\theta}))$: Entropy of the phase distribution

### 6.4 K-Step Langevin Iteration

For each batch sample (independent loop over batch dimension), K steps of iteration are executed:

```python
for step in range(K):
    # Phase update
    r_i, r_j = r.unsqueeze(1), r.unsqueeze(0)      # (N,N)
    th_i, th_j = theta.unsqueeze(1), theta.unsqueeze(0)
    sin_dth = sin(th_j - th_i - phi)               # Phase difference drive term
    coupling = (J * r_j * sin_dth).sum(dim=1)      # (N,) coupling from neighbors

    xi_theta = randn(N) * sqrt(2 * T_enc * dt)     # Thermal noise
    theta = theta + dt * (omega + coupling) + xi_theta

    # Amplitude update (along free energy gradient descent)
    F = free_energy(r, r0, theta, J, phi, T_enc)
    if requires_grad:
        grad_r = torch.autograd.grad(F, r, create_graph=training)
    else:
        grad_r = numerical_grad(F, r)              # Non-gradient fallback
    xi_r = randn(N, Da) * sqrt(2 * T_enc * dt)
    r = r - dt * grad_r + xi_r
    r = r.clamp(min=0.01)                          # Amplitude must be positive
```

Parameters:

- $K$: Number of iteration steps (Phase A=0/off, Phase B=2, Phase C=5)
- $dt = 0.33$: Time step size
- $T_{\text{enc}} = 0.05$: Encoding temperature (thermal noise strength)
- Top-k sparsification: Each atom retains only the top-k=16 strongest coupling neighbors

**Output**:

- $r \in \mathbb{R}^{B \times N_a}$: Converged amplitude
- $\theta \in \mathbb{R}^{B \times N_a}$: Converged phase
- $Z_{\text{phys}}$: Complex amplitude embedding $z = r\cos\theta \oplus r\sin\theta$ projected to D dimensions via MLP
- $F, dF$: Free energy scalar and its change (used for Phase C reward)

---

## 7. Dual Memory System (Inherited from 3.6)

### 7.1 Episodic Memory (EpisodicMemory)

**Ring KV Buffer** with capacity of 4096 slots, independent per domain (Logic/Poetic):

| Buffer       | Shape       | Description            |
| ------------ | ----------- | ---------------------- |
| slot_k       | (4096, D/2) | Key vectors            |
| slot_v       | (4096, D)   | Value vectors          |
| slot_t       | (4096,)     | Write timestamp        |
| slot_domain  | (4096,)     | Domain tag (0=L, 1=P)  |
| slot_ptr     | scalar      | Ring write pointer     |

**Retrieval** (attention-based):
$$\text{score} = \frac{Q K_{\text{slot}}^T}{\sqrt{d_k}} \cdot \mathbb{1}_{\text{domain}} - 10^9 \cdot (1 - \mathbb{1}_{\text{domain}})$$
$$\text{decay} = \exp(-\lambda_t \cdot (t - t_{\text{slot}}))$$
$$A = \text{softmax}(\text{score}) \odot \text{decay}$$
$$\text{mem} = A \cdot V_{\text{slot}}$$

Time decay $\lambda_t = 0.01$ causes old memories to naturally fade; domain mask ensures the logic domain only retrieves logic memories.

**Writing**: Each step projects the current x's KV, reshapes into tokens, and writes them to the ring buffer; the pointer advances modulo.

### 7.2 Structural Memory (StructuralMemory)

**Skill Key-Value Table** with 1024 learnable skill slots:

- skill_key: Embedding(1024, Da) — Skill keys
- skill_val: Embedding(1024, 2Da) — Skill values
- logit_bias: (1024,) — Bias

Retrieval is similar to attention: $A = \text{softmax}(q K^T/\sqrt{D_a} + b)$, output $A \cdot V$. Structural memory is learned via BP and represents reusable "skills".

---

## 8. Meta-Cognitive Gating (Inherited from 3.6)

### 8.1 TriPercept Three-Perceptual-Voter Voting

Three independent MLPs (same structure, shared parameters not shared) classify the fused domain features, voting to determine the weights of the Logic/Poetic domains:

```python
x = cat(z_L.mean(dim=1), z_P.mean(dim=1), task_emb)  # (B, 3D)
outs = [softmax(MLP_i(x), dim=-1) for i in range(3)]   # each: (B,2)
alphas_L = stack([out[:,0] for out in outs])           # (3,B)
alpha_L = median(alphas_L, dim=0)[0]                   # (B,) median vote
alpha_P = 1 - alpha_L
u_cog = var(alphas_L, dim=0) / 0.25                   # (B,) normalized variance
```

- **Median voting** is more robust than mean (resistant to outliers)
- **$u_{\text{cog}}$** (cognitive uncertainty) = variance of the three perceptual outputs / 0.25 (0.25 is the maximum variance of a Bernoulli distribution)
  - $u_{\text{cog}} \approx 0$: Three perceptual voters agree → high confidence
  - $u_{\text{cog}} \approx 1$: Three perceptual voters completely disagree → uncertain

### 8.2 Temperature-Uncertainty Coupling

$$T = T_0 \cdot (1 + \kappa \cdot u_{\text{cog}})$$
$$T = \text{clip}(T, T_{\min}, T_{\max})$$

| Phase | T0  | κ   | T Range     | Meaning                                        |
| ----- | --- | --- | ----------- | ---------------------------------------------- |
| A     | 1.0 | -   | 1.0         | High-temperature exploration                    |
| B     | 0.7 | 0.5 | 0.7\~1.05   | Medium temperature                              |
| C     | 0.2 | 2.0 | 0.2\~0.6    | Low-temperature convergence, but heats up when uncertain |

Safety mode: When $u_{\text{cog}} > 0.7$, force $\alpha_L = \alpha_P = 0.5$ (equal mixing), preventing over-reliance on one domain under high uncertainty.

### 8.3 SafetyFilter

64 danger anchor vectors: cosine similarity with input pattern → minimum similarity → safety_score. Dimensions with score < 0.3 are replaced with the pattern mean (deactivating dangerous patterns).

### 8.4 Fusion Output

$$Z_{\text{IB}} = \alpha_L \cdot [(1-\lambda_p) z_L' + \lambda_p Z_{\text{phys},L}] + \alpha_P \cdot [(1-\lambda_p) z_P' + \lambda_p Z_{\text{phys},P}]$$
$$Z_{\text{cog}} = \text{MLP}_{\text{fuse}}(\text{cat}(Z_{\text{IB}}, \text{epi\_mem}, Z_{\text{safe}}, \text{task\_exp}))$$

$\lambda_p = \lambda_{\text{phys}}$ controls the fusion ratio between IB representation and physics representation (Phase C = 0.2).

---

## 9. V4 Encoder (EncoderV4)

### 9.1 Structure

```
Input: input_ids (B,S)
  → token_emb: Embedding(V, D)              (B,S,D)
  → pos_emb:   Embedding(S_max, D)          (B,S,D)
  → x = LN(drop(tok + pos))                (B,S,D)
  → PGTA Block × 2:
      Pre-LN → PGTA self-attention → residual
      → Pre-LN → FFN(Linear→GELU→Linear→Dropout) → residual
  → Information Bottleneck (per-domain):
      z_L' = IB_L(z_IB_L): mu_L, logvar_L → sample → dec → out + KL_L
      z_P' = IB_P(z_IB_P): mu_P, logvar_P → sample → dec → out + KL_P
  → Auxiliary losses:
      loss_D (BCE D_score vs 0.5)
      loss_aff (λ·std(aff_score))
      loss_H (λ_H·ReLU(H_enc - H_target))
Output: {x_enc, z_IB_L', z_IB_P', loss_IB, loss_D, loss_aff, loss_H, attn_*}
```

### 9.2 Information Bottleneck (IB)

One VAE per domain:

$$\mu = W_{\mu}(x), \quad \log\sigma^2 = W_{\text{lv}}(x)$$
$$z = \mu + \epsilon \cdot \sigma, \quad \epsilon \sim \mathcal{N}(0, I) \quad \text{(reparameterization)}$$
$$\hat{x} = \text{MLP}_{\text{dec}}(z)$$
$$\mathcal{L}_{\text{KL}} = -\frac{1}{2}\sum_j (1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$$

The IB forces the encoder to compress input into compact latent representations, filtering noise. KL weight λ_IB gradually increases during Phase B/C.

### 9.3 Notes

Encoder PGTA temperature uses its own learnable log_temp (not overridden by meta-cognition), because the encoder runs before the meta-cognitive gate. The encoder entropy loss target is $H^* = 0.3 \cdot \log S$.

---

## 10. V4 Decoder (GLUBlockV4)

### 10.1 Complete Layer Structure

Each decoder layer $l \in \{0,...,n_{\text{layers}}-1\}$ contains:

```
Input: x (B,S,D), Z_cog (B,S,D), T (scalar), θ_tokens (B,S), s_TSR, s_PSR
  → 1. Pre-LN → PGTA self-attention (independent PGTA, temperature T)
     → residual: x = x + attn_out
  → 2. Pre-LN → GLU
      g = W_g(x_norm),  u, v = chunk(g, 2, dim=-1)  # each (B,S,d_ff)
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
  → 5. Z_cog layer-level injection + physics gating
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
  → 6. GLU output
      out = (σ(u) * v) @ W_proj.T
      x = x + out  (residual)
Output: x (B,S,D), {TSR:M_TSR, PSR:M_PSR}
```

### 10.2 TSR/PSR State Routing

TSR is a GRU-style gated state update, where Z_cog modulates the gate value γ — cognitive state influences each layer's internal memory. PSR uses phase difference Δθ as the gating signal: small phase difference (physics synchronization) retains physics state, while large phase difference resets it. Both states are passed cross-layer.

### 10.3 Physics Gating α

$\alpha_t$ is the data-driven gate (sigmoid output), while $\alpha_p$ is the physics-driven gate (based on consistency between token phase and layer-preferred phase $\theta_l$). $\lambda_T$ controls the physics gating weight (gradually increasing to 0.3 in Phase C).

The final $u$ is modulated by scale/bias and then passed through a GLU gate (sigmoid-gated): $\text{GLU}(u,v) = \sigma(u) \odot v$, which is more flexible than standard ReLU (gating information flow).

---

## 11. AetherMind4 Main Model

### 11.1 Initialization Essentials

```python
def __init__(config):
    # Auto-match d_anchor to d_atom if 0
    if config.d_anchor <= 0:
        config.d_anchor = config.d_atom

    # Core modules (inherited from 3.6)
    encoder = EncoderV4(config)
    dual_domain = DualDomainSystem(config)
    physics = LangevinOscillator(config)
    memory = DualMemorySystem(config)
    metacog = MetaCognitiveGate(config)

    # V4: n_layers independent PGTA + GLUBlockV4
    decoder_attns = ModuleList([PheromoneThermoAttention(...) for _ in range(n_layers)])
    decoder_layers = ModuleList([GLUBlockV4(config, i, decoder_attns[i]) for i in range(n_layers)])

    # Output heads
    pointer = MLP(D, D, 1)         # Copy gate
    counter_rnn = GRUCell(Dc, Dc)  # Counter
    counter_proj = Linear(Dc, 1)
    lm_head = Linear(D, V, bias=False)

    # V4: Evolution controller
    evolver = EvolvableWeightSystem(config)
    for attn in encoder.attn_layers + decoder_attns:
        evolver.register_attention(attn)

    # Weight tying: lm_head shares weights with token_emb
    lm_head.weight = encoder.token_emb.weight
```

### 11.2 Forward 12-Step Complete Data Flow

```
Step 1: token_emb = encoder.token_emb(input_ids)       # (B,S,D)
Step 2: domain_out = dual_domain(token_emb)             # → z_L/P, atom_w, D_score, etc.
Step 3: enc_out = encoder(input_ids, domain_out)        # → x_enc, z_IB_L', z_IB_P', aux losses
Step 4: phys_out = physics({**domain_out, "z_IB_L/P": enc_out["z_IB_L/P"]}, pos_emb)
       # → Z_phys_L/P, r, θ, F, dF for both domains
Step 5: mem_out = memory(x_enc, domain_out, t)
       epi_mem = mem_out["epi_L"] + mem_out["epi_P"]   # (B,S,D)
Step 6: Z_phys_L_tokens = einsum("bsn,bnd->bsd", atom_w_L, phys_out["L"]["Z_phys"])
       Z_phys_P_tokens = einsum("bsn,bnd->bsd", atom_w_P, phys_out["P"]["Z_phys"])
       # Physics features mapped from atom space to token space, shape alignment
Step 7: meta_out = metacog(z_IB_L', z_IB_P', Z_phys_tokens, epi_mem, task_id)
       # → Z_cog, T, u_cog, α, safety_score
Step 8: θ_L_tokens = (atom_w_L * th_L.unsqueeze(1)).sum(-1)  # (B,S)
       θ_P_tokens = (atom_w_P * th_P.unsqueeze(1)).sum(-1)
       θ_tokens = θ_L_tokens + θ_P_tokens
Step 9: T_val = meta_out["T"].mean().item()
       for attn in decoder_attns: attn.set_temperature(T_val)
       h = x_enc
       for layer in decoder_layers:
           h, state = layer(h, Z_cog, meta_out["T"], θ_tokens, s_TSR, s_PSR)
           s_TSR, s_PSR = state["TSR"], state["PSR"]
       h = final_norm(h)
Step 10: CounterSlot: GRUCell per token position (noise input, maintains counting ability)
       h = h + counter_proj(cnt_stack).tanh() * 0.1
Step 11: p_copy = sigmoid(pointer(h)).squeeze(-1)       # (B,S)
       logits = h @ lm_head.weight.T * D^{-0.5}        # (B,S,V) weight tying
       probs = safe_softmax(logits, T=T_val)
       # Copy mechanism: mix softmax distribution and input sequence cumulative distribution
       copy_prob = cumsum(one_hot(input_ids)) / arange(1,S+1)
       probs = (1-p_copy_gate)*probs + p_copy_gate*copy_prob
Step 12: Loss assembly:
       L_LM = CE(shift_logits, shift_labels, ignore_index=pad)
       L_total = L_LM + λ_IB·L_IB + λ_D·L_D + λ_aff·L_aff + λ_H·L_H
              + λ_mod·L_mod + λ_anc·L_anc + λ_phys·L_phys
```

### 11.3 Key Shape Alignment Handling

When the atom-space $Z_{\text{phys}}$ is mapped to token space, shapes may not match; defensive handling is implemented in the code:

- S-dimension padding (when physics output length ≠ sequence length)
- D-dimension: When $Z_{\text{phys}}$'s last dimension ≠ D, a Linear projection is dynamically created (note: this dynamic creation is not recommended for formal training; correct dimensions are guaranteed via phys_mlp)

When θ is mapped to token level: θ atoms have shape $(B, N_a)$, weighted sum via `(atom_w * θ_atoms.unsqueeze(1)).sum(-1)` yields $(B,S)$.

### 11.4 Evolution Step Interface

```python
def evolution_step(step, phase="A", loss_val=None):
    F_val = self._last_F if hasattr(self, '_last_F') else None
    evolver(step, free_energy=F_val, phase=phase, loss_val=loss_val)
```

Note: In the current implementation, `_last_F` needs to be saved from the forward pass (the forward return dict must contain the "F" key); the trainer calls evolution_step after each step.

### 11.5 Generative Inference

The generate() method: Phase C mode autoregressive generation with top-k + top-p sampling:

1. Take the last max_seq_len tokens as current input
2. Forward pass to get logits at the last position
3. Temperature comes from meta_out (can be externally overridden)
4. Top-k=50 filtering → top-p=0.9 nucleus sampling → multinomial sampling
5. Append generated token, stop at eos
6. During generation, PGTA pheromones continue to deposit (but without external reward, deposition is uniform by attention usage)

---

## 12. Training System

### 12.1 Three-Phase Training Recipe

| Dimension                | Phase A (0\~50k steps) | Phase B (50k\~150k steps) | Phase C (150k\~250k steps) |
| ------------------------ | ---------------------- | ------------------------- | -------------------------- |
| **Objective**            | Structural pre-training | Mixed learning              | Evolutionary convergence + consolidation |
| **W Learning**           | Full BP                | Elastic BP                 | Primarily evolution (BP weakened) |
| **τ Deposition η**        | 0.0 (off)              | 0.05×progress (gradual)    | 0.05 (full)                |
| **τ Evaporation ρ**        | -                      | 0.05                       | 0.03 (protect good paths)  |
| **langevin_K**            | 0 (off)                | 2                          | 5                          |
| **λ_IB**                 | 0                      | 0.5×progress               | 1.0                        |
| **λ_D**                  | 0                      | 0.3×progress               | 0.5                        |
| **λ_phys**               | 0                      | 0.1×progress               | 0.2                        |
| **λ_PSR**                | 0                      | 0.1×progress               | 0.3×progress               |
| **λ_T**                  | 0                      | 0.1×progress               | 0.3×progress               |
| **λ_align**              | 0                      | 0.3×progress               | 0.5                        |
| **T0**                   | 1.0                    | 0.7                        | 0.2                        |
| **κ**                    | -                      | 0.5                        | 2.0                        |
| **init_temperature**     | 1.0                    | 0.7+0.3p                   | 0.5→0.2 annealing          |
| **Credit Signal r**        | -                      | -loss (coarse)             | -dF (dense)                |
| **Consolidation**        | Off                    | Off                        | Every 100 steps            |
| **Analogy**              | Infant learning language | Teenager learning skills    | Expert consolidating experience |

progress ∈ [0,1] is the linear progress within each phase.

### 12.2 Optimizer and Mixed Precision

```python
optimizer = AdamW(params, lr=3e-4, betas=(0.9,0.95), weight_decay=0.01)
scheduler = SequentialLR(
    [LinearLR(warmup=1000, 1e-4→1.0),
     CosineAnnealingLR(T_max=249000, eta_min=1e-5)],
    milestones=[1000]
)
scaler = GradScaler("cuda")  # bf16 mixed precision
```

Training tips:

- CUDA: cudnn.benchmark=True, TF32=True
- DataParallel multi-GPU support
- pin_memory + non_blocking=True
- grad_accumulation=8 (on RTX 3050 4GB, batch=1 equivalent to batch=8)
- grad_clip_max_norm=1.0

### 12.3 Training Loop Pseudocode

```python
for each epoch:
    for batch in train_loader:
        steps += 1
        set_phase()  # Switch A/B/C based on steps, update all λ parameters

        # Forward+Backward
        with autocast(cuda, dtype=bfloat16):
            out = model(input_ids, labels, task_id, t=steps, phase=phase)
            loss = out["loss"] / grad_accum
        scaler.scale(loss).backward()

        # Update after gradient accumulation
        if steps % grad_accum == 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        # V4 core: Pheromone evolution step
        loss_val = out["loss"].detach()
        if hasattr(model, 'module'):
            model.module.evolution_step(steps, phase, loss_val)
        else:
            model.evolution_step(steps, phase, loss_val)

        # Logging/evaluation/save
        if steps % log_interval == 0: print_progress()
        if steps % eval_interval == 0: evaluate()
        if steps % save_interval == 0: save_checkpoint()
```

### 12.4 Data Pipeline

`StreamingTextDataset` streams JSONL files from the `F:/数据集` directory:

- Maximum 5000 lines per file (prevents OOM)
- No file size upper limit (12GB MOSS files read normally)
- Recursively extracts string fields from nested JSON
- HF_HUB_OFFLINE=1 (offline mode, no HuggingFace access)
- BPE tokenizer trained/loaded offline
- Train/eval split at 98%/2%

---

## 13. Complete Hyperparameter Reference

### 13.1 Model Structure Parameters

| Parameter                 | Default | RTX 3050 Config | Description          |
| ------------------------- | ------- | --------------- | -------------------- |
| vocab_size                | 50000   | 50000           | Vocabulary size      |
| d_model                   | 512     | 384             | Model hidden dimension |
| d_ff                      | 2048    | 1536            | Feed-forward network dimension (4×D) |
| n_layers                  | 8       | 6               | Number of decoder layers |
| n_heads                   | 8       | 6               | Number of attention heads |
| max_seq_len               | 1024    | 512             | Maximum sequence length |
| dropout                   | 0.1     | 0.1             | Dropout rate          |
| n_atoms                   | 1024    | 1024            | Number of GMM atoms  |
| d_atom                    | 64      | 64              | Atom dimension        |
| n_gmm_components          | 3       | 3               | Number of GMM components per atom |
| n_anchor                  | 512     | 512             | Number of anchors     |
| d_state                   | 128     | 128             | TSR/PSR state dimension |
| d_counter                 | 16      | 16              | Counter GRU dimension |
| pad/bos/eos               | 0/1/2   | 0/1/2           | Special token IDs      |

### 13.2 PGTA Parameters

| Parameter                      | Value                           | Description        |
| ------------------------------ | ------------------------------- | ------------------ |
| pheromone_rho                  | 0.05 (C: 0.03)                  | Evaporation rate    |
| pheromone_beta                 | 1.0                             | Pheromone sensitivity |
| pheromone_deposit              | 0→0.05                          | Deposition strength (by phase) |
| pheromone_tau_min              | 0.01                            | τ lower bound       |
| pheromone_tau_max              | 5.0                             | τ upper bound       |
| pheromone_credit_mode          | "soft_center"→"free_energy"   | Credit mode (B→C)  |
| pheromone_whiten              | True                            | Energy whitening    |
| init_temperature               | 1.0→0.2                         | Initial temperature (annealing) |
| target_entropy_ratio           | 0.3                             | Target entropy ratio |
| lambda_H                       | 0.05                            | Entropy regularization weight |

### 13.3 LTP Consolidation Parameters

| Parameter                     | Value | Description          |
| ----------------------------- | ----- | -------------------- |
| consolidate_threshold         | 1.5   | Consolidation trigger threshold |
| consolidate_lam               | 0.1   | Consolidation write strength λ |
| consolidate_gamma             | 0.5   | Post-consolidation τ decay γ |
| consolidate_interval          | 100   | Consolidation interval in steps |
| max_consolidations            | 4096  | Consolidation budget upper limit |

### 13.4 Training Parameters

| Parameter                     | Value          | Description       |
| ----------------------------- | -------------- | ----------------- |
| batch_size                    | 8 (3050: 1)     | Batch size        |
| gradient_accumulation         | 4 (3050: 8)     | Gradient accumulation |
| learning_rate                 | 3e-4           | Peak learning rate |
| min_lr                        | 1e-5           | Minimum learning rate |
| weight_decay                  | 0.01           | Weight decay      |
| warmup_steps                  | 1000           | LR warmup steps   |
| max_grad_norm                 | 1.0            | Gradient clipping |
| phase_A_steps                 | 50,000         | Structural pre-training |
| phase_B_steps                 | 100,000        | Mixed learning    |
| phase_C_steps                 | 100,000        | Evolutionary convergence |
| total_steps                   | 250,000        | Total steps       |
| log_interval                  | 10             | Logging interval  |
| eval_interval                 | 2000           | Evaluation interval |
| save_interval                 | 5000           | Save interval     |

---

## 14. VRAM and Compute Budget

### 14.1 VRAM Breakdown

Using d=384, h=6, n_layers=6, S=512, B=1, bf16 as example:

| Component                   | Calculation                    | Size          |
| --------------------------- | ------------------------------ | ------------- |
| Model parameters             | ~56M × 2B (bf16)               | ~112MB        |
| Gradients (fp32)             | ~56M × 4B                      | ~224MB        |
| AdamW states (m+v, fp32)     | 2×56M×4B                       | ~448MB        |
| Activations (bf16)           | ~B×S×D×n_layers×3.5           | ~80MB         |
| PGTA τ (8 layers×h×S×S×4B)   | 8×6×512×512×4B                | ~48MB         |
| PGTA C (same as τ)           | Same as above                   | ~48MB         |
| Episodic memory buffer (2 domains) | 2×4096×(D/2+D)×4B     | ~14MB         |
| CUDA context + fragmentation  | -                              | ~200MB        |
| **Total**                     |                                | **~1.1GB**    |

Full version d=512, n=8, S=1024:

- τ/C: (2+8)×8×1024×1024×4B×2 = ~640MB
- Total VRAM ~3.5GB (barely runnable on 4GB, requires smaller batch)

### 14.2 Computational Complexity

| Operation            | Complexity                          | Notes                               |
| -------------------- | ----------------------------------- | ----------------------------------- |
| QKV projection        | $O(BSD^2)$                        | Standard Transformer                 |
| Attention matrix multiply | $O(BS^2D)$                    | Standard self-attention               |
| Pheromone update       | $O(hS^2)$                         | Per step: only add+mul+clamp, extremely fast |
| Consolidation         | $O(hS^2)$                         | Once every 100 steps                 |
| Langevin iteration     | $O(K \cdot N_a^2 \cdot B)$        | K=5 steps, Na=1024, B=1 is acceptable |
| Total FLOPs per step  | ~$2BSD^2 + 2BS^2D$                | Close to standard Transformer         |

**Key point**: Pheromone update is a pure buffer operation (no matrix multiply, no gradients), with <5% additional overhead.

---

## 15. Complete Loss Function Derivation

Total loss:

$$\mathcal{L} = \mathcal{L}_{\text{LM}} + \lambda_{\text{IB}}\mathcal{L}_{\text{IB}} + \lambda_D\mathcal{L}_D + \lambda_{\text{aff}}\mathcal{L}_{\text{aff}} + \lambda_H\mathcal{L}_H + \lambda_{\text{mod}}\mathcal{L}_{\text{mod}} + \lambda_{\text{anc}}\mathcal{L}_{\text{anc}} + \lambda_{\text{phys}}\mathcal{L}_{\text{phys}}$$

| Loss                          | Formula                                                                   | Purpose                    | λ (Phase C) |
| ----------------------------- | ------------------------------------------------------------------------- | -------------------------- | ----------- |
| $\mathcal{L}_{\text{LM}}$     | $\text{CE}(\text{logits}[:,:-1,:], \text{labels}[:,1:])$                | Core language modeling      | 1.0 (fixed) |
| $\mathcal{L}_{\text{IB}}$     | $\sum \text{KL}[\mathcal{N}(\mu,\sigma^2)\|\mathcal{N}(0,I)]$           | Information bottleneck compression | 1.0       |
| $\mathcal{L}_D$               | $\text{BCE}(\text{D\_score}, 0.5)$                                       | Prevent domain collapse (GAN-style) | 0.5       |
| $\mathcal{L}_{\text{aff}}$    | $\text{std}(\text{affect\_score})$                                      | Emotional diversity         | 0.3       |
| $\mathcal{L}_H$               | $\text{ReLU}(H - H^*)$                                                  | Attention entropy anti-collapse | 0.05      |
| $\mathcal{L}_{\text{mod}}$    | Modularity constraint (see 5.6)                                         | Association matrix topology | λ_module (0→0) |
| $\mathcal{L}_{\text{anc}}$    | $\text{MSE}(z, \text{anchor})_L + \text{MSE}(z, \text{anchor})_P$        | Anchor uniform filling     | 0.5       |
| $\mathcal{L}_{\text{phys}}$   | $F_L^2 + F_P^2$                                                          | Physics free energy minimization | 0.2       |

Note: In code, the BCE loss is wrapped with `torch.amp.autocast("cuda", enabled=False)` and `.float()` to prevent numerical instability of BCE under bf16 precision (the "unsafe to autocast" error).

---

## 16. Comparison with V3.6 / Mainstream Models

| Dimension       | Standard Transformer | AetherMind 3.6   | AetherMind V4.0                              |
| --------------- | --------------------| ----------------- | --------------------------------------------- |
| Attention Mechanism | Scaled Dot-Product | ShapleyAttention | **PGTA (Boltzmann + Pheromone)**             |
| Weight System    | Single BP weights   | Single BP weights  | **Dual weights W/τ+C (LTP consolidation)**    |
| Temporal Scale   | Single-step forward  | Single-step + physics K-step iteration | **Three-level (Fast T / Medium τ / Slow C)** |
| Temperature Control | None / constant  | Meta-cognitive T   | **Learnable logT + meta-cognitive override + annealing** |
| Entropy Regularization | None             | Shapley value constraint | **Boltzmann entropy + free energy** |
| Physics Layer    | None                | Langevin oscillator | Langevin + free energy reward signal         |
| Memory System    | KV Cache            | Episodic + structural dual memory | Dual memory + **in-weight C (procedural memory)** |
| Training Phases  | Single phase        | Single phase (progressive λ) | **Three phases A/B/C**                       |
| Deployment Behavior | Static inference  | Static inference  | **Online agent continuous evolution**        |
| Credit Signal    | Gradient loss       | Gradient loss      | **soft_center → free_energy (intrinsic motivation)** |
| Path Memory      | None (recomputed each step) | None         | **τ cross-step persistent + C permanent consolidation** |
| Parameter Scale Ratio | 1.0×           | ~1.1× (+ physics layer) | **~1.15× (+PGTA buffers, <5% parameter increase)** |
| VRAM Overhead Ratio | 1.0×            | ~1.3×             | **~1.6× (τ/C buffers)**                       |
| Neuroscience Analogy | -               | Predictive coding + free energy | **Fast/slow weights + LTP + STDP**           |
| Swarm Intelligence | -                  | -                 | **Ant colony stigmergy paths**               |

---

## 17. Known Limitations and Future Directions

### 17.1 Current Limitations

1. **Pheromone O(S²) Storage**: At long sequence length S=4096, τ size reaches 8×4096×4096×4B = 512MB per layer, requiring low-rank decomposition or sparse attention. The design document reserves the DeepSeek CSA/HCA sparse direction.
2. **Credit Signal Density**: Phase C's free_energy reward, while denser than 0/1 correct/incorrect, is still not as dense as "mutual information / causal influence." Experiments in `pgtt_self_evolution.py` show that coarse credit signals can lock into degenerate solutions.
3. **Consolidation-BP Coordination**: Currently, C is a purely additive bias, without considering coordinated gradients between C and W. In the future, C might obtain gradients through Straight-Through Estimators (STE).
4. **Dynamic Linear Projection**: In the code, shape alignment dynamically creates Linear layers (`aethermind4.py#L361`); this should be removed or replaced with pre-defined projections for formal training.
5. **F_prev Tracking**: The current `_last_F` save mechanism needs improvement to ensure dF computation uses the correct previous-step free energy.

### 17.2 Future Directions

1. **τ Low-Rank Decomposition**: $\tau = UV^T$, U,V ∈ ℝ^{S×r}, reducing storage from O(S²) to O(Sr)
2. **Consolidation Hash Index**: Borrowing from DeepSeek-V4 hash routing, high-frequency consolidated paths can be directly looked up in a table
3. **MoE Integration**: τ serves as expert routing prior, consolidation = writing high-frequency expert selections into a hash table
4. **SNN Spiking Version**: E = spike potential, τ = synaptic efficacy, consolidation = STDP
5. **4-bit Quantization**: Borrowing from Kimi K3 QAT, making τ/C quantizable for storage
6. **Phase D Deployment Evolution**: Complete inference-time online evolution + periodic consolidation + LoRA export

---

## 18. File Structure and Code Index

```
d:\AetherMind-Nano3\
├── configs/
│   ├── aethermind36_config.py                  # 3.6 config (retained)
│   └── aethermind4_config.py                   # V4 config [NEW] L7-L170
├── src/
│   ├── model/
│   │   ├── aethermind36.py                     # 3.6 main model (retained)
│   │   ├── aethermind4.py                      # V4 main model [NEW] L263-515
│   │   │   ├── EncoderV4                       # [NEW] L34-138
│   │   │   └── GLUBlockV4                      # [NEW] L144-257
│   │   ├── attention/
│   │   │   └── pheromone_thermo.py             # PGTA [NEW] L20-218
│   │   ├── evolution/
│   │   │   └── evolvable_weight.py             # Evolution controller [NEW] L20-123
│   │   ├── domain/
│   │   │   └── dual_domain.py                  # GMM dual-domain (3.6) L9-196
│   │   ├── physics/
│   │   │   └── langevin.py                     # Langevin (3.6) L9-220
│   │   ├── memory/
│   │   │   └── dual_memory.py                  # Dual memory (3.6) L8-92
│   │   ├── metacog/
│   │   │   └── meta_gate.py                    # Meta-cognitive gate (3.6) L49-99
│   │   └── decoder/
│   │       └── glu_decoder.py                  # 3.6 GLU (V4 uses GLUBlockV4)
│   ├── training/
│   │   ├── train.py                            # 3.6 trainer
│   │   └── train_v4.py                         # V4 trainer [NEW]
│   ├── data/
│   │   └── dataset.py                          # Streaming dataset (3.6)
│   └── utils/
│       └── ops.py                              # Utility functions (3.6)
├── docs/
│   ├── AetherMind36_Architecture_Report.md     # 3.6 documentation
│   └── AetherMind4_Architecture_Report.md      # This document [NEW]
├── test_v4_smoke.py                            # V4 smoke test
├── 1212/                                       # V4 design inspiration sources
│   ├── Pheromone_Path_Network_Thermodynamic_Attention_Transformer_Design.md
│   ├── Evolvable_Weight_Evolution_Model_Architecture_Design.md
│   ├── Physics_Driven_Attention_Mechanism_Proposal.md
│   ├── pheromone_thermo_transformer.py         # PGTT reference implementation
│   ├── pgtt_self_evolution.py                  # Self-evolution experiment
│   ├── evolvable_weight.py                     # LTP consolidation MVP
│   └── thermo_info_attention.py               # Thermodynamic attention reference
└── train_gpu.cmd / train_v4_smoke.cmd          # Training scripts
```

---

## 19. Verification Status

### 19.1 Smoke Test Passed

```
Model params: 9,223,140 (d=128, n_layers=2, n_heads=4, batch=2, seq=64)
Phase A: loss=10.9589  Backward OK
Phase B: loss=11.0780  Evolution step OK
Phase C: loss=11.3571  Generate OK: torch.Size([1, 15])
ALL PASSED
```

### 19.2 Verified Items

- [x] PGTA forward/backward propagation normal
- [x] Pheromone stepping (evaporation + deposition) normal
- [x] LTP consolidation mechanism (code path correct, awaiting long-training validation)
- [x] Three-phase configuration switching normal
- [x] Weight tying (lm_head ↔ token_emb) normal
- [x] Autoregressive generate normal
- [x] Dual-domain GMM activation normal
- [x] Langevin oscillator K-step iteration normal
- [x] Meta-cognitive gating temperature computation normal
- [x] Dual memory retrieval/storage normal
- [ ] Full three-phase long training (pending execution)
- [ ] Free energy reward signal dF computation accuracy (pending verification)
- [ ] Post-consolidation effect quantification (pending Phase C running 100+ steps)
- [ ] 4GB VRAM d=384 config training stability