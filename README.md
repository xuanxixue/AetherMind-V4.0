# AetherMind V4.0 — Xirang

### Dual-Weight Evolutionary Cognitive System

<p align="center">
  <img src="https://img.shields.io/badge/Version-4.0.0-blue" alt="Version" />
  <img src="https://img.shields.io/badge/Python-3.9%2B-green" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange" alt="PyTorch" />
  <img src="https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey" alt="License" />
  <img src="https://img.shields.io/badge/Status-Coming%20Soon-yellow" alt="Status" />
</p>

> **Codename: Xirang (息壤)** — In Chinese mythology, Xirang is a magical soil that grows on its own. Likewise, AetherMind V4 grows its own cognitive pathways through online pheromone evolution and LTP consolidation, going beyond frozen-weight inference.

---

## What is AetherMind?

AetherMind is a novel neural language model architecture that draws inspiration from **biological neural systems**, **statistical physics**, and **swarm intelligence** to overcome the fundamental limitation of mainstream Transformers: **weights are frozen after deployment**.

V4.0 introduces three core innovations:

- **PGTA (Pheromone-Guided Thermodynamic Attention)** — Reformulates attention as a Boltzmann distribution over energy, with ant-colony-style pheromone paths that persist and evolve across timesteps
- **Evolvable Dual-Weight System (W/τ/C)** — A three-level memory hierarchy: fast gradient weights, medium-term pheromone traces, and long-term LTP consolidation
- **Three-Phase Training (A → B → C)** — Progressive training from structural pre-training to online evolutionary convergence with intrinsic motivation

### The Core Proposition

> *The model from pre-training is the "body"; the online-evolving pheromone network is the "experience." Experience should not be stored in external memory but grown into the weights.*

---

## Key Features

### 🔬 PGTA — Pheromone-Guided Thermodynamic Attention

Replaces standard softmax attention with a physics-grounded three-factor attention mechanism:

$$A_{ij} \propto \exp\left(-\hat{E}_{ij}/T\right) \cdot \tau_{ij}^{\beta} \cdot \exp(C_{ij})$$

| Factor | Source | Update Method |
|--------|--------|----------------|
| **Structural** $\exp(-\hat{E}/T)$ | QK similarity | Backpropagation |
| **Path** $\tau^{\beta}$ | Pheromone buffer | Deposition + Evaporation (non-gradient) |
| **Consolidated** $\exp(C)$ | Permanent bias | LTP threshold write |

- **Energy whitening** normalizes the energy matrix via running statistics, giving temperature a consistent physical meaning
- **Learnable temperature** $T = \exp(\log T)$ enables simulated annealing: high-T exploration → low-T convergence
- **Meta-cognitive override**: a three-voter uncertainty gate dynamically adjusts $T$ per inference step

### 🧠 Evolvable Dual-Weight System

Three temporal scales operating simultaneously — the essential difference from V3.6 and all mainstream Transformers:

| Scale | Variables | Update | Frequency | Analogy |
|-------|-----------|--------|-----------|---------|
| **Fast** (per-step) | Temperature $T$, Attention $A$ | Forward computation | Per token | Neuronal firing |
| **Medium** (cross-sample) | Pheromone $\tau$ | Deposition + Evaporation | Per step | Short-term synaptic efficacy |
| **Slow** (cross-session) | Consolidation $C$, Fast weights $W$ | LTP threshold write | Every 100 steps | Long-term memory |

**LTP Consolidation** — When pheromone $\tau > \theta_c = 1.5$ consistently, the excess is permanently written into $C$, then $\tau$ decays by 50%. This mirrors biological long-term potentiation: frequently activated synapses are permanently strengthened.

### ⚡ Three-Phase Training

| Phase | Steps | Objective | Credit Signal | Analogy |
|-------|-------|-----------|---------------|---------|
| **A** | 0–50K | Structural pre-training | — | Infant learning language |
| **B** | 50K–150K | Mixed learning | $r = -\mathcal{L}$ | Teenager learning skills |
| **C** | 150K–250K | Evolutionary convergence | $r = -dF/(|dF|+1)$ | Expert consolidating experience |

Phase C uses **free energy change** as an intrinsic motivation signal — the model actively reinforces attention patterns that reduce its own prediction uncertainty, requiring no external labels.

### 🌐 Dual-Domain Knowledge Atoms

Language understanding is decomposed into two independent GMM-based domains:

- **Logic Domain (L)** — Facts, reasoning, grammar (left brain)
- **Poetic Domain (P)** — Emotion, metaphor, style (right brain)

Each domain maintains 1,024 knowledge atoms, each a 3-component Gaussian Mixture Model, forming a soft-routed Mixture of Experts with 2,048 total expert atoms.

### 🌊 Langevin Oscillator

Knowledge atoms are modeled as coupled phase oscillators (Kuramoto-style). Over $K$ steps of Langevin iteration, amplitude $r$ and phase $\theta$ converge to a low free energy state, achieving cross-domain concept alignment and phase synchronization.

---

## Architecture Overview

```
input_ids (B,S)
      │
      ▼
┌─────────────────────┐
│  Token + Position    │
│  Embedding           │
└─────────┬───────────┘
          │
          ▼
┌──────────────────────────────────────────────┐
│  Dual-Domain GMM Knowledge Atoms             │
│  ┌────────────┐        ┌────────────┐        │
│  │ Logic (L)   │        │ Poetic (P)  │        │
│  │ 1024 atoms  │        │ 1024 atoms  │        │
│  │ 3-comp GMM  │        │ 3-comp GMM  │        │
│  └──────┬─────┘        └──────┬─────┘        │
│         └────────┬────────────┘               │
└──────────────────┼───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  Encoder V4 (PGTA × 2 + Information          │
│  Bottleneck VAE per domain)                   │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  Langevin Oscillator (K steps)                │
│  → Free energy F, phase θ, amplitude r        │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  Dual Memory + Meta-Cognitive Gate            │
│  Episodic (ring KV) + Structural (skill KV)   │
│  TriPercept voting → T, α_L, α_P, u_cog      │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  Decoder: GLUBlockV4 × n_layers               │
│  PGTA + GLU FFN + TSR + PSR + Z_cog injection │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  Output Head                                  │
│  CounterSlot + Weight-tied LM + Copy Mechanism │
└──────────────────────────────────────────────┘
```

---

## Project Structure

```
AetherMind-Nano3/
├── configs/
│   ├── aethermind36_config.py          # V3.6 config (retained)
│   └── aethermind4_config.py           # V4 config
├── src/
│   ├── model/
│   │   ├── aethermind4.py              # V4 main model
│   │   │   ├── EncoderV4               # V4 encoder (PGTA + IB VAE)
│   │   │   └── GLUBlockV4              # V4 decoder block
│   │   ├── attention/
│   │   │   └── pheromone_thermo.py      # PGTA attention [V4 NEW]
│   │   ├── evolution/
│   │   │   └── evolvable_weight.py      # Evolution controller [V4 NEW]
│   │   ├── domain/
│   │   │   └── dual_domain.py           # GMM dual-domain system
│   │   ├── physics/
│   │   │   └── langevin.py              # Langevin oscillator
│   │   ├── memory/
│   │   │   └── dual_memory.py           # Dual memory system
│   │   ├── metacog/
│   │   │   └── meta_gate.py             # Meta-cognitive gate
│   │   └── decoder/
│   │       └── glu_decoder.py           # V3.6 GLU decoder
│   ├── training/
│   │   ├── train.py                     # V3.6 trainer
│   │   └── train_v4.py                  # V4 trainer [V4 NEW]
│   ├── data/
│   │   └── dataset.py                   # Streaming text dataset
│   └── utils/
│       └── ops.py                       # Utility functions
├── docs/
│   ├── AetherMind36_Architecture_Report.md
│   └── AetherMind4_Architecture_Report.md
├── 1212/                               # Design documents & experiments
└── test_v4_smoke.py                    # V4 smoke test
```

---

## Getting Started

### Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (RTX 3050 4GB minimum; RTX 4090 recommended)

### Installation

```bash
git clone https://github.com/<your-username>/AetherMind-Nano3.git
cd AetherMind-Nano3
pip install -r requirements.txt
```

### Quick Start

```python
from configs.aethermind4_config import AetherMind4Config
from src.model.aethermind4 import AetherMind4

# Initialize model
config = AetherMind4Config()
model = AetherMind4(config)
model.cuda()

# Forward pass
import torch
input_ids = torch.randint(0, config.vocab_size, (1, 64)).cuda()
output = model(input_ids, labels=input_ids, task_id=0, t=0, phase="A")
print(f"Loss: {output['loss'].item():.4f}")

# Autoregressive generation (Phase C)
generated = model.generate(
    prompt=torch.tensor([[1]]).cuda(),  # BOS token
    max_new_tokens=64,
    phase="C"
)
print(f"Generated: {generated}")
```

### Training

```bash
# Full three-phase training
python src/training/train_v4.py --config configs/aethermind4_config.py

# Smoke test (small config, fast verification)
python test_v4_smoke.py
```

---

## Hyperparameters

### Model Architecture

| Parameter | Default | RTX 3050 | Description |
|-----------|---------|----------|-------------|
| `vocab_size` | 50,000 | 50,000 | Vocabulary size |
| `d_model` | 512 | 384 | Hidden dimension |
| `d_ff` | 2,048 | 1,536 | FFN dimension (4x D) |
| `n_layers` | 8 | 6 | Decoder layers |
| `n_heads` | 8 | 6 | Attention heads |
| `max_seq_len` | 1,024 | 512 | Max sequence length |
| `n_atoms` | 1,024 | 1,024 | GMM knowledge atoms per domain |
| `d_atom` | 64 | 64 | Atom dimension |

### PGTA Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `pheromone_rho` | 0.05 (Phase C: 0.03) | Evaporation rate |
| `pheromone_beta` | 1.0 | Pheromone sensitivity |
| `pheromone_deposit` | 0 → 0.05 | Deposition strength (by phase) |
| `pheromone_tau_min/max` | 0.01 / 5.0 | Pheromone bounds |
| `credit_mode` | "soft_center" → "free_energy" | Credit signal (Phase B → C) |

### LTP Consolidation

| Parameter | Value | Description |
|-----------|-------|-------------|
| `consolidate_threshold` | 1.5 | Trigger threshold |
| `consolidate_lam` | 0.1 | Write strength |
| `consolidate_gamma` | 0.5 | Post-consolidation τ decay |
| `consolidate_interval` | 100 steps | Consolidation frequency |
| `max_consolidations` | 4,096 | Budget limit |

### Training

| Parameter | Value | Description |
|-----------|-------|-------------|
| `learning_rate` | 3e-4 | Peak LR |
| `batch_size` | 8 (RTX 3050: 1 + grad_accum 8) | Batch size |
| `total_steps` | 250,000 | Total training steps |
| `phase_A_steps` | 50,000 | Structural pre-training |
| `phase_B_steps` | 100,000 | Mixed learning |
| `phase_C_steps` | 100,000 | Evolutionary convergence |

---

## Comparison

| Dimension | Standard Transformer | AetherMind 3.6 | **AetherMind V4.0** |
|-----------|----------------------|-----------------|---------------------|
| Attention | Scaled Dot-Product | ShapleyAttention | **PGTA (Boltzmann + Pheromone)** |
| Weight System | Single BP weights | Single BP weights | **Dual W/τ/C (LTP consolidation)** |
| Temporal Scale | Single-step | Single + physics K-step | **Three-level (Fast/Medium/Slow)** |
| Temperature | None / constant | Meta-cognitive T | **Learnable + meta-cog + annealing** |
| Memory | KV Cache | Episodic + Structural | **Dual memory + in-weight C** |
| Training | Single phase | Single phase (progressive) | **Three phases A/B/C** |
| Deployment | Static inference | Static inference | **Online continuous evolution** |
| Credit Signal | Gradient loss | Gradient loss | **Free energy (intrinsic motivation)** |
| Param Overhead | 1.0x | ~1.1x | **~1.15x** |
| VRAM Overhead | 1.0x | ~1.3x | **~1.6x** |

---

## Loss Functions

$$\mathcal{L} = \mathcal{L}_{\text{LM}} + \lambda_{\text{IB}}\mathcal{L}_{\text{IB}} + \lambda_D\mathcal{L}_D + \lambda_{\text{aff}}\mathcal{L}_{\text{aff}} + \lambda_H\mathcal{L}_H + \lambda_{\text{mod}}\mathcal{L}_{\text{mod}} + \lambda_{\text{anc}}\mathcal{L}_{\text{anc}} + \lambda_{\text{phys}}\mathcal{L}_{\text{phys}}$$

| Loss | Purpose | Phase C Weight |
|------|---------|----------------|
| $\mathcal{L}_{\text{LM}}$ | Core language modeling | 1.0 |
| $\mathcal{L}_{\text{IB}}$ | Information bottleneck compression | 1.0 |
| $\mathcal{L}_D$ | Prevent domain collapse (GAN-style) | 0.5 |
| $\mathcal{L}_{\text{aff}}$ | Emotional diversity | 0.3 |
| $\mathcal{L}_H$ | Attention entropy anti-collapse | 0.05 |
| $\mathcal{L}_{\text{mod}}$ | Association matrix topology | 0→0 |
| $\mathcal{L}_{\text{anc}}$ | Anchor uniform filling | 0.5 |
| $\mathcal{L}_{\text{phys}}$ | Physics free energy minimization | 0.2 |

---

## VRAM Budget

| Config | GPU | VRAM Usage |
|--------|-----|------------|
| d=384, h=6, n=6, S=512 | RTX 3050 4GB | ~1.1 GB |
| d=512, h=8, n=8, S=1024 | RTX 4090 24GB | ~3.5 GB |

> **Note**: Pheromone update is a pure buffer operation (add + multiply + clamp, no matrix multiply, no gradients) with <5% additional overhead per step.

---

## Verification Status

- [x] PGTA forward/backward propagation
- [x] Pheromone stepping (evaporation + deposition)
- [x] LTP consolidation mechanism (code path verified)
- [x] Three-phase configuration switching
- [x] Weight tying (lm_head <-> token_emb)
- [x] Autoregressive generation
- [x] Dual-domain GMM activation
- [x] Langevin oscillator K-step iteration
- [x] Meta-cognitive gating
- [x] Dual memory retrieval/storage
- [ ] Full three-phase long training
- [ ] Free energy reward signal dF accuracy
- [ ] Post-consolidation effect quantification
- [ ] 4GB VRAM training stability

---

## Known Limitations & Future Directions

### Current Limitations

1. **O(S²) pheromone storage** — At S=4096, τ reaches ~512MB per layer. Future work: low-rank decomposition $\tau = UV^T$ or sparse attention.
2. **Credit signal density** — Free energy reward is denser than 0/1 but still not as dense as mutual information / causal influence.
3. **Consolidation-BP coordination** — C is purely additive; future versions may use Straight-Through Estimators for gradient coordination.

### Planned Improvements

- τ low-rank decomposition for long sequences
- Consolidation hash index (inspired by DeepSeek-V4 hash routing)
- MoE integration: τ as expert routing prior, consolidation as high-frequency expert cache
- SNN spiking version: E = spike potential, τ = synaptic efficacy, consolidation = STDP
- 4-bit quantization for τ/C storage
- Phase D: deployment-time online evolution + periodic consolidation + LoRA export

---

## Citation

If you use AetherMind in your research or project, please cite:

```bibtex
@misc{aethermind4,
  author = {Xuan Xixue (Zhang Yue)},
  title  = {AetherMind V4.0: Xirang -- Dual-Weight Evolutionary Cognitive System},
  year   = {2026},
  note   = {Nanjing Qimeng Xinghui Technology Co., Ltd.}
}
```

---

## License

This project is licensed under **[CC BY 4.0 (Attribution + Citation)](https://creativecommons.org/licenses/by/4.0/deed.zh-hans)**.

You are free to use, modify, and distribute this work, including for commercial purposes, provided you credit the original author **Xuan Xixue (Zhang Yue)** and provide a link to the original source.

---

## Author

| | |
|---|---|
| **Author** | Xuan Xixue (Real Name: Zhang Yue) |
| **Region** | Sichuan, China |
| **Organization** | Nanjing Qimeng Xinghui Technology Co., Ltd. |

---

> **Open Source Announcement**: The model code is expected to be officially open-sourced between **September 5 – September 25, 2026**. Stay tuned!
