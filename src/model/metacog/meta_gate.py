import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from src.utils.ops import MLP, stopgrad, safe_softmax


class TriPercept(nn.Module):
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.percepts = nn.ModuleList([
            MLP(d * 3, d, 2, dropout) for _ in range(3)
        ])

    def forward(self, z_L: torch.Tensor, z_P: torch.Tensor, task_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([z_L.mean(dim=1), z_P.mean(dim=1), task_emb], dim=-1)
        outs = []
        for p in self.percepts:
            out = F.softmax(p(x), dim=-1)
            outs.append(out)

        outs_stack = torch.stack(outs, dim=0)
        alphas_L = outs_stack[..., 0]
        alpha_L = torch.median(alphas_L, dim=0)[0]
        alpha_P = 1.0 - alpha_L

        var = torch.var(alphas_L, dim=0, unbiased=False)
        max_var = 0.25
        u_cog = var / max_var
        return alpha_L.unsqueeze(-1).unsqueeze(-1), alpha_P.unsqueeze(-1).unsqueeze(-1), u_cog


class SafetyFilter(nn.Module):
    def __init__(self, d: int, n_danger: int = 64):
        super().__init__()
        self.danger_anchor = nn.Parameter(torch.randn(n_danger, d) * 0.02)

    def safety_score(self, x: torch.Tensor) -> torch.Tensor:
        sim = F.cosine_similarity(x.unsqueeze(2), self.danger_anchor.unsqueeze(0).unsqueeze(0), dim=-1)
        return (1.0 - sim).min(dim=-1)[0]

    def forward(self, pattern: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        score = self.safety_score(pattern)
        safe = (score > 0.3).float().unsqueeze(-1)
        skeleton = pattern * safe + pattern.mean(dim=-1, keepdim=True) * (1 - safe)
        return skeleton, score


class MetaCognitiveGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.d_model
        self.tri_percept = TriPercept(d, config.dropout)
        self.safety = SafetyFilter(d)
        self.fuse_mlp = MLP(d * 4, d * 2, d, config.dropout)
        self.task_emb = nn.Embedding(16, d)
        self.safe_pattern_bank = nn.Parameter(torch.randn(256, d) * 0.02)

    def get_temperature(self, u_cog: torch.Tensor) -> torch.Tensor:
        T = self.config.T0 * (1.0 + self.config.kappa * u_cog)
        return torch.clamp(T, min=self.config.T_min, max=self.config.T_max)

    def forward(self, z_IB_L: torch.Tensor, z_IB_P: torch.Tensor,
                Z_phys_L: torch.Tensor, Z_phys_P: torch.Tensor,
                epi_mem: torch.Tensor, task_id: int = 0) -> dict:
        B, S, D = z_IB_L.shape
        task = self.task_emb(torch.full((B,), task_id, dtype=torch.long, device=z_IB_L.device))

        a_L, a_P, u_cog = self.tri_percept(z_IB_L, z_IB_P, task)

        safe_mode = (u_cog > 0.7).float().unsqueeze(-1).unsqueeze(-1)
        a_L = a_L * (1 - safe_mode) + 0.5 * safe_mode
        a_P = 1.0 - a_L

        lam_phys = self.config.lambda_phys
        z_L = (1 - lam_phys) * z_IB_L + lam_phys * Z_phys_L
        z_P = (1 - lam_phys) * z_IB_P + lam_phys * Z_phys_P

        Z_IB = a_L * z_L + a_P * z_P

        pattern_lib = self.safe_pattern_bank.unsqueeze(0).unsqueeze(0)
        pattern_best = torch.einsum("bsd,pd->bsp", Z_IB, self.safe_pattern_bank)
        pattern_best = safe_softmax(pattern_best, dim=-1, T=0.5)
        pattern_sig = torch.einsum("bsp,pd->bsd", pattern_best, self.safe_pattern_bank)

        Z_safe, s_score = self.safety(pattern_sig)

        task_exp = task.unsqueeze(1).expand(-1, S, -1)
        Z_cog = self.fuse_mlp(torch.cat([Z_IB, epi_mem, Z_safe, task_exp], dim=-1))

        T = self.get_temperature(u_cog.mean())

        return {
            "Z_cog": Z_cog,
            "alpha_L": a_L, "alpha_P": a_P,
            "u_cog": u_cog, "T": T,
            "safety_score": s_score,
        }
