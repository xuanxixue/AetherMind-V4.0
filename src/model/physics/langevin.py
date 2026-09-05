import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from src.utils.ops import MLP, stopgrad


class LangevinOscillator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.d_atom
        d_model = config.d_model

        self.r_proj = nn.Linear(d, d)
        self.theta_proj = nn.Linear(d + config.d_atom, 1)
        self.omega_proj = nn.Linear(d, 1)

        # 将(r*cosθ, r*sinθ)复数表示(最后一维=2)投影到d_model维token空间
        self.phys_proj = nn.Sequential(
            nn.Linear(2, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d_model),
        )

    def _init_complex(self, mu: torch.Tensor, pos_emb: torch.Tensor, u_mod: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N = mu.shape[0]
        r = torch.norm(self.r_proj(mu), dim=-1) + 1e-6
        if r.ndim > 1:
            r = r.mean(dim=tuple(range(1, r.ndim)))
        if r.shape[0] != N:
            r = F.adaptive_avg_pool1d(r.view(1, 1, -1), N).view(-1)

        pe_sin = pos_emb.reshape(-1, pos_emb.shape[-1])[..., 0::2].mean(-1)
        pe_cos = pos_emb.reshape(-1, pos_emb.shape[-1])[..., 1::2].mean(-1)
        theta_pe = torch.atan2(pe_sin, pe_cos)
        if theta_pe.shape[0] != N:
            if theta_pe.shape[0] > N:
                theta_pe = theta_pe[:N]
            else:
                theta_pe = F.pad(theta_pe, (0, N - theta_pe.shape[0]))

        if u_mod.shape[0] == mu.shape[0]:
            u_flat = u_mod
        else:
            u_flat = u_mod.mean(0).unsqueeze(0).expand(mu.shape[0], -1)
        cat_in = torch.cat([mu, u_flat], dim=-1)
        theta_u = self.theta_proj(cat_in).view(-1)
        if theta_u.shape[0] != N:
            if theta_u.shape[0] > N:
                theta_u = theta_u[:N]
            else:
                theta_u = F.pad(theta_u, (0, N - theta_u.shape[0]))

        theta = theta_pe + theta_u
        assert r.shape == (N,), f"r shape {r.shape} != ({N},)"
        assert theta.shape == (N,), f"theta shape {theta.shape} != ({N},)"
        return r, theta

    def _free_energy(self, r: torch.Tensor, r0: torch.Tensor, theta: torch.Tensor,
                     J: torch.Tensor, phi: torch.Tensor, T: float) -> torch.Tensor:
        k = 1.0
        F1 = 0.5 * k * (r - r0).pow(2).mean()

        n = r.shape[0]
        r_i = r.unsqueeze(1)
        r_j = r.unsqueeze(0)
        th_i = theta.unsqueeze(1)
        th_j = theta.unsqueeze(0)
        cos_dth = torch.cos(th_i - th_j - phi)
        F2 = -(J * r_i * r_j * cos_dth).mean()

        theta_mean = theta.mean()
        p_theta = F.softmax(torch.cos(theta - theta_mean), dim=0)
        S_phase = -(p_theta * torch.log(p_theta + 1e-9)).sum()
        F3 = -T * S_phase

        return F1 + F2 + F3

    def run_langevin(self, mu: torch.Tensor, pos_emb: torch.Tensor, u_mod: torch.Tensor,
                     A: torch.Tensor, domain: str = "logic",
                     extra_grad: Optional[torch.Tensor] = None) -> dict:
        K = self.config.langevin_K
        B, N, D = mu.shape
        # K<=0: 物理层关闭，返回零张量（与正常路径同形状）
        if K <= 0:
            d_model = self.config.d_model
            return {
                "r": torch.ones(B, N, device=mu.device, dtype=mu.dtype),
                "theta": torch.zeros(B, N, device=mu.device, dtype=mu.dtype),
                "Z_phys": torch.zeros(B, N, d_model, device=mu.device, dtype=mu.dtype),
                "F": torch.tensor(0.0, device=mu.device, dtype=mu.dtype),
                "dF": torch.tensor(0.0, device=mu.device, dtype=mu.dtype),
            }

        dt = self.config.langevin_dt
        T_enc = self.config.langevin_T_enc
        topk = self.config.topk_neighbors

        r_all = []
        theta_all = []
        F_all = []

        for b in range(B):
            mu_b = mu[b]
            assert mu_b.shape == (N, D), f"mu_b shape mismatch: {mu_b.shape} vs expected ({N}, {D})"
            pe_b = pos_emb[b] if pos_emb.dim() == 3 else pos_emb
            A_b = A if A.dim() == 2 else A[b]

            J_b = A_b.clone()
            if J_b.shape[0] != N:
                J_b = J_b[:N, :N]
            if N > topk:
                val, idx = torch.topk(J_b, topk, dim=-1)
                mask = torch.zeros_like(J_b)
                mask.scatter_(-1, idx, 1.0)
                J_b = J_b * mask

            phi_b = torch.zeros(N, N, device=mu_b.device) if domain == "logic" else None

            r0, theta = self._init_complex(mu_b, pe_b, u_mod)
            if r0.shape[0] != N:
                if r0.shape[0] > N:
                    r0 = r0[:N]
                else:
                    r0 = F.pad(r0, (0, N - r0.shape[0]))
            if theta.shape[0] != N:
                if theta.shape[0] > N:
                    theta = theta[:N]
                else:
                    theta = F.pad(theta, (0, N - theta.shape[0]))
            r = r0.clone()

            omega = self.omega_proj(mu_b).squeeze(-1)

            F_prev = None
            for step in range(K):
                r_i = r.unsqueeze(1)
                r_j = r.unsqueeze(0)
                th_i = theta.unsqueeze(1)
                th_j = theta.unsqueeze(0)
                phi = phi_b if phi_b is not None else (th_i - th_j).detach()
                sin_dth = torch.sin(th_j - th_i - phi)
                coupling = (J_b * r_j * sin_dth).sum(dim=1)

                xi_theta = torch.randn_like(theta) * math.sqrt(2 * T_enc * dt)
                theta = theta + dt * (omega + coupling) + xi_theta

                F = self._free_energy(r, r0, theta, J_b, phi_b if domain == "logic" else torch.zeros_like(J_b), T_enc)
                if r.requires_grad and torch.is_grad_enabled():
                    if extra_grad is not None:
                        r_grad = -torch.autograd.grad(F.sum(), r, create_graph=True, retain_graph=True)[0]
                        r_grad = r_grad + extra_grad[b] * 0.1
                    else:
                        r_grad = -torch.autograd.grad(F.sum(), r, create_graph=True, retain_graph=True, allow_unused=True)[0]
                        if r_grad is None:
                            r_grad = torch.zeros_like(r)
                else:
                    r_grad = -2.0 * (r - r0)
                xi_r = torch.randn_like(r) * math.sqrt(2 * T_enc * dt)
                r = r + dt * r_grad + xi_r
                r = torch.clamp(r, min=1e-3)

                F_all.append(F.detach())
                if step == K - 1 and F_prev is not None:
                    dF = F - F_prev
                F_prev = F.detach()

            r_all.append(r)
            theta_all.append(theta)

        r = torch.stack(r_all, dim=0)
        theta = torch.stack(theta_all, dim=0)
        assert r.shape == (B, N), f"DEBUG r shape: {r.shape}, expected ({B}, {N})"
        assert theta.shape == (B, N), f"DEBUG theta shape: {theta.shape}, expected ({B}, {N})"
        F = torch.stack(F_all) if F_all else torch.tensor(0.0, device=mu.device)
        dF = (F[-1] - F[0]) if len(F) > 1 else torch.tensor(0.0, device=mu.device)

        re = r.unsqueeze(-1) * torch.cos(theta.unsqueeze(-1))
        im = r.unsqueeze(-1) * torch.sin(theta.unsqueeze(-1))
        z_complex = torch.cat([re, im], dim=-1)  # (B, N, 2)
        Z_phys = self.phys_proj(z_complex)  # (B, N, d_model)

        return {
            "r": r, "theta": theta,
            "Z_phys": Z_phys,
            "F": F.mean(),
            "dF": dF,
        }

    def forward(self, domain_out: dict, pos_emb: torch.Tensor) -> dict:
        if self.config.lambda_phys <= 0 and self.config.langevin_K <= 0:
            B = domain_out["z_IB_L"].shape[0]
            z_dtype = domain_out["z_IB_L"].dtype
            z_dev = domain_out["z_IB_L"].device
            N = domain_out["atom_w_L"].shape[-1]
            D = self.config.d_model
            Z_phys_L = torch.zeros(B, N, D, dtype=z_dtype, device=z_dev)
            Z_phys_P = torch.zeros(B, N, D, dtype=z_dtype, device=z_dev)
            return {
                "L": {"Z_phys": Z_phys_L, "F": torch.zeros((), dtype=z_dtype, device=z_dev),
                      "dF": torch.zeros((), dtype=z_dtype, device=z_dev), "r": None, "theta": None},
                "P": {"Z_phys": Z_phys_P, "F": torch.zeros((), dtype=z_dtype, device=z_dev),
                      "dF": torch.zeros((), dtype=z_dtype, device=z_dev), "r": None, "theta": None},
                "loss_phys": torch.zeros((), dtype=z_dtype, device=z_dev),
            }

        out_L = self.run_langevin(
            domain_out["mu_L"].unsqueeze(0).expand(domain_out["z_L"].shape[0], -1, -1),
            pos_emb, domain_out["u_L"],
            domain_out["A_L"], "logic"
        )
        out_P = self.run_langevin(
            domain_out["mu_P"].unsqueeze(0).expand(domain_out["z_P"].shape[0], -1, -1),
            pos_emb, domain_out["u_P"],
            domain_out["A_P"], "poetic"
        )

        F_total = out_L["F"] + out_P["F"]
        loss_phys = self.config.lambda_phys * (F_total.pow(2).mean())

        return {"L": out_L, "P": out_P, "loss_phys": loss_phys}
