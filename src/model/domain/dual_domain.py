import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from src.utils.ops import MLP, stopgrad, entropy


class GMMAtom(nn.Module):
    def __init__(self, n_atoms: int, d_atom: int, n_components: int = 3, d_token: int = 512):
        super().__init__()
        self.n_atoms = n_atoms
        self.d_atom = d_atom
        self.n_components = n_components

        self.mean_emb = nn.Parameter(torch.randn(n_atoms, n_components, d_atom) * 0.02)
        self.logvar_emb = nn.Parameter(torch.zeros(n_atoms, n_components, d_atom))
        self.mix_logits = nn.Parameter(torch.zeros(n_atoms, n_components))

        self.mass = nn.Parameter(torch.ones(n_atoms) * 0.5)
        self.tau = nn.Parameter(torch.ones(n_atoms) * 1.0)

        self.token_to_atom = nn.Linear(d_token, n_atoms)
        self.atom_to_token = nn.Linear(d_atom, d_token)

    def get_mixture_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.mean_emb
        sigma = F.softplus(self.logvar_emb) + 1e-6
        pi = F.softmax(self.mix_logits, dim=-1)
        return mu, sigma, pi

    def sample(self, atom_idx: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        mu, sigma, pi = self.get_mixture_params()
        mu_i = mu[atom_idx]
        sigma_i = sigma[atom_idx]
        pi_i = pi[atom_idx]

        B = atom_idx.shape[0]
        comp = torch.multinomial(pi_i.reshape(-1, self.n_components), n_samples, replacement=True)
        comp = comp.reshape(*atom_idx.shape, n_samples)

        mu_s = torch.gather(mu_i, -2, comp.unsqueeze(-1).expand(-1, -1, -1, self.d_atom))
        sigma_s = torch.gather(sigma_i, -2, comp.unsqueeze(-1).expand(-1, -1, -1, self.d_atom))

        eps = torch.randn_like(mu_s)
        return mu_s + eps * torch.sqrt(sigma_s)

    def log_prob(self, z: torch.Tensor, atom_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        mu, sigma, pi = self.get_mixture_params()
        if atom_idx is None:
            mu = mu.unsqueeze(0).unsqueeze(0)
            sigma = sigma.unsqueeze(0).unsqueeze(0)
            pi = pi.unsqueeze(0).unsqueeze(0)
            z_e = z.unsqueeze(2).unsqueeze(3)
        else:
            mu = mu[atom_idx].unsqueeze(2)
            sigma = sigma[atom_idx].unsqueeze(2)
            pi = pi[atom_idx].unsqueeze(2)
            z_e = z.unsqueeze(2)

        diff = z_e - mu
        log_pdf = -0.5 * ((diff * diff) / sigma + torch.log(2 * math.pi * sigma)).sum(-1)
        log_prob = torch.logsumexp(torch.log(pi + 1e-9) + log_pdf, dim=-1)
        return log_prob

    def activate(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        atom_logits = self.token_to_atom(x)
        atom_weight = F.softmax(atom_logits, dim=-1)

        mu, sigma, pi = self.get_mixture_params()
        mean_atom = (pi.unsqueeze(-1) * mu).sum(dim=1)
        z = torch.einsum("bsn,nd->bsd", atom_weight, mean_atom)

        return atom_weight, z

    def get_eigen_freq(self) -> torch.Tensor:
        _, sigma, _ = self.get_mixture_params()
        return torch.sqrt(sigma.mean(dim=[1, 2]) + 1e-6)


class DualDomainSystem(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.d_model

        self.logic_atoms = GMMAtom(config.n_atoms, config.d_atom, config.n_gmm_components, d)
        self.poetic_atoms = GMMAtom(config.n_atoms, config.d_atom, config.n_gmm_components, d)

        self.logic_assoc_C = nn.Parameter(torch.zeros(config.n_atoms, config.n_atoms))
        self.logic_assoc_E = nn.Parameter(torch.zeros(config.n_atoms, config.n_atoms))
        self.poetic_assoc_C = nn.Parameter(torch.zeros(config.n_atoms, config.n_atoms))
        self.poetic_assoc_E = nn.Parameter(torch.zeros(config.n_atoms, config.n_atoms))

        self.logic_world_model = MLP(d, d * 2, d, config.dropout)
        self.poetic_world_model = MLP(d, d * 2, d, config.dropout)
        self.poetic_discriminator = MLP(d, d, 1, config.dropout)

        self.module_emb_L = nn.Embedding(config.n_atoms, config.d_atom)
        self.module_emb_P = nn.Embedding(config.n_atoms, config.d_atom)

        self.anchor_emb = nn.Embedding(config.n_anchor, config.d_anchor)
        self.anchor_router_L = nn.Linear(config.d_atom, config.n_anchor)
        self.anchor_router_P = nn.Linear(config.d_atom, config.n_anchor)

        self.affect_mlp = MLP(config.d_atom, d // 2, 1, config.dropout)

    def get_assoc(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        C_L = torch.sigmoid(self.logic_assoc_C)
        E_L = torch.sigmoid(self.logic_assoc_E)
        C_P = torch.sigmoid(self.poetic_assoc_C)
        E_P = torch.sigmoid(self.poetic_assoc_E)
        return C_L, E_L, C_P, E_P

    def module_loss(self) -> torch.Tensor:
        C_L, E_L, C_P, E_P = self.get_assoc()
        A_L = C_L * (1 - E_L)
        A_P = C_P * (1 - E_P)

        def _loss(u, A, delta=1.0):
            diff = u.unsqueeze(0) - u.unsqueeze(1)
            dist2 = (diff * diff).sum(-1)
            l1 = (dist2 * A).mean()
            l2 = torch.clamp(delta - torch.sqrt(dist2 + 1e-9), min=0).pow(2) * (1 - A)
            return l1 + l2.mean()

        u_L = self.module_emb_L.weight
        u_P = self.module_emb_P.weight
        return _loss(u_L, A_L) + _loss(u_P, A_P)

    def anchor_align_loss(self, z_L: torch.Tensor, z_P: torch.Tensor,
                          atom_w_L: torch.Tensor, atom_w_P: torch.Tensor) -> torch.Tensor:
        if self.config.lambda_align <= 0 and self.config.lambda_phase <= 0:
            return torch.tensor(0.0, device=z_L.device)

        loss_align = torch.tensor(0.0, device=z_L.device)
        if self.config.lambda_align > 0:
            B = z_L.shape[0]
            z_Lf = z_L.reshape(-1, self.config.d_atom)
            z_Pf = z_P.reshape(-1, self.config.d_atom)
            anc_L = torch.argmax(self.anchor_router_L(z_Lf), dim=-1)
            anc_P = torch.argmax(self.anchor_router_P(z_Pf), dim=-1)
            z_Lanc = stopgrad(self.anchor_emb(anc_L))
            z_Panc = self.anchor_emb(anc_P)
            loss_align = ((z_Lf - z_Lanc) ** 2).mean() + ((z_Pf - z_Panc) ** 2).mean()
            loss_align = self.config.lambda_align * loss_align

        loss_phase = torch.tensor(0.0, device=z_L.device)
        return loss_align + loss_phase

    def forward(self, x: torch.Tensor) -> dict:
        B, S, D = x.shape

        atom_w_L, z_L = self.logic_atoms.activate(x)
        atom_w_P, z_P = self.poetic_atoms.activate(x)

        mu_L = (self.logic_atoms.mix_logits.softmax(-1).unsqueeze(-1) * self.logic_atoms.mean_emb).sum(1)
        mu_P = (self.poetic_atoms.mix_logits.softmax(-1).unsqueeze(-1) * self.poetic_atoms.mean_emb).sum(1)

        C_L, E_L, C_P, E_P = self.get_assoc()
        A_L = C_L * (1 - E_L)
        A_P = C_P * (1 - E_P)

        omega_L = self.logic_atoms.get_eigen_freq()
        omega_P = self.poetic_atoms.get_eigen_freq()

        z_L_token = self.logic_atoms.atom_to_token(z_L)
        z_P_token = self.poetic_atoms.atom_to_token(z_P)

        wm_L = self.logic_world_model(z_L_token)
        wm_P = self.poetic_world_model(z_P_token)

        D_score = torch.sigmoid(self.poetic_discriminator(z_P_token)).squeeze(-1)

        loss_mod = self.config.lambda_module * self.module_loss()
        loss_anc = self.anchor_align_loss(z_L, z_P, atom_w_L, atom_w_P)

        z_IB_L = z_L_token + wm_L
        z_IB_P = z_P_token + wm_P

        return {
            "z_L": z_L, "z_P": z_P,
            "z_IB_L": z_IB_L, "z_IB_P": z_IB_P,
            "atom_w_L": atom_w_L, "atom_w_P": atom_w_P,
            "mu_L": mu_L, "mu_P": mu_P,
            "u_L": self.module_emb_L.weight,
            "u_P": self.module_emb_P.weight,
            "C_L": C_L, "E_L": E_L, "C_P": C_P, "E_P": E_P,
            "A_L": A_L, "A_P": A_P,
            "omega_L": omega_L, "omega_P": omega_P,
            "D_score": D_score,
            "loss_module": loss_mod,
            "loss_anchor": loss_anc,
            "wm_L": wm_L, "wm_P": wm_P,
        }
