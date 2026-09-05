import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from src.utils.ops import MLP, stopgrad, safe_softmax


class EpisodicMemory(nn.Module):
    def __init__(self, d_model: int, max_slots: int = 4096, decay_lambda: float = 0.01):
        super().__init__()
        self.d_k = d_model // 2
        self.d_v = d_model
        self.max_slots = max_slots
        self.decay_lambda = decay_lambda

        self.k_proj = nn.Linear(d_model, self.d_k)
        self.v_proj = nn.Linear(d_model, self.d_v)
        self.q_proj = nn.Linear(d_model, self.d_k)

        self.register_buffer("slot_k", torch.randn(max_slots, self.d_k) * 0.02)
        self.register_buffer("slot_v", torch.randn(max_slots, self.d_v) * 0.02)
        self.register_buffer("slot_t", torch.zeros(max_slots))
        self.register_buffer("slot_domain", torch.zeros(max_slots, dtype=torch.long))
        self.register_buffer("slot_ptr", torch.tensor(0))

    def retrieve(self, q: torch.Tensor, current_domain: int = 0, t: float = 0.0) -> torch.Tensor:
        B, S, _ = q.shape
        q_k = self.q_proj(q)

        score = torch.einsum("bsd,md->bsm", q_k, self.slot_k) / (self.d_k ** 0.5)
        domain_mask = (self.slot_domain == current_domain).float().unsqueeze(0).unsqueeze(0)
        score = score * domain_mask - 1e9 * (1 - domain_mask)

        decay = torch.exp(-self.decay_lambda * (t - self.slot_t).clamp(min=0))
        decay = decay.unsqueeze(0).unsqueeze(0)
        attn = safe_softmax(score, T=1.0) * decay
        attn = attn / (attn.sum(-1, keepdim=True) + 1e-9)

        return torch.einsum("bsm,md->bsd", attn, self.slot_v)

    def store(self, x: torch.Tensor, domain: int = 0, t: float = 0.0):
        B, S, D = x.shape
        k = self.k_proj(x).reshape(-1, self.d_k)
        v = self.v_proj(x).reshape(-1, self.d_v)
        N = k.shape[0]
        ptr = int(self.slot_ptr.item())
        slots = min(N, self.max_slots)
        idx = torch.arange(ptr, ptr + slots, device=k.device) % self.max_slots
        self.slot_k[idx] = k[:slots].detach()
        self.slot_v[idx] = v[:slots].detach()
        self.slot_t[idx] = t
        self.slot_domain[idx] = domain
        self.slot_ptr = torch.tensor((ptr + slots) % self.max_slots, device=self.slot_ptr.device)


class StructuralMemory(nn.Module):
    def __init__(self, d_atom: int, max_skills: int = 1024):
        super().__init__()
        self.d_atom = d_atom
        self.max_skills = max_skills

        self.skill_key = nn.Embedding(max_skills, d_atom)
        self.skill_val = nn.Embedding(max_skills, d_atom * 2)
        self.logit_bias = nn.Parameter(torch.zeros(max_skills))

    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        keys = self.skill_key.weight
        score = torch.einsum("...d,kd->...k", query, keys) / (self.d_atom ** 0.5) + self.logit_bias.unsqueeze(0).unsqueeze(0)
        attn = safe_softmax(score, T=1.0)
        return torch.einsum("...k,kd->...d", attn, self.skill_val.weight)


class DualMemorySystem(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.epi_L = EpisodicMemory(config.d_model)
        self.epi_P = EpisodicMemory(config.d_model)
        self.struct_L = StructuralMemory(config.d_atom)
        self.struct_P = StructuralMemory(config.d_atom)

    def forward(self, x: torch.Tensor, domain_out: dict, t: float = 0.0) -> dict:
        mem_L = self.epi_L.retrieve(x, 0, t)
        mem_P = self.epi_P.retrieve(x, 1, t)

        skill_L = self.struct_L.retrieve(domain_out["z_L"])
        skill_P = self.struct_P.retrieve(domain_out["z_P"])

        return {
            "epi_L": mem_L, "epi_P": mem_P,
            "struct_L": skill_L, "struct_P": skill_P,
        }
