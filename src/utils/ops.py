import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def get_activation(name: str):
    if name == "silu":
        return F.silu
    elif name == "gelu":
        return F.gelu
    elif name == "relu":
        return F.relu
    elif name == "sigmoid":
        return torch.sigmoid
    else:
        return F.silu


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.0, activation: str = "silu"):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.dropout = nn.Dropout(dropout)
        self.act = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


def sinusoidal_positional_encoding(seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_causal_mask(x: torch.Tensor, fill_value: float = float("-inf")) -> torch.Tensor:
    seq_len = x.shape[-2]
    mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
    return x.masked_fill(mask, fill_value)


def stopgrad(x: torch.Tensor) -> torch.Tensor:
    return x.detach()


def entropy(p: torch.Tensor, dim: int = -1, eps: float = 1e-9) -> torch.Tensor:
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=dim)


def gradient_penalty(loss: torch.Tensor, params, weight: float = 1.0) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True, allow_unused=True)
    penalty = 0.0
    for g in grads:
        if g is not None:
            penalty = penalty + g.pow(2).sum()
    return weight * penalty


def safe_softmax(logits: torch.Tensor, T: float = 1.0, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    T = max(T, eps)
    scaled = logits / T
    scaled = scaled - scaled.max(dim=dim, keepdim=True)[0]
    return F.softmax(scaled, dim=dim)
