import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from src.utils.ops import MLP, stopgrad, entropy


class InformationBottleneck(nn.Module):
    def __init__(self, d_in: int, d_latent: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.enc_mu = nn.Linear(d_in, d_latent)
        self.enc_logvar = nn.Linear(d_in, d_latent)
        self.dec = MLP(d_latent, d_latent * 2, d_out, dropout)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, beta: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.enc_mu(x)
        logvar = self.enc_logvar(x)
        z = self.reparameterize(mu, logvar)
        out = self.dec(z)

        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        loss_IB = beta * kl
        return out, loss_IB


class Encoder36(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.d_model

        self.token_emb = nn.Embedding(config.vocab_size, d, padding_idx=config.pad_token_id)
        self.pos_emb = nn.Embedding(config.max_seq_len, d)

        self.input_norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(config.dropout)

        self.IB_L = InformationBottleneck(d, d, d, config.dropout)
        self.IB_P = InformationBottleneck(d, d, d, config.dropout)

        self.shapley_attn = nn.MultiheadAttention(d, config.n_heads, config.dropout, batch_first=True)

    def embed_tokens(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
        tok = self.token_emb(input_ids)
        pos_e = self.pos_emb(pos)
        x = self.input_norm(self.drop(tok + pos_e))
        return x, pos_e

    def forward(self, input_ids: torch.Tensor, domain_out: dict) -> dict:
        x, pos_e = self.embed_tokens(input_ids)

        z_IB_L_full, loss_IB_L = self.IB_L(domain_out["z_IB_L"], beta=self.config.lambda_IB)
        z_IB_P_full, loss_IB_P = self.IB_P(domain_out["z_IB_P"], beta=self.config.lambda_IB)
        loss_IB = loss_IB_L + loss_IB_P

        shap_out, shap_w = self.shapley_attn(x, x, x, need_weights=True)
        loss_shap = self.config.lambda_shap * (shap_w * torch.log(shap_w + 1e-9)).sum(-1).mean()

        D_score = domain_out["D_score"]
        with torch.amp.autocast("cuda", enabled=False):
            D_f = D_score.float().clamp(1e-6, 1 - 1e-6)
            tgt_f = torch.ones_like(D_f) * 0.5
            loss_D = self.config.lambda_D * F.binary_cross_entropy(D_f, tgt_f)

        aff_score = domain_out["z_P"].mean(dim=-1)
        loss_aff = self.config.lambda_aff * aff_score.std() if self.config.lambda_aff > 0 else torch.tensor(0.0)

        return {
            "x": x, "pos_emb": pos_e,
            "z_IB_L": z_IB_L_full, "z_IB_P": z_IB_P_full,
            "loss_IB": loss_IB, "loss_shap": loss_shap,
            "loss_D": loss_D, "loss_aff": loss_aff,
        }
