import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from configs.aethermind36_config import AetherMind36Config
from src.model.domain.dual_domain import DualDomainSystem
from src.model.physics.langevin import LangevinOscillator
from src.model.memory.dual_memory import DualMemorySystem
from src.model.metacog.meta_gate import MetaCognitiveGate
from src.model.encoder.ib_encoder import Encoder36
from src.model.decoder.glu_decoder import GLUDecoder36
from src.utils.ops import safe_softmax, sinusoidal_positional_encoding, stopgrad


class AetherMind36(nn.Module):
    def __init__(self, config: AetherMind36Config):
        super().__init__()
        self.config = config

        self.encoder = Encoder36(config)
        self.dual_domain = DualDomainSystem(config)
        self.physics = LangevinOscillator(config)
        self.memory = DualMemorySystem(config)
        self.metacog = MetaCognitiveGate(config)
        self.decoder = GLUDecoder36(config)

        self.decoder.tie_weights(self.encoder.token_emb)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None,
                task_id: int = 0, t: float = 0.0) -> Dict[str, torch.Tensor]:
        cfg = self.config
        B, S = input_ids.shape
        device = input_ids.device

        domain_out = self.dual_domain(self.encoder.token_emb(input_ids))
        enc_out = self.encoder(input_ids, domain_out)

        phys_out = self.physics({
            **domain_out,
            "z_IB_L": enc_out["z_IB_L"],
            "z_IB_P": enc_out["z_IB_P"],
        }, enc_out["pos_emb"])

        mem_out = self.memory(enc_out["x"], domain_out, t)
        epi_mem = domain_out["C_L"].new_zeros(B, S, cfg.d_model)
        epi_mem = epi_mem + mem_out["epi_L"] + mem_out["epi_P"]

        Z_phys_L_tokens = torch.einsum("bsn,bnd->bsd", domain_out["atom_w_L"], phys_out["L"]["Z_phys"])
        Z_phys_P_tokens = torch.einsum("bsn,bnd->bsd", domain_out["atom_w_P"], phys_out["P"]["Z_phys"])
        if Z_phys_L_tokens.shape[1] != enc_out["z_IB_L"].shape[1]:
            Z_phys_L_tokens = F.pad(Z_phys_L_tokens, (0, 0, 0, enc_out["z_IB_L"].shape[1] - Z_phys_L_tokens.shape[1]))
        if Z_phys_P_tokens.shape[1] != enc_out["z_IB_P"].shape[1]:
            Z_phys_P_tokens = F.pad(Z_phys_P_tokens, (0, 0, 0, enc_out["z_IB_P"].shape[1] - Z_phys_P_tokens.shape[1]))
        if Z_phys_L_tokens.shape[2] != enc_out["z_IB_L"].shape[2]:
            d_in, d_out = Z_phys_L_tokens.shape[2], enc_out["z_IB_L"].shape[2]
            proj = torch.nn.Linear(d_in, d_out, device=Z_phys_L_tokens.device, dtype=Z_phys_L_tokens.dtype)
            Z_phys_L_tokens = proj(Z_phys_L_tokens)
            Z_phys_P_tokens = proj(Z_phys_P_tokens)

        meta_out = self.metacog(
            enc_out["z_IB_L"], enc_out["z_IB_P"],
            Z_phys_L_tokens, Z_phys_P_tokens,
            epi_mem, task_id
        )

        th_L_atoms = phys_out["L"]["theta"]
        th_P_atoms = phys_out["P"]["theta"]
        B, S = input_ids.shape
        if th_L_atoms is not None and th_L_atoms.numel() > 0:
            th_L_tokens = (domain_out["atom_w_L"] * th_L_atoms.unsqueeze(1)).sum(-1)
            th_P_tokens = (domain_out["atom_w_P"] * th_P_atoms.unsqueeze(1)).sum(-1)
        else:
            th_L_tokens = torch.zeros(B, S, device=input_ids.device, dtype=domain_out["atom_w_L"].dtype)
            th_P_tokens = torch.zeros(B, S, device=input_ids.device, dtype=domain_out["atom_w_P"].dtype)

        dec_out = self.decoder(
            enc_out["x"], meta_out["Z_cog"], meta_out["T"],
            th_L_tokens, th_P_tokens, input_ids
        )

        loss_dict = {}
        loss_LM = torch.tensor(0.0, device=device)
        if labels is not None:
            shift_logits = dec_out["logits"][..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_LM = F.cross_entropy(
                shift_logits.view(-1, cfg.vocab_size),
                shift_labels.view(-1),
                ignore_index=cfg.pad_token_id
            )
            loss_dict["loss_LM"] = loss_LM

        losses = [
            loss_LM,
            enc_out["loss_IB"] if cfg.lambda_IB > 0 else 0.0,
            enc_out["loss_shap"] if cfg.lambda_shap > 0 else 0.0,
            enc_out["loss_D"] if cfg.lambda_D > 0 else 0.0,
            enc_out["loss_aff"] if cfg.lambda_aff > 0 else 0.0,
            domain_out["loss_module"] if cfg.lambda_module > 0 else 0.0,
            domain_out["loss_anchor"] if cfg.lambda_align > 0 else 0.0,
            phys_out["loss_phys"] if cfg.lambda_phys > 0 else 0.0,
        ]
        total = 0.0
        for l in losses:
            if isinstance(l, torch.Tensor):
                total = total + l

        loss_dict.update({
            "loss_IB": enc_out["loss_IB"],
            "loss_shap": enc_out["loss_shap"],
            "loss_D": enc_out["loss_D"],
            "loss_aff": enc_out["loss_aff"],
            "loss_module": domain_out["loss_module"],
            "loss_anchor": domain_out["loss_anchor"],
            "loss_phys": phys_out["loss_phys"],
            "loss": total,
            "u_cog": meta_out["u_cog"].mean(),
            "T": meta_out["T"].mean(),
        })

        return {
            **loss_dict,
            "logits": dec_out["logits"],
            "probs": dec_out["probs"],
            "Z_cog": meta_out["Z_cog"],
            "p_copy": dec_out["p_copy"],
            "last_hidden": dec_out["last_hidden"],
            "F": phys_out["L"]["F"] + phys_out["P"]["F"],
            "dF": phys_out["L"]["dF"] + phys_out["P"]["dF"],
            "safety_score": meta_out["safety_score"].mean(),
        }

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 256,
                 temperature: Optional[float] = None, top_k: int = 50,
                 top_p: float = 0.9, task_id: int = 0) -> torch.Tensor:
        self.eval()
        cfg = self.config
        B, S = input_ids.shape
        device = input_ids.device
        generated = input_ids.clone()

        for step in range(max_new_tokens):
            cur_ids = generated[:, -cfg.max_seq_len:]
            out = self.forward(cur_ids, task_id=task_id, t=step)
            logits = out["logits"][:, -1, :]
            T = temperature if temperature is not None else out["T"].item()
            T = max(T, 1e-3)
            probs = safe_softmax(logits, T=T)

            if top_k > 0:
                vals, idx = torch.topk(probs, top_k, dim=-1)
                probs = torch.zeros_like(probs).scatter_(-1, idx, vals)
                probs = probs / probs.sum(-1, keepdim=True)
            if top_p < 1.0:
                sv, si = torch.sort(probs, dim=-1, descending=True)
                cs = sv.cumsum(dim=-1)
                mask = cs - sv > top_p
                sv[mask] = 0
                sv = sv / sv.sum(-1, keepdim=True)
                probs = torch.zeros_like(probs).scatter_(-1, si, sv)

            nxt = torch.multinomial(probs, 1)
            generated = torch.cat([generated, nxt], dim=1)

            if (nxt == cfg.eos_token_id).all():
                break

        return generated

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
