import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from src.utils.ops import MLP, safe_softmax


class TokenStateRouter(nn.Module):
    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        self.W_M = nn.Linear(d_model, d_state)
        self.W_gamma_x = nn.Linear(d_model, d_state)
        self.W_gamma_z = nn.Linear(d_model, d_state)
        self.W_uM = nn.Linear(d_state, d_model)

    def forward(self, x: torch.Tensor, Z_cog: torch.Tensor,
                M_prev: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        d_s = self.W_M.out_features
        if M_prev is None:
            M_prev = torch.zeros(B, S, d_s, device=x.device, dtype=x.dtype)

        gamma = torch.sigmoid(x @ self.W_gamma_x.weight.T + (Z_cog @ self.W_gamma_z.weight.T).mean(dim=1, keepdim=True))
        x_exp = x
        M_curr = gamma * M_prev + (1 - gamma) * (x_exp @ self.W_M.weight.T + self.W_M.bias)

        u_aug = x + M_curr @ self.W_uM.weight.T + self.W_uM.bias
        return u_aug, M_curr


class PhaseStateRouter(nn.Module):
    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        self.W_M_phys = nn.Linear(d_model, d_state)
        self.W_uM_phys = nn.Linear(d_state, d_model)

    def forward(self, x: torch.Tensor, delta_theta: torch.Tensor, T: float,
                M_prev: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        d_s = self.W_M_phys.out_features
        if M_prev is None:
            M_prev = torch.zeros(B, S, d_s, device=x.device, dtype=x.dtype)

        gamma_phys = torch.exp(-0.5 * delta_theta.pow(2) / (T + 1e-6)).unsqueeze(-1)
        M_curr = gamma_phys * M_prev + (1 - gamma_phys) * (x @ self.W_M_phys.weight.T + self.W_M_phys.bias)
        return M_curr @ self.W_uM_phys.weight.T + self.W_uM_phys.bias, M_curr


class GLUBlock36(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.idx = layer_idx
        d = config.d_model
        d_ff = config.d_ff
        d_s = config.d_state

        self.norm = nn.LayerNorm(d)
        self.W_g = nn.Linear(d, d_ff * 2)
        self.W_proj = nn.Linear(d_ff, d)

        self.TSR = TokenStateRouter(d, d_s)
        self.PSR = PhaseStateRouter(d, d_s)

        self.decomp_mlp = MLP(d + 64, d * 2, d, config.dropout)
        self.pos_layer_emb = nn.Parameter(torch.randn(config.n_layers, 64) * 0.02)

        self.W_alpha_x = nn.Linear(d, 1)
        self.W_alpha_z = nn.Linear(d, 1)
        self.W_scale = nn.Linear(d, d_ff)
        self.W_bias = nn.Linear(d, d_ff)

        self.Theta_phase = nn.Parameter(torch.randn(config.n_layers, d_ff) * 0.02)

    def forward(self, x: torch.Tensor, Z_cog: torch.Tensor, T: float,
                theta: Optional[torch.Tensor] = None,
                state_TSR: Optional[torch.Tensor] = None,
                state_PSR: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        d_ff = self.config.d_ff
        x_norm = self.norm(x)
        g = x_norm @ self.W_g.weight.T + self.W_g.bias
        u, v = g.chunk(2, dim=-1)

        u_aug, new_TSR = self.TSR(x_norm, Z_cog, state_TSR)
        if u_aug.shape[-1] != d_ff:
            u_aug_proj = F.pad(u_aug, (0, d_ff - u_aug.shape[-1])) if u_aug.shape[-1] < d_ff else u_aug[..., :d_ff]
        else:
            u_aug_proj = u_aug
        u = u + u_aug_proj

        lam_PSR = self.config.lambda_PSR
        if theta is not None and lam_PSR > 0 and theta.numel() > 0:
            dth = theta[:, 1:] - theta[:, :-1]
            dth = F.pad(dth, (0, 1), value=0)
            if dth.shape[0] != u.shape[0]:
                dth = dth.mean(0, keepdim=True).expand(u.shape[0], -1)
            if dth.shape[1] != u.shape[1]:
                dth = F.pad(dth, (0, u.shape[1] - dth.shape[1]))
            psr_out, new_PSR = self.PSR(x_norm, dth, T, state_PSR)
            if psr_out.shape[-1] != d_ff:
                psr_proj = F.pad(psr_out, (0, d_ff - psr_out.shape[-1])) if psr_out.shape[-1] < d_ff else psr_out[..., :d_ff]
            else:
                psr_proj = psr_out
            u = u + lam_PSR * psr_proj
        else:
            new_PSR = state_PSR

        pel = self.pos_layer_emb[self.idx].unsqueeze(0).unsqueeze(0).expand(Z_cog.shape[0], Z_cog.shape[1], -1)
        Z_l = self.decomp_mlp(torch.cat([Z_cog, pel], dim=-1))

        logit_a = (x_norm @ self.W_alpha_x.weight.T + self.W_alpha_x.bias) + \
                  (Z_l @ self.W_alpha_z.weight.T + self.W_alpha_z.bias).mean(dim=1, keepdim=True)
        alpha_t = torch.sigmoid(logit_a)

        lam_T = self.config.lambda_T
        if theta is not None and lam_T > 0 and theta.numel() > 0:
            th = theta.reshape(x.shape[0], x.shape[1], -1).mean(-1, keepdim=True)
            th_layer = self.Theta_phase[self.idx].unsqueeze(0).unsqueeze(0).mean(-1, keepdim=True)
            if th.shape[1] != x.shape[1]:
                th = F.pad(th, (0, x.shape[1] - th.shape[1])) if th.shape[1] < x.shape[1] else th[:, :x.shape[1]]
            cos_term = torch.cos(th - th_layer)
            alpha_phys = torch.sigmoid(cos_term / (T + 1e-6))
            alpha_final = (1 - lam_T) * alpha_t + lam_T * alpha_phys
        else:
            alpha_final = alpha_t

        scale = (Z_l @ self.W_scale.weight.T + self.W_scale.bias).mean(dim=1, keepdim=True)
        bias = (Z_l @ self.W_bias.weight.T + self.W_bias.bias).mean(dim=1, keepdim=True)
        if scale.shape[1] != u.shape[1]:
            scale = scale.expand(-1, u.shape[1], -1)
            bias = bias.expand(-1, u.shape[1], -1)

        u_final = u * (1 + scale) * alpha_final + bias * (1 - alpha_final)
        out = (torch.sigmoid(u_final) * v) @ self.W_proj.weight.T + self.W_proj.bias
        x = x + out

        return x, {"TSR": new_TSR, "PSR": new_PSR}


class PointerGate(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.W_copy = nn.Linear(d_model, 1)
        self.W_copyz = nn.Linear(d_model, 1)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor, Z_cog: torch.Tensor, T: float) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = h.shape
        p_copy_base = torch.sigmoid((h @ self.W_copy.weight.T + self.W_copy.bias) +
                                     (Z_cog @ self.W_copyz.weight.T + self.W_copyz.bias).mean(dim=1, keepdim=True))
        return p_copy_base.squeeze(-1), None


class CounterSlot(nn.Module):
    def __init__(self, d_counter: int = 16):
        super().__init__()
        self.rnn = nn.GRUCell(d_counter, d_counter)
        self.proj = nn.Linear(d_counter, 1)

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        if h_prev is None:
            h = torch.zeros(B, self.rnn.hidden_size, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(S):
            inp = torch.randn(B, self.rnn.hidden_size, device=x.device, dtype=x.dtype) * 0.01
            h = self.rnn(inp, h)
            outs.append(h)
        return torch.stack(outs, dim=1), h


class GLUDecoder36(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.d_model

        self.layers = nn.ModuleList([GLUBlock36(config, i) for i in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(d)

        self.pointer = PointerGate(d)
        self.counter = CounterSlot(config.d_counter)

        self.lm_head = nn.Linear(d, config.vocab_size, bias=False)
        self.token_emb_scale = d ** -0.5

    def tie_weights(self, emb: nn.Embedding):
        self.lm_head.weight = emb.weight

    def forward(self, x: torch.Tensor, Z_cog: torch.Tensor, T: float,
                theta_L: Optional[torch.Tensor] = None,
                theta_P: Optional[torch.Tensor] = None,
                input_ids: Optional[torch.Tensor] = None) -> dict:
        B, S, D = x.shape
        states = []
        h = x
        theta = None
        if theta_L is not None:
            theta = theta_L
            if theta_P is not None:
                theta = torch.cat([theta_L.reshape(B, -1), theta_P.reshape(B, -1)], dim=-1)

        s_TSR = None
        s_PSR = None
        for layer in self.layers:
            h, st = layer(h, Z_cog, T, theta, s_TSR, s_PSR)
            s_TSR = st["TSR"]
            s_PSR = st["PSR"]
            states.append(h)

        h = self.final_norm(h)

        counter_out, _ = self.counter(h)
        h = h + self.counter.proj(counter_out).tanh() * 0.1

        p_copy, _ = self.pointer(h, Z_cog, T)
        logits = h @ self.lm_head.weight.T * self.token_emb_scale

        out_prob = safe_softmax(logits, T=T)

        if input_ids is not None and p_copy is not None:
            p_copy_gate = torch.sigmoid(p_copy).unsqueeze(-1)
            if input_ids.shape[1] == S:
                # 用半精度减少显存占用 (vocab_size 可能很大)
                src_onehot = F.one_hot(input_ids, num_classes=self.config.vocab_size).half()
                copy_prob = src_onehot.cumsum(dim=1) / (torch.arange(1, S + 1, device=input_ids.device).float().unsqueeze(0).unsqueeze(-1).half())
                out_prob = (1 - p_copy_gate.half()) * out_prob.half() + p_copy_gate.half() * copy_prob
                out_prob = out_prob.float()

        return {
            "logits": logits,
            "probs": out_prob,
            "p_copy": p_copy,
            "last_hidden": h,
            "layer_states": states,
        }
