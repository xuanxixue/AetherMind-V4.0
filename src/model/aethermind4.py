"""
AetherMind V4.0 — 息壤·双权重演化认知体
=========================================
在3.6基础上引入三大进化：
  1. 信息素调制热力学注意力 (PGTA) — 替换Shapley注意力，成为统一注意力机制
  2. 可演化双权重系统 (W/τ) — 快权重梯度学习 + 慢权重物理演化
  3. 固化机制 (LTP) — 经验从短期信息素写入长期权重，跨session保持

模块拓扑：
  Token Emb → PGTA Encoder(×n_layers_enc) → DualDomain(GMM+世界模型)
    → Langevin Oscillator(相位同步) → DualMemory(情景+结构)
    → MetaCognitiveGate(域权重+温度) → PGTA Decoder(×n_layers_dec)
    → EvolvableWeightSystem(信息素生命周期管理+固化)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from typing import Optional, Dict
from configs.aethermind4_config import AetherMind4Config
from src.model.domain.dual_domain import DualDomainSystem
from src.model.physics.langevin import LangevinOscillator
from src.model.memory.dual_memory import DualMemorySystem
from src.model.metacog.meta_gate import MetaCognitiveGate
from src.model.attention.pheromone_thermo import PheromoneThermoAttention
from src.model.attention.pheromone_thermo_inference import PheromoneThermoInference
from src.model.evolution.evolvable_weight import EvolvableWeightSystem
from src.utils.ops import safe_softmax, MLP, sinusoidal_positional_encoding


# ============================================================
# V4 编码器（使用PGTA替换Shapley注意力）
# ============================================================
class EncoderV4(nn.Module):
    def __init__(self, config: AetherMind4Config, attn_cls=PheromoneThermoAttention):
        super().__init__()
        self.config = config
        d = config.d_model

        self.token_emb = nn.Embedding(config.vocab_size, d, padding_idx=config.pad_token_id)
        self.pos_emb = nn.Embedding(config.max_seq_len, d)
        self.input_norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(config.dropout)

        # V4: 使用信息素热力学注意力层（推理架构可传入 PheromoneThermoInference）
        self.attn_layers = nn.ModuleList([
            attn_cls(
                d_model=d, num_heads=config.n_heads, max_seq_len=config.max_seq_len,
                init_temp=config.init_temperature, whiten=config.pheromone_whiten,
                rho=config.pheromone_rho, beta=config.pheromone_beta,
                deposit=config.pheromone_deposit,
                tau_min=config.pheromone_tau_min, tau_max=config.pheromone_tau_max,
                target_entropy_ratio=config.target_entropy_ratio, dropout=config.dropout,
            ) for _ in range(2)  # 编码器2层PGTA
        ])
        self.attn_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])
        self.attn_ffns = nn.ModuleList([
            nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d), nn.Dropout(config.dropout))
            for _ in range(2)
        ])
        self.attn_ffn_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])

        # 信息瓶颈
        self.IB_L = nn.ModuleDict({
            "enc_mu": nn.Linear(d, d), "enc_logvar": nn.Linear(d, d),
            "dec": MLP(d, d * 2, d, config.dropout),
        })
        self.IB_P = nn.ModuleDict({
            "enc_mu": nn.Linear(d, d), "enc_logvar": nn.Linear(d, d),
            "dec": MLP(d, d * 2, d, config.dropout),
        })

    def embed_tokens(self, input_ids: torch.Tensor):
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
        tok = self.token_emb(input_ids)
        pos_e = self.pos_emb(pos)
        x = self.input_norm(self.drop(tok + pos_e))
        return x, pos_e

    def _ib_forward(self, ib_dict, x, beta):
        mu = ib_dict["enc_mu"](x)
        logvar = ib_dict["enc_logvar"](x)
        std = torch.exp(0.5 * logvar)
        # 训练时采样（重参数化），推理时用均值——旧版推理也采样导致输出随机抖动
        z = mu + torch.randn_like(std) * std if self.training else mu
        out = ib_dict["dec"](z)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
        return out, beta * kl

    def forward(self, input_ids: torch.Tensor, domain_out: dict) -> dict:
        x, pos_e = self.embed_tokens(input_ids)
        B, S, D = x.shape
        cfg = self.config
        use_ckpt = cfg.gradient_checkpointing and self.training

        # PGTA 自注意力层
        attn_stats = {}
        entropy_total = 0.0
        fe_total = 0.0
        
        def _enc_layer_block(x_in, attn_mod, norm_mod, ffn_mod, ffn_norm_mod):
            """单层encoder block（供checkpoint使用）"""
            residual = x_in
            attn_out, stats = attn_mod(norm_mod(x_in))
            h = residual + attn_out
            h = h + ffn_mod(ffn_norm_mod(h))
            return h, stats
        
        for i, (attn, norm, ffn, ffn_norm) in enumerate(
            zip(self.attn_layers, self.attn_norms, self.attn_ffns, self.attn_ffn_norms)
        ):
            if use_ckpt:
                x, stats = torch_checkpoint(
                    _enc_layer_block, x, attn, norm, ffn, ffn_norm,
                    use_reentrant=False
                )
            else:
                x, stats = _enc_layer_block(x, attn, norm, ffn, ffn_norm)
            entropy_total += stats["entropy"]
            fe_total += stats["free_energy"]
        attn_stats["enc_entropy"] = entropy_total / len(self.attn_layers)
        attn_stats["enc_free_energy"] = fe_total / len(self.attn_layers)

        # 信息瓶颈压缩
        z_IB_L, loss_IB_L = self._ib_forward(self.IB_L, domain_out["z_IB_L"], cfg.lambda_IB)
        z_IB_P, loss_IB_P = self._ib_forward(self.IB_P, domain_out["z_IB_P"], cfg.lambda_IB)
        loss_IB = loss_IB_L + loss_IB_P

        # GMM 判别器损失 (bf16安全 + NaN防护)
        D_score = domain_out["D_score"]
        with torch.amp.autocast("cuda", enabled=False):
            D_f = D_score.float()
            # 替换 NaN/inf 为 0.5（无信息），再 clamp 到安全范围
            D_f = torch.where(torch.isfinite(D_f), D_f, torch.full_like(D_f, 0.5))
            D_f = D_f.clamp(1e-6, 1 - 1e-6)
            tgt_f = torch.ones_like(D_f) * 0.5
            loss_D = cfg.lambda_D * F.binary_cross_entropy(D_f, tgt_f)

        # 情感正则
        aff_score = domain_out["z_P"].mean(dim=-1)
        loss_aff = cfg.lambda_aff * aff_score.std() if cfg.lambda_aff > 0 else torch.tensor(0.0, device=x.device)

        # 熵正则（PGTA自带）
        target_H = cfg.target_entropy_ratio * math.log(S)
        loss_H = cfg.lambda_H * F.relu(attn_stats["enc_entropy"] - target_H)

        return {
            "x": x, "pos_emb": pos_e,
            "z_IB_L": z_IB_L, "z_IB_P": z_IB_P,
            "loss_IB": loss_IB, "loss_D": loss_D, "loss_aff": loss_aff,
            "loss_H": loss_H,
            "attn_entropy": attn_stats["enc_entropy"],
            "attn_FE": attn_stats["enc_free_energy"],
        }


# ============================================================
# V4 解码器（GLU+TSR+PSR + PGTA自注意力）
# ============================================================
class GLUBlockV4(nn.Module):
    def __init__(self, config: AetherMind4Config, layer_idx: int, attn_layer: PheromoneThermoAttention):
        super().__init__()
        self.config = config
        self.idx = layer_idx
        d = config.d_model
        d_ff = config.d_ff
        d_s = config.d_state

        self.norm1 = nn.LayerNorm(d)
        self.attn = attn_layer  # 共享/独立的PGTA层
        self.norm2 = nn.LayerNorm(d)

        self.W_g = nn.Linear(d, d_ff * 2)
        self.W_proj = nn.Linear(d_ff, d)

        # TSR (Token State Router)
        self.W_M = nn.Linear(d, d_s)
        self.W_gamma_x = nn.Linear(d, d_s)
        self.W_gamma_z = nn.Linear(d, d_s)
        self.W_uM = nn.Linear(d_s, d)

        # PSR (Phase State Router)
        self.W_M_phys = nn.Linear(d, d_s)
        self.W_uM_phys = nn.Linear(d_s, d)

        # 层级注入
        self.decomp_mlp = MLP(d + 64, d * 2, d, config.dropout)
        self.pos_layer_emb = nn.Parameter(torch.randn(config.n_layers, 64) * 0.02)
        self.W_alpha_x = nn.Linear(d, 1)
        self.W_alpha_z = nn.Linear(d, 1)
        self.W_scale = nn.Linear(d, d_ff)
        self.W_bias = nn.Linear(d, d_ff)
        self.Theta_phase = nn.Parameter(torch.randn(config.n_layers, d_ff) * 0.02)

    def _tsr(self, x, Z_cog, M_prev):
        d_s = self.W_M.out_features
        if M_prev is None:
            M_prev = torch.zeros(x.shape[0], x.shape[1], d_s, device=x.device, dtype=x.dtype)
        gamma = torch.sigmoid(x @ self.W_gamma_x.weight.T + (Z_cog @ self.W_gamma_z.weight.T).mean(1, keepdim=True))
        M_curr = gamma * M_prev + (1 - gamma) * (x @ self.W_M.weight.T + self.W_M.bias)
        return M_curr @ self.W_uM.weight.T + self.W_uM.bias, M_curr

    def _psr(self, x, delta_theta, T, M_prev):
        d_s = self.W_M_phys.out_features
        if M_prev is None:
            M_prev = torch.zeros(x.shape[0], x.shape[1], d_s, device=x.device, dtype=x.dtype)
        # T -> (B,1,1) 或 (1,1,1)，delta_theta -> (B,S,1)，保证除法广播到 (B,S,1)
        if isinstance(T, torch.Tensor):
            if T.numel() > 1:
                T_bc = T.reshape(-1)[:x.shape[0]].view(-1, 1, 1)
            else:
                T_bc = T.view(1, 1, 1)
        else:
            T_bc = torch.tensor(float(T), device=x.device, dtype=x.dtype).view(1, 1, 1)
        if delta_theta.dim() == 2:
            delta_theta = delta_theta.unsqueeze(-1)  # (B,S) -> (B,S,1)
        # delta_theta 已为 (B,S,1)，T_bc 为 (B,1,1) 或 (1,1,1)，除法广播安全
        gamma_phys = torch.exp(-0.5 * delta_theta.pow(2) / (T_bc + 1e-6))
        M_curr = gamma_phys * M_prev + (1 - gamma_phys) * (x @ self.W_M_phys.weight.T + self.W_M_phys.bias)
        return M_curr @ self.W_uM_phys.weight.T + self.W_uM_phys.bias, M_curr

    def forward(self, x, Z_cog, T, theta=None, s_TSR=None, s_PSR=None):
        B, S, D = x.shape
        d_ff = self.config.d_ff
        cfg = self.config

        # PGTA 自注意力（T 直接透传，保持可微；PGTA内部处理float/tensor）
        residual = x
        attn_out, _ = self.attn(self.norm1(x), temperature_override=T)
        x = residual + attn_out

        x_norm = self.norm2(x)
        g = x_norm @ self.W_g.weight.T + self.W_g.bias
        u, v = g.chunk(2, dim=-1)

        # TSR
        u_aug, new_TSR = self._tsr(x_norm, Z_cog, s_TSR)
        if u_aug.shape[-1] != d_ff:
            u_aug = F.pad(u_aug, (0, d_ff - u_aug.shape[-1])) if u_aug.shape[-1] < d_ff else u_aug[..., :d_ff]
        u = u + u_aug

        # PSR
        new_PSR = s_PSR
        if theta is not None and cfg.lambda_PSR > 0 and theta.numel() > 0:
            dth = theta[:, 1:] - theta[:, :-1]
            dth = F.pad(dth, (0, 1), value=0)
            if dth.shape[0] != u.shape[0]:
                dth = dth.mean(0, keepdim=True).expand(u.shape[0], -1)
            if dth.shape[1] != u.shape[1]:
                dth = F.pad(dth, (0, u.shape[1] - dth.shape[1])) if dth.shape[1] < u.shape[1] else dth[:, :u.shape[1]]
            psr_out, new_PSR = self._psr(x_norm, dth, T, s_PSR)
            if psr_out.shape[-1] != d_ff:
                psr_out = F.pad(psr_out, (0, d_ff - psr_out.shape[-1])) if psr_out.shape[-1] < d_ff else psr_out[..., :d_ff]
            u = u + cfg.lambda_PSR * psr_out

        # 层级注入 + 物理门控
        pel = self.pos_layer_emb[self.idx].unsqueeze(0).unsqueeze(0).expand(B, S, -1)
        Z_l = self.decomp_mlp(torch.cat([Z_cog, pel], dim=-1))
        logit_a = (x_norm @ self.W_alpha_x.weight.T + self.W_alpha_x.bias) + \
                  (Z_l @ self.W_alpha_z.weight.T + self.W_alpha_z.bias).mean(1, keepdim=True)
        alpha_t = torch.sigmoid(logit_a)

        if theta is not None and cfg.lambda_T > 0 and theta.numel() > 0:
            th = theta.reshape(B, S, -1).mean(-1, keepdim=True)
            th_layer = self.Theta_phase[self.idx].unsqueeze(0).unsqueeze(0).mean(-1, keepdim=True)
            if th.shape[1] != x.shape[1]:
                th = F.pad(th, (0, 0, 0, x.shape[1] - th.shape[1])) if th.shape[1] < x.shape[1] else th[:, :x.shape[1]]
            cos_term = torch.cos(th - th_layer)
            if isinstance(T, torch.Tensor):
                T_bc = T.reshape(-1)[:B].view(-1, 1, 1) if T.numel() > 1 else T.view(1, 1, 1)
            else:
                T_bc = float(T)
            alpha_phys = torch.sigmoid(cos_term / (T_bc + 1e-6))
            alpha_final = (1 - cfg.lambda_T) * alpha_t + cfg.lambda_T * alpha_phys
        else:
            alpha_final = alpha_t

        scale = (Z_l @ self.W_scale.weight.T + self.W_scale.bias).mean(1, keepdim=True)
        bias = (Z_l @ self.W_bias.weight.T + self.W_bias.bias).mean(1, keepdim=True)
        scale = scale.expand(-1, u.shape[1], -1)
        bias = bias.expand(-1, u.shape[1], -1)

        u_final = u * (1 + scale) * alpha_final + bias * (1 - alpha_final)
        out = (torch.sigmoid(u_final) * v) @ self.W_proj.weight.T + self.W_proj.bias
        x = x + out

        return x, {"TSR": new_TSR, "PSR": new_PSR}


# ============================================================
# AetherMind V4 主模型
# ============================================================
class AetherMind4(nn.Module):
    def __init__(self, config: AetherMind4Config, arch_mode: str = 'train'):
        super().__init__()
        self.config = config
        self.arch_mode = arch_mode
        # 推理架构：使用相对核+窗口/低秩+扩散的推理版注意力（三方案结合 A+B+C）
        attn_cls = PheromoneThermoInference if arch_mode == 'inference' else PheromoneThermoAttention

        # Auto-match d_anchor to d_atom if not set
        if config.d_anchor <= 0:
            config.d_anchor = config.d_atom

        # 核心模块（继承3.6）
        self.encoder = EncoderV4(config, attn_cls=attn_cls)
        self.dual_domain = DualDomainSystem(config)
        self.physics = LangevinOscillator(config)
        self.memory = DualMemorySystem(config)
        self.metacog = MetaCognitiveGate(config)

        # V4: 解码器PGTA层（GLUBlockV4自带注意力）
        self.decoder_attns = nn.ModuleList([
            attn_cls(
                d_model=config.d_model, num_heads=config.n_heads,
                max_seq_len=config.max_seq_len, init_temp=config.init_temperature,
                whiten=config.pheromone_whiten, rho=config.pheromone_rho,
                beta=config.pheromone_beta, deposit=config.pheromone_deposit,
                tau_min=config.pheromone_tau_min, tau_max=config.pheromone_tau_max,
                dropout=config.dropout,
            ) for _ in range(config.n_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            GLUBlockV4(config, i, self.decoder_attns[i]) for i in range(config.n_layers)
        ])
        self.final_norm = nn.LayerNorm(config.d_model)

        # 指针门 + 计数器（继承3.6，修复：计数器由h驱动而非随机噪声）
        self.pointer = nn.Sequential(nn.Linear(config.d_model, config.d_model), nn.GELU(), nn.Linear(config.d_model, 1))
        self.counter_inp = nn.Linear(config.d_model, config.d_counter)
        self.counter_rnn = nn.GRUCell(config.d_counter, config.d_counter)
        self.counter_proj = nn.Linear(config.d_counter, 1)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.token_emb_scale = config.d_model ** -0.5

        # V4: 演化系统
        self.evolver = EvolvableWeightSystem(config)
        # 注册所有注意力层到演化系统
        for attn in self.encoder.attn_layers:
            self.evolver.register_attention(attn)
        for attn in self.decoder_attns:
            self.evolver.register_attention(attn)

        # 权重绑定
        self.lm_head.weight = self.encoder.token_emb.weight
        self.apply(self._init_weights)

        # 演化系统上一步自由能缓存（供evolution_step读取dF）
        self._last_F = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def _get_temperature(self, meta_T: torch.Tensor) -> float:
        return float(meta_T.mean().item())

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None,
                task_id: int = 0, t: float = 0.0, phase: str = "A") -> Dict[str, torch.Tensor]:
        cfg = self.config
        B, S = input_ids.shape
        device = input_ids.device

        # 1. Token嵌入 → 双域GMM激活
        token_emb = self.encoder.token_emb(input_ids)
        domain_out = self.dual_domain(token_emb)

        # 2. 编码器 (PGTA自注意力 + IB)
        enc_out = self.encoder(input_ids, domain_out)

        # 3. 朗之万物理层（相位同步+自由能）
        phys_out = self.physics({
            **domain_out,
            "z_IB_L": enc_out["z_IB_L"],
            "z_IB_P": enc_out["z_IB_P"],
        }, enc_out["pos_emb"])

        # 4. 双记忆
        mem_out = self.memory(enc_out["x"], domain_out, t)
        epi_mem = mem_out["epi_L"] + mem_out["epi_P"]

        # 5. 物理特征 → token级映射 (Z_phys: (B,N,D), atom_w: (B,S,N) -> (B,S,D))
        Z_phys_L_tokens = torch.einsum("bsn,bnd->bsd", domain_out["atom_w_L"], phys_out["L"]["Z_phys"])
        Z_phys_P_tokens = torch.einsum("bsn,bnd->bsd", domain_out["atom_w_P"], phys_out["P"]["Z_phys"])
        # 防御性pad/截断到S和D（langvin输出已是(B,N,d_model)，正常路径无需投影）
        if Z_phys_L_tokens.shape[1] < S:
            Z_phys_L_tokens = F.pad(Z_phys_L_tokens, (0, 0, 0, S - Z_phys_L_tokens.shape[1]))
            Z_phys_P_tokens = F.pad(Z_phys_P_tokens, (0, 0, 0, S - Z_phys_P_tokens.shape[1]))
        elif Z_phys_L_tokens.shape[1] > S:
            Z_phys_L_tokens = Z_phys_L_tokens[:, :S, :]
            Z_phys_P_tokens = Z_phys_P_tokens[:, :S, :]

        # 6. 元认知门控（域权重+温度+安全过滤）
        meta_out = self.metacog(
            enc_out["z_IB_L"], enc_out["z_IB_P"],
            Z_phys_L_tokens, Z_phys_P_tokens,
            epi_mem, task_id
        )

        # 7. 相位 → token级
        th_L_atoms = phys_out["L"]["theta"]
        th_P_atoms = phys_out["P"]["theta"]
        if th_L_atoms is not None and th_L_atoms.numel() > 0:
            th_L_tokens = (domain_out["atom_w_L"] * th_L_atoms.unsqueeze(1)).sum(-1)
            th_P_tokens = (domain_out["atom_w_P"] * th_P_atoms.unsqueeze(1)).sum(-1)
            if th_L_tokens.shape[1] < S:
                th_L_tokens = F.pad(th_L_tokens, (0, S - th_L_tokens.shape[1]))
                th_P_tokens = F.pad(th_P_tokens, (0, S - th_P_tokens.shape[1]))
            elif th_L_tokens.shape[1] > S:
                th_L_tokens = th_L_tokens[:, :S]
                th_P_tokens = th_P_tokens[:, :S]
        else:
            x_dtype = enc_out["x"].dtype
            th_L_tokens = torch.zeros(B, S, device=device, dtype=x_dtype)
            th_P_tokens = torch.zeros(B, S, device=device, dtype=x_dtype)
        theta_tokens = th_L_tokens + th_P_tokens

        # 8. 解码器（GLU+PGTA+TSR+PSR）
        T_tensor = meta_out["T"]  # (B,)或() tensor，保持可微
        # Bug3修复: 不再用set_temperature覆写可学习log_temp
        # 温度通过temperature_override参数传入attention forward
        # 推理时（phase="eval"/"generate"）才使用元认知温度覆盖
        # 训练时保留可学习温度，元认知温度作为辅助损失的调制信号

        h = enc_out["x"]
        s_TSR = None
        s_PSR = None
        use_ckpt = cfg.gradient_checkpointing and self.training
        
        def _dec_layer_block(h_in, layer_mod, Z_cog, T_t, theta_t, s_TSR_in, s_PSR_in):
            """单层decoder block（供checkpoint使用）"""
            return layer_mod(h_in, Z_cog, T_t, theta_t, s_TSR_in, s_PSR_in)
        
        for layer in self.decoder_layers:
            if use_ckpt:
                h, st = torch_checkpoint(
                    _dec_layer_block, h, layer, meta_out["Z_cog"], T_tensor, theta_tokens, s_TSR, s_PSR,
                    use_reentrant=False
                )
            else:
                h, st = layer(h, meta_out["Z_cog"], T_tensor, theta_tokens, s_TSR, s_PSR)
            s_TSR = st["TSR"]
            s_PSR = st["PSR"]
        h = self.final_norm(h)

        # 9. 计数器（由当前token隐状态驱动，不是随机噪声）
        h_cnt = torch.zeros(B, self.counter_rnn.hidden_size, device=device, dtype=h.dtype)
        cnt_outs = []
        for ti in range(S):
            inp = self.counter_inp(h[:, ti, :])
            h_cnt = self.counter_rnn(inp, h_cnt)
            cnt_outs.append(h_cnt)
        cnt_stack = torch.stack(cnt_outs, dim=1)
        h = h + self.counter_proj(cnt_stack).tanh() * 0.1

        # 10. LM Head (大词表下禁用pointer copy防止OOM)
        logits = h @ self.lm_head.weight.T * self.token_emb_scale
        # logits极端值clamp防止softmax溢出
        logits = logits.clamp(-1e4, 1e4)
        
        # 训练时(labels is not None)跳过softmax/prob计算，省~300MB显存
        # out_prob和pointer copy只在推理/生成时需要
        out_prob = None
        if labels is None:
            with torch.no_grad():
                T_val = float(T_tensor.mean().item()) if isinstance(T_tensor, torch.Tensor) else float(T_tensor)
            T_safe = max(T_val, 1e-3)
            out_prob = safe_softmax(logits.float(), T=T_safe).to(h.dtype)
            
            # Pointer copy仅在词表小时启用（vocab_size <= 65536），大词表下one-hot张量会OOM
            use_pointer = (cfg.vocab_size <= 65536) and input_ids is not None and input_ids.shape[1] == S
            if use_pointer:
                with torch.amp.autocast("cuda", enabled=False):
                    p_copy = torch.sigmoid(self.pointer(h)).squeeze(-1)
                    p_copy_gate = torch.sigmoid(p_copy).unsqueeze(-1).float()
                    src_onehot = F.one_hot(input_ids, num_classes=cfg.vocab_size).float()
                    copy_prob = src_onehot.cumsum(dim=1) / (torch.arange(1, S + 1, device=device).float().unsqueeze(0).unsqueeze(-1))
                    out_prob_f = out_prob.float()
                    out_prob_f = (1 - p_copy_gate) * out_prob_f + p_copy_gate * copy_prob
                    out_prob = out_prob_f.to(h.dtype)
        else:
            T_val = float(T_tensor.mean().item()) if isinstance(T_tensor, torch.Tensor) else 1.0

        # 11. 语言模型损失
        loss_LM = torch.tensor(0.0, device=device, dtype=h.dtype)
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            loss_LM = F.cross_entropy(shift_logits.view(-1, cfg.vocab_size), shift_labels.view(-1),
                                       ignore_index=cfg.pad_token_id).to(h.dtype)
            del shift_logits, shift_labels

        # 12. 总损失（加isfinite保护）
        loss_dict = {"loss_LM": loss_LM}
        total = loss_LM

        aux_losses = {
            "loss_IB": enc_out["loss_IB"] if cfg.lambda_IB > 0 else None,
            "loss_shap": enc_out.get("loss_H", torch.zeros((), device=device, dtype=h.dtype)) if cfg.lambda_shap > 0 else None,
            "loss_D": enc_out["loss_D"] if cfg.lambda_D > 0 else None,
            "loss_aff": enc_out["loss_aff"] if cfg.lambda_aff > 0 else None,
            "loss_module": domain_out["loss_module"] if cfg.lambda_module > 0 else None,
            "loss_anchor": domain_out["loss_anchor"] if cfg.lambda_align > 0 else None,
            "loss_phys": phys_out["loss_phys"] if cfg.lambda_phys > 0 else None,
            "loss_H": enc_out.get("loss_H", torch.zeros((), device=device, dtype=h.dtype)) if cfg.lambda_H > 0 else None,
        }
        for name, l in aux_losses.items():
            if l is not None and isinstance(l, torch.Tensor):
                if torch.isfinite(l).all():
                    loss_dict[name] = l
                    total = total + l
                else:
                    loss_dict[name] = torch.zeros((), device=device, dtype=h.dtype)

        # 总损失数值稳定性保护（用nan_to_num替换, 保留计算图grad_fn, NaN位置梯度为0, 避免backward崩）
        if not torch.isfinite(total):
            total = torch.nan_to_num(total, nan=10.0, posinf=10.0, neginf=10.0)
            total = total + loss_LM * 0.0  # 确保梯度流经loss_LM路径

        F_val = phys_out["L"]["F"] + phys_out["P"]["F"]
        dF_val = phys_out["L"]["dF"] + phys_out["P"]["dF"]
        # 保存自由能供evolution_step计算dF奖励
        self._last_F = F_val.detach() if torch.isfinite(F_val).all() else None

        loss_dict.update({
            "loss": total,
            "u_cog": meta_out["u_cog"].mean(),
            "T": T_tensor.mean(),
            "safety_score": meta_out["safety_score"].mean(),
            "F": F_val,
            "dF": dF_val,
        })
        
        # p_copy: 训练时或大词表下禁用pointer copy时返回0
        use_pointer = (cfg.vocab_size <= 65536) and (labels is None) and input_ids is not None and input_ids.shape[1] == S
        p_copy_val = torch.sigmoid(self.pointer(h)).squeeze(-1).mean() if use_pointer else torch.tensor(0.0, device=device)
        
        # probs: 训练时不返回（省显存）
        probs_out = out_prob if out_prob is not None else logits[:1, :1, :1]  # 占位tensor

        return {
            **loss_dict,
            "logits": logits, "probs": probs_out,
            "Z_cog": meta_out["Z_cog"], "p_copy": p_copy_val,
            "last_hidden": h,
        }

    @torch.no_grad()
    def evolution_step(self, step: int, phase: str = "A", loss_val: Optional[torch.Tensor] = None,
                       free_energy: Optional[torch.Tensor] = None):
        """训练步结束后调用：推进信息素演化+固化。
        free_energy: Phase C的自由能值(用于dF奖励);
        loss_val: Phase B的损失值(用于dloss奖励);
        Phase A直接跳过。
        """
        # 优先使用传入的F；否则用forward中保存的_last_F
        F_val = free_energy if free_energy is not None else self._last_F
        self.evolver(step, free_energy=F_val, phase=phase, loss_val=loss_val)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 256,
                 temperature: Optional[float] = None, top_k: int = 50, top_p: float = 0.9,
                 task_id: int = 0) -> torch.Tensor:
        self.eval()
        cfg = self.config
        B = input_ids.shape[0]
        device = input_ids.device
        generated = input_ids.clone()

        for step in range(max_new_tokens):
            cur_ids = generated[:, -cfg.max_seq_len:]
            out = self.forward(cur_ids, task_id=task_id, t=step, phase="C")
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

        self.train()
        return generated

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_evolution_stats(self) -> dict:
        return self.evolver.get_evolution_stats()

    def apply_pheromone_config(self):
        """把config的pheromone参数推送到各注意力层。

        注意: 注意力层在构造时缓存了 deposit/rho/beta/tau_min/tau_max,
        后续 config.set_phase_*() 修改的是 config 对象本身, 不会自动反映到注意力层。
        若不显式同步, Phase C/D 设定的强沉积/低蒸发/阈值都不会生效(仍用构造时的值),
        这会导致τ塌缩、固化空转。每次切阶段后必须调用本方法同步。
        """
        cfg = self.config
        for attn in self.evolver._attention_layers:
            attn.deposit = cfg.pheromone_deposit
            attn.rho = cfg.pheromone_rho
            attn.beta = cfg.pheromone_beta
            attn.tau_min = cfg.pheromone_tau_min
            attn.tau_max = cfg.pheromone_tau_max
