"""
信息素调制热力学注意力 (Pheromone-Guided Thermodynamic Attention, PGTA)
=====================================================================
V4.0 核心注意力机制。融合三层：
  1. 热力学：A = softmax(-E_eff/T), E_eff = -QK/sqrt(d) - beta*T*log(tau)
  2. 信息论：熵H作为正则目标, 自由能F = <E> - T*H
  3. 群体智能：tau 跨步累积（沉积+蒸发），形成持久路径记忆

公式：A ∝ exp(-E/T) · τ^β
时间尺度：快(T/温度,每步) + 慢(τ/信息素,跨样本) + 结构(权重/固化,跨session)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class PheromoneThermoAttention(nn.Module):
    """信息素调制的热力学注意力 — V4的统一注意力层"""

    def __init__(self, d_model: int, num_heads: int = 8, max_seq_len: int = 1024,
                 init_temp: float = 1.0, whiten: bool = True,
                 rho: float = 0.05, beta: float = 1.0, deposit: float = 0.05,
                 tau_min: float = 1e-2, tau_max: float = 5.0,
                 target_entropy_ratio: float = 0.3,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.whiten = whiten
        self.rho = rho
        self.beta = beta
        self.deposit = deposit
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.target_entropy_ratio = target_entropy_ratio
        self._last_A = None
        self._last_E = None
        self._last_E_components = None  # 能量分量追踪：E_qk, E_tau, E_cons, E_eff
        self._step_count = 0

        # Q/K/V/O 投影
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

        # 可学习温度 (log参数化保证正值)
        self.log_temp = nn.Parameter(torch.log(torch.tensor(init_temp)))

        # 信息素缓冲：(num_heads, max_seq, max_seq), 跨forward持久保存
        self.register_buffer("tau", torch.ones(num_heads, max_seq_len, max_seq_len))

        # 固化偏置（LTP长期权重记忆）：(num_heads, max_seq, max_seq)
        self.register_buffer("consolidated", torch.zeros(num_heads, max_seq_len, max_seq_len))
        self.register_buffer("consolidation_count", torch.tensor(0))

        # 能量白化运行统计量
        self.register_buffer("energy_mean", torch.tensor(0.0))
        self.register_buffer("energy_std", torch.tensor(1.0))
        self.register_buffer("stats_momentum", torch.tensor(0.99))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temp).clamp(min=1e-2, max=1e2)

    def set_temperature(self, T: float):
        """外部设置温度（用于元认知门控）"""
        with torch.no_grad():
            self.log_temp.fill_(math.log(max(T, 1e-2)))

    def reset_pheromone(self):
        """清空信息素（session切换/推理前重置工作记忆）"""
        with torch.no_grad():
            self.tau.fill_(1.0)

    def consolidate(self, threshold: float = 1.5, lam: float = 0.1, gamma: float = 0.5, max_cons: int = 4096,
                    top_frac: Optional[float] = None):
        """固化(LTP)：将τ中超过阈值的稳定路径写入长期权重consolidated。
        τ局部衰减，短期记忆让位给长期权重。

        threshold模式: τ > threshold 绝对阈值判定（要求τ量级健康, 均值≈1）
        top_frac模式: 固化最强的top_frac比例路径（分位数自适应, 不依赖绝对量级）
        两者同时给出时top_frac优先。
        """
        if self.consolidation_count.item() >= max_cons:
            return
        with torch.no_grad():
            if top_frac is not None and top_frac > 0:
                k = max(1, int(self.tau.numel() * top_frac))
                thresh = self.tau.flatten().kthvalue(max(1, self.tau.numel() - k + 1)).values
                mask = self.tau >= thresh
            else:
                mask = self.tau > threshold
                if not mask.any():
                    # 绝对阈值无人达标(流式训练中τ常塌缩到下限, 永远够不到1.2)
                    # -> 退化为分位数固化, 固化当前最强top_frac比例路径, 避免固化空转
                    k = max(1, int(self.tau.numel() * 0.002))
                    thresh = self.tau.flatten().kthvalue(max(1, self.tau.numel() - k + 1)).values
                    mask = self.tau >= thresh
            if mask.any():
                n_new = mask.sum().item()
                # 写入长期权重
                self.consolidated[mask] += lam * (self.tau[mask] - threshold).clamp(min=0.05)
                # τ局部衰减
                self.tau[mask] *= (1 - gamma)
                self.consolidation_count += int(min(n_new, max_cons - self.consolidation_count.item()))

    def _credit_signal(self, loss_val: Optional[torch.Tensor], pred_correct: Optional[torch.Tensor] = None,
                       mode: str = "soft_center") -> torch.Tensor:
        """计算信用信号r（奖励门控的沉积量）。
        hard: r = 0/1 (预测对/错)
        soft: r = P(正确) ∈ (0,1)
        soft_center: r = P(正确) - 0.5 (零中心化, 对加错撤)
        free_energy: r = -dF (自由能下降=正奖励)
        """
        if pred_correct is not None:
            if mode == "hard":
                return pred_correct.float()
            elif mode == "soft":
                return pred_correct.float().clamp(0, 1)
            elif mode == "soft_center":
                return (pred_correct.float() - 0.5) * 2.0
        # 默认：自由能下降作为奖励
        if loss_val is not None:
            return torch.clamp(-loss_val, -1.0, 1.0)
        return torch.tensor(1.0)

    def step_pheromone(self, reward: Optional[torch.Tensor] = None):
        """信息素动力学：蒸发+沉积（非梯度，物理演化）

        量级修复: A是softmax分布(行和=1), 元素量级≈1/S。原版直接用A沉积,
        稳态τ≈deposit/(rho*S)≈0.01, 被clamp到tau_min下限, 固化阈值1.2永远无法达到。
        沉积量乘以S补偿后稳态τ≈deposit*gate*C/rho (C=注意力集中度, 均匀=1),
        恢复到设计量级(均值≈1), 固化阈值才有意义。
        reward为门控值(0~1): 均值≈0.5的净正沉积, 信号好满额, 信号差接近不沉积。
        """
        if self._last_A is None:
            return
        A = self._last_A.detach()  # (B, h, S, S)
        B, h, S, _ = A.shape

        with torch.no_grad():
            # 蒸发：所有边乘以(1-rho)，用进废退
            self.tau[:, :S, :S].mul_(1 - self.rho)

            # 沉积：gate * A * S（S补偿softmax归一化，恢复τ设计量级）
            if reward is not None:
                if reward.dim() == 0:
                    r = float(reward.item())
                    delta = self.deposit * r * A.mean(0)[:, :S, :S] * S
                elif reward.dim() == 1:
                    r = reward.detach().view(-1, 1, 1, 1)
                    delta = self.deposit * (r * A).mean(0)[:, :S, :S] * S
                else:
                    delta = self.deposit * reward.detach().mean(0)[:, :S, :S] * S
            else:
                # 无奖励时按注意力用量均匀沉积（gate=1）
                delta = self.deposit * A.mean(0)[:, :S, :S] * S

            self.tau[:, :S, :S].add_(delta)
            self.tau.clamp_(self.tau_min, self.tau_max)

        self._step_count += 1

    # ------------------------------------------------------------------
    # 因果掩码：语言建模必须是自回归的（位置 i 只能看到 0..i）
    # 旧版无掩码 → 训练时模型"偷看"未来token，推理时分布错位 → 生成乱码
    # ------------------------------------------------------------------
    def _causal_mask(self, S: int, device) -> torch.Tensor:
        rng = torch.arange(S, device=device)
        return rng[None, :] > rng[:, None]  # (S,S) True=未来位置

    def forward(self, x: torch.Tensor, extra_mask: Optional[torch.Tensor] = None,
                temperature_override=None,
                update_pheromone: bool = True,
                causal: bool = True) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, S, D = x.shape
        h = self.num_heads
        hd = self.head_dim

        # Q/K/V 投影
        q = self.Wq(x).view(B, S, h, hd).transpose(1, 2)  # (B, h, S, hd)
        k = self.Wk(x).view(B, S, h, hd).transpose(1, 2)
        v = self.Wv(x).view(B, S, h, hd).transpose(1, 2)

        # 能量 E = -Q·K/sqrt(d)（越相似能量越低）
        sim = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(hd)
        E = -sim

        # 能量白化（统计辅助：让T的物理意义与数据尺度解耦）
        if self.whiten and self.training:
            with torch.no_grad():
                em = E.mean()
                es = E.std(unbiased=False) + 1e-5
                self.energy_mean.mul_(self.stats_momentum).add_(em * (1 - self.stats_momentum))
                self.energy_std.mul_(self.stats_momentum).add_(es * (1 - self.stats_momentum))
                # 防止stats无约束膨胀（bf16下大数相减丢精度，实测encoder std可到1e5）
                self.energy_mean.clamp_(-1e3, 1e3)
                self.energy_std.clamp_(1e-3, 1e4)
            E = (E - self.energy_mean) / (self.energy_std + 1e-5)

        # 温度 —— 支持 float 或 tensor（元认知门可学习温度）
        if temperature_override is None:
            T = self.temperature
        elif isinstance(temperature_override, torch.Tensor):
            T = temperature_override
        else:
            T = torch.tensor(float(temperature_override), device=x.device, dtype=x.dtype)
        T = T.clamp(min=1e-2, max=1e2)
        # 将T reshape到可广播到 (B, h, S, S) 的形状: 支持 () / (B,) / (B,1) / (B,1,1,1)
        if T.dim() == 0:
            T_bc = T
        elif T.dim() == 1:
            T_bc = T.view(-1, 1, 1, 1)
        elif T.dim() == 2:
            T_bc = T.unsqueeze(1).unsqueeze(1)
        else:
            T_bc = T.view(B, 1, 1, 1) if T.numel() == B else T

        # 信息素调制：有效能量 E_eff = E - beta*T*log(tau) - T*consolidated
        tau_slice = self.tau[:h, :S, :S].unsqueeze(0)  # (1, h, S, S)
        log_tau = torch.log(tau_slice.clamp(min=self.tau_min))
        E_tau = -self.beta * T_bc * log_tau  # 信息素项能量
        E_eff = E + E_tau

        # 固化偏置（长期权重增强）
        cons_slice = self.consolidated[:h, :S, :S].unsqueeze(0)
        E_cons = -T_bc * cons_slice  # 固化项能量
        E_eff = E_eff + E_cons

        # 玻尔兹曼注意力分布
        # 关键修复：掩码用 -inf 填在 softmax 输入上（而非对 E_eff 填 +inf）。
        # 若 E_eff 含 +inf，-E_eff/T 反传时 DivBackward0 对 T 求导 = E_eff/T² 含 ±inf
        # → 温度梯度 NaN → 连锁污染 metacog/dual_domain 等全部参数梯度。
        scores = -E_eff / T_bc
        if causal:
            cmask = self._causal_mask(S, x.device)
            scores = scores.masked_fill(cmask.view(1, 1, S, S), float('-inf'))
        if extra_mask is not None:
            scores = scores.masked_fill(extra_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        A = F.softmax(scores, dim=-1)  # (B, h, S, S)
        A = self.attn_drop(A)

        # 输出 = 期望 Σ A·V
        out = torch.einsum("bhij,bhjd->bhid", A, v)
        out = out.transpose(1, 2).reshape(B, S, D)
        out = self.Wo(out)

        # 信息论量（E_eff 可能含掩码 +inf，统计前 clamp 防止 0*inf=NaN）
        entropy = -(A * (A + 1e-8).log()).sum(-1).mean()
        _E_eff_stats = E_eff.detach().clamp(-1e4, 1e4)
        free_energy = (A * _E_eff_stats).sum(-1).mean() - T.detach().mean() * entropy.detach()

        # 方差（不确定性）
        out_sq = torch.einsum("bhij,bhjd->bhid", A, v * v)
        variance = (out_sq - out.view(B, S, h, hd).transpose(1, 2) ** 2).mean()

        # 缓存本步注意力，供信息素沉积
        self._last_A = A.detach()
        self._last_E = E_eff.detach()
        # 缓存能量分量，供验证钩子分析物理占比
        self._last_E_components = {
            "E_qk": E.detach(),
            "E_tau": E_tau.detach(),
            "E_cons": E_cons.detach(),
            "E_eff": E_eff.detach(),
        }

        stats = {
            "entropy": entropy,
            "free_energy": free_energy,
            "temperature": T.mean() if isinstance(T, torch.Tensor) else T,
            "variance": variance,
            "attention": A.detach(),
            "tau_concentration": (self.tau[:h, :S, :S].max() / (self.tau[:h, :S, :S].mean() + 1e-9)).detach(),
        }
        return out, stats
