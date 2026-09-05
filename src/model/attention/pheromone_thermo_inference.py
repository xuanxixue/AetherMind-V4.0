"""
推理版信息素热力学注意力 (PheromoneThermoInference)
====================================================
三方案结合（A+B+C）的推理实现，与训练版 PheromoneThermoAttention 接口完全兼容，
但存储与计算采用物理化的相对坐标：

  方案A（表示）: τ / consolidated 从绝对坐标 (h,S,S) 矩阵 退化为 相对偏移核 (h, 2k-1)
                —— O(S²) 存储 → O(1)，且恢复平移对称性（上下文滑动不再错位）
  方案B（计算）: 默认"窗口掩码"(只算 |i-j|≤k，物理局域性) 或可选"低秩特征映射"(O(S·r))
  方案C（演化）: step_pheromone 增加扩散项 D·Lap(τ_rel)，长程结构保留、短程噪声抹平

核心权重 Wq/Wk/Wv/Wo 与训练版完全一致，可直接 load_state_dict 复用；
τ/C 由 convert_train_to_inference.py 做绝对→相对压缩迁移。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class PheromoneThermoInference(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, max_seq_len: int = 1024,
                 rel_k: int = 64, init_temp: float = 1.0, rho: float = 0.01,
                 beta: float = 1.0, deposit: float = 0.06,
                 tau_min: float = 1e-2, tau_max: float = 5.0,
                 dropout: float = 0.0, whiten: bool = False,
                 target_entropy_ratio: float = 0.3,
                 use_low_rank: bool = False, diffusion_D: float = 0.0,
                 **kwargs):  # kwargs 吸收训练版可能传入的额外参数，保证签名兼容
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.rel_k = rel_k
        self.rho = rho
        self.beta = beta
        self.deposit = deposit
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.whiten = whiten
        self.target_entropy_ratio = target_entropy_ratio
        self.use_low_rank = use_low_rank
        self.diffusion_D = diffusion_D
        self._last_A = None
        self._step_count = 0

        # 核心投影（与训练版完全一致，权重直接复用）
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

        # 可学习温度（与训练版一致）
        self.log_temp = nn.Parameter(torch.log(torch.tensor(init_temp)))

        # 方案A：相对偏移核（O(1) 存储，平移不变）
        # 桶索引 d = clip(i-j, -k+1, k-1) + k - 1 ∈ [0, 2k-2]
        self.register_buffer("tau_rel", torch.ones(num_heads, 2 * rel_k - 1))
        self.register_buffer("consolidated_rel", torch.zeros(num_heads, 2 * rel_k - 1))
        self.register_buffer("consolidation_count", torch.tensor(0))

        # 能量白化运行统计（与训练版一致，推理时默认不更新）
        self.register_buffer("energy_mean", torch.tensor(0.0))
        self.register_buffer("energy_std", torch.tensor(1.0))
        self.register_buffer("stats_momentum", torch.tensor(0.99))

    # ------------------------------------------------------------------
    # 兼容训练版的接口
    # ------------------------------------------------------------------
    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temp).clamp(min=1e-2, max=1e2)

    @property
    def tau(self):
        """兼容别名：训练版 evolver.get_evolution_stats 会访问 attn.tau"""
        return self.tau_rel

    @property
    def consolidated(self):
        """兼容别名：训练版 evolver 会访问 attn.consolidated"""
        return self.consolidated_rel

    def set_temperature(self, T: float):
        with torch.no_grad():
            self.log_temp.fill_(math.log(max(T, 1e-2)))

    def reset_pheromone(self):
        """清空信息素（仅清 τ_rel，长期固化 consolidated_rel 保留）"""
        with torch.no_grad():
            self.tau_rel.fill_(1.0)

    # ------------------------------------------------------------------
    # 相对坐标工具
    # ------------------------------------------------------------------
    def _rel_index(self, S: int, device) -> torch.Tensor:
        """构造 (S,S) 的相对桶索引：[i][j] = clip(i-j)+k-1 ∈ [0,2k-2]"""
        rng = torch.arange(S, device=device)
        rel = rng[:, None] - rng[None, :]           # (S,S)
        rel = rel.clamp(-self.rel_k + 1, self.rel_k - 1)
        return (rel + self.rel_k - 1).long()        # (S,S) int

    def _window_mask(self, S: int, device) -> torch.Tensor:
        """|i-j| > rel_k-1 或 j > i（未来）的位置置 True。
        窗口=物理局域性；因果=自回归正确性（旧版缺失导致训练偷看未来）。"""
        rng = torch.arange(S, device=device)
        rel = rng[:, None] - rng[None, :]
        return (rel.abs() > (self.rel_k - 1)) | (rel < 0)

    def _rel_bias(self, S: int, device):
        """由相对核 gather 成 (h,S,S) 的相对偏置（平移不变）"""
        idx = self._rel_index(S, device)            # (S,S)
        tau_b = self.tau_rel[:, idx]                # (h,S,S)
        cons_b = self.consolidated_rel[:, idx]      # (h,S,S)
        return tau_b, cons_b

    # ------------------------------------------------------------------
    # 方案B：低秩特征映射注意力（O(S·r)，实验性，不含相对偏置）
    # ------------------------------------------------------------------
    def _linear_attn(self, q, k, v):
        B, h, S, hd = q.shape
        phi_q = F.relu(q)                           # (B,h,S,hd)
        phi_k = F.relu(k)
        KV = torch.einsum("bhsd,bhse->bhde", phi_k, v)                 # (B,h,hd,hd)
        denom = torch.einsum("bhsd,bhd->bhs", phi_q, phi_k.sum(dim=2))  # (B,h,S)
        out = torch.einsum("bhsd,bhde->bhse", phi_q, KV)
        out = out / (denom.unsqueeze(-1) + 1e-9)
        return out

    # ------------------------------------------------------------------
    # 方案C：信息素扩散演化（在 step_pheromone 内）
    # ------------------------------------------------------------------
    def consolidate(self, threshold: float = 1.5, lam: float = 0.1, gamma: float = 0.5,
                    max_cons: int = 4096, top_frac: Optional[float] = None):
        """固化(LTP)：把 τ_rel 超阈值/最高分位的相对路径写入 consolidated_rel。"""
        if self.consolidation_count.item() >= max_cons:
            return
        with torch.no_grad():
            if top_frac is not None and top_frac > 0:
                k = max(1, int(self.tau_rel.numel() * top_frac))
                thresh = self.tau_rel.flatten().kthvalue(max(1, self.tau_rel.numel() - k + 1)).values
                mask = self.tau_rel >= thresh
            else:
                mask = self.tau_rel > threshold
                if not mask.any():
                    k = max(1, int(self.tau_rel.numel() * 0.002))
                    thresh = self.tau_rel.flatten().kthvalue(max(1, self.tau_rel.numel() - k + 1)).values
                    mask = self.tau_rel >= thresh
            if mask.any():
                self.consolidated_rel[mask] += lam * (self.tau_rel[mask] - threshold).clamp(min=0.05)
                self.tau_rel[mask] *= (1 - gamma)
                self.consolidation_count += int(min(mask.sum().item(), max_cons - self.consolidation_count.item()))

    def step_pheromone(self, reward: Optional[torch.Tensor] = None):
        """信息素演化：蒸发 + 沉积 + 扩散（方案C）。

        将注意力模式 A 按相对距离聚合到 τ_rel，再叠加一维扩散（二阶差分），
        使长程结构保留、短程噪声被抹平。
        """
        if self._last_A is None:
            return
        A = self._last_A.detach()                    # (B,h,S,S)
        B, h, S, _ = A.shape
        idx = self._rel_index(S, A.device)           # (S,S)
        with torch.no_grad():
            # 蒸发
            self.tau_rel.mul_(1 - self.rho)

            # 沉积：按相对距离聚合注意力用量
            A_mean = A.mean(0)                       # (h,S,S)
            src = A_mean.reshape(h, -1)              # (h, S*S)
            delta = torch.zeros(h, 2 * self.rel_k - 1, device=A.device, dtype=A.dtype)
            flat_idx = idx.reshape(-1).unsqueeze(0).expand(h, -1)
            delta.scatter_add_(1, flat_idx, src)
            # 归一化（每桶元素数）
            cnt = torch.zeros(2 * self.rel_k - 1, device=A.device, dtype=A.dtype)
            cnt.scatter_add_(0, idx.reshape(-1), torch.ones(S * S, device=A.device, dtype=A.dtype))
            delta = delta / (cnt.unsqueeze(0) + 1e-9)
            # 沉积（S 补偿 softmax 归一化）
            if reward is not None:
                r = float(reward.item()) if reward.dim() == 0 else float(reward.detach().mean().item())
            else:
                r = 1.0
            self.tau_rel.add_(self.deposit * r * delta * S)

            # 方案C：扩散（一维拉普拉斯，边界零流）
            if self.diffusion_D > 0 and self.tau_rel.shape[1] >= 3:
                lap = self.tau_rel[:, 2:] - 2 * self.tau_rel[:, 1:-1] + self.tau_rel[:, :-2]
                self.tau_rel[:, 1:-1] += self.diffusion_D * lap

            # 边界保护
            self.tau_rel.clamp_(self.tau_min, self.tau_max)
        self._step_count += 1

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, extra_mask: Optional[torch.Tensor] = None,
                temperature_override=None,
                update_pheromone: bool = True) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, S, D = x.shape
        h = self.num_heads
        hd = self.head_dim

        q = self.Wq(x).view(B, S, h, hd).transpose(1, 2)   # (B,h,S,hd)
        k = self.Wk(x).view(B, S, h, hd).transpose(1, 2)
        v = self.Wv(x).view(B, S, h, hd).transpose(1, 2)

        # 温度（兼容 float / tensor）
        if temperature_override is None:
            T = self.temperature
        elif isinstance(temperature_override, torch.Tensor):
            T = temperature_override
        else:
            T = torch.tensor(float(temperature_override), device=x.device, dtype=x.dtype)
        T = T.clamp(min=1e-2, max=1e2)
        if T.dim() == 0:
            T_bc = T
        elif T.dim() == 1:
            T_bc = T.view(-1, 1, 1, 1)
        elif T.dim() == 2:
            T_bc = T.unsqueeze(1).unsqueeze(1)
        else:
            T_bc = T.view(B, 1, 1, 1) if T.numel() == B else T

        if self.use_low_rank:
            # 方案B（低秩）：O(S·r)，无相对偏置（实验性，供长上下文快速推理）
            hd_out = self._linear_attn(q, k, v)
            A = None
            entropy = torch.tensor(math.log(S), device=x.device, dtype=x.dtype)
            free_energy = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        else:
            # 精确 softmax + 相对偏置（方案A）+ 窗口掩码（方案B的局域版）
            sim = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(hd)
            E = -sim                                       # (B,h,S,S)

            if self.whiten and self.training:
                with torch.no_grad():
                    em = E.mean()
                    es = E.std(unbiased=False) + 1e-5
                    self.energy_mean.mul_(self.stats_momentum).add_(em * (1 - self.stats_momentum))
                    self.energy_std.mul_(self.stats_momentum).add_(es * (1 - self.stats_momentum))
                E = (E - self.energy_mean) / (self.energy_std + 1e-5)

            # 相对偏置（平移不变，O(k) 存储）
            tau_b, cons_b = self._rel_bias(S, x.device)    # (h,S,S)
            # τ 归一化：log(1+τ) 饱和对数（τ→0 时→0 而非 -∞，τ 大时对数慢增）
            #            tanh(C) 把固化偏置压到 [-1,1]，防高浓度偏置淹没内容项 QK
            E_eff = E - self.beta * T_bc * torch.log1p(tau_b.unsqueeze(0)) \
                       - T_bc * torch.tanh(cons_b.unsqueeze(0))

            # 窗口+因果掩码（局域性 + 自回归正确性：只保留 |i-j| ≤ rel_k-1 且 j ≤ i）
            wmask = self._window_mask(S, x.device)         # (S,S)
            E_eff = E_eff.masked_fill(wmask.view(1, 1, S, S), float('inf'))

            if extra_mask is not None:
                E_eff = E_eff.masked_fill(extra_mask.unsqueeze(1).unsqueeze(2), float('inf'))

            A = F.softmax(-E_eff / T_bc, dim=-1)           # (B,h,S,S)

            # 信息论量
            entropy = -(A * (A + 1e-8).log()).sum(-1).mean()
            free_energy = (A * E_eff.detach()).sum(-1).mean() - T.detach().mean() * entropy.detach()

            # 输出
            hd_out = torch.einsum("bhij,bhjd->bhid", A, v)

        out = hd_out.transpose(1, 2).reshape(B, S, D)
        out = self.Wo(out)

        # 缓存供沉积
        if A is not None:
            self._last_A = A.detach()
        else:
            self._last_A = None

        stats = {
            "entropy": entropy,
            "free_energy": free_energy,
            "temperature": T.mean() if isinstance(T, torch.Tensor) else T,
            "variance": torch.tensor(0.0, device=x.device, dtype=x.dtype),
            "attention": A.detach() if A is not None else torch.zeros(B, h, S, S, device=x.device),
            "tau_concentration": (self.tau_rel.max() / (self.tau_rel.mean() + 1e-9)).detach(),
        }
        return out, stats