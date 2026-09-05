"""
可演化双权重系统 (Evolvable Dual-Weight System)
=================================================
管理V4模型的核心演化机制：
  - 快权重W: Transformer主干，梯度下降学习（BP），冻结后不再改变
  - 慢权重τ: 信息素路径网络，物理演化（沉积+蒸发+固化），在线更新
  - 固化(LTP): τ超阈值时写入consolidated永久偏置，经验长进权重

三阶段：
  Phase A: W可学,τ=均匀（纯BP预训练）
  Phase B: W弹性微调,τ开始沉积（混合学习）
  Phase C/D: W冻结,τ持续演化+固化（在线自演化）
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List


class EvolvableWeightSystem(nn.Module):
    """可演化双权重控制器 — 管理整个模型的信息素生命周期"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._attention_layers: List[nn.Module] = []
        self._global_step = 0
        self._consolidation_count = 0
        # 跨步状态，用于计算自由能奖励dF和损失改进dloss
        self.register_buffer("_F_prev", torch.tensor(0.0))
        self._F_initialized = False
        self._loss_prev = None
        self._loss_initialized = False

    def register_attention(self, attn_layer: nn.Module):
        """注册注意力层（用于统一管理信息素）"""
        self._attention_layers.append(attn_layer)

    def reset_all_pheromones(self):
        """清空所有信息素（session切换）"""
        for attn in self._attention_layers:
            if hasattr(attn, 'reset_pheromone'):
                attn.reset_pheromone()

    def set_global_temperature(self, T: float):
        """全局设置所有注意力层的温度"""
        for attn in self._attention_layers:
            if hasattr(attn, 'set_temperature'):
                attn.set_temperature(T)

    def step_pheromones(self, reward: Optional[torch.Tensor] = None, credit_mode: str = "soft_center"):
        """对所有注册的注意力层执行信息素沉积+蒸发"""
        for attn in self._attention_layers:
            if hasattr(attn, 'step_pheromone'):
                attn.step_pheromone(reward)

    def consolidate_all(self, threshold: Optional[float] = None,
                        lam: Optional[float] = None,
                        gamma: Optional[float] = None,
                        max_cons: Optional[int] = None,
                        top_frac: Optional[float] = None):
        """全局固化(LTP)：将所有层τ中超过阈值的路径写入长期权重"""
        th = threshold or self.config.consolidate_threshold
        lm = lam or self.config.consolidate_lam
        gm = gamma or self.config.consolidate_gamma
        mc = max_cons or self.config.max_consolidations
        tf = top_frac if top_frac is not None else self.config.consolidate_top_frac
        for attn in self._attention_layers:
            if hasattr(attn, 'consolidate'):
                attn.consolidate(th, lm, gm, mc, top_frac=tf)
        self._consolidation_count += 1

    def get_evolution_stats(self) -> Dict[str, float]:
        """获取演化状态统计"""
        stats = {}
        total_tau_conc = 0.0
        total_cons_mass = 0.0
        n = 0
        for attn in self._attention_layers:
            if hasattr(attn, 'tau'):
                tau = attn.tau
                total_tau_conc += (tau.max() / (tau.mean() + 1e-9)).item()
                n += 1
            if hasattr(attn, 'consolidated'):
                total_cons_mass += attn.consolidated.abs().sum().item()
        if n > 0:
            stats["tau_concentration"] = total_tau_conc / n
            stats["consolidation_mass"] = total_cons_mass
        stats["global_step"] = self._global_step
        stats["consolidation_rounds"] = self._consolidation_count
        return stats

    def compute_free_energy_reward(self, F_current: torch.Tensor, F_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        """用自由能下降量作为信用奖励信号。
        dF < 0（自由能降低）→ gate→1.0 满额沉积
        dF > 0（自由能升高）→ gate→0.1 接近不沉积

        修复: 原版零中心奖励(±1, 均值≈0)导致净沉积≈0, τ纯蒸发衰减到下限。
        改为门控形式: gate = 0.55 + 0.45*signal ∈ [0.1, 1.0], 均值≈0.55的正基线,
        保证信息素有净积累, 信号好坏只调制沉积强度（保留奖励门控语义）。
        """
        if F_prev is None:
            return torch.tensor(0.55)
        dF = F_current - F_prev
        signal = torch.clamp(-dF / (abs(dF.detach()) + 0.1), -1.0, 1.0)
        return (0.55 + 0.45 * signal).clamp(0.1, 1.0)

    def forward(self, step: int, free_energy: Optional[torch.Tensor] = None,
                phase: str = "A", loss_val: Optional[torch.Tensor] = None):
        """在训练步结束后调用：推进演化状态"""
        self._global_step = step

        # Phase A 不演化（纯BP）
        if phase == "A":
            return

        # 计算奖励信号
        reward = None
        if phase in ("C", "D") and free_energy is not None:
            # Phase C/D: 用自由能下降量作为信用信号
            if self._F_initialized:
                reward = self.compute_free_energy_reward(free_energy, self._F_prev)
            # 更新F_prev
            with torch.no_grad():
                self._F_prev.copy_(free_energy.detach().mean())
            self._F_initialized = True
        elif loss_val is not None:
            # Phase B: 用loss下降量作为奖励（改进越大沉积越满）
            lv = loss_val.detach().mean() if isinstance(loss_val, torch.Tensor) else torch.tensor(float(loss_val))
            if self._loss_initialized and self._loss_prev is not None:
                dloss = self._loss_prev - lv  # 正 = loss下降 = 改进
                signal = torch.clamp(dloss / (dloss.abs() + 0.1), -1.0, 1.0)
                reward = (0.55 + 0.45 * signal).clamp(0.1, 1.0)
            self._loss_prev = lv
            self._loss_initialized = True
        else:
            reward = None

        # 信息素更新
        self.step_pheromones(reward, self.config.pheromone_credit_mode)

        # 定期固化（warmup后开始, 避免τ尚未成型时把噪声写入长期权重）
        interval = self.config.consolidate_interval
        warmup = getattr(self.config, "consolidate_warmup", 0)
        if phase in ("C", "D") and step >= warmup and step > 0 and step % interval == 0:
            self.consolidate_all()
