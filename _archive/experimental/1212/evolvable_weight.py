"""
evolvable_weight.py —— 权重迭代 MVP（模型 vs Agent 的分水岭）
================================================================
演示：信息素 τ 从"逐样本激活"升级为"可固化的演化权重"。

核心机制（固化 / LTP 长时程增强）：
  consolidated[j] += λ        当 τ[CLS→j] 持续超过阈值（写入主干权重，跨 session 保持）
  τ[CLS→j] *= (1-γ)           局部衰减（短期记忆让位给长期权重）

对照实验（冻结主干 W，路由能力唯一载体 = τ，以排除梯度学习的干扰）：
  M1 演化模型：学路由 → 固化 → 清空工作记忆 τ → 路由还在（权重记住了）
  M2 对照模型：学路由 → 不固化 → 清空工作记忆 τ → 路由丢失（Agent 式失忆）

这正是一句话的区别：
  权重 = 持久的、改变模型本身 = 模型；
  激活 = 逐样本、用完即弃 = Agent 外挂记忆。

运行：python evolvable_weight.py
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pheromone_thermo_transformer import (  # noqa: E402
    PheromoneThermoAttention,
    MiniModel,
    make_data,
)


class EvolvablePheromoneAttention(PheromoneThermoAttention):
    """在 PGTT 信息素注意力之上，加'固化'：把高频稳定路径写入主干权重。"""

    def __init__(self, *a, consolidate_threshold=1.0, **kw):
        super().__init__(*a, **kw)
        N = self.tau.shape[-1]
        # consolidated[h, j]：已固化进权重的 CLS→列j 永久 logit 偏置（权重级记忆）
        self.register_buffer("consolidated", torch.zeros(self.num_heads, N))
        self.consolidate_threshold = consolidate_threshold

    def forward(self, x, update_pheromone=True, reward=1.0):
        B, N, D = x.shape
        h = self.num_heads
        q = self.Wq(x).view(B, N, h, self.head_dim).transpose(1, 2)
        k = self.Wk(x).view(B, N, h, self.head_dim).transpose(1, 2)
        v = self.Wv(x).view(B, N, h, self.head_dim).transpose(1, 2)
        sim = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.head_dim)
        E = -sim
        if self.whiten:
            em = E.mean(dim=(0, 2, 3), keepdim=True)
            es = E.std(dim=(0, 2, 3), unbiased=False, keepdim=True) + 1e-5
            E = (E - em) / es
        T = self.temperature
        log_tau = torch.log(self.tau.clamp(min=self.tau_min))
        Eeff = E - self.beta * T * log_tau
        # ★ 固化偏置：CLS(行0) 对每列 j 的永久 logit 增强 → 注意力 × exp(consolidated)
        Eeff[:, :, 0, :] = Eeff[:, :, 0, :] - T * self.consolidated[None, :, :]
        A = torch.softmax(-Eeff / T, dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", A, v).transpose(1, 2).reshape(B, N, D)
        out = self.Wo(out)
        entropy = -(A * (A + 1e-8).log()).sum(-1).mean()
        free_energy = (A * Eeff).sum(-1).mean() - T * entropy
        self._last_A = A.detach()
        return out, {"entropy": entropy, "free_energy": free_energy,
                     "temperature": T, "tau": self.tau,
                     "consolidated": self.consolidated}

    def consolidate(self, threshold=None, lam=4.0, gamma=0.5):
        """把 τ 中 CLS 行超过阈值的路由，固化进权重；τ 局部衰减（短期让位长期）。"""
        threshold = threshold if threshold is not None else self.consolidate_threshold
        with torch.no_grad():
            for hh in range(self.num_heads):
                row = self.tau[hh, 0]
                for j in range(row.shape[0]):
                    if row[j].item() > threshold:
                        self.consolidated[hh, j] += lam
                        self.tau[hh, 0, j] *= (1 - gamma)
        return self.consolidated

    def reset_tau(self):
        """清空工作记忆（模拟 session 切换 / 上下文清空）。"""
        with torch.no_grad():
            self.tau.fill_(1.0)


def consolidate_model(model, **kw):
    for blk in model.blocks:
        if hasattr(blk["attn"], "consolidate"):
            blk["attn"].consolidate(**kw)


def reset_tau_model(model):
    for blk in model.blocks:
        if hasattr(blk["attn"], "reset_tau"):
            blk["attn"].reset_tau()


def learn_route(model, k, steps=80, eta=0.2):
    """oracle 直接沉积：让 τ 的 CLS 行在正确路由列 k+1 长成尖峰。

    说明：本 demo 聚焦'固化机制'（激活→权重），路由学习本身用 oracle 直接沉积，
    不依赖 A 加权（冻结主干时 A 接近均匀，A 加权沉积太弱会顶不住蒸发）。
    真实场景里这一步换成 pgtt_self_evolution 的信用信号沉积即可。
    """
    with torch.no_grad():
        for _ in range(steps):
            for blk in model.blocks:
                attn = blk["attn"]
                attn.tau.mul_(1 - attn.rho)          # 蒸发（用进废退）
                attn.tau[:, 0, k + 1] += eta          # 沉积到正确路由列
                attn.tau.clamp_(attn.tau_min, attn.tau_max)


def route_top(model, topk=3):
    """清空工作记忆后，看 CLS 的注意力还指向哪几列（固化权重是否生效）。"""
    model.eval()
    x, _ = make_data(64, 12, 5)
    with torch.no_grad():
        model(x)
    A = model.blocks[0]["attn"]._last_A  # (B,h,N,N)
    row = A[:, 0, 0, :].mean(0)          # 头0、CLS 行，batch 平均
    top = row.argsort(descending=True)[:topk]
    return [(int(j), round(float(row[j]), 3)) for j in top]


def tau_peak(model):
    tau = model.blocks[0]["attn"].tau[0, 0].detach()
    return int(tau.argmax().item()), round(float(tau.max().item()), 2)


def main():
    torch.manual_seed(0)
    kA = 5
    mk = lambda: MiniModel(EvolvablePheromoneAttention, n_tokens=12,
                           init_temp=1.0, rho=0.05, beta=1.0, deposit=0.05)

    # M1：演化模型 —— 学 → 固化 → 清工作记忆
    M1 = mk()
    learn_route(M1, kA)
    col1, peak1 = tau_peak(M1)
    consolidate_model(M1, lam=4.0, gamma=0.5)
    reset_tau_model(M1)
    m1_route = route_top(M1)

    # M2：对照模型 —— 学 → 不固化 → 清工作记忆
    M2 = mk()
    learn_route(M2, kA)
    col2, peak2 = tau_peak(M2)
    reset_tau_model(M2)
    m2_route = route_top(M2)

    print("=" * 64)
    print("权重迭代 MVP：信息素 τ 从'激活'升级为'可固化的演化权重'")
    print("=" * 64)
    print(f"任务：路由到第 {kA} 个 token（正确列 = {kA+1}）\n")

    print(f"[演化模型 M1]  训练后 τ 峰值列={col1}（τ={peak1}）")
    print(f"  → 固化(写入主干权重) → 清空工作记忆 τ")
    print(f"  清空后 CLS 仍指向: {m1_route}  ← 固化的权重记住了路由")

    print(f"\n[对照模型 M2]  训练后 τ 峰值列={col2}（τ={peak2}）")
    print(f"  → 不固化 → 清空工作记忆 τ")
    print(f"  清空后 CLS 指向: {m2_route}  ← 记忆随之消失")

    print("\n" + "=" * 64)
    m1_top = m1_route[0][0] if m1_route else -1
    m2_top = m2_route[0][0] if m2_route else -1
    if m1_top == kA + 1 and m2_top != kA + 1:
        print("结论 ✓  固化 = 把经验写进权重，工作记忆清了还在；")
        print("        不固化 = Agent 式失忆。这就是'模型'和'Agent'的分水岭：")
        print("        权重迭代（长在模型里） vs 外挂记忆（挂在模型外）。")
    else:
        print(f"（提示：m1_top={m1_top}, m2_top={m2_top}，需调 threshold/lam 或沉积步数）")


if __name__ == "__main__":
    main()
