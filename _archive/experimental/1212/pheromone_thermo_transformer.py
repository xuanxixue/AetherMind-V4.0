"""
Pheromone-Guided Thermodynamic Transformer (PGTT)
==================================================
信息素路径网络 + 热力学注意力 + Transformer 的融合演示。
核心：注意力 = 信息素调制的玻尔兹曼分布 A ∝ exp(-E/T)·τ^β；
      τ 是持久缓冲，按 "沉积(使用量) + 蒸发" 跨步演化（stigmergy）。
玩具任务：固定路由 —— 目标恒等于第 k* 个 token 的值，逼模型学会 "CLS 必须走到 k*"。
运行：python pheromone_thermo_transformer.py   （需 PyTorch，已在受控 venv 装好）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# 1) 信息素调制的热力学注意力（本次主角）
# ----------------------------------------------------------------------------
class PheromoneThermoAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, n_tokens=12, init_temp=1.0,
                 target_entropy=None, whiten=True,
                 rho=0.05, beta=1.0, deposit=0.05,
                 tau_min=1e-2, tau_max=5.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model, self.num_heads = d_model, num_heads
        self.head_dim = d_model // num_heads
        self.whiten = whiten
        self.target_entropy = target_entropy
        self.rho, self.beta, self.deposit = rho, beta, deposit
        self.tau_min, self.tau_max = tau_min, tau_max
        self._last_A = None                        # 缓存本步注意力，供奖励门控沉积

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.log_temp = nn.Parameter(torch.log(torch.tensor(init_temp)))

        # 信息素：持久缓冲（含 CLS，故 N = n_tokens + 1），跨 forward 累积
        N = n_tokens + 1
        self.register_buffer("tau", torch.ones(num_heads, N, N))

    @property
    def temperature(self):
        return torch.exp(self.log_temp).clamp(min=1e-2, max=1e2)

    def forward(self, x, update_pheromone=True, reward=1.0):
        B, N, D = x.shape
        h = self.num_heads
        q = self.Wq(x).view(B, N, h, self.head_dim).transpose(1, 2)
        k = self.Wk(x).view(B, N, h, self.head_dim).transpose(1, 2)
        v = self.Wv(x).view(B, N, h, self.head_dim).transpose(1, 2)

        sim = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.head_dim)
        E = -sim                                   # 能量
        if self.whiten:
            em = E.mean(dim=(0, 2, 3), keepdim=True)
            es = E.std(dim=(0, 2, 3), unbiased=False, keepdim=True) + 1e-5
            E = (E - em) / es

        T = self.temperature
        # 信息素调制：有效能量 Ẽ = E - β·T·log τ  → A ∝ exp(-E/T)·τ^β
        log_tau = torch.log(self.tau.clamp(min=self.tau_min))
        Eeff = E - self.beta * T * log_tau
        A = F.softmax(-Eeff / T, dim=-1)

        out = torch.einsum("bhij,bhjd->bhid", A, v).transpose(1, 2).reshape(B, N, D)
        out = self.Wo(out)

        entropy = -(A * (A + 1e-8).log()).sum(-1).mean()
        free_energy = (A * Eeff).sum(-1).mean() - T * entropy

        # 缓存本步注意力，信息素沉积改为"奖励门控 + 延迟"（见 step_pheromone）
        self._last_A = A.detach()
        return out, {"entropy": entropy, "free_energy": free_energy,
                     "temperature": T, "tau": self.tau}

    def step_pheromone(self, reward):
        """奖励门控沉积：只在预测正确(reward≈1)的步上，沿注意力用量堆信息素；
        蒸发(1-ρ)始终发生。避免无门控时把早期随机错路锁死（蚁群停滞）。"""
        if not self.training or self._last_A is None:
            return
        if reward.dim() == 1:
            r = reward.detach().view(-1, 1, 1, 1)      # 逐样本标量奖励
        else:
            r = reward.detach()                          # 逐边掩码 (B,h,N,N)
        delta = self.deposit * (r * self._last_A).mean(0)
        with torch.no_grad():
            self.tau.mul_(1 - self.rho).add_(delta)
            self.tau.clamp_(self.tau_min, self.tau_max)


# ----------------------------------------------------------------------------
# 2) 标准 softmax 注意力（基线）
# ----------------------------------------------------------------------------
class VanillaAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, **kwargs):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model); self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model); self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, D = x.shape; h = self.num_heads
        q = self.Wq(x).view(B, N, h, self.head_dim).transpose(1, 2)
        k = self.Wk(x).view(B, N, h, self.head_dim).transpose(1, 2)
        v = self.Wv(x).view(B, N, h, self.head_dim).transpose(1, 2)
        s = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.head_dim)
        A = F.softmax(s, dim=-1)
        o = torch.einsum("bhij,bhjd->bhid", A, v).transpose(1, 2).reshape(B, N, D)
        return self.Wo(o)


# ----------------------------------------------------------------------------
# 3) 纯热力学注意力（上次，无信息素）
# ----------------------------------------------------------------------------
class ThermoAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, init_temp=1.0, target_entropy=None,
                 **kwargs):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, d_model // num_heads
        self.target_entropy = target_entropy
        self.Wq = nn.Linear(d_model, d_model); self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model); self.Wo = nn.Linear(d_model, d_model)
        self.log_temp = nn.Parameter(torch.log(torch.tensor(init_temp)))

    @property
    def temperature(self):
        return torch.exp(self.log_temp).clamp(min=1e-2, max=1e2)

    def forward(self, x):
        B, N, D = x.shape; h = self.num_heads
        q = self.Wq(x).view(B, N, h, self.head_dim).transpose(1, 2)
        k = self.Wk(x).view(B, N, h, self.head_dim).transpose(1, 2)
        v = self.Wv(x).view(B, N, h, self.head_dim).transpose(1, 2)
        E = -torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.head_dim)
        A = F.softmax(-E / self.temperature, dim=-1)
        o = torch.einsum("bhij,bhjd->bhid", A, v).transpose(1, 2).reshape(B, N, D)
        out = self.Wo(o)
        entropy = -(A * (A + 1e-8).log()).sum(-1).mean()
        return out, {"entropy": entropy, "temperature": self.temperature}


# ----------------------------------------------------------------------------
# 小模型（CLS 强制聚焦）
# ----------------------------------------------------------------------------
def _mk_blocks(attn_cls, d_model, num_heads, n_layers, n_tokens, **kw):
    return nn.ModuleList([
        nn.ModuleDict({
            "attn": attn_cls(d_model, num_heads, n_tokens=n_tokens, **kw),
            "ln1": nn.LayerNorm(d_model),
            "ffn": nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(),
                                 nn.Linear(d_model * 2, d_model)),
            "ln2": nn.LayerNorm(d_model),
        }) for _ in range(n_layers)
    ])


class MiniModel(nn.Module):
    def __init__(self, attn_cls, d_model=32, num_heads=4, n_layers=2,
                 n_tokens=12, **attn_kw):
        super().__init__()
        self.n_tokens = n_tokens
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.embed = nn.Linear(2, d_model)
        self.blocks = _mk_blocks(attn_cls, d_model, num_heads, n_layers,
                                 n_tokens, **attn_kw)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        B = x.shape[0]
        tok = self.embed(x)
        h = torch.cat([self.cls.expand(B, -1, -1), tok], dim=1)
        stats = {}; ent_acc = 0.0; nblk = 0; tau = None
        for blk in self.blocks:
            res = blk["attn"](blk["ln1"](h))
            if isinstance(res, tuple):
                a, s = res
                if isinstance(s, dict):
                    if "entropy" in s:
                        ent_acc += s["entropy"]; nblk += 1
                    if "tau" in s:
                        tau = s["tau"]
            else:
                a = res
            h = h + a
            h = h + blk["ffn"](blk["ln2"](h))
        if nblk > 0:
            stats["entropy"] = ent_acc / nblk
        if tau is not None:
            stats["tau"] = tau
        return self.head(h[:, 0]).squeeze(-1), stats

    def deposit(self, reward):
        """对各层信息素注意力做奖励门控沉积（仅 PGTT 有 step_pheromone）。"""
        for blk in self.blocks:
            attn = blk["attn"]
            if hasattr(attn, "step_pheromone"):
                attn.step_pheromone(reward)


# ----------------------------------------------------------------------------
# 固定路由任务：y = x[:, k*]（CLS 必须走到第 k* 个 token）
# ----------------------------------------------------------------------------
def make_data(B, n_tokens=12, k=5):
    val = torch.randn(B, n_tokens)
    pos = torch.arange(n_tokens, dtype=torch.float32).unsqueeze(0).expand(B, -1) \
        / (n_tokens - 1)
    x = torch.stack([val, pos], dim=-1)          # (B, n_tokens, 2): 值 + 位置
    y = (val[:, k] > 0).float()                  # 二分类：第 k 个位置的值是否为正
    return x, y


def train_model(model, name, epochs=200, lr=1e-3, k=5, n_tokens=12,
                entropy_reg=0.05, target_entropy=None, teacher_route=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    B = 128
    print(f"\n--- {name} ---")
    if teacher_route is not None:
        print("  [信息素] 教师奖励模式：仅在 CLS→正确路由上沉积（oracle，用于演示凝聚机制）")
    for ep in range(epochs):
        x, y = make_data(B, n_tokens, k)
        model.train()
        pred, stats = model(x)
        bce = F.binary_cross_entropy_with_logits(pred, y)
        reg = torch.tensor(0.0)
        if isinstance(stats, dict) and "entropy" in stats and target_entropy is not None:
            reg = entropy_reg * F.relu(stats["entropy"] - target_entropy)
        loss = bce + reg
        opt.zero_grad(); loss.backward(); opt.step()
        # 奖励门控的信息素沉积（仅 PGTT 生效）
        if hasattr(model, "deposit"):
            if teacher_route is not None:
                qi, tj = teacher_route
                ts = model.blocks[0]["attn"].tau.shape      # (h, N, N)
                mask = torch.zeros(B, *ts)
                mask[:, :, qi, tj] = 1.0                     # 只在正确边上沉积
                model.deposit(mask)
            else:
                with torch.no_grad():
                    reward = ((pred > 0).float() == (y > 0.5).float()).float()
                model.deposit(reward)
        if ep % 40 == 0 or ep == epochs - 1:
            extra = ""
            if isinstance(stats, dict):
                if "temperature" in stats:
                    extra += f"  T={stats['temperature'].item():.3f}"
                if "entropy" in stats:
                    extra += f"  H={stats['entropy'].item():.3f}"
            print(f"  ep{ep:3d}  bce={bce.item():.4f}{extra}")
    model.eval()
    xt, yt = make_data(2000, n_tokens, k)
    with torch.no_grad():
        pt, _ = model(xt)
        acc = ((pt > 0).float() == (yt > 0.5).float()).float().mean().item()
        return acc


def inspect_pheromone(model, k, n_tokens):
    # 拼接后 CLS 占列0，原 token k 落在列 k+1
    correct_col = k + 1
    tau = model.blocks[0]["attn"].tau[0, 0].detach()      # 头0, 从 CLS 出发的行
    mx = float(tau.max().item()); mean = float(tau.mean().item())
    order = tau.argsort(descending=True)[:3].tolist()
    argmax = int(tau.argmax().item())
    top_vals = [round(float(tau[i].item()), 3) for i in order]
    print(f"  [信息素] CLS行 τ：峰值={mx:.3f} @列{argmax}  "
          f"(正确路由列={correct_col})  集中度峰值/均值={mx/mean:.2f}")
    print(f"  [信息素] Top3列: {order} 对应τ: {top_vals}")
    hit = argmax == correct_col
    print(f"  [信息素] 尖峰是否落在正确路由: {'是 ✓' if hit else '否 ✗'}")


def main():
    torch.manual_seed(42)                  # 固定种子，结果可复现
    k, n_tokens = 5, 12
    target_entropy = math.log(3)

    vanilla = MiniModel(VanillaAttention, n_tokens=n_tokens)
    thermo = MiniModel(ThermoAttention, n_tokens=n_tokens,
                       init_temp=1.0, target_entropy=target_entropy)
    phero = MiniModel(PheromoneThermoAttention, n_tokens=n_tokens,
                      init_temp=1.0, target_entropy=target_entropy,
                      rho=0.05, beta=1.0, deposit=0.05)

    v_mse = train_model(vanilla, "Vanilla softmax 注意力（基线）")
    t_mse = train_model(thermo, "Thermo 注意力（热力学，无信息素）",
                        target_entropy=target_entropy)
    p_mse = train_model(phero, "PGTT 信息素+热力学+Transformer",
                        target_entropy=target_entropy, teacher_route=(0, k + 1))

    print("\n" + "=" * 60)
    print(f"测试集准确率  →  基线:{v_mse:.4f}  热力学:{t_mse:.4f}  "
          f"信息素+热力学:{p_mse:.4f}")
    print("=" * 60)
    print("\n信息素路径检查（PGTT）：")
    inspect_pheromone(phero, k, n_tokens)


if __name__ == "__main__":
    main()
