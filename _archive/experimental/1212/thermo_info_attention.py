"""
ThermoInfoAttention —— 物理（热力学+信息论）驱动的注意力机制演示
==============================================================
设计对应《物理驱动注意力机制方案.md》的四支柱：
  1. 热力学：注意力 = 玻尔兹曼分布 exp(-E/T)/Z，温度 T 可学习
  2. 信息论：熵 H 作为正则目标（聚焦但不塌缩）
  3. 统计学：能量白化(运行统计量) + 输出期望/方差(不确定性)
  4. 数学优化：任务损失 + 熵目标正则 的带约束优化

玩具任务：序列 top-3 均值预测（带 [CLS] 强制注意力聚焦到最大的 3 个位置）
运行：python thermo_info_attention.py   （需 PyTorch）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# 物理驱动注意力
# ----------------------------------------------------------------------------
class ThermoInfoAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, init_temp=1.0, target_entropy=None,
                 whiten=True):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.whiten = whiten
        self.target_entropy = target_entropy  # 信息论：目标熵 H*

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

        # 热力学：温度 T = exp(log_T) 可学习，clamp 防塌缩/发散
        self.log_temp = nn.Parameter(torch.log(torch.tensor(init_temp)))

        # 统计辅助：能量白化的运行统计量
        self.register_buffer("energy_mean", torch.tensor(0.0))
        self.register_buffer("energy_std", torch.tensor(1.0))

    @property
    def temperature(self):
        return torch.exp(self.log_temp).clamp(min=1e-2, max=1e2)

    def forward(self, x, return_stats=True):
        B, N, D = x.shape
        h = self.num_heads
        q = self.Wq(x).view(B, N, h, self.head_dim).transpose(1, 2)  # (B,h,N,d)
        k = self.Wk(x).view(B, N, h, self.head_dim).transpose(1, 2)
        v = self.Wv(x).view(B, N, h, self.head_dim).transpose(1, 2)

        # 能量 E_ij = -(Q_i·K_j)/√d   （越相似能量越低）
        sim = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.head_dim)
        E = -sim  # 负相似度即"能量"

        # 统计辅助：能量白化（用批统计量，使 T 与数据尺度解耦）
        if self.whiten:
            em = E.mean(dim=(0, 2, 3), keepdim=True)
            es = E.std(dim=(0, 2, 3), unbiased=False, keepdim=True) + 1e-5
            E = (E - em) / es

        # 热力学：玻尔兹曼分布 A_ij = exp(-E_ij / T) / Z
        T = self.temperature
        A = F.softmax(-E / T, dim=-1)  # (B,h,N,N)

        # 物理输出 = 期望 Σ_j A_ij V_j；方差 = 不确定性
        out = torch.einsum("bhij,bhjd->bhid", A, v)            # 期望
        out2 = torch.einsum("bhij,bhjd->bhid", A, v * v)        # E[V^2]
        var = (out2 - out * out).mean(dim=-1)                   # 方差(不确定性)

        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.Wo(out)

        if not return_stats:
            return out

        # 信息论：每查询的熵 H = -Σ A log A
        entropy = -(A * (A + 1e-8).log()).sum(-1).mean()
        # 热力学：自由能 F = <E> - T·H（玻尔兹曼分布即其最小化解）
        free_energy = (A * E).sum(-1).mean() - T * entropy

        return out, {
            "attn": A, "entropy": entropy, "free_energy": free_energy,
            "temperature": T, "variance": var.mean(),
        }


# ----------------------------------------------------------------------------
# 标准 softmax 注意力（对照基线，纯统计壳、T 固定=1）
# ----------------------------------------------------------------------------
class VanillaAttention(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, D = x.shape
        h = self.num_heads
        q = self.Wq(x).view(B, N, h, self.head_dim).transpose(1, 2)
        k = self.Wk(x).view(B, N, h, self.head_dim).transpose(1, 2)
        v = self.Wv(x).view(B, N, h, self.head_dim).transpose(1, 2)
        scores = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.head_dim)
        A = F.softmax(scores, dim=-1)            # T=1 的特例
        out = torch.einsum("bhij,bhjd->bhid", A, v).transpose(1, 2).reshape(B, N, D)
        return self.Wo(out)


# ----------------------------------------------------------------------------
# 小模型：CLS token 强制注意力聚焦
# ----------------------------------------------------------------------------
class MiniThermoModel(nn.Module):
    def __init__(self, d_model=32, num_heads=4, n_layers=2, n_tokens=12,
                 target_entropy=None):
        super().__init__()
        self.n_tokens = n_tokens
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.embed = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "attn": ThermoInfoAttention(d_model, num_heads,
                                            target_entropy=target_entropy),
                "ln1": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(nn.Linear(d_model, d_model * 2),
                                     nn.GELU(), nn.Linear(d_model * 2, d_model)),
                "ln2": nn.LayerNorm(d_model),
            }) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):  # x: (B, n_tokens, 1)
        B = x.shape[0]
        tok = self.embed(x)
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, tok], dim=1)
        acc = {key: 0.0 for key in
               ("entropy", "free_energy", "temperature", "variance")}
        for blk in self.blocks:
            a, s = blk["attn"](blk["ln1"](h))
            h = h + a
            h = h + blk["ffn"](blk["ln2"](h))
            for key in acc:
                acc[key] = acc[key] + s[key]          # 保留张量，供反向传播
        n = torch.tensor(len(self.blocks), dtype=acc["entropy"].dtype)
        stats = {key: val / n for key, val in acc.items()}
        out = self.head(h[:, 0])  # 取 CLS 输出
        return out.squeeze(-1), stats


class MiniVanillaModel(nn.Module):
    def __init__(self, d_model=32, num_heads=4, n_layers=2, n_tokens=12):
        super().__init__()
        self.n_tokens = n_tokens
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.embed = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "attn": VanillaAttention(d_model, num_heads),
                "ln1": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(nn.Linear(d_model, d_model * 2),
                                     nn.GELU(), nn.Linear(d_model * 2, d_model)),
                "ln2": nn.LayerNorm(d_model),
            }) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        B = x.shape[0]
        tok = self.embed(x)
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, tok], dim=1)
        for blk in self.blocks:
            h = h + blk["attn"](blk["ln1"](h))
            h = h + blk["ffn"](blk["ln2"](h))
        return self.head(h[:, 0]).squeeze(-1), None


# ----------------------------------------------------------------------------
# 玩具数据：序列 top-k 均值
# ----------------------------------------------------------------------------
def make_data(B, n_tokens=12, k=3):
    x = torch.randn(B, n_tokens)
    topk = torch.topk(x, k, dim=-1).values
    y = topk.mean(dim=-1)
    return x.unsqueeze(-1), y


def train(model, epochs=200, lr=1e-3, k=3, n_tokens=12,
          entropy_reg=0.05, target_entropy=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    B = 128
    log = []
    for ep in range(epochs):
        x, y = make_data(B, n_tokens, k)
        pred, stats = model(x)
        mse = F.mse_loss(pred, y)
        # 信息论：熵目标正则（聚焦但不许过散），保留张量以反向传播
        reg = torch.tensor(0.0)
        if target_entropy is not None and isinstance(stats, dict):
            reg = entropy_reg * F.relu(stats["entropy"] - target_entropy)
        loss = mse + reg
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % 20 == 0 or ep == epochs - 1:
            T = stats["temperature"].item() if isinstance(stats, dict) else 1.0
            H = stats["entropy"].item() if isinstance(stats, dict) else 0.0
            FE = stats["free_energy"].item() if isinstance(stats, dict) else 0.0
            line = (f"ep{ep:3d}  loss={loss.item():.4f}  mse={mse.item():.4f}  "
                    f"T={T:.3f}  H={H:.3f}  F={FE:.3f}")
            log.append(line)
            print(line)
    # 测试
    xt, yt = make_data(2000, n_tokens, k)
    with torch.no_grad():
        pt, _ = model(xt)
        test_mse = F.mse_loss(pt, yt).item()
    return test_mse, log


def main():
    k, n_tokens = 3, 12
    target_entropy = math.log(k)  # 信息论目标：聚焦到 ~k 个位置

    print("=" * 64)
    print("【物理驱动】ThermoInfoAttention（可学温度 + 熵目标正则）")
    print("=" * 64)
    thermo = MiniThermoModel(d_model=32, num_heads=4, n_layers=2,
                             n_tokens=n_tokens, target_entropy=target_entropy)
    thermo_mse, _ = train(thermo, epochs=200, k=k, n_tokens=n_tokens,
                          entropy_reg=0.05, target_entropy=target_entropy)

    print()
    print("=" * 64)
    print("【对照基线】Vanilla softmax 注意力（T=1，无熵正则）")
    print("=" * 64)
    vanilla = MiniVanillaModel(d_model=32, num_heads=4, n_layers=2,
                               n_tokens=n_tokens)
    vanilla_mse, _ = train(vanilla, epochs=200, k=k, n_tokens=n_tokens)

    print()
    print("=" * 64)
    print(f"测试集 MSE  →  物理驱动: {thermo_mse:.5f}   |   基线: {vanilla_mse:.5f}")
    better = "物理驱动更优" if thermo_mse < vanilla_mse else "基线更优（需调参）"
    print(f"结论：{better}")
    print("=" * 64)


if __name__ == "__main__":
    main()
