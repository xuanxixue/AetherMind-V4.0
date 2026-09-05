p = r'D:\AetherMind-Nano3\src\model\aethermind4.py'
with open(p, 'r', encoding='utf-8') as f:
    c = f.read()

# 修复 BCE 输入 NaN/inf 问题：增加 isfinite 保护
old = """        # GMM 判别器损失 (bf16安全)
        D_score = domain_out["D_score"]
        with torch.amp.autocast("cuda", enabled=False):
            D_f = D_score.float().clamp(1e-6, 1 - 1e-6)
            tgt_f = torch.ones_like(D_f) * 0.5
            loss_D = cfg.lambda_D * F.binary_cross_entropy(D_f, tgt_f)"""

new = """        # GMM 判别器损失 (bf16安全 + NaN防护)
        D_score = domain_out["D_score"]
        with torch.amp.autocast("cuda", enabled=False):
            D_f = D_score.float()
            # 替换 NaN/inf 为 0.5（无信息），再 clamp 到安全范围
            D_f = torch.where(torch.isfinite(D_f), D_f, torch.full_like(D_f, 0.5))
            D_f = D_f.clamp(1e-6, 1 - 1e-6)
            tgt_f = torch.ones_like(D_f) * 0.5
            loss_D = cfg.lambda_D * F.binary_cross_entropy(D_f, tgt_f)"""

c = c.replace(old, new)

with open(p, 'w', encoding='utf-8') as f:
    f.write(c)

print("BCE NaN fix applied:", 'torch.where(torch.isfinite' in c)
