"""诊断checkpoint中信息素tau的真实分布与固化状态"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

CKPT = r'd:\AetherMind-Nano3\checkpoints_v4_fixed\v4_checkpoint_35000_final.pt'

d = torch.load(CKPT, map_location='cpu', weights_only=False)
sd = d['model_state']

tau_keys = [k for k in sd if k.endswith('.tau')]
cons_keys = [k for k in sd if k.endswith('.consolidated')]
cnt_keys = [k for k in sd if 'consolidation_count' in k]

print(f'steps={d.get("steps")} phase={d.get("phase")}')
print(f'tau tensors: {len(tau_keys)} | consolidated tensors: {len(cons_keys)}')
print()

for k in tau_keys:
    t = sd[k]
    frac = float((t > 1.2).float().mean())
    print(f'{k}: min={t.min():.4f} max={t.max():.4f} mean={t.mean():.4f} '
          f'frac>1.2={frac:.5f}')

print()
for k in cons_keys:
    print(f'{k}: abs_sum={sd[k].abs().sum():.4f} nonzero={int((sd[k]!=0).sum())}')

print()
for k in cnt_keys:
    print(f'{k}: {sd[k]}')

print()
print('evolution_stats:', d.get('evolution_stats'))
