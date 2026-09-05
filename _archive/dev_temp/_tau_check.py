# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r'D:\AetherMind-Nano3')
import torch

p = r'D:\AetherMind-Nano3\checkpoints_v4_fixed\v4_checkpoint_12000.pt'
ckpt = torch.load(p, map_location='cpu', weights_only=False)
ms = ckpt['model_state']
print('keys containing tau:', [k for k in ms if 'tau' in k])
print('keys containing pheromone:', [k for k in ms if 'pheromone' in k])
print('keys containing log_temp:', [k for k in ms if 'log_temp' in k or 'temp' in k])
# print all buffer-like keys (non weight/bias) sample
bufs = [k for k in ms if not k.endswith('weight') and not k.endswith('bias') and 'attn' in k]
print('sample buffer keys:', bufs[:20])
# check tau in a fresh model
from configs.aethermind4_config import AetherMind4Config
from src.model.aethermind4 import AetherMind4
mc = AetherMind4Config(vocab_size=151680, d_model=384, d_ff=1536, n_layers=6, n_heads=6,
                       max_seq_len=256, dropout=0.1, pad_token_id=151643,
                       bos_token_id=151643, eos_token_id=151643, unk_token_id=151643)
model = AetherMind4(mc)
sd = model.state_dict()
print('fresh keys containing tau:', [k for k in sd if 'tau' in k])
