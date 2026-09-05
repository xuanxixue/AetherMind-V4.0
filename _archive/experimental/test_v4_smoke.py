"""Quick smoke test for V4 model (no dataset needed)."""
import os, sys, torch
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, r"d:\AetherMind-Nano3")

from configs.aethermind4_config import AetherMind4Config

def _make_cfg(phase_init="A"):
    cfg = AetherMind4Config(
        vocab_size=5000, d_model=128, d_ff=512, n_layers=2, n_heads=2,
        max_seq_len=64, n_atoms=64, d_atom=32, n_anchor=32, d_state=32, d_counter=16,
        langevin_K=2, dropout=0.0,
    )
    if phase_init == "A":
        cfg.set_phase_A()
    elif phase_init == "B":
        cfg.set_phase_B(0.5)
    else:
        cfg.set_phase_C(0.5)
    return cfg

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Smoke] device={device}")

# Phase A: forward + backward
cfg = _make_cfg("A")
from src.model.aethermind4 import AetherMind4
model = AetherMind4(cfg).to(device)
print(f"[Smoke] params={model.count_params():,}")

B, S = 2, 64
ids = torch.randint(3, 4999, (B, S), device=device)
lbl = torch.randint(3, 4999, (B, S), device=device)

out = model(ids, lbl, phase="A")
assert torch.isfinite(out["loss"]), f"Phase A loss not finite: {out['loss']}"
out["loss"].backward()
print(f"[Smoke] Phase A: loss={out['loss'].item():.4f}  backward OK")

# Phase B: forward + evolution_step
cfg = _make_cfg("B")
model2 = AetherMind4(cfg).to(device)
opt = torch.optim.AdamW(model2.parameters(), lr=1e-3)
for step in range(3):
    opt.zero_grad()
    out = model2(ids, lbl, phase="B")
    assert torch.isfinite(out["loss"]), f"Phase B step{step} loss={out['loss']}"
    out["loss"].backward()
    opt.step()
    model2.evolution_step(step, phase="B", loss_val=out["loss"].detach())
print(f"[Smoke] Phase B x3: loss={out['loss'].item():.4f}  evolution_step OK")

# Phase C: forward + evolution_step (with free_energy) + consolidate
cfg = _make_cfg("C")
model3 = AetherMind4(cfg).to(device)
opt = torch.optim.AdamW(model3.parameters(), lr=1e-3)
for step in range(5):
    opt.zero_grad()
    out = model3(ids, lbl, phase="C")
    assert torch.isfinite(out["loss"]), f"Phase C step{step} loss={out['loss']}"
    out["loss"].backward()
    opt.step()
    model3.evolution_step(step, phase="C", free_energy=out["F"].detach(), loss_val=out["loss"].detach())
ev_stats = model3.get_evolution_stats()
print(f"[Smoke] Phase C x5: loss={out['loss'].item():.4f}  tau_conc={ev_stats['tau_concentration']:.2f}  cons_mass={ev_stats['consolidation_mass']:.1f}")

# Generate
model3.eval()
with torch.no_grad():
    prompt = torch.randint(3, 4999, (1, 16), device=device)
    gen = model3.generate(prompt, max_new_tokens=10, top_k=20, top_p=0.9)
print(f"[Smoke] Generate OK: input={prompt.shape} -> output={gen.shape}")

# Save/Load (checkpoint round-trip)
ckpt_path = r"d:\AetherMind-Nano3\checkpoints_v4_smoke\test_ckpt.pt"
os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
evo = model3.evolver
torch.save({
    "steps": 5, "phase": "C", "phase_progress": 0.01,
    "model_state": model3.state_dict(),
    "optimizer_state": opt.state_dict(),
    "scheduler_state": {}, "scaler_state": None,
    "evo_F_prev": float(evo._F_prev.detach().cpu().item()),
    "evo_F_init": bool(evo._F_initialized),
    "evo_loss_init": bool(evo._loss_initialized),
    "evo_loss_prev": None,
}, ckpt_path)
model4 = AetherMind4(_make_cfg("C")).to(device)
ck = torch.load(ckpt_path, map_location=device, weights_only=False)
model4.load_state_dict(ck["model_state"], strict=False)
out4 = model4(ids, lbl, phase="C")
assert torch.isfinite(out4["loss"])
print(f"[Smoke] Checkpoint save/load OK: reloaded loss={out4['loss'].item():.4f}")
os.remove(ckpt_path)

print("\n[Smoke] ALL PASSED")
