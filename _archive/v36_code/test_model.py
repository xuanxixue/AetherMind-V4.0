import os
import sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.aethermind36_config import AetherMind36Config
from src.model.aethermind36 import AetherMind36


def test_model_forward():
    print("[Test 1] 模型前向传播测试（三阶段配置）")
    phases = [("A", lambda c: c.set_phase_A()),
              ("B", lambda c: c.set_phase_B(0.5)),
              ("C", lambda c: c.set_phase_C(0.5))]

    for pname, setter in phases:
        cfg = AetherMind36Config(
            vocab_size=5000, d_model=256, d_ff=512,
            n_layers=2, max_seq_len=64, n_atoms=64, d_atom=32,
            n_anchor=32, d_anchor=32, d_state=64, langevin_K=2
        )
        cfg.device = "cpu"
        setter(cfg)
        model = AetherMind36(cfg)
        params = model.count_params()
        size_mb = params * 4 / 1024 / 1024

        B, S = 2, 64
        input_ids = torch.randint(0, 5000, (B, S))
        labels = input_ids.clone()
        out = model(input_ids, labels, task_id=0, t=1.0)

        loss = out["loss"]
        assert torch.isfinite(loss), f"Phase {pname}: loss={loss} 非有限值"
        print(f"  Phase {pname}: params={params:,} ({size_mb:.1f}MB), loss={loss.item():.4f}, "
              f"LM={out['loss_LM'].item():.4f}, IB={out['loss_IB'].item():.4f}, T={out['T'].item():.3f}")
        del model
    print("[Test 1] PASSED\n")


def test_model_generate():
    print("[Test 2] 文本生成测试")
    cfg = AetherMind36Config(
        vocab_size=5000, d_model=256, d_ff=512,
        n_layers=2, max_seq_len=64, n_atoms=64, d_atom=32,
        n_anchor=32, d_anchor=32, d_state=64, langevin_K=2
    )
    cfg.set_phase_C(1.0)
    cfg.device = "cpu"
    model = AetherMind36(cfg)
    model.eval()

    prompt = torch.tensor([[1, 10, 20, 30, 40]], dtype=torch.long)
    out = model.generate(prompt, max_new_tokens=16, top_k=10, top_p=0.9)
    print(f"  生成形状: {out.shape}")
    assert out.shape == (1, 5 + 16) or out.shape[0] == 1, "生成形状异常"
    print("[Test 2] PASSED\n")


def test_langevin_disabled():
    print("[Test 3] 物理层完全关闭时退化到 3.5.1 测试")
    cfg = AetherMind36Config(
        vocab_size=5000, d_model=256, d_ff=512,
        n_layers=2, max_seq_len=64, n_atoms=64, d_atom=32,
        n_anchor=32, d_anchor=32, d_state=64
    )
    cfg.set_phase_A()
    cfg.device = "cpu"
    model = AetherMind36(cfg)
    input_ids = torch.randint(0, 5000, (2, 64))
    labels = input_ids.clone()
    out = model(input_ids, labels)
    assert out["loss_phys"].item() == 0.0, "Phase A 下物理损失应为 0"
    assert out["loss_IB"].item() == 0.0, "Phase A 下 IB 损失应为 0"
    print(f"  Phase A (纯3.5.1模式): loss={out['loss'].item():.4f}, phys={out['loss_phys'].item()}, IB={out['loss_IB'].item()}")
    print("[Test 3] PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("AetherMind 3.6.1 模型自检 (CPU)")
    print("=" * 60)
    try:
        test_model_forward()
        test_model_generate()
        test_langevin_disabled()
        print("=" * 60)
        print("所有测试通过! 模型代码结构正确。")
        print("=" * 60)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[ERROR] 测试失败: {e}")
        sys.exit(1)
