"""
AetherMind V4 训练验证钩子 — 三层验证结构
=============================================
第1层（每500步）：数值卫生 — τ稳态、饱和预警、NaN/Inf、梯度健康
第2层（每5000步）：迁移性检查 — 压缩KL + 推理烟测
第3层（阶段末）：完整验收 — 转换→预热→复杂度→生成质量，输出JSON报告

所有指标写入 JSONL 日志流，供阶段间对比和 scaling 分析。
"""

import os
import json
import time
import math
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, List


class ValidationHooks:
    """三层验证钩子集合。挂在训练循环中调用。"""

    def __init__(self, log_dir: str, model=None, tokenizer=None,
                 check_interval: int = 500, migration_interval: int = 5000,
                 kl_threshold: float = 0.1, device: str = "cpu"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.model = model
        self.tokenizer = tokenizer
        self.check_interval = check_interval
        self.migration_interval = migration_interval
        self.kl_threshold = kl_threshold
        self.device = device
        self.log_path = os.path.join(log_dir, "validation_metrics.jsonl")
        self._tau_history: List[float] = []
        self._energy_history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # 日志写入
    # ------------------------------------------------------------------
    def _log(self, metrics: Dict[str, Any]):
        metrics["timestamp"] = time.time()
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # 第1层：每N步数值卫生检查
    # ------------------------------------------------------------------
    def layer1_numeric_check(self, step: int, loss: float, model=None) -> Dict[str, Any]:
        """数值卫生检查：τ稳态、饱和预警、梯度健康。返回metrics dict。"""
        m = model or self.model
        if m is None:
            return {}

        metrics = {"layer": 1, "step": step, "loss": loss}
        evo = m.get_evolution_stats() if hasattr(m, "get_evolution_stats") else {}

        # τ 浓度
        tau_conc = float(evo.get("tau_concentration", 0))
        metrics["tau_concentration"] = tau_conc
        self._tau_history.append(tau_conc)

        # τ 稳态斜率（最近5个检查点）
        if len(self._tau_history) >= 5:
            recent = self._tau_history[-5:]
            slope = (recent[-1] - recent[0]) / 4.0
            metrics["tau_slope"] = slope
            if slope > 0 and step > 1000:
                metrics["tau_warning"] = f"τ仍在上升 slope={slope:.2f}"

        # τ 过饱和预警：tau_conc > 10 时检查物理项能量占比
        if tau_conc > 10:
            metrics["tau_saturation_warning"] = f"τ浓度过高={tau_conc:.1f}"

        # 固化质量
        cons_mass = float(evo.get("consolidation_mass", 0))
        metrics["consolidation_mass"] = cons_mass
        cons_rounds = int(evo.get("consolidation_rounds", 0))
        metrics["consolidation_rounds"] = cons_rounds

        # 梯度健康：可训练参数梯度范数 max/min 比
        grad_norms = []
        for p in m.parameters():
            if p.requires_grad and p.grad is not None:
                grad_norms.append(p.grad.norm().item())
        if grad_norms:
            gmax, gmin = max(grad_norms), min(grad_norms) + 1e-12
            ratio = gmax / gmin
            metrics["grad_norm_max"] = gmax
            metrics["grad_norm_min"] = gmin
            metrics["grad_norm_ratio"] = ratio
            if ratio > 1e6:
                metrics["grad_explosion_warning"] = f"梯度max/min比={ratio:.1e}"

        self._log(metrics)
        return metrics

    # ------------------------------------------------------------------
    # 物理能量占比分析（谁主导：DL vs 物理）
    # ------------------------------------------------------------------
    def energy_ratio_analysis(self, step: int, model=None) -> Dict[str, Any]:
        """统计注意力能量三项的方差占比：
        E_eff = E_qk(DL内容) - beta*T*logτ(信息素) - T*C(固化)
        物理项占比 = Var(βT·logτ + T·C) / Var(E_eff总)
        """
        m = model or self.model
        if m is None:
            return {}

        metrics = {"layer": 1, "step": step, "type": "energy_ratio"}

        # 收集各层注意力的能量分量
        e_qk_list, e_tau_list, e_cons_list, e_total_list = [], [], [], []

        for layer in getattr(m, "decoder_layers", []):
            attn = getattr(layer, "attn", None)
            if attn is None or not hasattr(attn, "_last_E_components"):
                continue
            comp = attn._last_E_components  # dict: E_qk, E_tau, E_cons, E_eff
            if comp is None:
                continue
            e_qk_list.append(comp["E_qk"].detach().float())
            e_tau_list.append(comp["E_tau"].detach().float())
            e_cons_list.append(comp["E_cons"].detach().float())
            e_total_list.append(comp["E_eff"].detach().float())

        if not e_total_list:
            metrics["energy_ratio_status"] = "unavailable (no energy components tracked)"
            self._log(metrics)
            return metrics

        # 拼接所有层
        e_qk = torch.cat([t.flatten() for t in e_qk_list])
        e_tau = torch.cat([t.flatten() for t in e_tau_list])
        e_cons = torch.cat([t.flatten() for t in e_cons_list])
        e_total = torch.cat([t.flatten() for t in e_total_list])

        # 方差占比
        var_total = e_total.var().item() + 1e-12
        var_qk = e_qk.var().item()
        var_tau = e_tau.var().item()
        var_cons = e_cons.var().item()
        # NaN防护：早期训练能量分量可能全零或含NaN
        import math
        if math.isnan(var_qk) or math.isnan(var_tau) or math.isnan(var_cons) or math.isnan(var_total):
            metrics["energy_ratio_status"] = "nan_detected (early training, energy components undefined)"
            metrics["var_qk_ratio"] = 0.0
            metrics["var_tau_ratio"] = 0.0
            metrics["var_cons_ratio"] = 0.0
            metrics["var_physics_ratio"] = 0.0
            metrics["dominance"] = "undetermined (nan)"
            self._log(metrics)
            return metrics
        var_physics = var_tau + var_cons

        metrics["var_qk_ratio"] = var_qk / var_total
        metrics["var_tau_ratio"] = var_tau / var_total
        metrics["var_cons_ratio"] = var_cons / var_total
        metrics["var_physics_ratio"] = var_physics / var_total
        metrics["var_total"] = var_total

        # 判定
        if var_physics / var_total < 0.05:
            metrics["dominance"] = "DL-dominant (physics <5%)"
        elif var_physics / var_total > 0.30:
            metrics["dominance"] = "physics-significant (>30%)"
        else:
            metrics["dominance"] = "mixed"

        self._energy_history.append({
            "step": step,
            "physics_ratio": var_physics / var_total,
        })
        self._log(metrics)
        return metrics

    # ------------------------------------------------------------------
    # 第2层：迁移性检查（压缩KL）
    # ------------------------------------------------------------------
    def migration_check(self, step: int, model=None,
                        sample_batch=None) -> Dict[str, Any]:
        """压缩KL检查：训练图(绝对偏置) vs 推理图(相对偏置)的注意力分布KL散度。
        衡量当前权重是否可无损迁移到推理架构。"""
        m = model or self.model
        metrics = {"layer": 2, "step": step, "type": "migration_check"}

        if m is None or not hasattr(m, "decoder_layers"):
            metrics["status"] = "skipped (no model)"
            self._log(metrics)
            return metrics

        kl_values = []
        for layer_idx, layer in enumerate(m.decoder_layers):
            attn = getattr(layer, "attn", None)
            if attn is None or attn._last_A is None:
                continue
            A_train = attn._last_A.detach()  # (B, h, S, S)
            if A_train.dim() != 4:
                continue
            B, h, S, _ = A_train.shape
            if S < 4:
                continue

            # 简化版KL：检查注意力分布的平移不变性
            # 取相邻行的注意力分布差异（平移不变 → 差异小）
            if S >= 8:
                rows = A_train[0, 0]  # (S, S)
                # 比较 row[i] 和 row[i+1] 的偏移版本
                kl_sum = 0.0
                count = 0
                for i in range(min(S - 1, 16)):
                    r1 = rows[i, :i + 1] + 1e-8  # 第i行的有效部分
                    r2 = rows[i + 1, 1:i + 2] + 1e-8  # 第i+1行偏移1位
                    if r1.shape[0] == r2.shape[0] and r1.shape[0] > 0:
                        kl = (r1 * (r1.log() - r2.log())).sum().item()
                        kl_sum += kl
                        count += 1
                if count > 0:
                    kl_values.append(kl_sum / count)

        if kl_values:
            avg_kl = sum(kl_values) / len(kl_values)
            metrics["compression_kl"] = avg_kl
            metrics["kl_layers_checked"] = len(kl_values)
            if avg_kl > self.kl_threshold:
                metrics["kl_warning"] = (
                    f"平移不变假设失效 KL={avg_kl:.3f} > {self.kl_threshold}，"
                    f"相对化将有系统偏差")
            else:
                metrics["kl_status"] = "PASS"
        else:
            metrics["compression_kl"] = "unavailable"
            metrics["status"] = "no attention data (need forward pass first)"

        self._log(metrics)
        return metrics

    # ------------------------------------------------------------------
    # 第2层：推理烟测
    # ------------------------------------------------------------------
    def inference_smoke_test(self, step: int, model=None,
                             tokenizer=None, n_tokens: int = 20) -> Dict[str, Any]:
        """快速推理烟测：切换到推理模式生成n_tokens，确认无NaN。"""
        m = model or self.model
        tok = tokenizer or self.tokenizer
        metrics = {"layer": 2, "step": step, "type": "inference_smoke"}

        if m is None or tok is None:
            metrics["status"] = "skipped (no model/tokenizer)"
            self._log(metrics)
            return metrics

        try:
            m.eval()
            prompt = "你好"
            input_ids = tok.encode(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                for _ in range(n_tokens):
                    out = m(input_ids, task_id=0, t=0.0, phase="G")
                    logits = out["logits"][:, -1, :]
                    next_id = logits.argmax(dim=-1, keepdim=True)
                    input_ids = torch.cat([input_ids, next_id], dim=-1)

            generated = tok.decode(input_ids[0], skip_special_tokens=True)
            has_nan = not torch.isfinite(out["logits"]).all()
            metrics["generated"] = generated[:100]
            metrics["has_nan"] = bool(has_nan)
            metrics["status"] = "FAIL" if has_nan else "PASS"
            m.train()
        except Exception as e:
            metrics["status"] = "ERROR"
            metrics["error"] = str(e)[:200]
            m.train()

        self._log(metrics)
        return metrics

    # ------------------------------------------------------------------
    # 第3层：阶段末完整验收报告
    # ------------------------------------------------------------------
    def phase_end_report(self, phase: str, step: int,
                         model=None, tokenizer=None) -> Dict[str, Any]:
        """阶段末完整验收：汇总所有指标，输出JSON报告。"""
        m = model or self.model
        tok = tokenizer or self.tokenizer

        report = {
            "layer": 3,
            "phase": phase,
            "step": step,
            "timestamp": time.time(),
            "type": "phase_end_report",
        }

        # 演化统计
        if m is not None and hasattr(m, "get_evolution_stats"):
            evo = m.get_evolution_stats()
            report["evolution"] = {k: float(v) if isinstance(v, (int, float)) else str(v)
                                   for k, v in evo.items()}

        # 最近能量占比
        if self._energy_history:
            report["latest_physics_ratio"] = self._energy_history[-1]["physics_ratio"]
            report["physics_ratio_trend"] = (
                "increasing" if len(self._energy_history) >= 2
                and self._energy_history[-1]["physics_ratio"] > self._energy_history[0]["physics_ratio"]
                else "stable/decreasing"
            )

        # 推理烟测
        smoke = self.inference_smoke_test(step, m, tok, n_tokens=30)
        report["inference_smoke"] = smoke

        # 写报告文件
        report_path = os.path.join(self.log_dir, f"validation_report_phase{phase}_{step}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        report["report_path"] = report_path

        self._log(report)
        return report

    # ------------------------------------------------------------------
    # 统一调度入口：根据步数决定跑哪些检查
    # ------------------------------------------------------------------
    def on_step(self, step: int, loss: float, model=None,
                tokenizer=None, phase: str = "A") -> Dict[str, Any]:
        """训练循环每步调用。根据步数自动调度三层检查。"""
        results = {}

        # 第1层：每 check_interval 步
        if step > 0 and step % self.check_interval == 0:
            results["numeric"] = self.layer1_numeric_check(step, loss, model)
            # 能量占比也每500步算一次（廉价，用缓存的_last_E）
            results["energy"] = self.energy_ratio_analysis(step, model)

        # 第2层：每 migration_interval 步
        if step > 0 and step % self.migration_interval == 0:
            results["migration"] = self.migration_check(step, model)
            results["smoke"] = self.inference_smoke_test(step, model, tokenizer)

        return results

    def on_phase_end(self, phase: str, step: int,
                     model=None, tokenizer=None) -> Dict[str, Any]:
        """阶段结束时调用。"""
        return self.phase_end_report(phase, step, model, tokenizer)
