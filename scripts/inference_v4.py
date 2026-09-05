"""
AetherMind V4 推理脚本（推理架构版）
=====================================
支持三方案结合的推理架构（相对核 τ/C + 窗口/低秩 + 扩散演化）：
  - 加载 arch_mode='inference' 的 checkpoint（由 convert_train_to_inference.py 生成）
  - 在线演化：对话时让信息素沉积/扩散/固化（部署后继续长脑子）
  - 预热校准：加载后跑少量前向演化，让相对 τ 适配窗口/低秩分布

命令（交互模式）：
  /learn       开/关在线演化（信息素沉积+固化）
  /stats       显示演化统计（τ浓度/固化质量/固化轮数）
  /consolidate 手动触发一次 LTP 固化
  /temp [值]   设置采样温度  /max [值] 设置最大生成长度
  /reset       清空对话记忆与信息素（保留长期固化）
  /exit        退出
"""

import os
import sys
import json
import argparse
import glob
import time
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from configs.aethermind4_config import AetherMind4Config
from src.model.aethermind4 import AetherMind4
from src.data.dataset import build_tokenizer
from src.agent.agent_memory import AgentMemory


# ---------------------------------------------------------------------------
# 轻量知识图谱检索增强（RAG-lite）
# 在推理时从 KG 中命中与用户输入相关的三元组，作为上下文化提示注入 prompt，
# 不改变模型权重，也不依赖 Phase F 训练，仅做“事实提示”。
# ---------------------------------------------------------------------------
REL_CN = {
    "HAS_ATTR": "具有属性",
    "BELONGS_TO": "属于",
    "LOCATED_IN": "位于",
    "ALSO_KNOWN_AS": "又称",
    "MADE_OF": "由…组成",
    "COMES_FROM": "来自",
    "CONTAINS": "包含",
    "RELATED_TO": "相关",
}


def triple_to_text(h: str, r: str, t: str) -> str:
    if r == "HAS_ATTR":
        return f"{h}具有{t}的属性"
    if r == "IS_A":
        return f"{h}是一种{t}"
    if r == "BELONGS_TO":
        return f"{h}属于{t}"
    if r == "LOCATED_IN":
        return f"{h}位于{t}"
    if r == "ALSO_KNOWN_AS":
        return f"{h}又称{t}"
    if r == "MADE_OF":
        return f"{h}由{t}组成"
    if r == "COMES_FROM":
        return f"{h}来自{t}"
    if r == "CONTAINS":
        return f"{h}包含{t}"
    if r == "RELATED_TO":
        return f"{h}与{t}相关"
    if r == "LIKES":
        return f"{h}喜欢{t}"
    if r == "DEVELOPED_BY":
        return f"{h}由{t}开发"
    if r == "GOAL":
        return f"{h}的目标是{t}"
    return f"{h}与{t}{r}"


def load_knowledge_graphs(paths):
    """加载并合并一个或多个 KG JSON 文件。

    返回 (entities, triples, prio)，其中 prio[(h,r,t)] = 来源优先级(0最高)。
    排在前面的 path 优先级更高，重复三元组会保留高优先级来源。
    """
    entities = set()
    seen = {}
    entries = []  # (prio, h, r, t)
    if paths:
        for prio, p in enumerate(paths):
            p = (p or "").strip()
            if not p:
                continue
            if not os.path.exists(p):
                print(f"[KG] 未找到图谱文件, 跳过: {p}")
                continue
            try:
                with open(p, encoding="utf-8") as f:
                    kg = json.load(f)
            except Exception as e:
                print(f"[KG] 读取失败 {p}: {e}")
                continue
            # 第一个 path 视为人工精选图谱，保留全部三元组；后续自动抽取图谱
            # 过滤 HAS_ATTR（抽取规则丢弃了中间项，语义已失真，噪音最大）。
            keep_hasattr = (prio == 0)
            for e in kg.get("entities", []):
                if e:
                    entities.add(e)
            for item in kg.get("triples", []):
                if len(item) != 3:
                    continue
                h, r, t = item
                if not keep_hasattr and r == "HAS_ATTR":
                    continue
                key = (h, r, t)
                if key in seen:  # 保留更高优先级
                    continue
                seen[key] = prio
                entries.append((prio, h, r, t))
    triples = [(h, r, t) for _, h, r, t in entries]
    prio = {key: p for key, p in seen.items()}
    return list(entities), triples, prio


def build_kg_index(entities, triples):
    head_idx = {}
    tail_idx = {}
    for h, r, t in triples:
        head_idx.setdefault(h, []).append((h, r, t))
        tail_idx.setdefault(t, []).append((h, r, t))
    return head_idx, tail_idx


def retrieve_kg_facts(query: str, entities, head_idx, tail_idx, prio, topk: int = 3):
    """按实体子串命中用户输入，返回相关三元组的自然语言事实列表。

    排序：优先高优先级的图谱来源，同来源下更长的实体（更具体）优先。
    """
    if not query:
        return []
    matched = {}
    for e in entities:
        if e and e in query:
            matched[e] = len(e)
    if not matched:
        return []
    results = []  # (prio, -len, text)
    seen = set()
    for e, elen in matched.items():
        for tri in head_idx.get(e, []):
            if tri in seen:
                continue
            seen.add(tri)
            results.append((prio.get(tri, 99), -elen, triple_to_text(*tri)))
        for tri in tail_idx.get(e, []):
            if tri in seen:
                continue
            seen.add(tri)
            results.append((prio.get(tri, 99), -elen, triple_to_text(*tri)))
    results.sort(key=lambda x: (x[0], x[1]))
    return [text for _, _, text in results[:topk]]


# ---------------------------------------------------------------------------
# 对话语料检索（RAG 对话版）
# 从 muice.jsonl 中按字级 bigram 命中相似的用户提问，把对应回复作为参考注入 prompt。
# ---------------------------------------------------------------------------
def parse_dialogue_turn(text: str):
    """解析 "<|Human|>: ...<eoh>\\n<|MOSS|>: ...<eom>" -> (human, assistant)。"""
    if not text or "<|Human|>:" not in text or "<|MOSS|>:" not in text:
        return None, None
    try:
        h = text.split("<|Human|>:", 1)[1].split("<eoh>", 1)[0].strip()
        a = text.split("<|MOSS|>:", 1)[1].split("<eom>", 1)[0].strip()
    except Exception:
        return None, None
    if not h or not a:
        return None, None
    return h, a


def _char_bigrams(s: str):
    s = (s or "").strip()
    if not s:
        return set()
    toks = set()
    for i in range(len(s) - 1):
        a, b = s[i], s[i + 1]
        if a.isspace() or b.isspace():
            continue
        toks.add(a + b)
    if len(s) == 1:
        toks.add(s)
    return toks


def build_dialogue_index(path: str):
    """读取 muice.jsonl，返回 {pairs: [(h,a), ...], hb: [bigram_set, ...]}。"""
    pairs = []
    if not path or not os.path.exists(path):
        return {"pairs": pairs, "hb": []}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = obj.get("text", "") if isinstance(obj, dict) else ""
            h, a = parse_dialogue_turn(text)
            if h and a:
                pairs.append((h, a))
    hb = [_char_bigrams(h) for h, _ in pairs]
    return {"pairs": pairs, "hb": hb}


def retrieve_dialogue(query: str, didx, topk: int = 2):
    """返回 [(score, human, assistant), ...]，按 bigram 重叠度排序。"""
    q = _char_bigrams(query)
    if not q or not didx or not didx.get("pairs"):
        return []
    scored = []
    for i, (h, a) in enumerate(didx["pairs"]):
        hb = didx["hb"][i]
        if not hb:
            continue
        inter = len(q & hb)
        if inter == 0:
            continue
        scored.append((inter / len(hb), h, a))
    scored.sort(key=lambda x: -x[0])
    return scored[:topk]


def find_latest_checkpoint(ckpt_dir: str, suffix: str = "_final.pt") -> str:
    """查找最新 checkpoint（优先 *_inference.pt，否则 *_final.pt）"""
    inf = sorted(glob.glob(os.path.join(ckpt_dir, "v4_checkpoint_*_inference.pt")),
                 key=os.path.getmtime)
    if inf:
        return inf[-1]
    patterns = [f"v4_checkpoint_*{suffix}", "v4_checkpoint_*.pt"]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(os.path.join(ckpt_dir, pat)))
    if not candidates:
        return None
    finals = [c for c in candidates if c.endswith(suffix)]
    if finals:
        return max(finals, key=os.path.getmtime)
    return max(candidates, key=lambda p: int(os.path.basename(p).split("_")[-1].replace(".pt", "")))


def load_model(ckpt_path: str, device: str = "cuda", arch_mode: str = None):
    """加载模型。检测 checkpoint 的 arch_mode，推理架构用相对核。"""
    print(f"[Load] 读取检查点: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    steps = ckpt.get("steps", 0)
    phase = ckpt.get("phase", "?")
    model_cfg = ckpt.get("model_config", None)
    ckpt_arch = ckpt.get("arch_mode", "train")

    if model_cfg is None:
        model_cfg = AetherMind4Config()
        print("[Load] 警告: checkpoint 中无 model_config，使用默认配置")
    if arch_mode is None:
        arch_mode = ckpt_arch

    print("[Load] 加载Qwen2.5 tokenizer...")
    tokenizer = build_tokenizer()
    if hasattr(tokenizer, "truncation_side"):
        tokenizer.truncation_side = "left"  # 长输入时截掉最旧上下文，保留当前提问
    model_cfg.pad_token_id = tokenizer.pad_token_id
    model_cfg.eos_token_id = tokenizer.eos_token_id
    model_cfg.bos_token_id = tokenizer.eos_token_id
    model_cfg.unk_token_id = tokenizer.pad_token_id

    print(f"[Load] 训练步数: {steps}, 阶段: {phase}, 架构: {arch_mode}")
    print(f"[Load] vocab_size={model_cfg.vocab_size}, d_model={model_cfg.d_model}, "
          f"n_layers={model_cfg.n_layers}, max_seq_len={model_cfg.max_seq_len}")

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model_cfg.device = str(dev)

    # 按阶段 + 架构设置推理配置
    if arch_mode == "inference":
        # 推理架构：开启在线演化（强沉积/低蒸发/固化开）
        model_cfg.set_phase_D(progress=1.0)
        model = AetherMind4(model_cfg, arch_mode="inference")
        rel_k = ckpt.get("rel_k", 64)
        # 同步演化参数到推理注意力层（相对核 + 扩散）
        for attn in list(model.encoder.attn_layers) + list(model.decoder_attns):
            attn.rel_k = rel_k
            if rel_k != 64:
                attn.register_buffer("tau_rel", torch.ones(attn.num_heads, 2 * rel_k - 1))
                attn.register_buffer("consolidated_rel", torch.zeros(attn.num_heads, 2 * rel_k - 1))
            attn.deposit = getattr(model_cfg, "pheromone_deposit", 0.06)
            attn.rho = getattr(model_cfg, "pheromone_rho", 0.01)
            attn.diffusion_D = 0.1  # 方案C：扩散系数（长程结构保留、短程噪声抹平）
    else:
        ph = str(phase)
        if ph == "D":
            model_cfg.set_phase_D(progress=1.0)
        elif ph == "E":
            model_cfg.set_phase_E(progress=1.0)
        elif ph == "F":
            model_cfg.set_phase_F(progress=1.0)
        elif ph == "G":
            model_cfg.set_phase_G(progress=1.0)
        else:
            model_cfg.set_phase_C(progress=1.0)
        model = AetherMind4(model_cfg, arch_mode="train")

    # 载入 state_dict（arch_mode 不同时相对核 key 不同，strict=False）
    state = ckpt["model_state"]
    if any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}
    model_state = model.state_dict()
    loaded = 0
    skipped = 0
    for k, v in state.items():
        if k in model_state and v.shape == model_state[k].shape:
            model_state[k].copy_(v)
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(model_state, strict=False)
    print(f"[Load] 载入 {loaded} 张量, 跳过 {skipped} 个")

    # 【验证】打印实际注意力类 + 相对核是否存在（肉眼确认双架构是否真正生效）
    first_attn = model.encoder.attn_layers[0]
    has_rel = any('tau_rel' in k for k in model.state_dict().keys())
    has_abs = any(k.endswith('.tau') and '.tau_rel' not in k and v.dim() == 3
                  for k, v in model.state_dict().items())
    print(f"[Verify] 注意力类={type(first_attn).__name__} | "
          f"相对核tau_rel={'有' if has_rel else '无'} | 绝对τ(3D)={'有' if has_abs else '无'}")

    # train 架构：绝对坐标 τ/C 无法跨会话迁移，清空（防止训练残影污染注意力）
    if arch_mode != 'inference':
        print("[Load] train架构推理: 清空绝对 τ/C（绝对坐标偏置不可跨会话迁移）")
        model.evolver.reset_all_pheromones()
        for attn in list(model.encoder.attn_layers) + list(model.decoder_attns):
            if hasattr(attn, 'consolidated'):
                with torch.no_grad():
                    attn.consolidated.zero_()

    model = model.to(dev)
    model.eval()

    # 演化统计：清空后以实时统计为准（ckpt 里的绝对值已不可用于新会话）
    cur = model.evolver.get_evolution_stats()
    print(f"[Load] 演化统计(实时): tau_conc={cur.get('tau_concentration', 1.0):.3f}, "
          f"cons_mass={cur.get('consolidation_mass', 0.0):.3f}, "
          f"固化轮数={cur.get('consolidation_rounds', 0)}")

    params = sum(p.numel() for p in model.parameters())
    print(f"[Device] {dev}")
    print(f"[Model] 参数量: {params:,} ({params * 4 / 1024 / 1024:.1f} MB)")

    return model, tokenizer, model_cfg, dev, arch_mode


def calibrate(model, tokenizer, model_cfg, device, steps: int = 50):
    """预热校准：冻结 W，跑少量前向 + 演化，让相对 τ 适配新计算图（无 BP）。"""
    if steps <= 0:
        return
    print(f"[Calibrate] 预热校准 {steps} 步（冻结W，仅信息素演化）...")
    for p in model.parameters():
        p.requires_grad = False
    warm_prompts = [
        "你好", "今天天气怎么样", "你能帮我写一段诗吗", "介绍一下你自己",
        "1+1等于几", "有什么有趣的知识", "讲个故事", "什么是物理",
    ]
    n = 0
    for i in range(steps):
        prompt = warm_prompts[i % len(warm_prompts)]
        enc = tokenizer(prompt, truncation=True, max_length=model_cfg.max_seq_len - 16,
                        padding=False, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        with torch.no_grad():
            out = model(ids, task_id=0, t=i, phase="generate")
            _ = out["logits"]
        # 自由能驱动的信息素演化（方案C：扩散）
        model.evolution_step(i, phase="D")
        n += 1
    cur = model.evolver.get_evolution_stats()
    print(f"[Calibrate] 完成 {n} 步, tau_conc={cur.get('tau_concentration', 1.0):.3f}, "
          f"cons_mass={cur.get('consolidation_mass', 0.0):.3f}")


@torch.no_grad()
def generate(model, tokenizer, model_cfg, device, prompt: str,
             max_new_tokens: int = 200, temperature: float = 0.8,
             top_k: int = 40, repetition_penalty: float = 1.1, learn: bool = False):
    """生成文本。learn=True 时每步 forward 后做自由能驱动的信息素演化。"""
    enc = tokenizer(prompt, truncation=True, max_length=model_cfg.max_seq_len - max_new_tokens,
                    padding=False, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    B, S = input_ids.shape
    enc_full = tokenizer(prompt, padding=False, return_tensors="pt")
    S_orig = enc_full["input_ids"].shape[1]
    print(f"[Encode] prompt原始={S_orig} tokens, 实际送入={S} tokens"
          + ("(已截断)" if S_orig > S else ""))

    # 回合结束标记：训练语料用 <eom> 结束 assistant 回复、<eoh> 结束 Human 回合。
    # 它们不是词表特殊 token，而是普通 BPE 片段（<eom>=[27,68,316,29]），
    # 生成时必须在这些序列处停止，否则模型会自行"演"出后续多轮假对话。
    stop_seqs = []
    for marker in ("<eom>", "<eoh>"):
        mids = tokenizer(marker, add_special_tokens=False)["input_ids"]
        if mids:
            stop_seqs.append(mids)

    generated = input_ids.clone()
    start_time = time.time()
    tok_count = 0
    stop_reason = "max_len"

    for step in range(max_new_tokens):
        cur_ids = generated[:, -model_cfg.max_seq_len:]
        out = model(cur_ids, task_id=0, t=step, phase="generate")
        if learn:
            # 在线演化：自由能下降 dF 作为信用信号，驱动信息素沉积/扩散/固化
            model.evolution_step(step, phase="D")
        logits = out["logits"][:, -1, :].float()

        T = max(temperature, 1e-3)
        logits = logits / T
        if repetition_penalty > 1.0:
            for prev_id in generated[0].tolist():
                logits[0, prev_id] /= repetition_penalty
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
        tok_count += 1
        if (next_token == model_cfg.eos_token_id).all():
            stop_reason = "eos"
            break
        # 检测 <eom>/<eoh> 结束序列（出现在新生成部分的尾部即停止）
        new_so_far = generated[0, S:].tolist()
        hit = False
        for seq in stop_seqs:
            if len(new_so_far) >= len(seq) and new_so_far[-len(seq):] == seq:
                stop_reason = tokenizer.decode(seq).strip()
                hit = True
                break
        if hit:
            break

    elapsed = time.time() - start_time
    speed = tok_count / elapsed if elapsed > 0 else 0
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2 if device.type == 'cuda' else 0.0
    print(f"[Generate] 温度={temperature:.3f}, 生成{tok_count}tokens, {speed:.1f} tok/s | "
          f"S={S}->{generated.shape[1]}, 停止={stop_reason}, 峰值显存={peak_mem:.1f}MB, 耗时={elapsed:.2f}s")

    new_ids = generated[0, S:].tolist()
    if model_cfg.eos_token_id in new_ids:
        new_ids = new_ids[:new_ids.index(model_cfg.eos_token_id)]
    # 截掉 <eom>/<eoh> 结束序列及其后的所有 token（防止模型自演多轮对话）
    for seq in stop_seqs:
        cut = None
        for i in range(len(new_ids) - len(seq) + 1):
            if new_ids[i:i + len(seq)] == seq:
                cut = i
                break
        if cut is not None:
            new_ids = new_ids[:cut]
    try:
        text = tokenizer.decode(new_ids, skip_special_tokens=True, errors="ignore")
    except TypeError:
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
    # 文本级兜底：若格式标记以碎片形式出现，在第一个标记处截断
    for marker in ("<eom>", "<eoh>", "<|Human|>", "<|MOSS|>"):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text.replace("\ufffd", "").strip()


def interactive_mode(model, tokenizer, model_cfg, device, arch_mode, kg=None, kg_topk=3,
                     didx=None, dialogue_topk=2, rag_first=True, rag_min_score=0.6):
    print()
    arch_label = "推理架构[相对核]" if arch_mode == "inference" else "训练架构[绝对坐标]"
    print("=" * 60)
    print(f"  AetherMind V4 交互推理 · {arch_label}")
    print(f"  架构: {arch_mode} | 输入空行退出")
    print("  命令: /learn /stats /consolidate /temp [值] /max [值] /reset /memory /kg /ref /rag /exit")
    print("=" * 60)

    temperature = 0.7
    max_new = 200
    learn = False
    kg_enabled = kg is not None
    if kg is not None:
        print(f"[KG] 知识图谱检索增强已就绪 ({len(kg['entities'])} 实体), 输入 /kg 切换")
    dialogue_enabled = didx is not None and bool(didx.get("pairs"))
    if dialogue_enabled:
        print(f"[DLG] 对话检索增强已就绪 ({len(didx['pairs'])} 条), 输入 /ref 切换")
    if rag_first and dialogue_enabled:
        print(f"[RAG] 检索直答已开启 (置信度>={rag_min_score}), 输入 /rag 切换")

    # Agent式对话记忆：完整缓存+每轮压缩+按需检索，替代字符串拼接+硬截断
    memory = AgentMemory(tokenizer=tokenizer,
                         token_budget=192,
                         recent_turns=3,
                         retrieve_topk=2,
                         retrieve_min_score=0.12,
                         summary_max_items=10)

    def show_stats():
        s = model.evolver.get_evolution_stats()
        print(f"[Stats] τ浓度={s.get('tau_concentration', 1.0):.3f} | "
              f"固化质量={s.get('consolidation_mass', 0.0):.3f} | "
              f"固化轮数={s.get('consolidation_rounds', 0)}")

    while True:
        try:
            user_input = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            print("再见！")
            break

        low = user_input.lower()
        if low == "/exit":
            break
        elif low == "/learn":
            learn = not learn
            print(f"[在线演化 {'开启' if learn else '关闭'}]")
            continue
        elif low == "/kg":
            kg_enabled = not kg_enabled
            print(f"[知识图谱增强 {'开启' if kg_enabled else '关闭'}]")
            continue
        elif low == "/ref":
            dialogue_enabled = not dialogue_enabled
            print(f"[对话检索增强 {'开启' if dialogue_enabled else '关闭'}]")
            continue
        elif low == "/rag":
            rag_first = not rag_first
            print(f"[检索直答 {'开启' if rag_first else '关闭'}] (置信度>={rag_min_score})")
            continue
        elif low == "/stats":
            show_stats()
            continue
        elif low == "/memory":
            st = memory.stats()
            print(f"[Memory] {st} | 摘要: {memory.summary if memory.summary else '(空)'}")
            continue
        elif low == "/consolidate":
            model.evolver.consolidate_all()
            print("[固化] 已手动触发 LTP 固化")
            show_stats()
            continue
        elif low == "/reset":
            memory.reset()
            if hasattr(model, 'evolver'):
                model.evolver.reset_all_pheromones()
            print("[记忆已清空]（长期固化保留）")
            continue
        elif low.startswith("/temp"):
            parts = user_input.split()
            if len(parts) >= 2:
                try:
                    temperature = float(parts[1])
                    print(f"[温度设置为 {temperature}]")
                except ValueError:
                    print("[用法: /temp 0.8]")
            continue
        elif low.startswith("/max"):
            parts = user_input.split()
            if len(parts) >= 2:
                try:
                    max_new = int(parts[1])
                    print(f"[最大生成长度设置为 {max_new}]")
                except ValueError:
                    print("[用法: /max 200]")
            continue

        kg_ctx = ""
        if kg_enabled and kg is not None:
            facts = retrieve_kg_facts(user_input, kg["entities"], kg["head_idx"], kg["tail_idx"], kg["prio"], kg_topk)
            if facts:
                kg_ctx = "【相关知识】" + "；".join(facts) + "。"
                print(f"[KG] 命中 {len(facts)} 条: {facts}")

        refs = []
        if dialogue_enabled and didx is not None:
            refs = retrieve_dialogue(user_input, didx, dialogue_topk)
            if refs:
                print(f"[DLG] 命中 {len(refs)} 条: {[h[:20] for _, h, _ in refs]}")

        # 检索直答：高置信命中参考对话时直接返回参考答案，绕过生成
        # （当前检查点未训练成功时仍可正常对话；重训后可 /rag 关闭）
        if rag_first and refs and refs[0][0] >= rag_min_score:
            best_score, best_h, best_a = refs[0]
            print(f"[RAG] 高置信命中 ({best_score:.2f})「{best_h[:40]}」，直接返回参考答案")
            print("-" * 50)
            print(f"AI> {best_a}")
            memory.update(user_input, best_a)
            continue

        dialogue_ctx = ""
        if refs:
            dialogue_ctx = "【参考对话】" + "；".join(
                f"「{h[:40]}」→「{a[:80]}」" for _, h, a in refs) + "。"

        # 检索命中作为记忆源：DLG参考也写入压缩摘要（模型学到的关键回复）
        extra_ctx = "\n".join(x for x in (kg_ctx, dialogue_ctx) if x)

        # Agent记忆组装上下文（预算内），替代 conversation_history 拼接+截断
        cur_turn = f"<|Human|>: {user_input}<eoh>\n<|MOSS|>:"
        cur_toks = len(tokenizer(cur_turn, add_special_tokens=False)["input_ids"])
        memory.token_budget = max(64, model_cfg.max_seq_len - max_new - cur_toks - 24)
        mem_ctx = memory.build_context(user_input, extra_ctx=extra_ctx)

        prompt = (mem_ctx + "\n" if mem_ctx else "") + cur_turn

        print("-" * 50)
        response = generate(model, tokenizer, model_cfg, device, prompt,
                            max_new_tokens=max_new, temperature=temperature, learn=learn)
        print(f"AI> {response}")
        memory.update(user_input, response)


def main():
    parser = argparse.ArgumentParser(description="AetherMind V4 推理（推理架构）")
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint路径（优先 *_inference.pt）")
    parser.add_argument("--ckpt_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "checkpoints_v4_fixed"),
                        help="checkpoint目录（默认相对项目根，解压到任意路径均可）")
    parser.add_argument("--prompt", type=str, default=None, help="单次生成prompt")
    parser.add_argument("--max_new", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--arch_mode", type=str, default=None,
                        help="'inference' 用相对核推理架构, 默认自动检测 checkpoint")
    parser.add_argument("--warmup", type=int, default=0,
                        help="预热校准步数（推理架构加载后跑，默认0跳过）")
    parser.add_argument("--learn", action="store_true", help="默认开启在线演化")
    parser.add_argument("--kg_path", type=str, default=None,
                        help="知识图谱JSON路径(逗号分隔多个, 传空字符串可关闭)")
    parser.add_argument("--kg_topk", type=int, default=3,
                        help="每次检索注入的最大三元组条数")
    parser.add_argument("--dialogue_path", type=str, default=None,
                        help="对话语料JSONL路径(如muice.jsonl, 用于对话级检索; 传空字符串可关闭)")
    parser.add_argument("--dialogue_topk", type=int, default=2,
                        help="每次注入的最大参考对话条数")
    parser.add_argument("--rag_first", action="store_true", default=True,
                        help="检索直答：DLG高置信命中时直接返回参考答案（默认开）")
    parser.add_argument("--no_rag_first", action="store_true",
                        help="关闭检索直答，全部走模型生成")
    parser.add_argument("--rag_min_score", type=float, default=0.6,
                        help="检索直答的置信度阈值")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[提示] 已忽略无法识别的参数: {unknown}")

    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpt_path = find_latest_checkpoint(args.ckpt_dir)
    if ckpt_path is None or not os.path.exists(ckpt_path):
        print("错误: 未找到checkpoint！请先训练模型或运行 convert_train_to_inference.py")
        print(f"查找目录: {args.ckpt_dir}")
        sys.exit(1)

    model, tokenizer, model_cfg, device, arch_mode = load_model(ckpt_path, args.device, args.arch_mode)

    # 加载知识图谱（可选，RAG-lite 提示增强）
    kg = None
    if args.kg_path is not None:
        kg_paths = [p for p in args.kg_path.split(",") if p.strip()]
        kg_entities, kg_triples, kg_prio = load_knowledge_graphs(kg_paths)
        if kg_entities:
            head_idx, tail_idx = build_kg_index(kg_entities, kg_triples)
            kg = {"entities": kg_entities, "head_idx": head_idx, "tail_idx": tail_idx,
                  "prio": kg_prio}
            print(f"[KG] 已加载 {len(kg_triples)} 三元组 / {len(kg_entities)} 实体 "
                  f"(topk={args.kg_topk}) | 来源: {kg_paths}")

    # 加载对话语料（可选，RAG 对话版）
    didx = None
    if args.dialogue_path is not None:
        dialogue_paths = [p for p in args.dialogue_path.split(",") if p.strip()]
        didx = {"pairs": [], "hb": []}
        for p in dialogue_paths:
            d = build_dialogue_index(p)
            didx["pairs"].extend(d["pairs"])
            didx["hb"].extend(d["hb"])
        if didx["pairs"]:
            print(f"[DLG] 已加载 {len(didx['pairs'])} 条参考对话 "
                  f"(topk={args.dialogue_topk}) | 来源: {dialogue_paths}")
        else:
            didx = None

    if args.warmup > 0 and arch_mode == "inference":
        calibrate(model, tokenizer, model_cfg, device, steps=args.warmup)

    if args.prompt:
        kg_ctx = ""
        if kg is not None:
            facts = retrieve_kg_facts(args.prompt, kg["entities"], kg["head_idx"], kg["tail_idx"], kg["prio"], args.kg_topk)
            if facts:
                kg_ctx = "【相关知识】" + "；".join(facts) + "。"
                print(f"[KG] 命中 {len(facts)} 条: {facts}")
        dialogue_ctx = ""
        if didx is not None:
            refs = retrieve_dialogue(args.prompt, didx, args.dialogue_topk)
            if refs:
                dialogue_ctx = "【参考对话】" + "；".join(
                    f"「{h[:40]}」→「{a[:80]}」" for _, h, a in refs) + "。"
                print(f"[DLG] 命中 {len(refs)} 条: {[h[:20] for _, h, _ in refs]}")
        prompt = f"<|Human|>: {args.prompt}<eoh>\n<|MOSS|>:"
        if kg_ctx:
            prompt = kg_ctx + "\n" + prompt
        if dialogue_ctx:
            prompt = dialogue_ctx + "\n" + prompt
        print(f"\nPrompt: {args.prompt}\n" + "-" * 50)
        response = generate(model, tokenizer, model_cfg, device, prompt,
                            max_new_tokens=args.max_new, temperature=args.temperature,
                            learn=args.learn)
        print(f"AI> {response}")
    else:
        rag_first = args.rag_first and not args.no_rag_first
        interactive_mode(model, tokenizer, model_cfg, device, arch_mode, kg=kg, kg_topk=args.kg_topk,
                         didx=didx, dialogue_topk=args.dialogue_topk,
                         rag_first=rag_first, rag_min_score=args.rag_min_score)


if __name__ == "__main__":
    main()