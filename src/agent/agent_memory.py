"""
Agent式对话记忆系统
====================
替代旧的"字符串拼接 + 超长硬截断"方案：
  1. 完整缓存（Full Cache）  —— 所有对话轮次永久保存，信息不丢失
  2. 每轮压缩（Compression） —— 滚动摘要只保留关键信息（自称/偏好/事实/数字）
  3. 按需检索（Retrieval）   —— 当前输入与历史轮相似度命中时才召回该轮细节
  4. 预算组装（Budgeting）   —— 摘要+检索记忆+最近K轮原文在token预算内组装，
                                 从源头控制长度，不依赖tokenizer截断

设计原则（用户需求）：
  "截断不如Agent方式：每轮只压缩保留关键信息，其余缓存，需要时命中召回。"
"""

import re
from typing import List, Tuple, Optional, Dict

# 压缩时优先保留的关键信息模式（按显著度排序）
_KEY_PATTERNS = [
    (re.compile(r"(我叫|我是|我的名字[是叫])[^，。？！\s]{1,12}"), 3),   # 自称/身份
    (re.compile(r"(我喜欢|我爱|我讨厌|我不喜欢)[^，。？！]{1,16}"), 3),  # 偏好
    (re.compile(r"(我住在|我来自|我在)[^，。？！]{1,16}"), 2),          # 位置
    (re.compile(r"(记得|别忘了|记住)[^，。？！]{1,20}"), 2),            # 显式记忆请求
    (re.compile(r"\d+"), 1),                                            # 数字
    (re.compile(r"[？?]$"), 1),                                          # 问句
]

_SENT_SPLIT = re.compile(r"[。！？!?；;\n]+")
_CLS = "【对话摘要】"
_RCL = "【相关记忆】"


def _char_bigrams(s: str) -> set:
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


def _salience(sentence: str) -> int:
    score = 0
    for pat, w in _KEY_PATTERNS:
        if pat.search(sentence):
            score += w
    return score


def _compress_text(text: str, max_chars: int) -> str:
    """规则式压缩：按显著度挑选句子，拼接并限长。"""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if not sents:
        return text[:max_chars]
    scored = sorted(enumerate(sents), key=lambda x: -_salience(x[1]))
    chosen, used, budget = [], 0, max_chars
    for idx, s in scored:
        if used + len(s) + 1 <= budget:
            chosen.append((idx, s))
            used += len(s) + 1
        if used >= budget:
            break
    if not chosen:  # 没有短句可挑，直接截断最显著句
        return scored[0][1][:max_chars]
    chosen.sort()  # 恢复原句序
    return "；".join(s for _, s in chosen)


class AgentMemory:
    """Agent式对话记忆：完整缓存 + 每轮压缩 + 按需检索 + 预算组装"""

    def __init__(self, tokenizer=None,
                 token_budget: int = 256,
                 recent_turns: int = 3,
                 retrieve_topk: int = 2,
                 retrieve_min_score: float = 0.12,
                 summary_max_items: int = 10,
                 turn_max_chars: int = 64):
        self.tokenizer = tokenizer
        self.token_budget = token_budget          # 记忆上下文总token预算
        self.recent_turns = recent_turns          # 最近K轮保留原文
        self.retrieve_topk = retrieve_topk        # 检索召回条数
        self.retrieve_min_score = retrieve_min_score
        self.summary_max_items = summary_max_items
        self.turn_max_chars = turn_max_chars      # 摘要中每轮压缩上限

        self.turns: List[Tuple[str, str]] = []    # 完整缓存 [(user, ai), ...] 永不丢弃
        self.summary: List[str] = []              # 滚动压缩摘要（每轮一行）

    # ------------------------------------------------------------------
    def reset(self):
        self.turns.clear()
        self.summary.clear()

    def _n_tokens(self, text: str) -> int:
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
            except Exception:
                pass
        return max(1, len(text))  # 中文近似 1字≈1token

    # ------------------------------------------------------------------
    def update(self, user_text: str, ai_text: str):
        """记录一轮对话：进完整缓存 + 生成压缩摘要行。"""
        user_text, ai_text = (user_text or "").strip(), (ai_text or "").strip()
        self.turns.append((user_text, ai_text))
        u_key = _compress_text(user_text, self.turn_max_chars)
        a_key = _compress_text(ai_text, max(24, self.turn_max_chars // 2))
        line = f"用户:{u_key}" + (f"｜AI:{a_key}" if a_key else "")
        self.summary.append(line)
        # 摘要超预算：淘汰最不显著且最旧的行
        while len(self.summary) > self.summary_max_items:
            scored = [(i, _salience(s)) for i, s in enumerate(self.summary[:-1])]
            scored.sort(key=lambda x: (x[1], x[0]))  # 显著度低且旧的先淘汰
            self.summary.pop(scored[0][0])

    # ------------------------------------------------------------------
    def retrieve(self, query: str, exclude_recent: int = 0) -> List[Tuple[float, str, str]]:
        """按 bigram 相似度在完整缓存中检索相关历史轮（排除最近K轮）。"""
        q = _char_bigrams(query)
        if not q or not self.turns:
            return []
        n_excl = len(self.turns) - max(0, exclude_recent)
        scored = []
        for i, (h, a) in enumerate(self.turns[:n_excl]):
            hb = _char_bigrams(h)
            if not hb:
                continue
            inter = len(q & hb)
            if inter == 0:
                continue
            score = inter / len(hb)  # 历史轮被当前输入覆盖的比例
            if score >= self.retrieve_min_score:
                scored.append((score, h, a))
        scored.sort(key=lambda x: -x[0])
        return scored[:self.retrieve_topk]

    # ------------------------------------------------------------------
    def build_context(self, user_text: str, extra_ctx: str = "") -> str:
        """在token预算内组装记忆上下文（不含当前轮的对话模板）。

        结构（旧→新）：
          【对话摘要】  滚动压缩的关键信息
          【相关记忆】  当前输入命中的历史轮（按需召回）
          extra_ctx     外部RAG上下文（KG/DLG参考）
          最近K轮对话原文（MOSS格式，逐轮加入直到预算耗尽）
        当前轮 prompt 由调用方拼接，其预算已在本方法外扣除。
        """
        blocks: List[str] = []

        if self.summary:
            blocks.append(_CLS + "\n" + "\n".join(f"- {s}" for s in self.summary))

        hits = self.retrieve(user_text, exclude_recent=self.recent_turns)
        if hits:
            lines = [f"- 用户曾问「{_compress_text(h, 32)}」→ AI答「{_compress_text(a, 48)}」"
                     for _, h, a in hits]
            blocks.append(_RCL + "\n" + "\n".join(lines))

        if extra_ctx:
            blocks.append(extra_ctx.strip())

        # 最近K轮原文（从新到旧收集，组装时恢复旧→新顺序）
        recent_texts = []
        for h, a in self.turns[-self.recent_turns:]:
            recent_texts.append(f"<|Human|>: {h}<eoh>\n<|MOSS|>: {a}<eom>")

        # 预算分配：摘要/检索是压缩信息优先保留，最近轮原文用剩余预算
        head = "\n\n".join(blocks)
        head_toks = self._n_tokens(head)
        remaining = self.token_budget - head_toks

        kept = []
        for txt in reversed(recent_texts):  # 新→旧
            t = self._n_tokens(txt)
            if remaining - t >= 0:
                kept.append(txt)
                remaining -= t
            else:
                break
        kept.reverse()  # 恢复旧→新

        parts = [head] if head else []
        parts.extend(kept)
        return "\n".join(p for p in parts if p.strip())

    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, int]:
        return {
            "缓存轮数": len(self.turns),
            "摘要条数": len(self.summary),
            "预算": self.token_budget,
        }
