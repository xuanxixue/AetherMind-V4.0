"""
对话数据清洗脚本 — Phase G 前置
================================
把 MOSS + 沐雪(Muice) 数据清洗成统一纯对话格式，剥离所有 system prompt。

输入格式（自动识别）：
  1) MOSS:        {"meta_instruction": "...", "chat": {"turn_1": {"Human": "...", "MOSS": "..."}, ...}}
  2) 沐雪 muice-dataset: {"system": "...", "conversation": [{"human": "...", "assistant": "..."}, ...]}
  3) 沐雪 muice-jsonl:   {"text": "<|Human|>: ...<eoh>\n<|MOSS|>: ...<eom>"}

输出格式（统一，每行一条单轮对话）：
  {"text": "<|Human|>: {用户}<eoh>\n<|MOSS|>: {助手}<eom>"}

关键点：
  - 剥离 meta_instruction / system / Inner Thoughts / Commands / Tool Responses
  - 只保留 Human -> MOSS（用户 -> 助手）真实对话对
  - 分片输出（每片 <= shard_size 条），因为训练端每个 jsonl 最多读 5000 行
"""

import os
import re
import json
import glob
import argparse

# 对话标记
H_PRE = "<|Human|>:"
H_SUF = "<eoh>"
M_PRE = "<|MOSS|>:"
M_SUF = "<eom>"

# 需要剥离 / 排除的字段名
DROP_KEYS = {"meta_instruction", "Inner Thoughts", "Commands", "Tool Responses", "system"}

# 空/无效内容（去除标记后如果等于这些则丢弃）
_INVALID_CONTENT = {"", "None", "none", "null", "<None>", "<none>"}


def _clean_content(s):
    """去除首尾空白与多余的空白，规范化。"""
    if s is None:
        return ""
    s = str(s).strip()
    # 折叠连续空白
    s = re.sub(r"\s+", " ", s)
    return s


def _strip_markers(s, pre, suf):
    """从形如 '<|Human|>: X<eoh>' 中提取 X。找不到标记则返回空。"""
    s = s or ""
    s = s.strip()
    if pre in s:
        s = s.split(pre, 1)[1]
    if suf in s:
        s = s.split(suf, 1)[0]
    return _clean_content(s)


def _make_pair(human, assistant):
    """构造统一对话对。返回字符串或 None。"""
    h = _clean_content(human)
    a = _clean_content(assistant)
    # 再次剥一次标记，防止输入已含标记
    h = _strip_markers(h, H_PRE, H_SUF)
    a = _strip_markers(a, M_PRE, M_SUF)
    if h in _INVALID_CONTENT or a in _INVALID_CONTENT:
        return None
    # 丢弃过短/过长噪声
    if len(h) < 1 or len(a) < 1:
        return None
    if len(h) > 2000 or len(a) > 4000:
        return None
    return f"{H_PRE} {h}{H_SUF}\n{M_PRE} {a}{M_SUF}"


def clean_moss_obj(obj):
    """MOSS 格式 -> 若干条纯对话对（每条一个 turn）。"""
    pairs = []
    chat = obj.get("chat")
    if not isinstance(chat, dict):
        return pairs
    for key, turn in chat.items():
        if not isinstance(turn, dict):
            continue
        if key in DROP_KEYS:
            continue
        human = turn.get("Human")
        assistant = turn.get("MOSS")
        p = _make_pair(human, assistant)
        if p:
            pairs.append(p)
    return pairs


def clean_muice_dataset_obj(obj):
    """沐雪 muice-dataset 格式 -> 若干条纯对话对。"""
    pairs = []
    conv = obj.get("conversation")
    if not isinstance(conv, list):
        return pairs
    for item in conv:
        if not isinstance(item, dict):
            continue
        human = item.get("human")
        assistant = item.get("assistant")
        p = _make_pair(human, assistant)
        if p:
            pairs.append(p)
    return pairs


def clean_text_obj(obj):
    """沐雪 muice-jsonl 格式（已有 text，含标记）。"""
    text = obj.get("text")
    if not isinstance(text, str):
        return []
    text = text.strip()
    if not text:
        return []
    # 按 <eoh>/<eom> 拆轮，逐轮规范化，丢弃无意义的推文/任务标签外的空回复
    results = []
    # 尝试按 MOSS 回复分段：找到所有 human 与 moss 的配对
    # 简单稳健做法：用正则匹配所有 <|Human|>: ...<eoh> 与 <|MOSS|>: ...<eom>
    humans = re.findall(r"<\|Human\|>:\s*(.*?)<eoh>", text, flags=re.S)
    mosses = re.findall(r"<\|MOSS\|>:\s*(.*?)<eom>", text, flags=re.S)
    n = min(len(humans), len(mosses))
    for i in range(n):
        p = _make_pair(humans[i], mosses[i])
        if p:
            results.append(p)
    return results


def clean_line(line):
    """解析一行 JSON 并返回该行能产出的所有对话对列表。"""
    line = line.strip()
    if not line:
        return []
    try:
        obj = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(obj, dict):
        return []
    if "chat" in obj:
        return clean_moss_obj(obj)
    if "conversation" in obj:
        return clean_muice_dataset_obj(obj)
    if "text" in obj:
        return clean_text_obj(obj)
    return []


class ShardWriter:
    """分片写入器：每片最多 shard_size 条，超过则新开文件。"""

    def __init__(self, output_dir, shard_size=4000):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.shard_idx = 0
        self.count = 0
        self.total = 0
        self.fh = None
        os.makedirs(output_dir, exist_ok=True)

    def _open_next(self):
        if self.fh is not None:
            self.fh.close()
        path = os.path.join(self.output_dir, f"clean_part_{self.shard_idx:06d}.jsonl")
        self.fh = open(path, "w", encoding="utf-8")
        self.shard_idx += 1
        self.count = 0

    def write(self, text):
        if self.fh is None or self.count >= self.shard_size:
            self._open_next()
        self.fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        self.count += 1
        self.total += 1

    def close(self):
        if self.fh is not None:
            self.fh.close()
            self.fh = None


def main():
    p = argparse.ArgumentParser(description="清洗 MOSS + 沐雪对话数据")
    p.add_argument("--input_dir", default="d:/AetherMind-Nano3/03_dialogue")
    p.add_argument("--output_dir", default="d:/AetherMind-Nano3/03_dialogue_clean")
    p.add_argument("--shard_size", type=int, default=4000,
                   help="每片最多条数（训练端每文件最多读5000行，务必<=5000）")
    p.add_argument("--moss_max_lines", type=int, default=30000,
                   help="MOSS 大文件最多扫描的行数（每行=一个多轮对话）")
    p.add_argument("--max_total", type=int, default=None,
                   help="清洗总条数上限（None=不限制）")
    args = p.parse_args()

    # 1) 收集中等文件（muice 全部）与 MOSS 大文件分开处理
    small_files = []
    moss_files = []
    for ext in ["*.jsonl"]:
        for fp in glob.glob(os.path.join(args.input_dir, "**", ext), recursive=True):
            base = os.path.basename(fp)
            if base.startswith("clean_part_"):
                continue
            low = fp.lower()
            # MOSS 大文件
            if "moss-003" in low:
                moss_files.append(fp)
            else:
                small_files.append(fp)

    print(f"[Clean] 找到 {len(moss_files)} 个 MOSS 文件, {len(small_files)} 个沐雪/其他文件")
    writer = ShardWriter(args.output_dir, shard_size=args.shard_size)

    def _process_file(fp, max_lines=None, tag=""):
        line_no = 0
        kept = 0
        with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if max_lines is not None and line_no >= max_lines:
                    break
                line_no += 1
                for text in clean_line(line):
                    if args.max_total is not None and writer.total >= args.max_total:
                        return kept
                    writer.write(text)
                    kept += 1
        return kept

    # 2) 先清洗沐雪小文件（确保基础对话风格）
    for fp in sorted(small_files):
        kept = _process_file(fp, tag="muice")
        print(f"[Clean] 沐雪文件 {os.path.basename(fp)}: 读归零产出 {kept} 条")

    # 3) 清洗 MOSS 大文件（限行）
    for fp in sorted(moss_files):
        kept = _process_file(fp, max_lines=args.moss_max_lines, tag="moss")
        print(f"[Clean] MOSS 文件 {os.path.basename(fp)}: 扫描 {args.moss_max_lines} 行产出 {kept} 条")

    writer.close()
    print(f"[Clean] 完成! 总计 {writer.total} 条对话, "
          f"{writer.shard_idx} 个分片 -> {args.output_dir}")


if __name__ == "__main__":
    main()