# -*- coding: utf-8 -*-
"""
convert_muice.py - 沐雪数据集(Muice-Dataset)转训练格式
把 Moemuu/Muice-Dataset 的对话 jsonl 转成与推理一致的模板文本:
    <|Human|>: xxx<eoh>
    <|MOSS|>: yyy<eom>
输出到 03_dialogue/muice-jsonl/muice.jsonl (每行 {"text": ...}),
StreamingTextDataset 的 _extract_text 递归提取即可直接训练。
用法: C:/Python312/python.exe scripts/convert_muice.py
"""
import os
import json
import glob
import sys

SRC = "D:/AetherMind-Nano3/03_dialogue/muice-dataset"
DST_DIR = "D:/AetherMind-Nano3/03_dialogue/muice-jsonl"
DST = os.path.join(DST_DIR, "muice.jsonl")
MAX_CHARS = 8192  # 与 dataset._extract_text 的截断一致


def to_template(conversation):
    """conversation: [{"human":..., "assistant":...}, ...] -> 模板文本"""
    parts = []
    for turn in conversation:
        h = (turn.get("human") or "").strip()
        a = (turn.get("assistant") or "").strip()
        if h:
            parts.append(f"<|Human|>: {h}<eoh>")
        if a:
            parts.append(f"<|MOSS|>: {a}<eom>")
    return "\n".join(parts)


def main():
    files = glob.glob(os.path.join(SRC, "**", "*.jsonl"), recursive=True)
    files = [f for f in files if os.path.basename(f).startswith(("train", "test"))
             or "Customized" in f]
    os.makedirs(DST_DIR, exist_ok=True)
    n = 0
    with open(DST, "w", encoding="utf-8") as out:
        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        conv = obj.get("conversation")
                        if not isinstance(conv, list) or not conv:
                            continue
                        text = to_template(conv)
                        if len(text) < 8:
                            continue
                        text = text[:MAX_CHARS]
                        out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                        n += 1
            except Exception as e:
                print(f"[convert] 跳过 {fp}: {e}", file=sys.stderr)
    print(f"[convert] 完成: {n} 条对话 -> {DST}")
    print(f"[convert] 文件大小: {os.path.getsize(DST)/1024:.1f} KB")


if __name__ == "__main__":
    main()
