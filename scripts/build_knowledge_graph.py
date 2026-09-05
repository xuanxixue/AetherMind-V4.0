# -*- coding: utf-8 -*-
"""
build_knowledge_graph.py - 从 MOSS + 沐雪对话数据搭建轻量知识图谱
规则抽取三元组 (h, r, t), 输出 knowledge_graph.json:
    {"entities": [...], "triples": [[h, r, t], ...]}
供 Phase F (KG对齐) 训练使用。
用法: C:/Python312/python.exe scripts/build_knowledge_graph.py
"""
import os
import re
import json
import glob
import sys

DATA_DIRS = [
    "D:/AetherMind-Nano3/03_dialogue/moss-003-sft-data",
    "D:/AetherMind-Nano3/03_dialogue/muice-jsonl",
]
OUT = "D:/AetherMind-Nano3/03_dialogue/knowledge_graph.json"
MAX_LINES_PER_FILE = 3000  # 每个文件最多扫3000行, 控制耗时

# 关系抽取规则: (正则, 关系名, h组, t组)
RULES = [
    (re.compile(r"([\u4e00-\u9fa5A-Za-z0-9]{2,12})是([\u4e00-\u9fa5A-Za-z0-9]{2,20})的([\u4e00-\u9fa5A-Za-z0-9]{2,12})"), "HAS_ATTR", 1, 3),
    (re.compile(r"([\u4e00-\u9fa5A-Za-z0-9]{2,12})属于([\u4e00-\u9fa5A-Za-z0-9]{2,20})"), "BELONGS_TO", 1, 2),
    (re.compile(r"([\u4e00-\u9fa5A-Za-z0-9]{2,12})位于([\u4e00-\u9fa5A-Za-z0-9]{2,20})"), "LOCATED_IN", 1, 2),
    (re.compile(r"([\u4e00-\u9fa5A-Za-z0-9]{2,12})被称为([\u4e00-\u9fa5A-Za-z0-9]{2,20})"), "ALSO_KNOWN_AS", 1, 2),
    (re.compile(r"([\u4e00-\u9fa5A-Za-z0-9]{2,12})由([\u4e00-\u9fa5A-Za-z0-9]{2,20})组成"), "MADE_OF", 1, 2),
    (re.compile(r"([\u4e00-\u9fa5A-Za-z0-9]{2,12})来自([\u4e00-\u9fa5A-Za-z0-9]{2,20})"), "COMES_FROM", 1, 2),
    (re.compile(r"([\u4e00-\u9fa5A-Za-z0-9]{2,12})包含([\u4e00-\u9fa5A-Za-z0-9]{2,20})"), "CONTAINS", 1, 2),
    (re.compile(r"([\u4e00-\u9fa5A-Za-z0-9]{2,12})与([\u4e00-\u9fa5A-Za-z0-9]{2,20})相关"), "RELATED_TO", 1, 2),
]

# 过滤纯数字/常见停用词(避免垃圾实体)
STOP = {"什么", "为什么", "怎么", "如何", "这个", "那个", "你们", "我们", "他们", "自己",
        "一个", "不是", "没有", "可以", "就是", "这么", "那样", "的话", "时候"}
# 虚词/代词/常见碎片前缀/后缀(实体必须是实义名词短语)
BAD_PREFIX = ("其", "这", "那", "我", "你", "他", "她", "它", "们", "我们", "你们", "他们",
              "它们", "它的", "一个", "一种", "这个", "那个", "本书", "本文", "特别", "有些",
              "很多", "一些", "别的", "任何", "所有", "一切", "这样", "那样", "怎么", "为什么",
              "什么", "如何", "请", "让", "把", "被", "对", "对于", "关于", "通过", "使用",
              "利用", "根据", "按照", "由于", "因为", "所以", "但是", "然而", "如果", "虽然",
              "只要", "无论", "以及", "并", "而", "且", "或")
BAD_SUFFIX = ("的", "了", "着", "过", "吧", "吗", "呢", "啊", "呀", "哦", "和", "与", "或",
              "在", "是", "有", "不", "没", "到", "中", "上", "下", "里", "内", "外", "后",
              "前", "时", "就", "都", "也", "很", "最", "及", "以及", "们")


def is_valid_ent(e):
    if not e or len(e) < 2 or len(e) > 10 or e in STOP:
        return False
    if re.fullmatch(r"[\d\W]+", e):
        return False
    if e.startswith(BAD_PREFIX):
        return False
    if e.endswith(BAD_SUFFIX):
        return False
    # 含明显碎片符号
    if re.search(r"[，。、；：！？,.;:!?()（）「」『』<>《》]", e):
        return False
    return True


def extract_texts(fp):
    """逐行读jsonl, 递归提取字符串"""
    try:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i >= MAX_LINES_PER_FILE:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    obj = None
                parts = []

                def _collect(o):
                    if isinstance(o, str):
                        if len(o) > 1:
                            parts.append(o)
                    elif isinstance(o, dict):
                        for v in o.values():
                            _collect(v)
                    elif isinstance(o, list):
                        for v in o:
                            _collect(v)
                _collect(obj)
                yield " ".join(parts)
    except Exception as e:
        print(f"[kg] 跳过 {fp}: {e}", file=sys.stderr)


def main():
    triples = []
    seen = set()
    n_files = 0
    for d in DATA_DIRS:
        files = glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True)
        for fp in files:
            if os.path.basename(fp).startswith("."):
                continue
            n_files += 1
            for text in extract_texts(fp):
                for rule, rel, g_h, g_t in RULES:
                    for m in rule.finditer(text):
                        h = m.group(g_h).strip()
                        t = m.group(g_t).strip()
                        if not (is_valid_ent(h) and is_valid_ent(t)):
                            continue
                        key = (h, rel, t)
                        if key in seen:
                            continue
                        seen.add(key)
                        triples.append([h, rel, t])

    entities = set()
    for h, _, t in triples:
        entities.add(h)
        entities.add(t)
    entities = sorted(entities)

    kg = {"entities": entities, "triples": triples}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(kg, f, ensure_ascii=False, indent=1)

    print(f"[kg] 扫描文件: {n_files}")
    print(f"[kg] 三元组: {len(triples)}")
    print(f"[kg] 实体: {len(entities)}")
    print(f"[kg] 输出: {OUT} ({os.path.getsize(OUT)/1024:.1f} KB)")
    # 打印样例
    for t in triples[:8]:
        print("   ", t)


if __name__ == "__main__":
    main()
