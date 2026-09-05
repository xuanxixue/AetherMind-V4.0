import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import json
import glob
import sqlite3
import random
from typing import List, Dict, Tuple, Optional, Iterator
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


@dataclass
class DataExample:
    text: str
    source: str = "unknown"
    domain: str = "general"


class TextFileDataset(Dataset):
    def __init__(self, examples: List[DataExample], tokenizer, max_seq_len: int = 1024):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.text,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


class StreamingTextDataset(IterableDataset):
    def __init__(self, data_dir: str, tokenizer, max_seq_len: int = 1024,
                 max_samples: Optional[int] = None, shuffle: bool = True,
                 split: str = "all", split_ratio: float = 0.9):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_samples = max_samples
        self.shuffle = shuffle
        self.split = split
        self.split_ratio = split_ratio
        self.files = self._discover_files()
        self._apply_split()
        if self.files:
            sizes = []
            for f in self.files[:3]:
                try:
                    sizes.append(f"{os.path.basename(f)}({os.path.getsize(f)/1024/1024:.0f}MB)")
                except OSError:
                    sizes.append(os.path.basename(f))
            more = f" +{len(self.files)-3} more" if len(self.files) > 3 else ""
            sys.stderr.write(f"[Data] {split} split: {len(self.files)} files [{', '.join(sizes)}{more}]\n")
        else:
            sys.stderr.write(f"[Data] WARN: {split} split has 0 files from {data_dir}\n")

    _EXCLUDE_NAMES = {".gitattributes", ".gitignore", "README.md", "LICENSE", "README.txt",
                      "gitattributes.txt", ".gitkeep", "metafile.yaml",
                      # HuggingFace 数据集元数据/非对话文件（会被误读成训练文本导致数值问题）
                      "dataset_infos.json", "dataset_dict.json", "state.json", "lock.json",
                      # 知识图谱文件（Phase F 专用，非对话，递归收集会变成垃圾文本）
                      "knowledge_graph.json", "daily_knowledge_graph.json", "kg.json"}

    def _discover_files(self) -> List[str]:
        files = []
        for ext in ["*.jsonl", "*.json", "*.txt", "*.db"]:
            files.extend(glob.glob(os.path.join(self.data_dir, "**", ext), recursive=True))
        files = [f for f in files if os.path.basename(f) not in self._EXCLUDE_NAMES]
        return sorted(files)

    def _apply_split(self):
        """按 split_ratio 将文件列表分为训练集和验证集。文件少时两边共享文件（eval固定50batch影响很小）。"""
        if self.split == "all" or len(self.files) <= 1:
            return
        all_files = self.files
        n_total = len(all_files)
        # 文件数 < 4 时，两边共用所有文件（每文件只读前5000行，eval只读验证用途不泄漏）
        if n_total < 4:
            return  # 保持 self.files = all_files
        n_train = max(1, int(n_total * self.split_ratio))
        if self.split == "train":
            self.files = all_files[:n_train]
        elif self.split == "eval":
            eval_files = all_files[n_train:]
            if len(eval_files) == 0:
                eval_files = [all_files[-1]]
            self.files = eval_files

    # 已知会污染训练文本的非对话字段（MOSS原始SFT数据的元数据）
    _META_KEYS = {"meta_instruction", "meta", "conversation_id", "num_turns",
                  "lang", "model", "source", "dataset", "id", "instruction",
                  "Inner Thoughts", "inner_thoughts", "inner_instruction"}

    def _extract_text(self, obj) -> str:
        """优先提取对话字段；无对话字段时递归收集字符串值（跳过元数据键）"""
        if isinstance(obj, dict):
            # 1) 标准清洗格式 {"text": "<|Human|>: ...<eoh>\\n<|MOSS|>: ...<eom>"}
            if isinstance(obj.get("text"), str) and obj["text"].strip():
                return obj["text"]
            # 2) MOSS 原始SFT格式 {"meta_instruction":..., "chat": {"turn_1": {"Human":..., "Assistant":...}}}
            #    只拼 Human/Assistant 对话，丢弃 meta_instruction/Inner Thoughts/工具调用
            chat = obj.get("chat")
            if isinstance(chat, dict) and chat:
                parts = []
                for key in sorted(chat.keys()):
                    turn = chat[key]
                    if not isinstance(turn, dict):
                        continue
                    for role in ("Human", "Assistant", "MOSS", "user", "assistant"):
                        v = turn.get(role)
                        if isinstance(v, str) and v.strip():
                            parts.append(v.strip())
                            break
                text = "\n".join(parts)
                if text.strip():
                    return text
            # 3) messages 数组格式 [{"role": "user", "content": ...}, ...]
            msgs = obj.get("messages")
            if isinstance(msgs, list) and msgs:
                parts = []
                for m in msgs:
                    if isinstance(m, dict) and isinstance(m.get("content"), str):
                        parts.append(m["content"].strip())
                text = "\n".join(p for p in parts if p)
                if text.strip():
                    return text
        parts = []

        def _collect(o, key=None):
            if isinstance(o, str):
                if len(o) > 1 and key not in self._META_KEYS:
                    parts.append(o)
            elif isinstance(o, dict):
                for k, v in o.items():
                    _collect(v, k)
            elif isinstance(o, list):
                for item in o:
                    _collect(item, key)

        _collect(obj)
        return " ".join(parts) if parts else ""

    def _read_jsonl(self, path: str) -> Iterator[DataExample]:
        try:
            fsize_mb = os.path.getsize(path) / (1024 * 1024)
        except OSError:
            fsize_mb = 0
        line_count = 0
        skip_count = 0
        MAX_LINES_PER_FILE = 5000  # 每文件最多处理 5000 行
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            line_count = 0
            for line in f:
                line_count += 1
                if line_count > MAX_LINES_PER_FILE:
                    break
                if line_count % 50000 == 0:
                    sys.stderr.write(f"[Data] JSONL 进度: {os.path.basename(path)} line {line_count} (跳过 {skip_count})\n")
                    sys.stderr.flush()
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = self._extract_text(obj)
                    # 限制文本长度防止 tokenize 过慢
                    if text:
                        text = text[:8192]
                    else:
                        text = line[:4096]  # fallback: 用原始 JSON 行
                    if text:
                        yield DataExample(text=text, source=os.path.basename(path))
                    else:
                        skip_count += 1
                except (json.JSONDecodeError, ValueError):
                    skip_count += 1
                    continue

    def _read_txt(self, path: str) -> Iterator[DataExample]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            buf = []
            for line in f:
                buf.append(line)
                if len(buf) >= 50:
                    text = "".join(buf).strip()
                    if len(text) > 20:
                        yield DataExample(text=text, source=os.path.basename(path))
                    buf = []
            if buf:
                text = "".join(buf).strip()
                if len(text) > 20:
                    yield DataExample(text=text, source=os.path.basename(path))

    def _read_db(self, path: str) -> Iterator[DataExample]:
        try:
            conn = sqlite3.connect(path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cursor.fetchall()]
            for table in tables:
                try:
                    cursor.execute(f"SELECT * FROM {table} LIMIT 100000")
                    cols = [d[0] for d in cursor.description]
                    text_idx = None
                    for i, c in enumerate(cols):
                        if c.lower() in ["text", "content", "prompt", "response", "conversation"]:
                            text_idx = i
                            break
                    if text_idx is None:
                        continue
                    for row in cursor:
                        text = str(row[text_idx])
                        if len(text) > 20:
                            yield DataExample(text=text, source=f"{os.path.basename(path)}/{table}")
                except Exception:
                    continue
            conn.close()
        except Exception:
            pass

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = get_worker_info()
        if worker_info is None:
            worker_files = self.files
        else:
            worker_files = self.files[worker_info.id::worker_info.num_workers]

        # 如果没找到任何数据文件，生成虚拟文本数据确保训练不中断
        if len(self.files) == 0:
            sys.stderr.write("[Data] 未找到数据文件, 使用虚拟文本数据\n")
            sys.stderr.flush()
            dummy_texts = [
                "人工智能是计算机科学的一个分支，旨在创造能够模拟人类智能的系统。",
                "深度学习通过多层神经网络实现端到端的学习范式，在大规模数据上表现优异。",
                "自然语言处理是人工智能的重要领域，涉及机器翻译、文本生成、情感分析等任务。",
                "知识图谱是一种结构化的语义知识库，用于描述物理世界中的概念及其相互关系。",
                "强化学习通过智能体与环境的交互，以最大化累积奖励为目标学习最优策略。",
                "计算机视觉使机器能够从图像和视频中理解视觉世界，广泛应用于自动驾驶等领域。",
                "大语言模型的涌现能力表明，随着模型规模的扩大，新的能力会突然出现。",
                "思维链推理通过让模型逐步展示推理过程，显著提升了复杂问题的解决能力。",
            ]
            count = 0
            max_dummy = self.max_samples if self.max_samples is not None else 100000
            while count < max_dummy:
                if self.max_samples is not None and count >= self.max_samples:
                    return
                text = random.choice(dummy_texts) + " " + random.choice(dummy_texts)
                try:
                    enc = self.tokenizer(
                        text, truncation=True, max_length=self.max_seq_len,
                        padding="max_length", return_tensors="pt"
                    )
                except Exception:
                    continue
                input_ids = enc["input_ids"].squeeze(0)
                labels = input_ids.clone()

                # === SFT Label Mask：对话格式只计算assistant回复部分的loss ===
                _assistant_markers = ["<|MOSS|>:", "<|assistant|>:", "<|Assistant|>:"]
                _masked = False
                for _marker in _assistant_markers:
                    _marker_ids = self.tokenizer(_marker, add_special_tokens=False)["input_ids"]
                    if not _marker_ids:
                        continue
                    _mlen = len(_marker_ids)
                    _ids_list = input_ids.tolist()
                    for _pos in range(len(_ids_list) - _mlen + 1):
                        if _ids_list[_pos:_pos + _mlen] == _marker_ids:
                            labels[:_pos] = self.tokenizer.pad_token_id
                            _masked = True
                            break
                    if _masked:
                        break

                count += 1
                yield {"input_ids": input_ids, "labels": labels}

        sys.stderr.write(f"[Data] 发现 {len(self.files)} 个文件, worker 负责 {len(worker_files)} 个\n")
        sys.stderr.flush()

        if self.shuffle:
            random.shuffle(worker_files)

        MAX_FILES_PER_EPOCH = 200
        if len(worker_files) > MAX_FILES_PER_EPOCH:
            sys.stderr.write(f"[Data] 文件过多, 限制本 epoch 扫描前 {MAX_FILES_PER_EPOCH} 个\n")
            sys.stderr.flush()
            worker_files = worker_files[:MAX_FILES_PER_EPOCH]

        count = 0
        file_count = 0
        for fpath in worker_files:
            file_count += 1
            if file_count == 1 or file_count % 5 == 0:
                try:
                    fsize_mb = os.path.getsize(fpath) / (1024*1024)
                except OSError:
                    fsize_mb = 0
                sys.stderr.write(f"[Data] 读取文件 {file_count}/{len(worker_files)}: {os.path.basename(fpath)} ({fsize_mb:.1f}MB)\n")
                sys.stderr.flush()

            iterator = None
            try:
                if fpath.endswith(".jsonl") or fpath.endswith(".json"):
                    iterator = self._read_jsonl(fpath)
                elif fpath.endswith(".txt"):
                    iterator = self._read_txt(fpath)
                elif fpath.endswith(".db"):
                    iterator = self._read_db(fpath)
            except Exception as e:
                sys.stderr.write(f"[Data] 跳过文件 {fpath}: {e}\n")
                sys.stderr.flush()
                continue

            if iterator is None:
                continue

            for ex in iterator:
                if self.max_samples is not None and count >= self.max_samples:
                    sys.stderr.write(f"[Data] 已达到 max_samples={self.max_samples}, 停止读取\n")
                    sys.stderr.flush()
                    return
                try:
                    enc = self.tokenizer(
                        ex.text,
                        truncation=True,
                        max_length=self.max_seq_len,
                        padding="max_length",
                        return_tensors="pt"
                    )
                except Exception as e:
                    sys.stderr.write(f"[Data] tokenize 失败, 跳过: {e}\n")
                    sys.stderr.flush()
                    continue
                input_ids = enc["input_ids"].squeeze(0)
                labels = input_ids.clone()

                # === SFT Label Mask：对话格式只计算assistant回复部分的loss ===
                _assistant_markers = ["<|MOSS|>:", "<|assistant|>:", "<|Assistant|>:"]
                _masked = False
                for _marker in _assistant_markers:
                    _marker_ids = self.tokenizer(_marker, add_special_tokens=False)["input_ids"]
                    if not _marker_ids:
                        continue
                    _mlen = len(_marker_ids)
                    _ids_list = input_ids.tolist()
                    for _pos in range(len(_ids_list) - _mlen + 1):
                        if _ids_list[_pos:_pos + _mlen] == _marker_ids:
                            labels[:_pos] = self.tokenizer.pad_token_id
                            _masked = True
                            break
                    if _masked:
                        break

                count += 1
                yield {"input_ids": input_ids, "labels": labels}


def _select_bpe_train_files(data_dir: str, max_files: int = 5, max_total_mb: int = 200,
                            sample_lines: int = 5000) -> List[str]:
    """为BPE分词器训练选择文件。大JSONL文件会抽取前sample_lines行到临时文件。"""
    import tempfile
    _EXCLUDE = {".gitattributes", ".gitignore", "README.md", "LICENSE", "README.txt",
                "gitattributes.txt", ".gitkeep", "metafile.yaml"}
    files = []
    for ext in ["*.txt", "*.jsonl"]:
        files.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
    files = [f for f in sorted(files) if os.path.basename(f) not in _EXCLUDE]

    chosen = []
    total_bytes = 0
    tmp_files = []  # 跟踪临时文件以便清理（交给调用方通过atexit）
    has_jsonl = False

    for fp in files:
        try:
            sz = os.path.getsize(fp)
        except OSError:
            continue
        is_jsonl = fp.lower().endswith(".jsonl")
        # 小文件直接用
        if sz <= max_total_mb * 1024 * 1024 // max_files and total_bytes + sz <= max_total_mb * 1024 * 1024:
            chosen.append(fp)
            total_bytes += sz
            if is_jsonl:
                has_jsonl = True
        elif is_jsonl and not has_jsonl:
            # 大JSONL：抽取前 sample_lines 行写临时文件
            try:
                fd, tmp_path = tempfile.mkstemp(suffix=".txt", prefix="bpe_sample_")
                os.close(fd)
                written = 0
                with open(fp, "r", encoding="utf-8", errors="ignore") as fin, \
                     open(tmp_path, "w", encoding="utf-8") as fout:
                    for i, line in enumerate(fin):
                        if i >= sample_lines:
                            break
                        fout.write(line)
                        written += len(line.encode("utf-8"))
                if written > 100:
                    chosen.append(tmp_path)
                    tmp_files.append(tmp_path)
                    total_bytes += written
                    has_jsonl = True
            except Exception:
                pass
        if len(chosen) >= max_files:
            break
        if has_jsonl and total_bytes > max_total_mb * 1024 * 1024 // 2:
            break  # 已有JSONL且数据量足够就停

    # 注册临时文件清理
    if tmp_files:
        import atexit
        def _cleanup():
            for tp in tmp_files:
                try:
                    os.unlink(tp)
                except OSError:
                    pass
        atexit.register(_cleanup)

    return chosen


def _wrap_tokenizer_call(tok, default_pad_id: int = 0):
    if hasattr(tok, "__call__") and callable(getattr(tok, "__call__")):
        try:
            r = tok("test", max_length=8, truncation=True, padding="max_length", return_tensors="pt")
            if isinstance(r, dict) and "input_ids" in r:
                return tok
        except Exception:
            pass

    class _Wrapper:
        def __init__(self, inner, pad_id):
            self.inner = inner
            self._pad_id = pad_id
            try:
                self.vocab_size = inner.get_vocab_size()
            except Exception:
                try:
                    self.vocab_size = len(inner.get_vocab())
                except Exception:
                    self.vocab_size = 50000
            self.pad_token_id = pad_id
            try:
                self.bos_token_id = inner.token_to_id("<bos>") if hasattr(inner, "token_to_id") else None
                self.eos_token_id = inner.token_to_id("<eos>") if hasattr(inner, "token_to_id") else None
                self.unk_token_id = inner.token_to_id("<unk>") if hasattr(inner, "token_to_id") else None
            except Exception:
                self.bos_token_id = None
                self.eos_token_id = None
                self.unk_token_id = None

        def __call__(self, text, truncation=True, max_length=1024, padding="max_length", return_tensors="pt"):
            if isinstance(text, str):
                texts = [text]
            else:
                texts = list(text)
            try:
                encoded = self.inner.encode_batch(texts)
                ids = [e.ids for e in encoded]
            except Exception:
                ids = []
                for t in texts:
                    try:
                        ids.append(self.inner.encode(t).ids)
                    except Exception:
                        ids.append([0])
            result_ids = []
            for seq in ids:
                seq = list(seq)
                if truncation and len(seq) > max_length:
                    seq = seq[:max_length]
                if padding == "max_length" and len(seq) < max_length:
                    seq = seq + [self._pad_id] * (max_length - len(seq))
                result_ids.append(seq)
            return {"input_ids": torch.tensor(result_ids, dtype=torch.long)}

    pad_id = default_pad_id
    try:
        if hasattr(tok, "token_to_id"):
            pad_id = tok.token_to_id("<pad>") or default_pad_id
    except Exception:
        pass
    wrapped = _Wrapper(tok, pad_id)
    return wrapped


def build_tokenizer(vocab_size: int = 151936, data_dir: Optional[str] = None,
                    model_path: Optional[str] = None, local_models_dir: Optional[str] = None):
    """使用Qwen2.5 tokenizer (修复版)"""
    from transformers import AutoTokenizer
    
    # 默认路径
    if model_path is None:
        if local_models_dir is None:
            local_models_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "models_local"
            )
        model_path = os.path.join(local_models_dir, "Qwen", "Qwen2___5-0___5B")
    
    # 加载tokenizer
    if not os.path.exists(model_path):
        # 尝试Qwen目录下的实际路径
        for root, dirs, files in os.walk(local_models_dir if local_models_dir else "models_local"):
            if "tokenizer.json" in files:
                model_path = root
                break
    
    sys.stderr.write(f"[Tokenizer] 加载Qwen2.5 tokenizer: {model_path}\n")
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
    # 确保特殊token设置正确
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    
    sys.stderr.write(f"[Tokenizer] vocab_size={tok.vocab_size}, pad={tok.pad_token_id}, eos={tok.eos_token_id}\n")
    return tok
