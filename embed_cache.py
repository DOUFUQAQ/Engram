"""
embed_cache.py —— 磁盘嵌入缓存

为什么必须有：LongMemEval 的每个问题有约 50 个会话，按题型抽样后同一个 haystack 会被
反复建立索引。不做缓存的话，500 条问题会打出几万次嵌入调用 —— 又慢又费。

缓存键 = 模型名 + 文本哈希。所以换嵌入模型不会读到旧向量（这是必须的：
换模型的向量空间不兼容，读旧缓存会得到静默错误的分数）。
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from typing import Dict, List, Sequence, Optional

DEFAULT_PATH = os.path.join("bench_data", ".embed_cache.pkl")


class EmbeddingCache:
    def __init__(self, embedder, path: str = DEFAULT_PATH):
        self.embedder = embedder
        self.path = path
        self.model = getattr(embedder, "model", type(embedder).__name__)
        self.store: Dict[str, List[float]] = {}
        self.hits = 0
        self.misses = 0
        self._load()

    # ------------------------------------------------------------------
    def _key(self, text: str) -> str:
        h = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
        return f"{self.model}|{h}"

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "rb") as f:
                payload = pickle.load(f)
            if payload.get("model") == self.model:
                self.store = payload.get("store", {})
            # 模型不同则整体丢弃，避免用错向量空间
        except Exception:
            self.store = {}

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"model": self.model, "store": self.store}, f)
        os.replace(tmp, self.path)

    # ------------------------------------------------------------------
    def embed_docs(self, texts: Sequence[str]) -> List[List[float]]:
        """批量取向量：命中的直读，未命中的批量调 API 后写回。"""
        keys = [self._key(t) for t in texts]
        todo_texts, todo_keys = [], []
        seen = set()
        for t, k in zip(texts, keys):
            if k in self.store:
                self.hits += 1
            elif k not in seen:
                seen.add(k)
                todo_texts.append(t)
                todo_keys.append(k)
        if todo_texts:
            self.misses += len(todo_texts)
            vecs = (self.embedder.embed_batch(todo_texts,
                                              prefix=getattr(self.embedder, "doc_prefix", ""))
                    if hasattr(self.embedder, "embed_batch")
                    else [self.embedder.embed(t) for t in todo_texts])
            # 硬校验：数量不匹配必须炸掉。静默少给向量会让索引"半边失明"，
            # 而且不会报错 —— 这类静默降级是最难查的 bug（本原型踩过一次）。
            if len(vecs) != len(todo_texts):
                raise RuntimeError(
                    f"嵌入返回数量不匹配: 请求 {len(todo_texts)} 得到 {len(vecs)}。"
                    f"该端点可能不支持多文本批量，请在 .env 里设 EMBED_FORCE_SINGLE=1")
            for k, v in zip(todo_keys, vecs):
                self.store[k] = v
        return [self.store[k] for k in keys]

    def embed_query(self, text: str) -> List[float]:
        qk = "Q|" + self._key(text)
        if qk in self.store:
            self.hits += 1
            return self.store[qk]
        self.misses += 1
        fn = getattr(self.embedder, "embed_query", None)
        if fn is not None:
            v = fn(text)
        elif getattr(self.embedder, "force_single", False) or not hasattr(self.embedder, "embed_batch"):
            v = self.embedder.embed(text)
        else:
            # 大批量端点：也要防"只返回 1 个向量"的聚合行为
            vs = self.embedder.embed_batch([text])
            if len(vs) != 1:
                raise RuntimeError(f"查询嵌入返回数量异常: {len(vs)}")
            v = vs[0]
        self.store[qk] = v
        return v

    def describe(self) -> str:
        return (f"嵌入缓存 命中 {self.hits} / 未命中 {self.misses} "
                f"（缓存 {len(self.store)} 条向量）")
