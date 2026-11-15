"""
textutil.py —— 共享的零依赖文本工具（避免 hub / embedder 循环导入）

只放"两边都要用、且没有任何状态"的东西：分词、n-gram、token 估算、稳定哈希。
Cell / Hub 等有状态的实体仍留在 hub.py。
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Dict, List, Optional, Sequence

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+(?:\.[0-9]+)?|[\u4e00-\u9fff]")

# 高频功能字：几乎无处不在，算覆盖率时只会稀释信号
STOP = set("的了吗呢吧啊是不我你他这那就都也还要和与或在有为对个上下中被把从以及可以什么怎么多少"
           "哪个哪里请问帮我一下然后其实就是因为所以如果那么这些那些一个两种")


def tokenize(text: str) -> List[str]:
    """CJK 单字 + 拉丁词/数字。BM25 的 IDF 会自动把常见汉字压下去，罕见标识符顶上来。"""
    return _WORD_RE.findall(text.lower())


def char_ngrams(text: str, lo: int = 1, hi: int = 3) -> List[str]:
    """字符 n-gram（长度 1..3）—— 对中英文都不需要分词器。"""
    s = re.sub(r"\s+", " ", text.lower())
    grams: List[str] = []
    grams.extend(s.split())  # 词级也留一份，拉丁标识符靠它抓
    for n in range(lo, hi + 1):
        if n == 1:
            grams.extend(s)
        else:
            grams.extend(s[i : i + n] for i in range(max(0, len(s) - n + 1)))
    return grams


def stable_hash(token: str, dim: int, salt: str = "") -> int:
    """跨进程稳定的哈希（Python 内置 hash 带随机盐，不能用于持久化索引）。"""
    h = hashlib.blake2b((salt + token).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % dim


def est_tokens(text: str) -> int:
    """粗略 token 估计：CJK ~0.6 tok/字，拉丁 ~0.28 tok/字符。只用于预算，不追求精确。"""
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return int(cjk * 0.6 + other * 0.28) + 4


class HashingEmbedder:
    """离线兜底嵌入：signed hashing trick + IDF 加权 + 最大分量归一化。

    ⚠️ 实测结论（calibrate2.py 参数扫描）：在纯字符级实现下，这一路的**判别力接近于零**
    （最优权重扫描一致给出 dense_w = 0），正负样本 AUC 只有 0.75~0.84。
    它的作用只是"没有在线嵌入时还能跑"，**不要指望它做语义匹配**。

    两个必须踩过才知道的细节：
      ① 若对特征做等权平均，800 字的神经元会被大量无信息量字符稀释，余弦只剩 0.05 量级，
         中枢就永远"想不起来"；
      ② 用**最大分量归一化**而非 L2 归一化，"查询词是否命中"才能直接体现为分数。
    """

    def __init__(self, dim: int = 4096):
        self.dim = dim
        self.df: Dict[int, int] = {}   # 维度 → 命中文档数（IDF 用）
        self.n_docs = 0
        self._cells_ref: Optional[Dict[str, Any]] = None
        self._df_cache: Dict[str, int] = {}
        self._df_stamp = -1

    def _weight(self, idx: int) -> float:
        df = self.df.get(idx, 0)
        return math.log((1 + self.n_docs) / (1 + df)) + 1.0

    def df_count(self, token: str) -> int:
        """真实文档频率（BM25 用，与哈希桶碰撞无关）。"""
        cells = self._cells_ref or {}
        if self._df_stamp != len(cells):
            self._df_cache.clear()
            self._df_stamp = len(cells)
        if token in self._df_cache:
            return self._df_cache[token]
        n = sum(1 for c in cells.values() if token in c.tf)
        self._df_cache[token] = n
        return n

    def embed(self, text: str, learn: bool = False) -> List[float]:
        vec = [0.0] * self.dim
        seen: Dict[int, int] = {}
        for g in char_ngrams(text):
            idx = stable_hash(g, self.dim, "idx")
            sign = 1.0 if stable_hash(g, self.dim, "sgn") % 2 == 0 else -1.0
            vec[idx] += sign * self._weight(idx)
            seen[idx] = 1
        if learn:
            self.n_docs += 1
            for idx in seen:
                self.df[idx] = self.df.get(idx, 0) + 1
        peak = max((abs(v) for v in vec), default=0.0) or 1.0
        return [v / peak for v in vec]

    def embed_query(self, text: str) -> List[float]:
        """与 embed 一致——接口对齐在线嵌入，方便热插拔。"""
        return self.embed(text)

    @staticmethod
    def cos(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(x * y for x, y in zip(a, b))
