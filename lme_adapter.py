"""
lme_adapter.py —— LongMemEval 适配层：把公开评测集接进我们的 Cell/Hub

LongMemEval（ICLR 2025，xiaowu0162/LongMemEval）是这个领域最对口的公开评测集：
  · 500 条问题，6 种题型
      single-session-user / single-session-assistant / single-session-preference
      multi-session（跨会话聚合）/ temporal-reasoning（时序）/ knowledge-update（知识更新）
  · 每条问题自带 **证据会话标注** `answer_session_ids` —— 这就是我们要的 ground truth
  · haystack 里同时有证据会话和大量干扰会话（S 版约 53 个会话 / 115k tokens）

本适配层只做「会话 → 神经元」的映射与检索评测：
    ground truth 用官方标注的**会话级**命中，不掺任何我们自己的判断。
    这样"架构有没有效"就不再由我构造的语料说了算。

局限（必须说清）：
  · 官方评测是"生成答案 + LLM 判分"，本层只测**检索**这一环，
    因为检索是长会话记忆的瓶颈，也是我们架构唯一改变的东西。
  · 不联网跑生成，所以答案正确率不测（那需要给每条问题调一次 LLM，500 条成本不可忽略）。

文件：bench_data/longmemeval_{oracle,s_cleaned}.json
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from textutil import est_tokens

DATA_DIR = "bench_data"
FILES = {
    "oracle": "longmemeval_oracle.json",     # 只含证据会话（约 3 个）
    "s": "longmemeval_s_cleaned.json",       # 完整 haystack（约 53 个会话）
    "m": "longmemeval_m_cleaned.json",       # 约 500 个会话（未下载）
}


@dataclass
class LmeSession:
    session_id: str
    date: str
    turns: List[Dict[str, str]] = field(default_factory=list)

    def text(self) -> str:
        lines = []
        for t in self.turns:
            role = t.get("role", "?")
            content = t.get("content", "")
            lines.append(f"<{role}> {content}")
        return f"# session {self.session_id}  ({self.date})\n" + "\n".join(lines)

    @property
    def tokens(self) -> int:
        return est_tokens(self.text())


@dataclass
class LmeQuestion:
    qid: str
    qtype: str
    question: str
    answer: str
    question_date: str
    answer_session_ids: List[str]
    sessions: List[LmeSession] = field(default_factory=list)

    @property
    def gold(self) -> List[str]:
        return list(self.answer_session_ids)

    def haystack_tokens(self) -> int:
        return sum(s.tokens for s in self.sessions)


def available() -> List[str]:
    out = []
    for key, fn in FILES.items():
        p = os.path.join(DATA_DIR, fn)
        if os.path.exists(p):
            out.append(key)
    return out


def load_questions(version: str = "s", *, limit: Optional[int] = None,
                   per_type: Optional[int] = None, types: Optional[Sequence[str]] = None,
                   seed: int = 17) -> List[LmeQuestion]:
    """加载问题。per_type 用于按题型均衡抽样（控制 API 成本）。

    为什么要按题型抽样：LongMemEval 的题型分布不均衡（multi-session 与 temporal 各 133 条，
    preference 只有 30 条）。不均衡抽样会让"总分"被多数组型主导，
    掩盖我们最想知道的差异 —— 跨会话推理与时序推理。
    """
    import random
    fn = FILES.get(version)
    if not fn:
        raise ValueError(f"未知版本 {version}，可选 {list(FILES)}")
    path = os.path.join(DATA_DIR, fn)
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少 {path}。下载方式见 DESIGN.md §7.2")

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    rng = random.Random(seed)
    by_type: Dict[str, List[dict]] = {}
    for item in raw:
        qt = item.get("question_type", "unknown")
        if types and qt not in types:
            continue
        by_type.setdefault(qt, []).append(item)

    picked: List[dict] = []
    for qt, items in by_type.items():
        rng.shuffle(items)
        picked += items[: per_type] if per_type else items
    if limit:
        picked = picked[:limit]

    out: List[LmeQuestion] = []
    for item in picked:
        sessions = []
        for sid, date, turns in zip(item.get("haystack_session_ids", []),
                                    item.get("haystack_dates", []),
                                    item.get("haystack_sessions", [])):
            sessions.append(LmeSession(session_id=sid, date=date, turns=turns))
        out.append(LmeQuestion(
            qid=item.get("question_id", "?"),
            qtype=item.get("question_type", "unknown"),
            question=item.get("question", ""),
            answer=str(item.get("answer", "")),
            question_date=item.get("question_date", ""),
            answer_session_ids=list(item.get("answer_session_ids", [])),
            sessions=sessions,
        ))
    return out


def stats(questions: Sequence[LmeQuestion]) -> Dict[str, object]:
    from collections import Counter
    per_type = Counter(q.qtype for q in questions)
    hay = [q.haystack_tokens() for q in questions]
    return {
        "n": len(questions),
        "types": dict(per_type),
        "avg_haystack_tokens": int(sum(hay) / max(1, len(hay))),
        "max_haystack_tokens": max(hay) if hay else 0,
        "avg_sessions": int(sum(len(q.sessions) for q in questions) / max(1, len(questions))),
        "avg_gold": round(sum(len(q.gold) for q in questions) / max(1, len(questions)), 2),
    }
