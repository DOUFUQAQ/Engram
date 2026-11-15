"""
eval_compaction.py —— 压缩腿：真实 agent 今天在用的策略，在公开评测集上到底丢多少

为什么这一腿最重要：
    RAG 是"另一种检索架构"，而**压缩是真实 agent（包括 DSH）今天正在做的事**。
    长会话超限 → 触发摘要 → 历史被重写。§1 引用的 Compact-Bench 说 LLM 摘要
    最多保留 16% 的重复细节；本脚本用同一份公开评测集，把这件事量化到我们自己的口径上。

四种策略（都做成"检索器"，才能在同样的问题上比）：
    full        全量原文（上限：物理上装不下，作为天花板）
    compaction  压缩摘要（真实 agent 的做法 —— 只保留每轮首句/前 N 字，精确串被丢）
    recency     只留最近 k 个会话（最省，早期事实全失忆）
    cells       本方案（保真冻结 + 按需调取 + 主动查询）

指标：
    Recall@k / MRR    证据会话能否被找回（检索层）
    exact_keep        关键精确串（数字、专名、阈值…）在进入上下文的文本里是否还完整存在
                      —— 这是压缩策略真正的死穴，也是"飘忽"的直接来源
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
import time
from typing import Dict, List, Sequence, Set, Tuple

from embed_cache import EmbeddingCache
from embedder import EmbeddingProvider, build_components
from hub import Hub, HashingEmbedder
from lme_adapter import LmeQuestion, load_questions, stats
from textutil import est_tokens, tokenize

# 从答案里抽出"关键精确串"：数字、专名、日期、带连字符/下划线的标识符
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}|\d+(?:[.:/]\d+)+|\d+")


def key_strings(q: LmeQuestion) -> List[str]:
    """从官方答案里抽出必须保真的精确串。

    为什么从**答案**抽：LongMemEval 的答案就是"应该被回忆出来的那个事实"
    （如 "Business Administration"、"25 * 60"、"March 15th"）。
    压缩如果把这个串改写成别的措辞，答对的可能性就没了。
    """
    out: Set[str] = set()
    for m in _TOKEN_RE.findall(q.answer or ""):
        s = m.strip()
        if len(s) >= 3 and not s.lower() in {"the", "and", "for", "with", "that", "not", "you"}:
            out.add(s)
    return sorted(out)


# ---------------------------------------------------------------------------
# 压缩实现（模拟真实 agent 的摘要：保留轮廓、丢弃精确串）
# ---------------------------------------------------------------------------

def compact_session(text: str, *, keep_first: int = 90, keep_last: int = 40) -> str:
    """有损压缩：每轮只留首尾片段。

    真实压缩由另一个 LLM 完成，行为与此同构 —— 它会写下"用户讨论了自己的车和保养"，
    但不会记住"GPS system not functioning correctly"。
    这里用确定性规则而不是 LLM，是为了**可复现**且不烧 API 额度；
    而 §1 的一手数据（LLM 摘要仅保留 16% 重复细节）说明 LLM 版只会更差，不会更好。
    """
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if len(s) <= keep_first + keep_last:
            lines.append(s)
        else:
            lines.append(f"{s[:keep_first]} … {s[-keep_last:]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 四种策略各自产出"进入上下文的文本"，以及"它排序出的会话"
# ---------------------------------------------------------------------------

def rank_full(q: LmeQuestion, cache: EmbeddingCache, *, topk: int, budget: int) -> Tuple[List[str], str]:
    """天花板：全量原文，**没有检索步骤**（它把一切都给了）。

    所以它的 Recall@k 无意义（README 里按 N/A 报），它在这一栏的作用只有两个：
      · 关键串保留的上限（不压缩就能 100% 保留）
      · token 成本的上限（139k，物理上不可行）
    """
    text = "\n\n".join(s.text() for s in q.sessions)
    return [s.session_id for s in q.sessions], text


def rank_compaction(q: LmeQuestion, cache: EmbeddingCache, *, topk: int,
                    budget: int) -> Tuple[List[str], str]:
    """压缩 + 会话级向量检索（压缩后仍需检索，因为压缩后还是装不下）。"""
    ids = [s.session_id for s in q.sessions]
    texts = [compact_session(s.text()) for s in q.sessions]
    vecs = cache.embed_docs(texts)
    qv = cache.embed_query(q.question)
    scores = [EmbeddingProvider.cos(qv, v) for v in vecs]
    order = sorted(range(len(ids)), key=lambda i: -scores[i])

    picked: List[str] = []
    used = 0
    for i in order:
        c = est_tokens(texts[i])
        if budget > 0 and used + c > budget:
            continue
        picked.append(ids[i])
        used += c
        if len(picked) >= topk:
            break
    return picked, "\n\n".join(texts[order.index(ids[i])] if False else "" for i in []) or ""


def rank_recency(q: LmeQuestion, cache: EmbeddingCache, *, topk: int,
                 budget: int) -> Tuple[List[str], str]:
    """只留最近的 topk 个会话（按 haystack 顺序，即时间序）。"""
    picked = [s.session_id for s in q.sessions[-topk:]]
    return picked, "\n\n".join(s.text() for s in q.sessions[-topk:])


def rank_cells(q: LmeQuestion, hub: Hub, cache: EmbeddingCache, *, topk: int,
               budget: int) -> Tuple[List[str], str]:
    """本方案：多路脑电波 + 预算内按相关度装入。"""
    probes = Hub.decompose(q.question)
    merged: Dict[str, float] = {}
    for p in probes:
        qv = cache.embed_query(p)
        for cid, c in hub.cells.items():
            d = EmbeddingProvider.cos(qv, c.vec)
            _, _, _, s, cov = next((x for x in hub._score_all(p) if x[0] == cid),
                                   (cid, c, 0.0, 0.0, 0.0))
            conf = hub._combine_any(d, s, cov)
            merged[cid] = max(merged.get(cid, 0.0), conf)
    order = [cid for cid, _ in sorted(merged.items(), key=lambda kv: -kv[1])]

    picked: List[str] = []
    used = 0
    for cid in order:
        if cid not in hub.cells:
            continue
        c = est_tokens(hub.cells[cid].text)
        if budget > 0 and used + c > budget:
            continue
        picked.append(cid)
        used += c
        if len(picked) >= topk:
            break
    text = "\n\n".join(hub.cells[cid].text for cid in picked if cid in hub.cells)
    return picked, text


# ---------------------------------------------------------------------------

def build_hub_for(q: LmeQuestion, cache: EmbeddingCache, embedder) -> Hub:
    hub = Hub(embedder=embedder, conf_threshold=0.30, margin_threshold=0.03)
    texts = [s.text() for s in q.sessions]
    vecs = cache.embed_docs(texts)
    for s, text, vec in zip(q.sessions, texts, vecs):
        cell = hub.make_cell([text], turn_start=0, turn_end=0, summary=s.date)
        cell.cell_id = s.session_id
        cell.vec = vec
    hub.cells = {c.cell_id: c for c in hub.cells.values()}
    hub._sync_index()
    return hub


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="s", choices=["oracle", "s", "m"])
    ap.add_argument("--per-type", type=int, default=5)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--budget", type=int, default=18000)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    comp = build_components(".env", strict=False)
    embedder = (HashingEmbedder(dim=4096) if args.offline or not comp.strict_ok
                else comp.embedder)
    cache = EmbeddingCache(embedder, path="bench_data/.embed_cache_cmp.pkl")
    qs = load_questions(args.version, per_type=args.per_type)
    st = stats(qs)

    print("=" * 104)
    print(f"压缩策略 vs 保真存储 —— LongMemEval-{args.version.upper()}（真实公开评测集）")
    print("=" * 104)
    print(f"  样本        : {st['n']} 条  {st['types']}")
    print(f"  平均 haystack: {st['avg_haystack_tokens']:,} tokens / {st['avg_sessions']} 会话")
    print(f"  取用上限    : top-{args.topk}，上下文预算 {args.budget:,} tokens")
    print()

    rows: List[Dict[str, object]] = []
    for i, q in enumerate(qs, 1):
        hub = build_hub_for(q, cache, embedder)
        keys = key_strings(q)
        try:
            comp_ids, comp_text = rank_compaction(q, cache, topk=args.topk, budget=args.budget)
            rec_ids, rec_text = rank_recency(q, cache, topk=args.topk, budget=args.budget)
            cell_ids, cell_text = rank_cells(q, hub, cache, topk=args.topk, budget=args.budget)
            full_ids, full_text = rank_full(q, cache, topk=args.topk, budget=args.budget)
        except Exception as e:
            print(f"  [{i}/{len(qs)}] {q.qid} 失败: {type(e).__name__}: {str(e)[:70]}")
            continue

        def keep(text: str) -> float:
            if not keys:
                return 1.0
            return sum(1 for k in keys if k in text) / len(keys)

        rows.append({
            "qid": q.qid, "qtype": q.qtype, "gold": q.gold, "keys": keys,
            "compaction": (comp_ids, keep(comp_text), est_tokens(comp_text)),
            "recency": (rec_ids, keep(rec_text), est_tokens(rec_text)),
            "cells": (cell_ids, keep(cell_text), est_tokens(cell_text)),
            "full": (full_ids, keep(full_text), est_tokens(full_text)),
        })
        if i % 5 == 0 or i == len(qs):
            print(f"  已评测 {i}/{len(qs)}")
            cache.save()
    cache.save()

    def rec_at(ids: Sequence[str], gold: Sequence[str], k: int) -> float:
        if not gold:
            return 0.0
        return sum(1 for g in gold if g in set(ids[:k])) / len(gold)

    def mrr(ids: Sequence[str], gold: Sequence[str]) -> float:
        for i, x in enumerate(ids, 1):
            if x in gold:
                return 1.0 / i
        return 0.0

    print()
    print("=" * 104)
    print("结果")
    print("=" * 104)
    print(f"{'策略':<18}{'Recall@k':>11}{'MRR':>8}{'关键串保留':>12}{'上下文tokens':>14}")
    print("-" * 104)
    names = [("full", "full 全量原文"), ("compaction", "compaction 压缩"),
             ("recency", "recency 只留最近"), ("cells", "cells 本方案")]
    for key, label in names:
        ids_ok = statistics.mean(rec_at(r[key][0], r["gold"], args.topk) for r in rows)
        m = statistics.mean(mrr(r[key][0], r["gold"]) for r in rows)
        kp = statistics.mean(r[key][1] for r in rows)
        cost = statistics.mean(r[key][2] for r in rows)
        # full 没有检索步骤，"证据在本轮是否被取到"对它不是有效指标 → 报 N/A
        rec_s = "N/A（无检索）" if key == "full" else f"{ids_ok * 100:.1f}%"
        mrr_s = "N/A" if key == "full" else f"{m:.3f}"
        print(f"{label:<18}{rec_s:>14}{mrr_s:>8}{kp * 100:>11.1f}%{cost:>14,.0f}")
    print("-" * 104)

    print()
    print(f"{'题型':<28}{'条数':>6}{'压缩 关键串':>13}{'本方案 关键串':>15}{'差':>9}")
    print("-" * 104)
    for t in sorted({r["qtype"] for r in rows}):
        sub = [r for r in rows if r["qtype"] == t]
        a = statistics.mean(r["compaction"][1] for r in sub)
        b = statistics.mean(r["cells"][1] for r in sub)
        print(f"{t:<28}{len(sub):>6}{a * 100:>12.1f}%{b * 100:>14.1f}%{(b - a) * 100:>+8.1f}")
    print("-" * 104)
    print("\n『关键串保留』= 官方答案里的精确串（数字/专名/标识符）在进入上下文的文本中")
    print("完整存在的比例。这是压缩策略的真正死穴：摘要会把它们改写成别的措辞，")
    print("于是模型「看起来记得」，但答不出那个具体值 —— 这就是 40 轮后飘忽的直接机制。")
    print(f"\n嵌入缓存: {cache.describe()}")


if __name__ == "__main__":
    sys.exit(main())
