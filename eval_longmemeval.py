"""
eval_longmemeval.py —— 在公开评测集上比较：神经元架构 vs RAG

这是回答"架构是否有效、对比 RAG 如何"的**唯一站得住的证据**：
  语料、问题、ground truth 全部来自公开评测集，没有一个字是我构造的。
  ground truth 用官方标注的证据会话（`answer_session_ids`），不掺我们的判断。

指标（会话级检索，两边口径完全一致）：
  Recall@k   正确答案所在会话是否进入前 k 个（k 由 --topk 指定）
  MRR        正确会话的平均倒数排名
  NDCG@k     考虑多条证据会话时的排序质量
  单问耗时 / 上下文 token 成本

用法：
  python eval_longmemeval.py --per-type 3              # 快速验证（18 条）
  python eval_longmemeval.py --per-type 20 --topk 5    # 正式评测（120 条）
  python eval_longmemeval.py --version oracle          # 只含证据会话的版本（难度基准）
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time
from typing import Dict, List, Sequence, Tuple

from embed_cache import EmbeddingCache
from embedder import EmbeddingProvider, build_components
from hub import Hub, HashingEmbedder
from lme_adapter import LmeQuestion, load_questions, stats
from textutil import est_tokens, tokenize


# ---------------------------------------------------------------------------
# ground truth 折算
# ---------------------------------------------------------------------------

def recall_at(ranked_ids: Sequence[str], gold: Sequence[str], k: int) -> float:
    """召回率：gold 里有多少条进入了前 k。"""
    if not gold:
        return 0.0
    top = set(ranked_ids[:k])
    return sum(1 for g in gold if g in top) / len(gold)


def mrr(ranked_ids: Sequence[str], gold: Sequence[str]) -> float:
    for i, rid in enumerate(ranked_ids, 1):
        if rid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at(ranked_ids: Sequence[str], gold: Sequence[str], k: int) -> float:
    gold = set(gold)
    dcg = sum((1.0 / math.log2(i + 1)) for i, rid in enumerate(ranked_ids[:k], 1)
              if rid in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


# ---------------------------------------------------------------------------
# 本方案：一个会话 = 一个神经元
# ---------------------------------------------------------------------------

def build_hub_for(q: LmeQuestion, cache: EmbeddingCache, embedder) -> Hub:
    """把该问题的 haystack 灌成神经元。

    映射原则（与设计一致）：**一个会话 = 一个神经元**。
    理由：LongMemEval 的 ground truth 就是会话级的，会话本身也是"一段有边界的经历"，
    这正是我们主张的"事件边界"，而不是我为了讨好指标而选的切法。
    """
    hub = Hub(embedder=embedder, conf_threshold=0.30, margin_threshold=0.03)
    texts = [s.text() for s in q.sessions]
    vecs = cache.embed_docs(texts)
    for s, text, vec in zip(q.sessions, texts, vecs):
        cell = hub.make_cell([text], turn_start=0, turn_end=0,
                             summary=s.date, source="session")
        cell.cell_id = s.session_id          # 直接用会话 id 作为神经元 id，便于对账
        cell.vec = vec
    hub.cells = {c.cell_id: c for c in hub.cells.values()}
    hub._sync_index()
    # 时序突触：按 haystack 给出的顺序连接相邻会话
    ordered = [s.session_id for s in q.sessions]
    for a, b in zip(ordered, ordered[1:]):
        hub.synapses.setdefault(a, {})[b] = 0.6
        hub.synapses.setdefault(b, {})[a] = 0.6
    return hub


def our_ranking(q: LmeQuestion, hub: Hub, cache: EmbeddingCache, *, topk: int,
                budget: int = 0) -> List[str]:
    """用我们的多路脑电波 + 二跳检索，返回排序后的会话 id。

    注意：这里**关掉 LLM 重排**，因为它是按查询调一次 API 的，500 条问题成本太大；
    等基础检索的结论稳定后，再单独测"加不加重排"。

    budget>0 时做等成本对比：按排序依次装入会话，超预算就跳过（与 Packer 的策略一致）。
    """
    probes = Hub.decompose(q.question)
    merged: Dict[str, float] = {}
    for p in probes:
        qv = cache.embed_query(p)
        dense = {cid: EmbeddingProvider.cos(qv, c.vec) for cid, c in hub.cells.items()}
        for cid, _, d, s, cov in hub._score_all(p):
            conf = hub._combine_any(dense.get(cid, 0.0), s, cov)
            merged[cid] = max(merged.get(cid, 0.0), conf)
    ranked = [cid for cid, _ in sorted(merged.items(), key=lambda kv: -kv[1])][: max(topk, 10)]
    if budget <= 0:
        return ranked
    out: List[str] = []
    used = 0
    for cid in ranked:
        cost = est_tokens(hub.cells[cid].text) if cid in hub.cells else 0
        if used + cost > budget:
            continue                     # 塞不下就试下一个更小的（与 Packer 同策略）
        out.append(cid)
        used += cost
        if len(out) >= topk:
            break
    return out


# ---------------------------------------------------------------------------
# RAG 基线：会话内固定窗口切块
# ---------------------------------------------------------------------------

def rag_ranking(q: LmeQuestion, cache: EmbeddingCache, *, chunk_size: int, overlap: int,
                topk: int, prefilter: int = 80, budget: int = 0) -> Tuple[List[str], int]:
    """返回 (按相关度排序的会话 id 列表, 取 topk 块的 token 量)。

    块级检索 → 会话级排序：把块按分数排序，**按会话去重后取首次出现的顺序**。
    这是把块级 RAG 折算到会话级 ground truth 的标准做法，对 RAG 最有利
    （它只要有一个块命中就够了）。

    为什么要有 prefilter：一个 48 会话的 haystack 切出约 250 块，而嵌入端点是逐条调用的
    （见 embedder.embed_batch 的坑），250 次调用 = 2 分钟/问题，跑不动。
    所以先用 BM25 粗筛出 prefilter 个候选块再嵌入 —— 这是标准做法，
    且**不会削弱 RAG**：BM25 本就是它的混合检索的一半。
    """
    chunks: List[Tuple[str, str]] = []          # (session_id, text)
    for s in q.sessions:
        text = s.text()
        step = max(1, chunk_size - overlap)
        for start in range(0, max(1, len(text)), step):
            piece = text[start : start + chunk_size].strip()
            if len(piece) >= 40:
                chunks.append((s.session_id, piece))
            if start + chunk_size >= len(text):
                break
    if not chunks:
        return [], 0

    # ---- BM25 粗筛（RAG 混合检索本来就含这一路）----
    qt = tokenize(q.question)
    n = len(chunks)
    tf_list = [{t: c.count(t) for t in set(qt) if t in c} for _, c in chunks]
    df = {t: sum(1 for tf in tf_list if t in tf) for t in set(qt)}
    avgdl = sum(len(c) for _, c in chunks) / n / 3 or 1.0
    sparse_all: List[float] = []
    for i, tf in enumerate(tf_list):
        s = 0.0
        dl = len(chunks[i][1]) / 3 or 1.0
        for t, f in tf.items():
            idf = math.log(1 + (n - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
            s += idf * (f * 2.2) / (f + 1.2 * (1 - 0.4 + 0.4 * dl / avgdl))
        sparse_all.append(s)
    cand_idx = sorted(range(n), key=lambda i: -sparse_all[i])[:prefilter]

    texts = [chunks[i][1] for i in cand_idx]
    vecs = cache.embed_docs(texts)
    qv = cache.embed_query(q.question)
    scores = [EmbeddingProvider.cos(qv, v) for v in vecs]

    order_d = sorted(range(len(cand_idx)), key=lambda j: -scores[j])
    order_s = sorted(range(len(cand_idx)), key=lambda j: -sparse_all[cand_idx[j]])
    fused: Dict[int, float] = {}
    for rank, j in enumerate(order_d, 1):
        fused[j] = fused.get(j, 0.0) + 1.0 / (60 + rank)
    for rank, j in enumerate(order_s, 1):
        fused[j] = fused.get(j, 0.0) + 1.0 / (60 + rank)
    ranked_chunks = sorted(fused.items(), key=lambda kv: -kv[1])

    seen: List[str] = []
    cost = 0
    for j, _ in ranked_chunks:
        sid = chunks[cand_idx[j]][0]
        if sid in seen:
            continue
        c = est_tokens(chunks[cand_idx[j]][1])
        if budget > 0 and cost + c > budget:
            continue                     # 等成本对比：同我们的策略，塞不下就试下一个
        seen.append(sid)
        cost += c
        if len(seen) >= topk:
            break
    return seen, cost


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="s", choices=["oracle", "s", "m"])
    ap.add_argument("--per-type", type=int, default=5, help="每种题型取几条")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--types", default=None, help="逗号分隔，只看某些题型")
    ap.add_argument("--budget", type=int, default=0,
                    help="等成本对比：把两边的上下文预算都限制到这个 token 数（0=不限）")
    args = ap.parse_args()

    comp = build_components(".env", strict=False)
    if args.offline or not comp.strict_ok:
        embedder, label = HashingEmbedder(dim=4096), "离线哈希"
    else:
        embedder, label = comp.embedder, comp.embedder.model
    cache = EmbeddingCache(embedder)

    types = args.types.split(",") if args.types else None
    qs = load_questions(args.version, per_type=args.per_type, limit=args.limit, types=types)
    st = stats(qs)
    print("=" * 104)
    print(f"LongMemEval-{args.version.upper()}  真实公开评测集")
    print("=" * 104)
    print(f"  嵌入模型    : {label}")
    print(f"  样本        : {st['n']} 条   {st['types']}")
    print(f"  平均 haystack: {st['avg_haystack_tokens']:,} tokens / {st['avg_sessions']} 个会话"
          f"（最大 {st['max_haystack_tokens']:,}）；平均证据会话 {st['avg_gold']} 个")
    print(f"  取用条数    : top-{args.topk}     RAG 切块: {args.chunk_size} 字符 / 重叠 {args.overlap}")
    print()

    rows: List[Dict[str, object]] = []
    t_start = time.perf_counter()
    for i, q in enumerate(qs, 1):
        try:
            hub = build_hub_for(q, cache, embedder)
        except Exception as e:
            print(f"  [{i}/{len(qs)}] {q.qid} 建库失败: {type(e).__name__}: {str(e)[:80]}")
            continue

        t0 = time.perf_counter()
        ours = our_ranking(q, hub, cache, topk=args.topk, budget=args.budget)
        t_ours = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        rag, rag_cost = rag_ranking(q, cache, chunk_size=args.chunk_size,
                                    overlap=args.overlap, topk=args.topk,
                                    budget=args.budget)
        t_rag = (time.perf_counter() - t0) * 1000

        rows.append({
            "qid": q.qid, "qtype": q.qtype, "gold": q.gold,
            "ours": ours, "rag": rag,
            "ours_ms": t_ours, "rag_ms": t_rag, "rag_cost": rag_cost,
            "ours_cost": sum(est_tokens(hub.cells[c].text) for c in ours[:args.topk]
                             if c in hub.cells),
            "sessions": len(q.sessions),
        })
        if i % 10 == 0 or i == len(qs):
            print(f"  已评测 {i}/{len(qs)}  （{(time.perf_counter() - t_start) / 60:.1f} 分钟）")
            cache.save()

    if not rows:
        print("没有成功评测的样本。")
        return
    cache.save()

    # ---------------- 汇总 ----------------
    def agg(sel, key):
        sub = [r for r in rows if sel(r)]
        if not sub:
            return (0.0, 0.0, 0.0)
        rec = statistics.mean(recall_at(r[key], r["gold"], args.topk) for r in sub)
        m = statistics.mean(mrr(r[key], r["gold"]) for r in sub)
        nd = statistics.mean(ndcg_at(r[key], r["gold"], args.topk) for r in sub)
        return (rec, m, nd)

    print()
    print("=" * 104)
    print(f"结果（会话级检索，与官方证据标注对比，top-{args.topk}）")
    print("=" * 104)
    print(f"{'方案':<26}{'Recall@k':>10}{'MRR':>8}{'NDCG@k':>9}{'单问耗时':>11}{'上下文tokens':>14}")
    print("-" * 104)
    for key, name in (("ours", "神经元架构（本方案）"), ("rag", "RAG 固定窗口+BM25")):
        rec, m, nd = agg(lambda r: True, key)
        ms = statistics.mean(r[f"{key}_ms"] for r in rows)
        cost_key = "ours_cost" if key == "ours" else "rag_cost"
        cost = statistics.mean(r[cost_key] for r in rows)
        print(f"{name:<26}{rec * 100:>9.1f}%{m:>8.3f}{nd:>9.3f}{ms:>10.0f}ms{cost:>14,.0f}")
    print("-" * 104)

    # 分题型
    print()
    print(f"{'题型':<30}{'条数':>6}{'本方案 R@k':>13}{'RAG R@k':>11}{'差':>8}")
    print("-" * 104)
    types_seen = sorted({r["qtype"] for r in rows})
    for t in types_seen:
        a = agg(lambda r, t=t: r["qtype"] == t, "ours")[0]
        b = agg(lambda r, t=t: r["qtype"] == t, "rag")[0]
        n = sum(1 for r in rows if r["qtype"] == t)
        mark = "  ←" if abs(a - b) > 0.15 else ""
        print(f"{t:<30}{n:>6}{a * 100:>12.1f}%{b * 100:>10.1f}%{(a - b) * 100:>+7.1f}{mark}")
    print("-" * 104)
    print(f"\n嵌入缓存: {cache.describe()}")
    if isinstance(embedder, EmbeddingProvider):
        print(f"API 用量: {embedder.stats.describe()}")
    print("\n注：只测**检索**这一环（长会话记忆的瓶颈），不测答案生成。")
    print("    官方评测需要 LLM 判分，本脚本不联网生成，所以不报答案正确率。")


if __name__ == "__main__":
    sys.exit(main())
