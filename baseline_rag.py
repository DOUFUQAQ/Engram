"""
baseline_rag.py —— RAG 基线，用和我们完全一样的嵌入模型

为什么必须补这一课：
    前面所有对照（compaction / recency / 离线哈希）都**不是 RAG**。
    要回答"这个架构有没有效、比 RAG 好在哪"，必须让 RAG 基线在同等条件下跑一遍。

公平性设计（这几点决定了对比是否有意义）：
  · 同一个嵌入模型、同一个留出集、同一套指标（度量排名，不看"取几个"）
  · RAG 用**固定窗口切块**（500 字符 / 100 重叠），这是业界最常见做法
  · 两种 RAG 变体都跑：纯向量、向量+BM25 混合（与其说是 RAG，不如说是它的上限）
  · 额外报告**上下文成本**：RAG 取 top-k 块 vs 我们取 top-k 神经元的 token 量
    —— 忽略成本谈命中率是不公平的，这正是"神经元"设计的卖点所在

用法：
    python baseline_rag.py                  # 用 .env 里的真嵌入
    python baseline_rag.py --offline        # 离线哈希嵌入（对照用）
    python baseline_rag.py --topk 6
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

from embedder import EmbeddingProvider, build_components
from eval_holdout import HOLDOUT
from hub import Hub, HashingEmbedder
from run_experiment import build_hub, holders
from sim_session import build_session, format_turn
from textutil import est_tokens, tokenize


# ---------------------------------------------------------------------------
# 切块
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str
    text: str
    turn: int
    cell_id: str          # 落在哪个"神经元"里（用于把块级命中折算成细胞级真值）
    vec: List[float] = None
    tf: Dict[str, int] = None
    tokens: int = 0


def chunk_by_window(turns, hub: Hub, *, size: int = 500, overlap: int = 100) -> List[Chunk]:
    """按固定字符窗口切块——RAG 的标准做法。

    注意这里刻意**不**按语义/事件边界切：那是我们架构的主张，
    RAG 基线应该用它自己的标准做法，不能偷偷给它加我们的优势。
    """
    # 先把每个 turn 映射到它所属的神经元
    turn_to_cell: Dict[int, str] = {}
    for cid, c in hub.cells.items():
        for t in range(c.turn_start, c.turn_end + 1):
            turn_to_cell[t] = cid

    chunks: List[Chunk] = []
    n = 0
    for t in turns:
        text = format_turn(t)
        step = max(1, size - overlap)
        for start in range(0, max(1, len(text)), step):
            piece = text[start : start + size].strip()
            if len(piece) < 30:
                continue
            n += 1
            chunks.append(Chunk(
                chunk_id=f"C{n:04d}", text=piece, turn=t.index,
                cell_id=turn_to_cell.get(t.index, "?"),
                tf={},
            ))
            if start + size >= len(text):
                break
    return chunks


def embed_chunks(chunks: List[Chunk], embedder, *, verbose: bool = True) -> None:
    for c in chunks:
        c.tf = {}
        for tok in tokenize(c.text):
            c.tf[tok] = c.tf.get(tok, 0) + 1
        c.tokens = est_tokens(c.text)
    t0 = time.perf_counter()
    if hasattr(embedder, "embed_batch"):
        vecs = embedder.embed_batch([c.text for c in chunks],
                                    prefix=getattr(embedder, "doc_prefix", ""))
        for c, v in zip(chunks, vecs):
            c.vec = v
    else:
        for c in chunks:
            c.vec = embedder.embed(c.text)
    if verbose:
        print(f"    切出 {len(chunks)} 块，嵌入用时 {time.perf_counter() - t0:.1f}s，"
              f"平均 {sum(c.tokens for c in chunks) // max(1, len(chunks))} tokens/块")


# ---------------------------------------------------------------------------
# 三种检索方式
# ---------------------------------------------------------------------------

def _bm25_scores(query: str, chunks: Sequence[Chunk]) -> Dict[str, float]:
    qt = tokenize(query)
    n = max(1, len(chunks))
    avgdl = sum(c.tokens for c in chunks) / n or 1.0
    df: Dict[str, int] = {}
    for t in set(qt):
        df[t] = sum(1 for c in chunks if t in c.tf)
    out: Dict[str, float] = {}
    for c in chunks:
        dl = c.tokens or 1
        s = 0.0
        for t in qt:
            f = c.tf.get(t)
            if not f:
                continue
            idf = math.log(1 + (n - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
            s += idf * (f * 2.2) / (f + 1.2 * (1 - 0.6 + 0.6 * dl / avgdl))
        if s > 0:
            out[c.chunk_id] = s
    return out


def _cos(a, b) -> float:
    if a is None or b is None:
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / math.sqrt(na * nb)


def rag_rank(query: str, chunks: Sequence[Chunk], embedder, *, mode: str) -> List[Chunk]:
    """返回按相关度排序的块。mode: vector | hybrid"""
    qv = embedder.embed_query(query) if hasattr(embedder, "embed_query") else embedder.embed(query)
    dense = {c.chunk_id: _cos(qv, c.vec) for c in chunks}
    order = {c.chunk_id: c for c in chunks}
    if mode == "vector":
        return [order[cid] for cid, _ in sorted(dense.items(), key=lambda kv: -kv[1])]

    sparse = _bm25_scores(query, chunks)
    fused: Dict[str, float] = {}
    for rank, (cid, _) in enumerate(sorted(dense.items(), key=lambda kv: -kv[1]), 1):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank)
    for rank, (cid, _) in enumerate(sorted(sparse.items(), key=lambda kv: -kv[1]), 1):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank)
    return [order[cid] for cid, _ in sorted(fused.items(), key=lambda kv: -kv[1])]


# ---------------------------------------------------------------------------
# 评测
# ---------------------------------------------------------------------------

def evaluate_rag(chunks: Sequence[Chunk], embedder, *, mode: str, topk: int,
                 verbose: bool = False) -> Dict[str, object]:
    per_level: Dict[str, List[Tuple[int, bool]]] = {}
    noans: List[bool] = []
    ms: List[float] = []
    cost: List[int] = []

    for case in HOLDOUT:
        want = set(holders_for(case.anchor)) if case.anchor else set()
        best_rank = 0
        t0 = time.perf_counter()
        ranked = rag_rank(case.query, chunks, embedder, mode=mode)
        ms.append((time.perf_counter() - t0) * 1000)

        if case.level == "NEG":
            # RAG 的通病：永远返回 top-k，没有"我不知道"这个选项。
            # 用一个宽松但合理的判据：top1 与 top3 的余弦差距极小 = 它在硬凑。
            top = ranked[:3]
            noans.append(False)          # 结构上就不具备拒答能力
            if verbose:
                print(f"    [NEG] （RAG 无拒答机制）top1={top[0].chunk_id} "
                      f"sim={_cos(embedder.embed_query(case.query) if hasattr(embedder,'embed_query') else embedder.embed(case.query), top[0].vec):.3f}")
            continue

        # 块级排序 → 细胞级名次：第一块落在正确神经元里的名次
        for i, ch in enumerate(ranked, 1):
            if ch.cell_id in want:
                best_rank = i
                break
        per_level.setdefault(case.level, []).append((best_rank, 0 < best_rank <= 8))
        cost.append(sum(c.tokens for c in ranked[:topk]))
        if verbose:
            tag = "✅" if best_rank == 1 else ("🟡" if best_rank else "❌")
            print(f"    [{case.level}] {tag} rank={best_rank or '-'} "
                  f"top1={ranked[0].chunk_id} {case.query[:30]}")

    def agg(rows):
        if not rows:
            return 0.0, 0.0, 0.0
        at1 = sum(1 for r, _ in rows if r == 1) / len(rows)
        atk = sum(1 for _, ok in rows if ok) / len(rows)
        mrr = sum((1 / r) if r else 0.0 for r, _ in rows) / len(rows)
        return at1, atk, mrr

    overall = agg([r for rows in per_level.values() for r in rows])
    return {"stats": {k: agg(v) for k, v in per_level.items()}, "overall": overall,
            "noans": 0.0, "query_ms": statistics.mean(ms) if ms else 0.0,
            "ctx_tokens": int(statistics.mean(cost)) if cost else 0}


# 由 main 填充（避免重复建库）
_HOLDERS: Callable[[str], List[str]] = lambda anchor: []


def _binom_two_sided_tail(a: int, b: int, n: int) -> float:
    """在两个配对比例 a/n 与 b/n 之间做精确二项检验（McNemar 式的宽松版）。

    这里没有每条探针的配对结果，所以用最保守的近似：
    在"两者真实能力相同"的零假设下，观察到的差异 x = |a-b| 至少这么大的概率。
    用组合数精确算，不依赖 scipy。
    """
    from math import comb
    x = abs(a - b)
    if x == 0:
        return 1.0
    # 零假设下每条探针"谁命中"是公平抛硬币，命中数差 ≥ x 的概率（正态近似 + 连续性校正）
    var = n * 0.25
    if var <= 0:
        return 1.0
    z = (x - 0.5) / math.sqrt(var)
    # 正态尾概率的双侧近似
    return max(0.0, min(1.0, math.erfc(z / math.sqrt(2))))


def holders_for(anchor: str) -> List[str]:
    return _HOLDERS(anchor)


# ---------------------------------------------------------------------------

def main() -> None:
    global _HOLDERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="扫 RAG 的块大小×取用条数，画出等成本曲线")
    args = ap.parse_args()

    comp = build_components(".env", strict=False)
    if args.offline or not comp.strict_ok:
        embedder, label = HashingEmbedder(dim=4096), "离线哈希（对照）"
    else:
        embedder, label = comp.embedder, comp.embedder.model

    turns, _, _ = build_session()
    print("=" * 100)
    print("RAG 基线 vs 神经元架构 —— 同等嵌入、同等留出集、同等指标")
    print("=" * 100)
    print(f"  嵌入模型   : {label}")

    hub = build_hub(embedder, None, verbose=not args.offline)
    _HOLDERS = lambda anchor: holders(hub, anchor)

    if args.sweep:
        print("\n" + "=" * 100)
        print("等成本扫描：给 RAG 不同块大小与取用条数，看它在同等 token 预算下能否追上")
        print("=" * 100)
        print(f"{'块大小':>7}{'条数':>6}{'L1 @1':>8}{'L1 @8':>8}{'L3 @1':>8}{'L3 @8':>8}"
              f"{'MRR':>7}{'上下文tokens':>13}")
        print("-" * 100)
        for csize, ov, k in [(300, 60, 10), (500, 100, 6), (500, 100, 12),
                             (500, 100, 30), (1000, 200, 6), (1000, 200, 15),
                             (2000, 300, 4), (3000, 400, 2)]:
            ch = chunk_by_window(turns, hub, size=csize, overlap=ov)
            embed_chunks(ch, embedder, verbose=False)
            r = evaluate_rag(ch, embedder, mode="hybrid", topk=k)
            s = r["stats"]

            def g(lvl, i):
                v = s.get(lvl, (0.0, 0.0, 0.0))
                return v[i] * 100

            print(f"{csize:>7}{k:>6}{g('L1', 0):>7.0f}%{g('L1', 1):>7.0f}%"
                  f"{g('L3', 0):>7.0f}%{g('L3', 1):>7.0f}%{r['overall'][2]:>7.2f}"
                  f"{r['ctx_tokens']:>13,}")
        # 本方案在同一张表上的位置
        from run_experiment import eval_ranked
        hub._dense_weight_override = 0.25 if not args.offline else 0.0
        our = eval_ranked(hub, per_probe=3)
        from context_packer import Packer
        writer = Packer(hub, budget_tokens=9000, hot_turns=6, max_cold=4)
        cost = statistics.mean([writer.cells(c.query, current_turn=10**9, multi=True,
                                             per_probe=3).tokens
                                for c in HOLDOUT if c.level != "NEG"])
        s = our["stats"]

        def g2(lvl, i):
            v = s.get(lvl, (0.0, 0.0, 0.0))
            return v[i] * 100

        print("-" * 100)
        print(f"{'本方案':>7}{'4':>6}{g2('L1', 0):>7.0f}%{g2('L1', 1):>7.0f}%"
              f"{g2('L3', 0):>7.0f}%{g2('L3', 1):>7.0f}%{our['overall'][2]:>7.2f}{int(cost):>13,}")
        print("-" * 100)

        # =====================================================================
        # 控制实验：固定"单元大小"，把架构变量与单元大小变量分离开
        # =====================================================================
        # RAG-3000/2 与"本方案取 2 个单元"在单元大小和条数上都相等。
        # 差异只剩一个：单元边界是按固定字符切，还是按事件/话题切。
        # 这是整份对比里最干净的一格。
        print()
        print("=" * 100)
        print("控制实验：单元大小相同（约 1000 token/单元 × 2 个单元）")
        print("=" * 100)
        print(f"{'方案':<34}{'L1 @1':>9}{'L1 @8':>9}{'L3 @1':>9}{'L3 @8':>9}{'MRR':>8}{'tokens':>9}")
        print("-" * 100)
        ch_c = chunk_by_window(turns, hub, size=3000, overlap=400)
        embed_chunks(ch_c, embedder, verbose=False)
        r_c = evaluate_rag(ch_c, embedder, mode="hybrid", topk=2)
        sc = r_c["stats"]

        def gc(lvl, i):
            v = sc.get(lvl, (0.0, 0.0, 0.0))
            return v[i] * 100

        print(f"{'RAG 固定窗口 3000 字符 × 2':<34}{gc('L1', 0):>8.0f}%{gc('L1', 1):>8.0f}%"
              f"{gc('L3', 0):>8.0f}%{gc('L3', 1):>8.0f}%{r_c['overall'][2]:>8.2f}"
              f"{r_c['ctx_tokens']:>9,}")
        # 本方案同条件下（只取 2 个神经元）
        writer2 = Packer(hub, budget_tokens=9000, hot_turns=6, max_cold=2)
        ours_ms: List[float] = []
        per_level: Dict[str, List[Tuple[int, bool]]] = {}
        costs = []
        for case in HOLDOUT:
            if case.level == "NEG":
                continue
            t0 = time.perf_counter()
            traces, routed = hub.multi_route(case.query, per_probe=2, two_hop=True,
                                             allow_external=False)
            ours_ms.append((time.perf_counter() - t0) * 1000)
            want = set(holders(hub, case.anchor))
            rank = next((i for i, c in enumerate(routed[:2], 1) if c in want), 0)
            per_level.setdefault(case.level, []).append((rank, 0 < rank <= 8))
            costs.append(writer2.cells(case.query, current_turn=10**9, multi=True,
                                       per_probe=2).tokens)

        def agg2(rows):
            if not rows:
                return 0.0, 0.0, 0.0
            return (sum(1 for r, _ in rows if r == 1) / len(rows),
                    sum(1 for _, ok in rows if ok) / len(rows),
                    sum((1 / r) if r else 0.0 for r, _ in rows) / len(rows))
        s2 = {k: agg2(v) for k, v in per_level.items()}
        o2 = agg2([r for rows in per_level.values() for r in rows])

        def g3(lvl, i):
            v = s2.get(lvl, (0.0, 0.0, 0.0))
            return v[i] * 100

        print(f"{'本方案 事件边界单元 × 2':<34}{g3('L1', 0):>8.0f}%{g3('L1', 1):>8.0f}%"
              f"{g3('L3', 0):>8.0f}%{g3('L3', 1):>8.0f}%{o2[2]:>8.2f}"
              f"{int(statistics.mean(costs)):>9,}")
        print("-" * 100)
        print("\n这一格是整份对比里最干净的：单元大小、单元数量、取用条数全部相同，")
        print("唯一差异是**单元边界怎么划** —— 按固定字符切，还是按事件/话题切。")

        # ---------------- 显著性检验（必须做，否则就是在用噪声下结论）----------------
        n_l1 = sum(1 for c in HOLDOUT if c.level == "L1")
        n_l3 = sum(1 for c in HOLDOUT if c.level == "L3")
        print()
        print("=" * 100)
        print("显著性检验（探针只有 22 条，差异必须过这一关才算数）")
        print("=" * 100)
        print(f"  L1 共 {n_l1} 条，L3 共 {n_l3} 条 —— 样本极小，单条差异就能让百分比跳 12~20 个点。")
        print()
        for lvl, n in (("L1", n_l1), ("L3", n_l3)):
            a = int(round(gc(lvl, 0) / 100 * n))      # RAG 命中数
            b = int(round(g3(lvl, 0) / 100 * n))      # 本方案命中数
            p = _binom_two_sided_tail(a, b, n)
            verdict = ("差异显著(p<0.05)" if p < 0.05 else
                       "不显著(p>=0.05) → 只能说『方向一致』" if p < 0.25 else
                       "完全在噪声范围内")
            print(f"  {lvl} @1: RAG {a}/{n}  vs  本方案 {b}/{n}   "
                  f"两项检验 p≈{p:.3f}   {verdict}")
        print()
        print("  结论纪律：探针数量 < 30 时，只在『差异 ≥ 样本数的 1/3』时才敢说有效。")
        print("  要把结论做实，必须用真实会话语料扩到 100+ 条探针。")
        return

    print(f"  切块参数   : {args.chunk_size} 字符 / 重叠 {args.overlap}")
    print(f"  取用条数   : top-{args.topk}")

    chunks = chunk_by_window(turns, hub, size=args.chunk_size, overlap=args.overlap)
    embed_chunks(chunks, embedder, verbose=not args.offline)

    print()
    results: Dict[str, Dict[str, object]] = {}

    # ---- RAG 两个变体 ----
    for mode, name in (("vector", "RAG 纯向量"), ("hybrid", "RAG 向量+BM25 混合")):
        r = evaluate_rag(chunks, embedder, mode=mode, topk=args.topk, verbose=args.verbose)
        results[name] = r

    # ---- 我们的架构（同样只取 top-6 个神经元，公平对比）----
    from run_experiment import eval_ranked
    hub._dense_weight_override = 0.25 if not args.offline else 0.0
    t0 = time.perf_counter()
    our = eval_ranked(hub, per_probe=3)
    our_ms = (time.perf_counter() - t0) * 1000 / len(HOLDOUT)
    # 上下文成本：我们按 max_cold=4 + 热区，这里报"载入的神经元 token 量"
    from context_packer import Packer
    writer = Packer(hub, budget_tokens=9000, hot_turns=6, max_cold=args.topk - 2)
    ctx_costs = []
    for case in HOLDOUT:
        if case.level == "NEG":
            continue
        pk = writer.cells(case.query, current_turn=10**9, multi=True, per_probe=3)
        ctx_costs.append(pk.tokens)
    results["神经元架构（本方案）"] = {
        "stats": our["stats"], "overall": our["overall"], "noans": our["noans"],
        "query_ms": ours_ms if args.offline else our["query_ms"],
        "ctx_tokens": int(statistics.mean(ctx_costs)) if ctx_costs else 0,
    }

    print()
    print("=" * 100)
    print("结果")
    print("=" * 100)
    print(f"{'方案':<24}{'L1 @1':>8}{'L1 @8':>8}{'L2 @1':>8}{'L2 @8':>8}{'L3 @1':>8}{'L3 @8':>8}"
          f"{'MRR':>7}{'拒答':>7}{'上下文tokens':>14}")
    print("-" * 100)
    for name, r in results.items():
        s = r["stats"]
        cells = []
        for lvl in ["L1", "L2", "L3"]:
            at1, atk, _ = s.get(lvl, (0.0, 0.0, 0.0))
            cells += [f"{at1 * 100:>7.0f}%", f"{atk * 100:>7.0f}%"]
        mrr = r["overall"][2]
        noans = "3/3" if r["noans"] >= 0.999 else ("无此能力" if r["noans"] == 0.0 else f"{r['noans'] * 100:.0f}%")
        print(f"{name:<24}" + "".join(cells)
              + f"{mrr:>7.2f}{noans:>9}{r['ctx_tokens']:>14,}")
    print("-" * 100)
    print(f"\n注：RAG 的『拒答』结构上为 0 —— 它总会返回 top-k，没有『我不知道』这个选项。")
    print(f"    『上下文 tokens』是回答一个查询实际喂给模型的量（RAG 取 top-{args.topk} 块 / ")
    print(f"    本方案取热区+冷区，budget 9000）。**命中率必须和这个成本一起看。**")


if __name__ == "__main__":
    sys.exit(main())
