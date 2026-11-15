"""
evaluate.py —— 留出集评测（度量排名，不看"取几个"）

为什么用**排名**而不是 recall@k：
    上一版评测里，baseline 在闸门拦下时偷偷回退成"全库重排取前 6"，等于绕过闸门作弊，
    于是基线虚高到 19/22。改成度量"正确神经元排在第几"，谁都无法靠多取几个来刷分。

四条被对比的管线（都是真实可用的组合）：
  single          单指针路由：闸门通过则用融合名次，被拦下则空
  multi           多路脑电波：拆线索 → 各取 top-3 → 全局重排
  multi+2hop      多路脑电波 + 二跳联想扩散
  multi+2hop+exp  再加上查询扩展

指标：
  @1 / @6         正确神经元落在第 1 位 / 前 6 位的比例（L1/L2/L3 分列）
  MRR             平均倒数排名（越大越好，0 表示完全没召回到）
  NoAns           负样本正确"举手"的比例（闸门是否知道自己在不知道）
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Callable, Dict, List, Tuple

from eval_holdout import HOLDOUT, EvalCase
from hub import Hub
from sim_session import build_session, format_turn

TOPK = 6


def build_hub() -> Hub:
    turns, _, _ = build_session()
    hub = Hub()
    for t in turns:
        hub.write(format_turn(t), turn_index=t.index, files=t.files)
    hub.consolidate()
    hub.decay()
    hub._sync_index()
    return hub


def holders(hub: Hub, anchor: str) -> List[str]:
    return [cid for cid, c in hub.cells.items() if anchor in c.text]


# ---------------------------------------------------------------------------
# 四条管线的"排名函数"：输入查询，输出按相关度排序的 cell_id 列表
# ---------------------------------------------------------------------------

def rank_single(hub: Hub, q: str) -> Tuple[List[str], bool, float]:
    t0 = time.perf_counter()
    tr = hub.route(q, allow_external=False)
    if tr.stage == "cell":
        ranked = [cid for cid, _ in tr.hits]
        ranked += [cid for cid in hub.rerank(q, list(hub.cells)) if cid not in ranked]
    else:
        ranked = []                     # 闸门拦下 = 承认召回不到，不做兜底
    return ranked, tr.stage == "cell", (time.perf_counter() - t0) * 1000


def rank_multi(hub: Hub, q: str, *, two_hop: bool = False, expand: bool = False
               ) -> Tuple[List[str], bool, float]:
    t0 = time.perf_counter()
    traces, routed = hub.multi_route(q, per_probe=3, two_hop=two_hop, expand=expand,
                                     allow_external=False)
    accepted = any(t.stage == "cell" for t in traces)
    return routed, accepted, (time.perf_counter() - t0) * 1000


PIPELINES: Dict[str, Callable[[Hub, str], Tuple[List[str], bool, float]]] = {
    "single": rank_single,
    "multi": lambda h, q: rank_multi(h, q),
    "multi+2hop": lambda h, q: rank_multi(h, q, two_hop=True),
    "multi+2hop+exp": lambda h, q: rank_multi(h, q, two_hop=True, expand=True),
}


def evaluate(hub: Hub, fn, *, verbose: bool = False) -> Dict[str, object]:
    per_level: Dict[str, List[Tuple[int, bool]]] = {}
    lat: List[float] = []
    noans: List[bool] = []
    detail: List[Tuple[EvalCase, int, bool]] = []

    for case in HOLDOUT:
        ranked, accepted, ms = fn(hub, case.query)
        lat.append(ms)
        if case.level == "NEG":
            noans.append(not accepted)
            detail.append((case, -1, not accepted))
            if verbose:
                print(f"  [NEG] {'✅' if not accepted else '❌'} {case.query[:32]:<34} "
                      f"stage={'cell' if accepted else 'escalate'}")
            continue
        want = set(holders(hub, case.anchor))
        rank = 0
        for i, cid in enumerate(ranked, 1):
            if cid in want:
                rank = i
                break
        per_level.setdefault(case.level, []).append((rank, rank <= TOPK))
        detail.append((case, rank, rank > 0))
        if verbose:
            tag = "✅" if rank == 1 else ("🟡" if 0 < rank <= TOPK else "❌")
            print(f"  [{case.level}] {tag} {case.query[:32]:<34} rank={rank or '-'}"
                  f"{'  ← ' + case.note if case.note and rank != 1 else ''}")

    def agg(rows: List[Tuple[int, bool]]) -> Tuple[float, float, float]:
        if not rows:
            return 0.0, 0.0, 0.0
        at1 = sum(1 for r, _ in rows if r == 1) / len(rows)
        atk = sum(1 for _, ok in rows if ok) / len(rows)
        mrr = sum((1 / r) if r else 0.0 for r, _ in rows) / len(rows)
        return at1, atk, mrr

    stats = {lvl: agg(rows) for lvl, rows in per_level.items()}
    overall_rows = [r for rows in per_level.values() for r in rows]
    return {
        "stats": stats, "overall": agg(overall_rows),
        "noans": (sum(noans) / len(noans)) if noans else 0.0,
        "noans_n": f"{sum(noans)}/{len(noans)}",
        "latency_ms": statistics.mean(lat), "detail": detail,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    hub = build_hub()
    print("=" * 100)
    print(f"留出集评测   神经元 {len(hub.cells)} 个   冻结 {hub.total_tokens():,} tokens   "
          f"探针 {len(HOLDOUT)} 条（L1 字面 {sum(1 for c in HOLDOUT if c.level=='L1')} / "
          f"L2 概念 {sum(1 for c in HOLDOUT if c.level=='L2')} / "
          f"L3 因果 {sum(1 for c in HOLDOUT if c.level=='L3')} / "
          f"NEG {sum(1 for c in HOLDOUT if c.level=='NEG')}）")
    print("=" * 100)

    pipes = {k: v for k, v in PIPELINES.items() if not args.only or k == args.only}
    results = {}
    for name, fn in pipes.items():
        if args.verbose:
            print(f"\n--- {name} ---")
        results[name] = evaluate(hub, fn, verbose=args.verbose)

    print()
    print(f"{'管线':<16}{'L1 @1':>8}{'L1 @6':>8}{'L2 @1':>8}{'L2 @6':>8}{'L3 @1':>8}{'L3 @6':>8}"
          f"{'MRR':>8}{'总计@6':>10}{'NEG拒绝':>10}{'延迟':>10}")
    print("-" * 100)
    for name, r in results.items():
        s = r["stats"]
        cells = []
        for lvl in ["L1", "L2", "L3"]:
            at1, atk, _ = s.get(lvl, (0.0, 0.0, 0.0))
            cells.append(f"{at1 * 100:>7.0f}%")
            cells.append(f"{atk * 100:>7.0f}%")
        _, _, mrr = r["overall"]
        at1, at6, _ = r["overall"]
        print(f"{name:<16}" + "".join(cells)
              + f"{mrr:>8.2f}{at6 * 100:>9.0f}%{r['noans_n']:>10}{r['latency_ms']:>8.1f}ms")
    print("-" * 100)

    if len(results) > 1:
        base = results.get("single") or results[list(results)[0]]
        best = max(results.items(), key=lambda kv: kv[1]["overall"][2])
        print(f"\n单指针 → 最佳管线({best[0]}): MRR {base['overall'][2]:.2f} → {best[1]['overall'][2]:.2f}   "
              f"@6 {base['overall'][1] * 100:.0f}% → {best[1]['overall'][1] * 100:.0f}%")
        for lvl in ["L1", "L2", "L3"]:
            b = base["stats"].get(lvl, (0, 0, 0))
            t = best[1]["stats"].get(lvl, (0, 0, 0))
            print(f"  {lvl}: @1 {b[0] * 100:.0f}% → {t[0] * 100:.0f}%   "
                  f"@6 {b[1] * 100:.0f}% → {t[1] * 100:.0f}%   MRR {b[2]:.2f} → {t[2]:.2f}")
        print("\n注：NEG 拒绝率衡量「闸门是否知道自己在不知道」，它是安全指标，不是召回指标。")


if __name__ == "__main__":
    sys.exit(main())
