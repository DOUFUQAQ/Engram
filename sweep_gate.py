"""
sweep_gate.py —— 标定置信门（conf_threshold × dense_weight）

为什么单独做这件事：
    debug_l3.py 揭示了一个关键事实：L3 因果查询里，**正确的神经元在全库重排里常排第 1-2 名**，
    但每条探针都因为 conf 落在 0.31~0.34 而被 0.35 的阈值拦下，于是并集为空、压根没进候选池。
    这不是召回能力问题，是**闸门标定问题** —— 而闸门是在离线弱打分上标出来的。

    所以换真嵌入/加二跳之后，闸门必须重新标定，否则前面所有改进都被它挡在门外。

指标里 NEG 拒绝率是安全底线：阈值调低不能以"开始对不知道的问题瞎答"为代价。

用法：
    python sweep_gate.py                  # 真嵌入（读 .env）
    python sweep_gate.py --offline        # 离线对比
    python sweep_gate.py --with-rerank    # 带上 LLM 重排（慢，会调 API）
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Tuple

from embedder import build_components
from hub import HashingEmbedder
from run_experiment import build_hub, eval_ranked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--with-rerank", action="store_true")
    ap.add_argument("--env", default=".env")
    args = ap.parse_args()

    comp = build_components(args.env, strict=False)
    if args.offline or not comp.strict_ok:
        embedder, reranker, label = HashingEmbedder(dim=4096), None, "离线哈希"
    else:
        embedder = comp.embedder
        reranker = comp.reranker if args.with_rerank else None
        label = comp.embedder.model + (" + 重排" if reranker else "")
    hub = build_hub(embedder, reranker, verbose=not args.offline)
    print(f"嵌入: {label}\n")

    grid: List[Tuple[float, float]] = [
        (dw, ct) for dw in (0.0, 0.25, 0.4)
        for ct in (0.20, 0.25, 0.30, 0.35)
    ]
    print("=" * 104)
    print("置信门标定：conf_threshold × dense_weight")
    print("=" * 104)
    print(f"{'dense_w':>8}{'conf':>7}{'MRR':>7}{'@1':>7}{'@6':>7}{'L1 MRR':>9}{'L2 MRR':>9}"
          f"{'L3 MRR':>9}{'NEG拒绝':>9}{'查询ms':>9}")
    print("-" * 104)
    best = None
    for dw, ct in grid:
        hub._dense_weight_override = dw
        hub.conf_threshold = ct
        hub.margin_threshold = 0.02
        r = eval_ranked(hub, per_probe=3)
        lv = {k: v[2] for k, v in r["stats"].items()}
        mrr = r["overall"][2]
        # 安全约束：NEG 拒绝必须是 100%，否则这个组合不能选
        safe = r["noans"] >= 0.999
        flag = ""
        if safe and (best is None or mrr > best[0]):
            best, flag = (mrr, dw, ct), "  ★"
        print(f"{dw:>8.2f}{ct:>7.2f}{mrr:>7.2f}{r['overall'][0] * 100:>6.0f}%"
              f"{r['overall'][1] * 100:>6.0f}%{lv.get('L1', 0):>9.2f}{lv.get('L2', 0):>9.2f}"
              f"{lv.get('L3', 0):>9.2f}{r['noans'] * 100:>8.0f}%{r['query_ms']:>9.0f}{flag}")
    print("-" * 104)
    if best:
        print(f"\n在 NEG 拒绝 100% 的约束下，最优: dense_weight={best[1]:.2f}, "
              f"conf_threshold={best[2]:.2f}  →  MRR {best[0]:.2f}")
    else:
        print("\n没有组合能同时保住 NEG 拒绝 100% —— 阈值降得过头了。")
    if reranker:
        print(f"重排 API 用量: {reranker.stats.describe()}")


if __name__ == "__main__":
    sys.exit(main())
