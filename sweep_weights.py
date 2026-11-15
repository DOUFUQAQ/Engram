"""
sweep_weights.py —— 用真嵌入重新标定稠密路权重

为什么必须做这一步：
    离线哈希嵌入的稠密路实测无判别力，权重扫描一致给出 0。
    换上真嵌入后余弦尺度完全变了（实测豆包 2048 维：同义对 0.62 / 无关对 0.39），
    原来那个"接上真嵌入就设 0.5"只是猜的。**换组件之后必须重新标定，否则你分不清
    是组件不行还是权重不对。** 这正是本项目一开始踩过的坑的第二次出现。

用法：
    python sweep_weights.py                  # 扫 dense_weight
    python sweep_weights.py --mode both      # 同时扫 dense_weight 与 conf_threshold
"""

from __future__ import annotations

import argparse
import statistics
import sys
from typing import Dict, List, Tuple

from embedder import build_components
from eval_holdout import HOLDOUT
from hub import Hub
from run_experiment import build_hub, eval_ranked, holders
from sim_session import build_session, format_turn


def evaluate_with(hub: Hub, *, dense_w: float, conf: float, margin: float,
                  per_probe: int = 3) -> Dict[str, float]:
    hub._dense_weight_override = dense_w
    hub.conf_threshold = conf
    hub.margin_threshold = margin
    r = eval_ranked(hub, per_probe=per_probe)
    return {"mrr": r["overall"][2], "at6": r["overall"][1],
            "noans": sum(1 for x in r["noans_n"].split("/") if x == "3"),  # 占位，见下
            "noans_rate": r["noans"], "at1": r["overall"][0],
            "levels": {k: v[2] for k, v in r["stats"].items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="dense", choices=["dense", "both"])
    ap.add_argument("--env", default=".env")
    args = ap.parse_args()

    comp = build_components(args.env, strict=False)
    if not comp.strict_ok:
        print("没有配置在线嵌入，扫描无意义（离线稠密路权重恒为 0）。")
        print("请先在 .env 里配好 EMBED_* ，或参考 .env.example 的方案 A。")
        return

    print("=" * 96)
    print("用真嵌入标定稠密路权重")
    print("=" * 96)
    print(comp.describe())
    print()
    hub = build_hub(comp.embedder, None, verbose=True)

    if args.mode == "dense":
        print(f"\n{'dense_w':>9}{'MRR':>8}{'@1':>8}{'@6':>8}{'NEG拒绝':>10}"
              f"{'L1 MRR':>9}{'L2 MRR':>9}{'L3 MRR':>9}")
        print("-" * 96)
        best = None
        for dw in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            r = evaluate_with(hub, dense_w=dw, conf=0.35, margin=0.03)
            lv = r["levels"]
            flag = ""
            if best is None or r["mrr"] > best[1]:
                best, flag = (dw, r["mrr"]), "  ★"
            print(f"{dw:>9.2f}{r['mrr']:>8.2f}{r['at1'] * 100:>7.0f}%{r['at6'] * 100:>7.0f}%"
                  f"{r['noans_rate'] * 100:>9.0f}%{lv.get('L1', 0):>9.2f}{lv.get('L2', 0):>9.2f}"
                  f"{lv.get('L3', 0):>9.2f}{flag}")
        print("-" * 96)
        print(f"\n最优 dense_weight = {best[0]:.2f}（MRR {best[1]:.2f}）")
        print("把 run_experiment / Hub 里的默认值改成这个数（或设 EMBED 侧的常量）。")
        return

    # mode == both：先粗扫 dense，再在最优附近扫 conf
    print("第 1 步：扫 dense_weight")
    rows = []
    for dw in [0.0, 0.2, 0.35, 0.5, 0.65, 0.8]:
        r = evaluate_with(hub, dense_w=dw, conf=0.35, margin=0.03)
        rows.append((dw, r))
        print(f"  dense_w={dw:.2f}  MRR={r['mrr']:.2f}  @6={r['at6'] * 100:.0f}%  "
              f"NEG={r['noans_rate'] * 100:.0f}%")
    best_dw = max(rows, key=lambda t: t[1]["mrr"])[0]

    print(f"\n第 2 步：在 dense_w={best_dw:.2f} 下扫 conf_threshold")
    print(f"\n{'conf':>8}{'MRR':>8}{'@1':>8}{'@6':>8}{'NEG拒绝':>10}")
    print("-" * 48)
    for conf in [0.15, 0.25, 0.35, 0.45, 0.55]:
        r = evaluate_with(hub, dense_w=best_dw, conf=conf, margin=0.03)
        print(f"{conf:>8.2f}{r['mrr']:>8.2f}{r['at1'] * 100:>7.0f}%{r['at6'] * 100:>7.0f}%"
              f"{r['noans_rate'] * 100:>9.0f}%")
    print("-" * 48)
    print(f"\n建议：dense_weight={best_dw:.2f}，conf_threshold 取'NEG 拒绝仍为 100%'里的最小值。")


if __name__ == "__main__":
    sys.exit(main())
