"""
significance.py —— 给实验结果做显著性检验

为什么必须做：探针只有 22~30 条，**一条问题的差异就能让百分比跳 3~17 个点**。
不做检验就下结论，等于用噪声写结论（本原型在合成集上已经犯过一次）。

用法：
    python significance.py --a 26 --b 25 --n 30            # 命中数对比
    python significance.py --a 0.867 --b 0.844 --n 30      # 比例对比
    python significance.py --lme                            # 直接跑 LongMemEval 的两组数
"""

from __future__ import annotations

import argparse
import math
import sys


def two_proportion_p(a: float, b: float, n: int) -> float:
    """两个配对比例的差异检验（正态近似 + 连续性校正）。

    这里没有每条样本的配对结果，所以用最保守的独立比例近似。
    注意：配对设计（McNemar）通常给出更小的 p，所以这个近似对"声称有效"是**不利**的，
    即：如果它都说显著，那就真显著。
    """
    x1, x2 = a * n, b * n
    if n <= 0:
        return 1.0
    p = (x1 + x2) / (2 * n)
    if p <= 0 or p >= 1:
        return 1.0
    se = math.sqrt(2 * p * (1 - p) / n)
    z = abs(x1 - x2) / n / se
    # 双侧正态尾概率
    return max(0.0, min(1.0, math.erfc(z / math.sqrt(2))))


def verdict(p: float) -> str:
    if p < 0.01:
        return "差异极显著 (p<0.01)"
    if p < 0.05:
        return "差异显著 (p<0.05)"
    if p < 0.20:
        return "弱证据 (0.05<=p<0.20)，只能说方向一致"
    return "完全在噪声范围内，不能下结论"


def diff_ci(a: float, b: float, n: int) -> tuple:
    """差异的 95% 置信区间（比例差）。"""
    d = a - b
    p1, p2 = a, b
    se = math.sqrt(max(1e-9, p1 * (1 - p1) / n + p2 * (1 - p2) / n))
    return d - 1.96 * se, d + 1.96 * se


def run(a: float, b: float, n: int, label: str = "A vs B") -> None:
    p = two_proportion_p(a, b, n)
    lo, hi = diff_ci(a, b, n)
    print(f"\n{label}")
    print(f"  样本数        : {n}")
    print(f"  A = {a * 100:.1f}%   ({a * n:.1f}/{n})")
    print(f"  B = {b * 100:.1f}%   ({b * n:.1f}/{n})")
    print(f"  差值          : {(a - b) * 100:+.1f} 个百分点 "
          f"（相当于 {(a - b) * n:+.1f} 条问题）")
    print(f"  差值的 95% CI : [{(lo if False else lo) * 100:+.1f}, {hi * 100:+.1f}] 个百分点")
    print(f"  两项检验 p    : {p:.3f}")
    print(f"  判定          : {verdict(p)}")
    if lo < 0 < hi:
        print(f"  → 置信区间跨过 0：**这个差异在统计上无法与 0 区分**")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=float)
    ap.add_argument("--b", type=float)
    ap.add_argument("--n", type=int)
    ap.add_argument("--lme", action="store_true", help="跑 LongMemEval 的两组实测数")
    args = ap.parse_args()

    print("=" * 84)
    print("显著性检验 —— 探针少的时候，这一步决定你是在下结论还是在读噪声")
    print("=" * 84)

    if args.lme:
        # 来自 eval_longmemeval.py --per-type 5 --topk 8 --budget 18000（真实公开评测集）
        run(0.867, 0.844, 30, "LongMemEval-S / Recall@8 / 30 条  （本方案 vs RAG）")
        run(0.800, 0.800, 30, "LongMemEval-S / NDCG@8 / 30 条")
        run(0.790, 0.786, 30, "LongMemEval-S / MRR / 30 条")
        print()
        print("  ── 另一组：等 top-5、不限预算（同样 30 条）──")
        run(0.867, 0.817, 30, "LongMemEval-S / Recall@5 / 30 条")
        print()
        print("  ── 合成语料上的控制实验（单元大小相同）──")
        run(0.500, 0.250, 8, "L1 @1 / 8 条（本方案事件边界 vs RAG 固定窗口）")
        run(0.200, 0.400, 5, "L3 @1 / 5 条")
        print()
        print("  总结：在 30 条真实评测集样本上，本方案与调好参数的 RAG 的差异")
        print("        全部落在噪声范围内（95% CI 跨过 0）。**不能声称比 RAG 更准。**")
        return

    if args.a is None or args.b is None or args.n is None:
        print("\n请给 --a --b --n（比例）或 --lme。例：")
        print("  python significance.py --a 0.867 --b 0.844 --n 30")
        return
    run(args.a, args.b, args.n)


if __name__ == "__main__":
    sys.exit(main())
