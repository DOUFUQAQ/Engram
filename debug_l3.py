"""
debug_l3.py —— 专门追查 L3 因果查询：正确神经元到底有没有进候选池

诊断比调参重要：如果正确答案压根没被召回，那再强的重排也没用；
如果它进了候选池但被排到后面，问题才在排序。

用法：
    python debug_l3.py                 # 只用 .env 里的真嵌入
    python debug_l3.py --offline       # 用离线嵌入对比
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from embedder import build_components
from eval_holdout import HOLDOUT
from hub import Hub, HashingEmbedder
from run_experiment import build_hub, holders


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--env", default=".env")
    args = ap.parse_args()

    comp = build_components(args.env, strict=False)
    if args.offline or not comp.strict_ok:
        embedder, label = HashingEmbedder(dim=4096), "离线哈希"
    else:
        embedder, label = comp.embedder, comp.embedder.model
    hub = build_hub(embedder, None, verbose=not args.offline)
    hub._dense_weight_override = 0.25
    print(f"嵌入: {label}   神经元 {len(hub.cells)} 个\n")

    print("=" * 100)
    print("L3 因果查询：分段诊断")
    print("=" * 100)
    for case in [c for c in HOLDOUT if c.level == "L3"]:
        want = set(holders(hub, case.anchor))
        print(f"\n【{case.query}】")
        print(f"  期望锚点: {case.anchor[:60]}")
        print(f"  持有神经元: {sorted(want)}")
        if case.note:
            print(f"  难度说明: {case.note}")

        # ① 拆出了哪些探针
        probes = Hub.decompose(case.query)
        print(f"  ① 拆出 {len(probes)} 条探针:")
        for p in probes:
            print(f"       · {p[:66]}")

        # ② 每条探针单独召回的结果 + 正确神经元是否在池里
        pool: List[str] = []
        print(f"  ② 逐条探针召回:")
        for p in probes:
            tr = hub.route(p, top_k=3, allow_external=False)
            ids = [c for c, _ in tr.hits]
            pool += ids
            hit = "✅" if want & set(ids) else "  "
            print(f"       {hit} conf={tr.confidence:.3f} {tr.stage:<9} -> {ids}   << {p[:34]}")

        # ③ 二跳扩散有没有救回来。
        #    注意：第一跳若被置信门拦下，二跳就不会启动（这是刻意设计：种子不可信不扩散），
        #    所以这里必须把"闸门拦下"和"扩散失败"区分开，否则会误判成召回能力问题。
        tr1 = hub.route(case.query, top_k=3, allow_external=False)
        if tr1.stage == "cell":
            _, spread_ids = hub.route_with_spread(case.query, top_k=8, allow_external=False)
            print(f"  ③ 二跳后候选前 8: {spread_ids[:8]}")
            print(f"      正确神经元在二跳候选里吗: {'✅ 在' if want & set(spread_ids[:8]) else '❌ 不在'}")
        else:
            spread_ids = []
            print(f"  ③ 二跳未启动：第一跳被闸门拦下（conf={tr1.confidence:.3f} < {hub.conf_threshold}）")
            print(f"      → 种子不可信时不扩散，这是刻意设计；此时应交给外呼通道")

        # ④ 全库按当前打分排序，看正确神经元的名次
        ordered = hub.rerank(case.query, list(hub.cells))
        rank = next((i for i, c in enumerate(ordered, 1) if c in want), None)
        print(f"  ④ 全库重排名次: {rank or '未出现'} / {len(ordered)}")
        if rank and rank > 8:
            print(f"      → 它排在前面这些之下: {ordered[:6]}")

        # ⑤ 判定瓶颈
        if not spread_ids and not pool:
            print(f"  >>> 结论: 全部探针被**置信门拦下** —— 并集为空，正确神经元根本没进候选池。"
                  f"这是阈值问题，不是召回能力问题（见 ④ 的全库名次）")
        elif want & set(spread_ids[:8]):
            print(f"  >>> 结论: 召回没问题（候选池里有），是**排序**把它压下去了")
        elif want & set(pool):
            print(f"  >>> 结论: 探针召回里有，但合并/重排丢了")
        else:
            print(f"  >>> 结论: **压根没召回** —— 词面与语义检索都够不着，只能靠 LLM 重排/模型自己提问")


if __name__ == "__main__":
    sys.exit(main())
