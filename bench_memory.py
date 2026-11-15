"""
bench_memory.py —— 主实验：四种上下文策略的「精确细节存活率 vs 上下文成本」

对照的四种策略，都是真实 agent 里能见到的东西：
  full        全量原文进上下文（今天的理想，物理上撑不过长会话）
  compaction  压缩摘要进上下文（今天的主流做法 → 40 轮后的飘忽）
  recency     只留最近 N 轮（省钱，但早期事实直接失忆）
  cells       本方案：神经元保真冻结 + 中枢按需调取（稀疏激活）

两个指标必须分开看，否则容易自欺：
  fact_recall   以**神经元**为召回单位：某条事实所在的神经元被载入，就算召回
                （载入一个神经元 = 它包含的全部事实都回来了，这才是记忆的真实粒度）
  exact_string  以**精确串**为召回单位：那个字符串真的出现在最终上下文里
                （压缩摘要最容易丢的就是这一类）

用法：
  python bench_memory.py                # 跑全部
  python bench_memory.py --only cells --verbose
"""

from __future__ import annotations

import argparse
import statistics
import sys
from typing import Dict, List, Optional, Tuple

from context_packer import Packer
from hub import Hub, est_tokens
from metrics import Anchor, Score, score_pack
from sim_session import PROBE_WORKING_SET, build_session, format_turn

BUDGET = 9000       # 当前工作区预算（token）
HOT_TURNS = 6       # 永远保留原文的最近轮数


# ---------------------------------------------------------------------------
# 基线策略
# ---------------------------------------------------------------------------

def run_full(turns, probe, anchors, anchor_cells) -> Dict[str, object]:
    pk = Packer(Hub(), budget_tokens=BUDGET, hot_turns=HOT_TURNS)
    pack = pk.full([format_turn(t) for t in turns])
    s = score_pack(pack, anchors, probe.index)
    return {"mode": "full", "score": s, "tokens": pack.tokens, "peak": pack.tokens,
            "latency_ms": 0.0, "extra": {}, "packed_cells": None, "anchor_cells": anchor_cells}


def run_compaction(turns, probe, anchors, anchor_cells) -> Dict[str, object]:
    pk = Packer(Hub(), budget_tokens=BUDGET, hot_turns=HOT_TURNS)
    pack = pk.compaction([format_turn(t) for t in turns])
    s = score_pack(pack, anchors, probe.index)
    return {"mode": "compaction", "score": s, "tokens": pack.tokens, "peak": pack.tokens,
            "latency_ms": 0.0, "extra": {}, "packed_cells": None, "anchor_cells": anchor_cells}


def run_recency(turns, probe, anchors, anchor_cells) -> Dict[str, object]:
    pk = Packer(Hub(), budget_tokens=BUDGET, hot_turns=HOT_TURNS)
    pack = pk.recency([format_turn(t) for t in turns])
    s = score_pack(pack, anchors, probe.index)
    return {"mode": "recency", "score": s, "tokens": pack.tokens, "peak": pack.tokens,
            "latency_ms": 0.0, "extra": {}, "packed_cells": None, "anchor_cells": anchor_cells}


# ---------------------------------------------------------------------------
# 本方案
# ---------------------------------------------------------------------------

def run_cells(turns, probe, anchors, anchor_cells, *, verbose: bool = False,
              per_probe: int = 3, expand: bool = False, two_hop: bool = False,
              max_cold: int = 6) -> Dict[str, object]:
    hub = Hub()
    writer = Packer(hub, budget_tokens=BUDGET, hot_turns=HOT_TURNS, max_cold=max_cold)

    offsite: List[Dict[str, object]] = []
    hub.external_fn = lambda q: (offsite.append({"query": q[:80]}) or None)

    tokens_hist: List[int] = []
    lat_hist: List[float] = []
    esc = hit = 0
    for i, t in enumerate(turns):
        hub.write(format_turn(t), turn_index=t.index, files=t.files)
        if (i + 1) % 12 == 0:                      # 周期性"睡眠"：巩固 + 遗忘
            hub.consolidate()
            hub.decay()
        if t.role == "user":                       # 用户提问 = 一次脑电波（只拿原话）
            tr = hub.route(t.utterance or format_turn(t), top_k=3, allow_external=False)
            lat_hist.append(tr.latency_ms)
            hit += tr.stage == "cell"
            esc += tr.stage != "cell"
        nxt = turns[i + 1] if i + 1 < len(turns) else probe
        nxt_q = (nxt.utterance or format_turn(nxt)) if nxt.role == "user" else format_turn(nxt)
        pack = writer.cells(nxt_q, current_turn=t.index, allow_external=False,
                            per_probe=per_probe, expand=expand, two_hop=two_hop)
        tokens_hist.append(pack.tokens)

    # 最终：第 72 轮事故提问（同样只拿人类原话做路由）
    pack = writer.cells(probe.utterance, current_turn=probe.index, allow_external=True,
                        per_probe=per_probe, expand=expand, two_hop=two_hop)
    tr = pack.trace
    s = score_pack(pack, anchors, probe.index)
    packed = set(pack.cell_ids)
    # 事实落在哪个神经元，以**运行结束时**的实际划分为准
    # （巩固合并会把事实搬进合并后的神经元；只要信息还在库里，就算召回到了）
    final_anchor_cells = {}
    for a in anchors:
        for cid, c in hub.cells.items():
            if a.text in c.text:
                final_anchor_cells[a.text] = cid
                break
    recalled = sum(1 for a in anchors if final_anchor_cells.get(a.text) in packed)

    if verbose:
        print("\n" + "=" * 100)
        print("【cells 模式最终打包内容（前 5000 字符）】")
        print("=" * 100)
        print(pack.text[:5000])
        print("\n--- 路由轨迹 ---")
        print(tr.describe() if tr else "-")
        print(f"载入神经元: {sorted(packed)}")
        print(f"精确串存活 {s.alive}/{s.total}   神经元级召回 {recalled}/{len(anchors)}")
        print(f"缺失: {s.missing}")

    return {
        "mode": "cells", "score": s,
        "tokens": int(statistics.mean(tokens_hist)), "peak": max(tokens_hist),
        "latency_ms": statistics.mean(lat_hist) if lat_hist else 0.0,
        "extra": {
            # 用中枢自己的计数器算，避免选择性统计自我美化
            "local_hit_rate": hub.stats["local_hit"] / max(1, hub.stats["route_calls"]),
            "escalation_rate": hub.stats["escalated"] / max(1, hub.stats["route_calls"]),
            "route_calls": hub.stats["route_calls"],
            "escalated_total": hub.stats["escalated"],
            "final_stage": tr.stage if tr else "-",
            "cells_final": len(hub.cells),
            "frozen_tokens": hub.total_tokens(),
            "final_pack_tokens": pack.tokens,
            "final_route": tr.describe() if tr else "-",
            "hub_stats": dict(hub.stats),
            "offsite_calls": len(offsite),
            "compression_ratio": hub.total_tokens() / max(1, int(statistics.mean(tokens_hist))),
            "packed_cells": sorted(packed),
            "anchor_cells_final": final_anchor_cells,
            "working_set_hit": sum(
                1 for t in PROBE_WORKING_SET
                # 任意一个被载入的神经元含有该事实即算命中。
                # 不能用"第一个持有者"记账：多持有者的锚点（如「20 分钟」在 N0002/N0003）
                # 会因为取错持有者而假性失分。
                if any(t in hub.cells[c].text for c in packed if c in hub.cells)
            ),
            "working_set_total": len(PROBE_WORKING_SET),
        },
        "packed_cells": packed,
        "anchor_cells": final_anchor_cells,
    }


# ---------------------------------------------------------------------------

# 模型在看到事故报告后，会自己发起的聚焦提问（真实 agent 里的记忆工具调用）
RECALL_FACETS = [
    "session TTL 过期时间是多少？",
    "webhook 读超时和连接超时分别是多少？",
    "数据库连接串来自哪个环境变量？",
]


def run_recall_loop(turns, probe, anchors, anchor_cells, *, verbose: bool = False) -> Dict[str, object]:
    """记忆工具闭环：被动打包一次 + 模型主动 recall N 次，合并后的上下文。"""
    hub = Hub()
    writer = Packer(hub, budget_tokens=BUDGET, hot_turns=HOT_TURNS, max_cold=4)
    hub.external_fn = lambda q: None
    for i, t in enumerate(turns):
        hub.write(format_turn(t), turn_index=t.index, files=t.files)
        if (i + 1) % 12 == 0:
            hub.consolidate()
            hub.decay()

    base = writer.cells(probe.utterance, current_turn=probe.index, allow_external=True,
                        per_probe=3)
    packed = set(base.cell_ids)
    recall_tokens = 0
    facets = []
    for q in RECALL_FACETS:
        rp = writer.recall(q, top_k=4, current_turn=probe.index)
        recall_tokens += rp.tokens
        facets.append((q, rp.cell_ids, rp.trace.stage if rp.trace else "-"))
        packed |= set(rp.cell_ids)

    final_anchor_cells = {}
    for a in anchors:
        for cid, c in hub.cells.items():
            if a.text in c.text:
                final_anchor_cells[a.text] = cid
                break
    recalled = [a.text for a in anchors if final_anchor_cells.get(a.text) in packed]
    ws_hit = [t for t in PROBE_WORKING_SET
              if any(t in hub.cells[c].text for c in packed if c in hub.cells)]

    if verbose:
        print("\n" + "=" * 100)
        print("【记忆工具闭环：模型主动提问的结果】")
        print("=" * 100)
        for q, ids, stage in facets:
            print(f"  [{stage}] {q:<34} -> {ids}")
        print(f"\n  被动打包: {base.cell_ids} ({base.tokens:,} tokens)")
        print(f"  recall 追加: {recall_tokens:,} tokens")
        print(f"  工作集命中: {len(ws_hit)}/{len(PROBE_WORKING_SET)}  {ws_hit}")

    return {
        "mode": "cells+recall", "score": base.score if hasattr(base, "score") else None,
        "tokens": base.tokens + recall_tokens, "peak": base.tokens + recall_tokens,
        "latency_ms": 0.0,
        "extra": {
            "recall_tokens": recall_tokens,
            "base_tokens": base.tokens,
            "packed_cells": sorted(packed),
            "working_set_hit": len(ws_hit), "working_set_total": len(PROBE_WORKING_SET),
            "anchor_recall": len(recalled),
            "facets": facets,
        },
        "packed_cells": packed,
        "anchor_cells": final_anchor_cells,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--per-probe", type=int, default=3)
    ap.add_argument("--expand", action="store_true", help="打开查询扩展（实测降精度，用于复现结论）")
    ap.add_argument("--two-hop", action="store_true", help="打开二跳联想检索（种子神经元扩散罕见词）")
    ap.add_argument("--max-cold", type=int, default=6, help="冷区最多调取几个神经元")
    ap.add_argument("--sweep-cold", action="store_true", help="扫冷区数量的成本-收益曲线")
    ap.add_argument("--recall-loop", action="store_true",
                    help="跑记忆工具闭环：被动打包 + 模型主动 recall")
    args = ap.parse_args()

    turns, probe, anchors = build_session()
    raw_tokens = sum(est_tokens(format_turn(t)) for t in turns)

    # 先建一份"权威"神经元划分，记录每条事实落在哪个神经元里
    ref = Hub()
    anchor_cells: Dict[str, str] = {}
    for t in turns:
        cid = ref.write(format_turn(t), turn_index=t.index, files=t.files)
        for a in anchors:
            if a.turn == t.index and a.text in ref.cells[cid].text:
                anchor_cells[a.text] = cid

    print("=" * 104)
    print("实验设定")
    print("=" * 104)
    print(f"  会话长度        : {len(turns)} 轮（最后追加第 {probe.index} 轮事故提问）")
    print(f"  全量原文        : {raw_tokens:,} tokens")
    print(f"  工作区预算      : {BUDGET:,} tokens（热区 {HOT_TURNS} 轮原文 + 冷区按需调取 + 指针目录）")
    print(f"  待考精确细节    : {len(anchors)} 条，分布在 {len(set(anchor_cells.values()))} 个神经元里")
    print(f"  最早埋点        : 第 {min(a.turn for a in anchors)} 轮（离最终提问 {probe.index - min(a.turn for a in anchors)} 轮）")
    print()

    runners = {
        "full": run_full, "compaction": run_compaction, "recency": run_recency,
        "cells": lambda *a: run_cells(*a, verbose=args.verbose, per_probe=args.per_probe,
                                      expand=args.expand, two_hop=args.two_hop,
                                      max_cold=args.max_cold),
    }
    if args.only:
        runners = {k: v for k, v in runners.items() if k == args.only}

    if args.recall_loop:
        print("=" * 104)
        print("记忆工具闭环：被动打包一次 + 模型主动 recall 三次（同一事故提问）")
        print("=" * 104)
        r = run_recall_loop(turns, probe, anchors, anchor_cells, verbose=True)
        ex = r["extra"]
        print()
        print(f"  上下文总量      : {r['tokens']:,} tokens"
              f"（被动 {ex['base_tokens']:,} + recall {ex['recall_tokens']:,}）")
        print(f"  载入神经元      : {len(ex['packed_cells'])} 个")
        print(f"  事故工作集命中  : {ex['working_set_hit']}/{ex['working_set_total']}   "
              f"（被动管线只有 3/6）")
        print(f"  精确串召回      : {ex['anchor_recall']}/{len(anchors)}   （被动管线 8/23）")
        return

    if args.sweep_cold:
        print("=" * 104)
        print("冷区数量成本-收益曲线（预算 9,000 token 固定）")
        print("=" * 104)
        print(f"{'max_cold':>9}{'神经元级召回':>14}{'精确串存活':>12}{'平均上下文':>12}{'最终上下文':>12}{'载入神经元数':>13}")
        print("-" * 104)
        for mc in [2, 3, 4, 6, 8, 12]:
            r = run_cells(turns, probe, anchors, anchor_cells, two_hop=args.two_hop,
                          max_cold=mc)
            packed = r["packed_cells"]
            n = sum(1 for a in anchors if r["anchor_cells"].get(a.text) in packed)
            s: Score = r["score"]
            print(f"{mc:>9}{n / len(anchors) * 100:>13.0f}%{s.survival * 100:>11.0f}%"
                  f"{r['tokens']:>12,}{r['extra']['final_pack_tokens']:>12,}"
                  f"{len(packed):>13}")
        print("-" * 104)
        return

    results = {name: fn(turns, probe, anchors, anchor_cells) for name, fn in runners.items()}

    print("=" * 104)
    print("结果对比")
    print("=" * 104)
    print(f"{'策略':<13}{'神经元级召回':>14}{'精确串存活':>12}{'存活数':>10}"
          f"{'上下文tokens':>14}{'峰值':>10}{'中枢延迟':>11}{'本地命中':>10}{'外呼率':>9}")
    print("-" * 104)
    for name, r in results.items():
        s: Score = r["score"]
        ex = r["extra"]
        packed = r.get("packed_cells")
        if packed is None:
            rec = "100% (全量)" if name == "full" else "-"
        else:
            n = sum(1 for a in anchors if anchor_cells.get(a.text) in packed)
            rec = f"{n / len(anchors) * 100:.0f}%"
        lat = f"{r['latency_ms']:.2f} ms" if r["latency_ms"] else "-"
        lh = f"{ex['local_hit_rate'] * 100:.0f}%" if "local_hit_rate" in ex else "-"
        er = f"{ex['escalation_rate'] * 100:.0f}%" if "escalation_rate" in ex else "-"
        print(f"{name:<13}{rec:>14}{s.survival * 100:>11.0f}%{s.alive:>7}/{s.total:<3}"
              f"{r['tokens']:>14,}{r['peak']:>10,}{lat:>11}{lh:>10}{er:>9}")
    print("-" * 104)

    print()
    print("缺失明细")
    print("-" * 104)
    for name, r in results.items():
        s: Score = r["score"]
        packed = r.get("packed_cells")
        lost_cells = []
        if packed is not None:
            lost_cells = [a.text for a in anchors if anchor_cells.get(a.text) not in packed]
        print(f"  [{name}] 精确串缺 {len(s.missing)}/{s.total}"
              + (f"，神经元级缺 {len(lost_cells)}/{len(anchors)}" if packed is not None else ""))
        for m in s.missing[:6]:
            tag = "" if packed is None or anchor_cells.get(m) in packed else "  (所在神经元没被载入)"
            print(f"       ✗ {m}{tag}")
        if len(s.missing) > 6:
            print(f"       … 另有 {len(s.missing) - 6} 条")

    if "cells" in results:
        ex = results["cells"]["extra"]
        print()
        print("=" * 104)
        print("cells 模式内部状态")
        print("=" * 104)
        print(f"  神经元个数      : {ex['cells_final']}")
        print(f"  冻结原文总量    : {ex['frozen_tokens']:,} tokens（保真，永不重写）")
        print(f"  平均请求上下文  : {results['cells']['tokens']:,} tokens   最终请求: {ex['final_pack_tokens']:,}")
        print(f"  压缩比          : {ex['compression_ratio']:.1f}x（冻结量 / 单次请求量）")
        print(f"  最终路由轨迹    : {ex['final_route']}")
        print(f"  本地命中率      : {ex['local_hit_rate'] * 100:.0f}%   "
              f"外呼率: {ex['escalation_rate'] * 100:.0f}%  "
              f"（{ex['route_calls']} 次路由中 {ex['escalated_total']} 次举手说'不知道'）")
        print(f"  最终一次路由    : {ex['final_stage']}")
        print(f"  最终载入神经元  : {', '.join(ex['packed_cells'])}")
        print(f"  事故工作集命中  : {ex['working_set_hit']}/{ex['working_set_total']}"
              f"  （解决第 72 轮事故真正需要的那几条事实）")
        print(f"  Hub 统计        : {ex['hub_stats']}")


if __name__ == "__main__":
    sys.exit(main())
