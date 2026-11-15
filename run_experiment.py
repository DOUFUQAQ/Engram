"""
run_experiment.py —— 组件 A/B 实验台

回答一个问题：**换上真嵌入（和 LLM 重排）到底值多少？**

三条对照腿，在**同一份会话、同一套留出探针**上跑：
    offline        离线哈希嵌入（基线，无 API 调用）
    embed          真嵌入（.env 里的 EMBED_*）
    embed+rerank   真嵌入 + chat 模型重排（.env 里的 DEEPSEEK_*）

指标沿用 evaluate.py 的口径（度量排名，不看"取几个"）：
    L1/L2/L3 的 @1 / @6、MRR、负样本拒绝率、每次查询的嵌入延迟。

用法：
    python run_experiment.py                     # 自动读 .env，能跑几条跑几条
    python run_experiment.py --strict             # 缺在线嵌入直接报错（推荐验证时用）
    python run_experiment.py --legs offline,embed # 只跑指定腿
    python run_experiment.py --check              # 只做连通性体检，不跑完整实验
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Dict, List, Optional, Tuple

from embedder import ChatReranker, Components, EmbeddingProvider, build_components, load_dotenv
from eval_holdout import HOLDOUT
from hub import Hub, HashingEmbedder
from sim_session import PROBE_WORKING_SET, build_session, format_turn
from textutil import est_tokens


# ---------------------------------------------------------------------------

def build_hub(embedder, reranker: Optional[ChatReranker], *, verbose: bool = True) -> Hub:
    """建库。在线嵌入下入库用批量预计算，避免逐轮调 API。"""
    turns, _, _ = build_session()
    hub = Hub(embedder=embedder, reranker=reranker)
    t0 = time.perf_counter()
    for t in turns:
        hub.write(format_turn(t), turn_index=t.index, files=t.files)
    hub.consolidate()
    hub.decay()
    hub._sync_index()
    if hub.emb_online:
        n = hub.precompute_embeddings()
        if verbose:
            print(f"    批量预计算 {n} 个神经元向量，用时 {time.perf_counter() - t0:.1f}s")
    return hub


def holders(hub: Hub, anchor: str) -> List[str]:
    return [cid for cid, c in hub.cells.items() if anchor in c.text]


def eval_ranked(hub: Hub, *, per_probe: int = 3, verbose: bool = False) -> Dict[str, object]:
    """留出集：度量排名。用 multi(+二跳) 管线，这是当前的推荐配置。"""
    per_level: Dict[str, List[Tuple[int, bool]]] = {}
    noans: List[bool] = []
    embed_ms: List[float] = []
    for case in HOLDOUT:
        t0 = time.perf_counter()
        traces, routed = hub.multi_route(case.query, per_probe=per_probe,
                                         two_hop=True, allow_external=False)
        embed_ms.append((time.perf_counter() - t0) * 1000)
        accepted = any(t.stage == "cell" for t in traces)
        if case.level == "NEG":
            noans.append(not accepted)
            if verbose:
                print(f"    [NEG] {'✅' if not accepted else '❌'} {case.query[:30]}")
            continue
        want = set(holders(hub, case.anchor))
        rank = 0
        for i, cid in enumerate(routed, 1):
            if cid in want:
                rank = i
                break
        per_level.setdefault(case.level, []).append((rank, 0 < rank <= 6))
        if verbose:
            tag = "✅" if rank == 1 else ("🟡" if 0 < rank <= 6 else "❌")
            print(f"    [{case.level}] {tag} rank={rank or '-':<3} {case.query[:34]}")

    def agg(rows):
        if not rows:
            return 0.0, 0.0, 0.0
        at1 = sum(1 for r, _ in rows if r == 1) / len(rows)
        atk = sum(1 for _, ok in rows if ok) / len(rows)
        mrr = sum((1 / r) if r else 0.0 for r, _ in rows) / len(rows)
        return at1, atk, mrr

    overall = agg([r for rows in per_level.values() for r in rows])
    return {"stats": {k: agg(v) for k, v in per_level.items()}, "overall": overall,
            "noans": (sum(noans) / len(noans)) if noans else 0.0,
            "noans_n": f"{sum(noans)}/{len(noans)}",
            "query_ms": statistics.mean(embed_ms) if embed_ms else 0.0}


def eval_working_set(hub: Hub, *, hot_turns: int = 6) -> Dict[str, object]:
    """事故工作集：被动打包 + 模型主动 recall 三次，看 6 条关键事实进没进上下文。"""
    from context_packer import Packer
    turns, probe, anchors = build_session()
    writer = Packer(hub, budget_tokens=9000, hot_turns=hot_turns, max_cold=4)
    base = writer.cells(probe.utterance, current_turn=probe.index, allow_external=False,
                        per_probe=3, two_hop=True)
    packed = set(base.cell_ids)
    recall_tokens = 0
    for q in ["session TTL 过期时间是多少？", "webhook 读超时和连接超时分别是多少？",
              "数据库连接串来自哪个环境变量？"]:
        rp = writer.recall(q, top_k=4, current_turn=probe.index)
        packed |= set(rp.cell_ids)
        recall_tokens += rp.tokens
    ws = [t for t in PROBE_WORKING_SET
          if any(t in hub.cells[c].text for c in packed if c in hub.cells)]
    exact = [a.text for a in anchors
             if any(a.text in hub.cells[c].text for c in packed if c in hub.cells)]
    return {"working_set": len(ws), "working_set_total": len(PROBE_WORKING_SET),
            "exact": len(exact), "exact_total": len(anchors),
            "base_tokens": base.tokens, "recall_tokens": recall_tokens,
            "total_tokens": base.tokens + recall_tokens}


# ---------------------------------------------------------------------------

def make_leg(name: str, comp: Components) -> Tuple[object, Optional[ChatReranker], str]:
    if name == "offline":
        return HashingEmbedder(dim=4096), None, "离线哈希嵌入（基线）"
    if name == "embed":
        if not isinstance(comp.embedder, EmbeddingProvider):
            raise RuntimeError("embed 这一腿需要 .env 里的 EMBED_* 配置")
        return comp.embedder, None, f"真嵌入 {comp.embedder.model}"
    if name == "embed+rerank":
        if not isinstance(comp.embedder, EmbeddingProvider):
            raise RuntimeError("embed+rerank 这一腿需要 EMBED_* 配置")
        if not (comp.reranker and comp.reranker.enabled):
            raise RuntimeError("embed+rerank 这一腿需要 DEEPSEEK_API_KEY（用于重排）")
        return comp.embedder, comp.reranker, f"真嵌入 + LLM 重排 {comp.reranker.model}"
    raise ValueError(f"未知的腿: {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", default="offline,embed,embed+rerank")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--check", action="store_true", help="只做连通性体检")
    ap.add_argument("--env", default=".env")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    load_dotenv(args.env)
    comp = build_components(args.env, strict=args.strict)

    print("=" * 100)
    print("组件体检")
    print("=" * 100)
    print(comp.describe())
    if isinstance(comp.embedder, EmbeddingProvider):
        ok, msg = comp.embedder.probe()
        print(f"  嵌入连通性 : {'✅' if ok else '❌'} {msg}")
        if ok:
            # 语义近邻检查：这是判断"嵌入到底有没有用"最快的信号
            a = comp.embedder.embed_query("session TTL 过期")
            b = comp.embedder.embed_query("登录态失效时间")
            c = comp.embedder.embed_query("库存扣减")
            print(f"  语义近邻   : 同义对 {EmbeddingProvider.cos(a, b):.3f}   "
                  f"无关对 {EmbeddingProvider.cos(a, c):.3f}   "
                  f"（同义对必须明显更高，否则这个模型不适合中文语义匹配）")
        if not ok:
            print("\n嵌入不可用，终止。请检查 .env 里的 EMBED_BASE_URL / EMBED_API_KEY / EMBED_MODEL。")
            return
    else:
        print("  （离线模式，跳过连通性检查）")

    if args.check:
        return

    legs = [x.strip() for x in args.legs.split(",") if x.strip()]
    results: Dict[str, Dict[str, object]] = {}
    print()
    print("=" * 100)
    print("A/B 实验：同一份会话 + 同一套留出探针")
    print("=" * 100)
    for leg in legs:
        try:
            embedder, reranker, desc = make_leg(leg, comp)
        except RuntimeError as e:
            print(f"\n[{leg}] 跳过：{e}")
            continue
        print(f"\n[{leg}] {desc}")
        hub = build_hub(embedder, reranker, verbose=True)
        rank = eval_ranked(hub, verbose=args.verbose)
        ws = eval_working_set(hub)
        results[leg] = {"rank": rank, "ws": ws, "desc": desc}
        print(f"    留出集 MRR={rank['overall'][2]:.2f}  @6={rank['overall'][1]*100:.0f}%  "
              f"NEG={rank['noans_n']}  查询 {rank['query_ms']:.0f}ms")
        print(f"    事故工作集 {ws['working_set']}/{ws['working_set_total']}   "
              f"精确串 {ws['exact']}/{ws['exact_total']}   上下文 {ws['total_tokens']:,} tokens")

    if len(results) < 2:
        return

    print()
    print("=" * 100)
    print("汇总")
    print("=" * 100)
    print(f"{'腿':<16}{'L1 @1':>8}{'L1 @6':>8}{'L2 @1':>8}{'L2 @6':>8}{'L3 @1':>8}{'L3 @6':>8}"
          f"{'MRR':>7}{'NEG':>7}{'工作集':>9}{'精确串':>9}{'查询ms':>9}")
    print("-" * 100)
    for leg, r in results.items():
        s = r["rank"]["stats"]
        cells = []
        for lvl in ["L1", "L2", "L3"]:
            at1, atk, _ = s.get(lvl, (0.0, 0.0, 0.0))
            cells += [f"{at1 * 100:>7.0f}%", f"{atk * 100:>7.0f}%"]
        mrr = r["rank"]["overall"][2]
        ws = r["ws"]
        ws_str = f"{ws['working_set']}/{ws['working_set_total']}"
        ex_str = f"{ws['exact']}/{ws['exact_total']}"
        print(f"{leg:<16}" + "".join(cells)
              + f"{mrr:>7.2f}{r['rank']['noans_n']:>7}"
              + f"{ws_str:>9}{ex_str:>9}{r['rank']['query_ms']:>9.0f}")
    print("-" * 100)

    base = results.get("offline")
    if base and "embed" in results:
        b, t = base["rank"]["overall"], results["embed"]["rank"]["overall"]
        print(f"\n换真嵌入的净收益：MRR {b[2]:.2f} → {t[2]:.2f}   "
              f"@6 {b[1]*100:.0f}% → {t[1]*100:.0f}%")
        for lvl in ["L1", "L2", "L3"]:
            bb = base["rank"]["stats"].get(lvl, (0, 0, 0))
            tt = results["embed"]["rank"]["stats"].get(lvl, (0, 0, 0))
            print(f"  {lvl}: @1 {bb[0]*100:.0f}% → {tt[0]*100:.0f}%   MRR {bb[2]:.2f} → {tt[2]:.2f}")
    if "embed+rerank" in results and "embed" in results:
        b, t = results["embed"]["rank"]["overall"], results["embed+rerank"]["rank"]["overall"]
        print(f"再加 LLM 重排的净收益：MRR {b[2]:.2f} → {t[2]:.2f}")
        for lvl in ["L1", "L2", "L3"]:
            bb = results["embed"]["rank"]["stats"].get(lvl, (0, 0, 0))
            tt = results["embed+rerank"]["rank"]["stats"].get(lvl, (0, 0, 0))
            print(f"  {lvl}: @1 {bb[0]*100:.0f}% → {tt[0]*100:.0f}%   MRR {bb[2]:.2f} → {tt[2]:.2f}")

    for leg, r in results.items():
        if isinstance(comp.embedder, EmbeddingProvider) and leg != "offline":
            print(f"\n[{leg}] API 用量：{comp.embedder.stats.describe()}")
            if comp.reranker and leg.endswith("rerank"):
                print(f"          重排：{comp.reranker.stats.describe()}")


if __name__ == "__main__":
    sys.exit(main())
