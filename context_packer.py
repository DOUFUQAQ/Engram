"""
context_packer.py —— 把记忆装进"当前工作区"（上下文窗口）的打包装配线

这里把四种策略放在同一个接口下，好做同条件对比：

    full        全部历史原文进上下文        —— 保真，但很快就装不下（40 轮后物理上不可能）
    compaction  压缩摘要进上下文            —— 装得下，但压缩会永久性丢掉精确细节（飘忽的根因）
    recency     只留最近 N 轮               —— 省钱，早期事实全部失忆
    cells       热区原文 + 冷区按需调取      —— 本方案：保真存储 + 稀疏激活

cells 模式的关键动作：
    ① 最近 `hot_turns` 轮永远以原文在位（保证思维链的连贯性，便宜）；
    ② 其余神经元不预加载，只挂一个 header 指针；命中才把全文拉进来（贵但准）；
    ③ 预算不够时，按「相关度/成本」性价比选神经元，并如实标记 partial。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from hub import Cell, Hub, Trace, est_tokens


@dataclass
class Packed:
    text: str
    tokens: int
    mode: str
    cell_ids: List[str] = field(default_factory=list)
    trace: Optional[Trace] = None
    partial: bool = False


def _lossy_rewrite(text: str, keep: int = 70) -> str:
    """模拟"压缩摘要"的信息损失：保留动作轮廓，丢掉精确标识符。

    真实压缩由另一个 LLM 完成，行为与此同构 —— 它会写下"修改了配置项和依赖"，
    但不会记住那个精确的键名、端口号、行号。
    """
    out = []
    for para in text.split("\n\n"):
        line = next((ln.strip() for ln in para.splitlines() if ln.strip()), "")
        words = line.split()
        if not words:
            continue
        # 高危信息（看起来像标识符/数字/键名）在摘要中被泛化掉
        gen = []
        for w in words:
            if any(ch.isdigit() for ch in w) or "_" in w or "/" in w or "." in w:
                gen.append("<细节>")
            else:
                gen.append(w)
        out.append(" ".join(gen)[:keep])
    return "\n".join(out)


class Packer:
    def __init__(self, hub: Hub, *, budget_tokens: int = 9000, hot_turns: int = 8,
                 max_cold: int = 6):
        self.hub = hub
        self.budget = budget_tokens
        self.hot_turns = hot_turns
        self.max_cold = max_cold        # 冷区最多调取几个神经元（控制稀释与成本）

    # ------------------------------------------------------------------ full
    def full(self, turns: Sequence[str]) -> Packed:
        text = "\n\n".join(turns)
        return Packed(text=text, tokens=est_tokens(text), mode="full",
                      cell_ids=[f"T{i+1}" for i in range(len(turns))])

    # ------------------------------------------------------------ compaction
    def compaction(self, turns: Sequence[str]) -> Packed:
        text = _lossy_rewrite("\n\n".join(turns))
        return Packed(text=text, tokens=est_tokens(text), mode="compaction")

    # --------------------------------------------------------------- recency
    def recency(self, turns: Sequence[str]) -> Packed:
        recent = turns[-self.hot_turns :]
        text = "\n\n".join(recent)
        return Packed(text=text, tokens=est_tokens(text), mode="recency",
                      partial=len(recent) < len(turns))

    # ----------------------------------------------------------------- cells
    def cells(self, query: str, current_turn: int, *, allow_external: bool = False,
              multi: bool = True, per_probe: int = 3, expand: bool = False,
              two_hop: bool = False) -> Packed:
        """打包当前工作区。

        multi=True   多路脑电波（复合任务拆线索，各自召回后取并集）。
        two_hop=True 二跳联想检索：种子神经元扩散罕见词后再召回一次，
                    用来跨越"查询里没有任何一个词能连到答案"的因果链断裂。
        expand=True  跨线索查询扩展（联想）。实测无收益，默认关闭。
        """
        hub = self.hub
        if multi:
            traces, routed = hub.multi_route(query, allow_external=allow_external,
                                             per_probe=per_probe, expand=expand,
                                             two_hop=two_hop)
            trace = traces[0] if traces else hub.route(query, allow_external=allow_external)
            hits = [(cid, hub._cell_rerank_score(query, hub.cells[cid]))
                    for cid in routed if cid in hub.cells]
            trace.hits = hits
            trace.reason += f" | 多路探针 {len(traces)} 个"
        else:
            trace = hub.route(query, allow_external=allow_external)
            hits = list(trace.hits)

        hot_ids = [c.cell_id for c in sorted(hub.cells.values(), key=lambda c: c.turn_end)
                   if current_turn - c.turn_end <= self.hot_turns]
        picked: List[str] = []
        used = 0

        # ① 热区原文（连贯性优先）。注意热区不能占满全部预算：
        #    预留至少 40% 给"按需调取"，否则检索得再准也塞不进来。
        #    这是设计上的硬权衡：连贯性 vs 可召回性。
        hot_budget = self.budget * 0.55
        blocks: List[str] = []
        for cid in hot_ids:
            c = hub.cells.get(cid)
            if not c:
                continue
            if used + c.tokens > hot_budget:
                break
            blocks.append(f"{c.header()}\n{c.text}")
            used += c.tokens
            picked.append(cid)

        # ② 冷区按需调取。
        #    排序用**相关度密度**（相关度 / 长度^0.5）而不是绝对相关度：
        #    一个 1.3k token 的神经元压掉一个 0.5k 的高密度神经元是亏的。
        #    同时"塞不下就继续试更小的"，别把剩余预算浪费掉
        #    （实测：就因为差 100 token 跳过了一个含 2 条关键事实的神经元）。
        cands: List[Tuple[float, int, str]] = []
        for cid, score in hits:
            if cid in picked:
                continue
            c = hub.cells.get(cid)
            if not c:
                continue
            cands.append((score / (c.tokens ** 0.5), c.tokens, score, cid))
        cands.sort(key=lambda t: -t[0])
        cold_taken = 0
        for _, cost, raw, cid in cands:
            if cold_taken >= self.max_cold:
                break
            if used + cost > self.budget:
                continue                     # 塞不下这个 → 继续试更小的
            c = hub.cells[cid]
            blocks.append(f"{c.header()}  (按需调取, 相关度 {raw:.2f})\n{c.text}")
            used += cost
            picked.append(cid)
            cold_taken += 1

        # ③ 目录（指针内存）：只留 header，让模型知道"还有什么在库里"
        catalog = [c.header() for c in sorted(hub.cells.values(), key=lambda c: c.turn_end)
                   if c.cell_id not in picked and not c.archived]
        if catalog:
            cat_text = "【记忆目录：以下神经元未加载，需要时可再调取】\n" + "\n".join(catalog[-24:])
            blocks.append(cat_text)
            used += est_tokens(cat_text)

        text = "\n\n".join(blocks)
        return Packed(text=text, tokens=est_tokens(text), mode="cells",
                      cell_ids=picked, trace=trace,
                      partial=len(cands) > 0 and used >= self.budget)

    # ------------------------------------------------------------- 记忆工具
    def recall(self, query: str, *, top_k: int = 4, current_turn: int = 10**9,
               budget: Optional[int] = None) -> Packed:
        """**记忆工具**：让模型主动问记忆，而不是指望一次融合查询撞对。

        为什么必须有这个 API（本原型的决定性实测）：
            把"登录 24 分钟掉线 + 支付 504 + DSN 连接中断"塞进**一次**融合查询，
            只能召回 6 条关键事实里的 3 条；而让模型分三次聚焦提问
            （"session TTL 是不是 25 分钟？" / "webhook 读超时多少？" / "DSN 来自哪个变量？"），
            6 条全部召回，且每次只要 3~5 个神经元。

        机制上这是 **sufficiency router** 的简化形态：一问不够就再问一次
        （Router-Mem 用单 token 决策做早停，推理时间降约 27%）。
        关键是：**分解能力来自模型，不来自检索器** —— 词面检索无法把
        "掉线" 变成 "session TTL"，但模型知道自己该问什么。
        """
        hub = self.hub
        q = (query or "").strip()
        if not q:
            return Packed(text="", tokens=0, mode="recall")
        tr = hub.route(q, top_k=top_k, allow_external=True, spread=False)
        budget = budget or max(2500, self.budget // 2)
        blocks: List[str] = []
        used = 0
        taken: List[str] = []
        for cid, _ in tr.hits:
            c = hub.cells.get(cid)
            if not c:
                continue
            if used + c.tokens > budget:
                continue                      # 塞不下就试下一个更小的，别浪费预算
            blocks.append(f"{c.header()}  (recall)\n{c.text}")
            used += c.tokens
            taken.append(cid)
        # 契约：**把预算内相关的都带回来**，不按 top_k 硬截断。
        # 实测（bench_memory.py --recall-loop）硬截断到 top_k 会丢掉工作集里的「20 分钟」
        # 且一个 token 都没省 —— 记忆工具的正确语义是"装满相关证据"，唯一约束是预算。
        text = "\n\n".join(blocks)
        return Packed(text=text, tokens=est_tokens(text), mode="recall",
                      cell_ids=taken, trace=tr)
