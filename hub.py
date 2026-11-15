"""
hub.py —— 「神经元 + 神经中枢」记忆内核（零依赖，纯标准库）

设计对应关系（与你提的想法一一对应）:

    * 神经元 (Cell)      = 一段被写满、自包含、保真存储的上下文片段（不压缩，只冻结）
    * 神经中枢 (Hub)     = 极便宜的定序器：把当前「脑电波」(query) 匹配到相关神经元
    * 树突/索引 (Index)  = 每个神经元的检索指纹（dense 向量 + 稀疏词项 + 元数据）
    * 阈值门 (Gate)      = 置信度不足时向外求援（联网 / 新推理 / 请问用户）
    * 突触 (Synapse)     = 神经元之间的有向相关边，支持「顺着一条线把整串记忆拉回来」
    * 巩固 (Consolidate) = 反复共激活的神经元合并（睡眠期的记忆固化）
    * 遗忘 (Decay)       = 未激活的神经元权重按艾宾浩斯曲线衰减，长期不用则降级/删除

保存与加载分离是本设计的核心主张:
    ① 保存 (ingest) 必须是保真的 —— 一旦重写就永久性丢信息，这正是 40 轮后飘忽的根因；
    ② 压缩只允许发生在「当前工作区」这一层，且必须可逆地指回神经元（指针内存）。
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from textutil import STOP as _STOP, HashingEmbedder, char_ngrams, est_tokens, stable_hash, tokenize

# 兼容旧名字（早期代码里叫 _h / _STOP）
_h = stable_hash

# --------------------------------------------------------------------------------------
# 1. 文本 → 指纹
# --------------------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+(?:\.[0-9]+)?|[\u4e00-\u9fff]")


def tokenize(text: str) -> List[str]:
    """CJK 单字 + 拉丁词/数字。BM25 的 IDF 会自动把常见汉字压下去，罕见标识符顶上来。"""
    return _WORD_RE.findall(text.lower())


# 文本工具（tokenize / char_ngrams / est_tokens / HashingEmbedder）已抽到 textutil.py，
# 这样 embedder.py 可以独立于 hub.py 被导入，避免循环依赖。

# --------------------------------------------------------------------------------------
# 2. 神经元
# --------------------------------------------------------------------------------------

@dataclass
class Cell:
    """一段饱和上下文。text 是保真原文，永不重写；只有 summary 允许被重写。"""

    cell_id: str
    turn_start: int
    turn_end: int
    text: str                      # 冻结全文（保真）
    summary: str                   # 一到两行索引摘要（允许丢细节，只用来给中枢看）
    topics: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    vec: List[float] = field(default_factory=list)
    tf: Dict[str, int] = field(default_factory=dict)
    tokens: int = 0
    created: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    weight: float = 1.0            # 突触强度 / 记忆巩固度，随"使用"上升、随"闲置"衰减
    pinned: bool = False           # 钉住 = 永不遗忘（用户显式要求记住的东西）
    source: str = "session"        # session | external | merged
    archived: bool = False
    merged_from: List[str] = field(default_factory=list)

    def header(self) -> str:
        span = f"T{self.turn_start}" if self.turn_start == self.turn_end else f"T{self.turn_start}-{self.turn_end}"
        bits = [f"[{self.cell_id} {span}"] 
        if self.files:
            bits.append(f" files:{','.join(self.files[:4])}")
        bits.append(f" w={self.weight:.2f} hits={self.access_count}]")
        return "".join(bits)


# --------------------------------------------------------------------------------------
# 3. 神经中枢
# --------------------------------------------------------------------------------------

@dataclass
class Trace:
    """一次「脑电波」的完整放电路径，可直接打印成可解释的调用链。"""

    query: str
    stage: str = "cell"             # cell | escalate
    hits: List[Tuple[str, float]] = field(default_factory=list)
    confidence: float = 0.0
    margin: float = 0.0
    reason: str = ""
    external_used: bool = False
    external_calls: int = 0
    latency_ms: float = 0.0
    partial: bool = False

    def describe(self) -> str:
        ids = ", ".join(f"{cid}:{s:.3f}" for cid, s in self.hits) or "-"
        ext = " -> 外呼" if self.external_used else ""
        return (f"[{self.stage}] conf={self.confidence:.3f} margin={self.margin:.3f} "
                f"hits=({ids}){ext} latency={self.latency_ms:.1f}ms  <{self.reason}>")


class Hub:
    """神经中枢：掌管全部神经元，只做三件事 —— 定序、路由、元管理。"""

    def __init__(
        self,
        embedder: Optional[HashingEmbedder] = None,
        *,
        top_k: int = 6,
        conf_threshold: float = 0.35,
        margin_threshold: float = 0.03,
        sure_threshold: float = 0.75,
        tau_floor: int = 220,
        tau_ceiling: int = 1400,
        consolidate_sim: float = 0.72,
        archive_weight: float = 0.18,
        drop_weight: float = 0.06,
        external_fn: Optional[Callable[[str], Optional[str]]] = None,
        reranker: Optional[Any] = None,
    ):
        self.emb = embedder or HashingEmbedder(dim=4096)
        # 在线嵌入（有 embed_batch 的提供者）与离线哈希嵌入的行为差异集中在这一个标志上
        self.emb_online = hasattr(self.emb, "embed_batch") and not isinstance(
            self.emb, HashingEmbedder)
        self.cells: Dict[str, Cell] = {}
        self.synapses: Dict[str, Dict[str, float]] = {}   # 有向相关边
        self.top_k = top_k
        self.conf_threshold = conf_threshold
        self.margin_threshold = margin_threshold
        self.sure_threshold = sure_threshold   # 置信度高于此值时不再因 margin 小而降级
        self.tau_floor = tau_floor          # 神经元的"饱和容量"下限（token）
        self.tau_ceiling = tau_ceiling      # 上限：超过就必须分裂
        self.consolidate_sim = consolidate_sim
        self.archive_weight = archive_weight
        self.drop_weight = drop_weight
        self.external_fn = external_fn      # 用户后续注入真实联网检索
        self.reranker: Optional[Any] = reranker   # ChatReranker（在线 LLM 重排，可选）
        self._init_counters()

    @property
    def dense_weight(self) -> float:
        """稠密路的权重。

        离线哈希嵌入的稠密路实测无判别力（参数扫描一致给出 0），所以默认 0，
        免得它污染正负分层；接入真嵌入后应调到 0.4~0.5 —— 这是"换组件"最直接的体现。
        """
        return 0.5 if self.emb_online else 0.0

    def _llm_rerank(self, query: str, cands: Sequence[str]) -> Dict[str, float]:
        """把候选交给在线 LLM 打分（治因果链）。失败静默回退，绝不阻塞路由。"""
        if not self.reranker or not getattr(self.reranker, "enabled", False):
            return {}
        pool = [(cid, self.cells[cid].text) for cid in cands if cid in self.cells]
        if len(pool) < 2:
            return {}
        try:
            return self.reranker.score(query, pool)
        except Exception:
            return {}

    def _init_counters(self) -> None:
        """计数器与编号序。单独一个方法，避免 __init__ 被后续插入的方法体截断
        （本原型真踩过：_llm_rerank 插在 __init__ 中间，把 stats 变成了死代码）。"""
        self.stats = {
            "ingested": 0, "merged": 0, "archived": 0, "dropped": 0,
            "route_calls": 0, "local_hit": 0, "escalated": 0, "external_calls": 0,
            "tokens_saved_vs_full": 0,
        }
        self._seq = 0

    # ---------------------------------------------------------------- 写入（保真）
    def _new_id(self) -> str:
        self._seq += 1
        return f"N{self._seq:04d}"

    # ------------------------------------------------------------ 嵌入抽象
    def _doc_vec(self, text: str) -> List[float]:
        """文档侧嵌入。离线哈希嵌入是"边写边学"的，在线嵌入不是。"""
        if self.emb_online:
            return self.emb.embed_batch([text])[0]
        return self.emb.embed(text, learn=True)

    def _query_vec(self, text: str) -> List[float]:
        """查询侧嵌入。有些模型（E5 系、bge instruct 变体）要求查询/文档用不同前缀。"""
        fn = getattr(self.emb, "embed_query", None)
        if callable(fn):
            return fn(text)
        return self.emb.embed(text)

    def precompute_embeddings(self, *, only_missing: bool = True) -> int:
        """入库结束后一次性重算全部文档向量（在线嵌入的正确用法：批量、便宜、无状态漂移）。

        离线哈希嵌入不需要这一步（它的事后 IDF 会变，但向量已按旧 IDF 算好——
        这是它的已知精度损失，也是为什么它只能当兜底）。
        """
        if not self.emb_online:
            return 0
        ids = [cid for cid, c in self.cells.items() if c.text and (not only_missing or not c.vec)]
        if not ids:
            return 0
        vecs = self.emb.embed_batch([self.cells[cid].text for cid in ids],
                                    prefix=getattr(self.emb, "doc_prefix", ""))
        for cid, v in zip(ids, vecs):
            self.cells[cid].vec = v
        return len(ids)

    def make_cell(
        self, turns: Sequence[str], *, turn_start: int, turn_end: int,
        summary: str = "", topics: Optional[Sequence[str]] = None,
        files: Optional[Sequence[str]] = None, source: str = "session",
        pinned: bool = False,
    ) -> Cell:
        text = "\n\n".join(turns)
        vec = self._doc_vec(text)
        tf: Dict[str, int] = {}
        for t in tokenize(text):
            tf[t] = tf.get(t, 0) + 1
        cid = self._new_id()
        cell = Cell(
            cell_id=cid, turn_start=turn_start, turn_end=turn_end, text=text,
            summary=summary or self._free_summary(turns),
            topics=list(topics or []), files=list(files or []),
            vec=vec, tf=tf, tokens=est_tokens(text), source=source, pinned=pinned,
        )
        self.cells[cid] = cell
        self.synapses.setdefault(cid, {})
        self.stats["ingested"] += 1
        return cell

    @staticmethod
    def _free_summary(turns: Sequence[str]) -> str:
        """零成本摘要：取每轮首个动作行。真正的系统这里调一次小模型，且允许它丢细节。"""
        out = []
        for t in turns:
            first = next((ln.strip() for ln in t.splitlines() if ln.strip()), "")
            out.append(first[:90])
        return " | ".join(out)[:400]

    def write(
        self, turn_text: str, *, turn_index: int, files: Optional[Sequence[str]] = None,
        summary: str = "", pinned: bool = False,
    ) -> str:
        """把一个 turn 追加进当前"未满"的神经元；满了就分裂成新神经元。

        返回被写入或被切分出的 cell_id。分裂点选择「相邻句窗口语义差最大处」，
        与大脑把一段经历切成事件（event segmentation）的机制同构。
        """
        cur = self._active_cell()
        if cur is not None:
            merged_text = cur.text + "\n\n" + turn_text
            if est_tokens(merged_text) <= self.tau_ceiling and not cur.pinned:
                cur.text = merged_text
                cur.turn_end = turn_index
                # 在线嵌入下这里**故意不重算**：每次追加都调一次 API 太贵，
                # 统一由 precompute_embeddings() 在会话结束/入库后批量刷新。
                # 期间该神经元的向量略旧，但稀疏路（BM25）仍然准确，且它通常还在热区里。
                if not self.emb_online:
                    cur.vec = self.emb.embed(merged_text, learn=True)
                cur.tf = {}
                for t in tokenize(merged_text):
                    cur.tf[t] = cur.tf.get(t, 0) + 1
                cur.tokens = est_tokens(merged_text)
                if files:
                    cur.files = sorted(set(cur.files) | set(files))
                return cur.cell_id
            # 满了 → 冻结当前神经元，开一个新的
        cell = self.make_cell(
            [turn_text], turn_start=turn_index, turn_end=turn_index,
            summary=summary, files=files, pinned=pinned,
        )
        self._link_recent(cell)
        return cell.cell_id

    def _active_cell(self) -> Optional[Cell]:
        if not self.cells:
            return None
        c = max(self.cells.values(), key=lambda c: (c.turn_end, c.created))
        if c.pinned:
            return None
        return c

    def _link_recent(self, cell: Cell, window: int = 3) -> None:
        """时间相邻的神经元默认连边（时序突触）。"""
        others = sorted(self.cells.values(), key=lambda c: c.turn_end)
        idx = others.index(cell)
        for nb in others[max(0, idx - window) : idx] + others[idx + 1 : idx + 1 + window]:
            self.synapses.setdefault(cell.cell_id, {})[nb.cell_id] = 0.6
            self.synapses.setdefault(nb.cell_id, {})[cell.cell_id] = 0.6

    def split_if_oversized(self) -> List[Cell]:
        """兜底：手工灌入的超长文本（超过 tau_ceiling）按段落边界切分成合法神经元。

        注意阈值必须是 tau_ceiling —— 用 tau_floor 会把每个正常神经元都误切，
        这条 bug 会直接把检索质量打回原形（细胞过小 → 上下文碎片化）。
        """
        created = []
        for c in list(self.cells.values()):
            if c.tokens <= self.tau_ceiling:
                continue
            paras = [p for p in c.text.split("\n\n") if p.strip()]
            if len(paras) < 2:
                continue
            self.cells.pop(c.cell_id, None)
            self.synapses.pop(c.cell_id, None)
            chunk: List[str] = []
            start = c.turn_start
            for p in paras:
                chunk.append(p)
                if est_tokens("\n\n".join(chunk)) >= self.tau_ceiling * 0.9:
                    created.append(self.make_cell(["\n\n".join(chunk)], turn_start=start,
                                                  turn_end=c.turn_end, source=c.source,
                                                  pinned=c.pinned))
                    chunk = []
            if chunk:
                created.append(self.make_cell(["\n\n".join(chunk)], turn_start=start,
                                              turn_end=c.turn_end, source=c.source,
                                              pinned=c.pinned))
        return created

    # ---------------------------------------------------------------- 读取（定序）
    def _sync_index(self) -> None:
        """把神经元集合引用同步给嵌入器，供 BM25 统计真实 df（零拷贝）。"""
        self.emb._cells_ref = self.cells

    def _bm25(self, query_tokens: Sequence[str], k1: float = 1.2, b: float = 0.6) -> Dict[str, float]:
        """真正的 BM25（IDF 用真实 df 统计）。负责抓精确标识符：端口号、环境变量名、文件名。"""
        n = max(1, len(self.cells))
        avgdl = sum(c.tokens for c in self.cells.values()) / n or 1.0
        scores: Dict[str, float] = {}
        for cid, c in self.cells.items():
            dl = c.tokens or 1
            s = 0.0
            for t in query_tokens:
                f = c.tf.get(t)
                if not f:
                    continue
                df = self.emb.df_count(t)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
            if s > 0:
                scores[cid] = s
        return scores

    def explain(self, query: str, top: int = 6) -> List[Tuple[str, float, float, float, int, int]]:
        """可解释打分：返回 (cell_id, dense, sparse, conf, turn_start, turn_end)，按 conf 排序。

        这是"神经中枢自我报告"的接口——调试路由必须能看到每一路的分数，
        否则阈值门就是玄学。
        """
        self._sync_index()
        rows = []
        for cid, c, d, s, cov in self._score_all(query):
            conf = self._combine_any(d, s, cov)
            rows.append((cid, d, s, conf, c.turn_start, c.turn_end))
        rows.sort(key=lambda r: -r[3])
        return rows[:top]

    def _q_terms(self, query: str) -> List[str]:
        """查询词项：只踢掉高频功能字，保留全部实义汉字（单个汉字也有信息量）。"""
        return [t for t in set(tokenize(query)) if t not in _STOP]

    def _score_all(self, query: str) -> List[Tuple[str, Cell, float, float, float]]:
        """三路打分：(cell, dense 相似, BM25 稀疏, IDF 质量覆盖率)。

        覆盖率另加一路"头部词覆盖"：只统计查询里 IDF 最高的少数几个词是否命中。
        这几个词才是"这个问题真正在问什么"——负样本（语料里根本没有的概念）
        通常在头部词上直接落空，于是被干净地拒绝。
        """
        qv = self._query_vec(query)
        terms = self._q_terms(query)
        sparse = self._bm25(tokenize(query))
        n = max(1, len(self.cells))
        idf_of: Dict[str, float] = {}
        for t in terms:
            df = self.emb.df_count(t)
            idf_of[t] = max(0.0, math.log(1 + (n - df + 0.5) / (df + 0.5)))
        total_idf = sum(idf_of.values()) or 1.0
        head = sorted(idf_of.items(), key=lambda kv: -kv[1])[:3]
        head_total = sum(v for _, v in head) or 1.0
        out = []
        for cid, c in self.cells.items():
            d = self.emb.cos(qv, c.vec)
            s = sparse.get(cid, 0.0)
            cov = sum(v for t, v in idf_of.items() if t in c.tf) / total_idf
            head_cov = sum(v for t, v in head if t in c.tf) / head_total
            # 头部词落空时强行压分：这是"记忆里没有这个概念"的强信号
            cov = min(cov, 0.35 + 0.65 * head_cov) * (0.4 + 0.6 * head_cov)
            out.append((cid, c, d, s, cov))
        return out

    @staticmethod
    def _combine(d: float, s: float, cov: float, *, w_cov: float = 0.65, w_sparse: float = 0.35,
                 w_dense: float = 0.0) -> float:
        """置信度 = IDF 质量覆盖 + log 压缩后的稀疏分（+ 稠密分）。

        经验结论（见 calibrate2.py 的参数扫描）：在**离线字符级**实现下，
        稠密哈希路对"意思像"几乎没有分辨力，甚至会拉低正负分离度，
        真正有判别力的是「罕见词命中」。
        接入**真嵌入**后，稠密路的权重由 Hub.dense_weight 打开（0.5），
        因为这时它才真的有语义信息。
        """
        return (w_cov * cov
                + w_sparse * (math.log1p(max(0.0, s)) / math.log1p(30.0))
                + w_dense * d)

    def _combine_any(self, d: float, s: float, cov: float) -> float:
        """按当前嵌入类型选择权重（离线 0 / 在线 0.5）。"""
        w = self.dense_weight
        w_cov = 0.65 * (1 - w)
        w_sparse = 0.35 * (1 - w)
        return self._combine(d, s, cov, w_cov=w_cov, w_sparse=w_sparse, w_dense=w)

    def multi_route(self, task: str, *, max_probes: int = 5, per_probe: int = 3,
                    expand: bool = False, two_hop: bool = False,
                    allow_external: bool = True,
                    spread: bool = False) -> Tuple[List[Trace], List[str]]:
        """多路脑电波：把复合任务拆成若干单实体探针，各自召回后再全局重排。

        三个机制：
          ① 每条线索独立放电（大脑不会只放一次电）；
          ② 可选**查询扩展**：借用同批次其他线索的高 IDF 词（联想）。
             ⚠️ 实测（bench_memory.py --expand）在"新鲜查询"上**没有可测收益**：
             借来的词要么已经在原句里，要么库中不存在（df=0 被过滤）。
             保留开关是为了在"线索很短、且都是库内既有概念"时再试；
          ③ 全局重排，并给每条线索的最优神经元**保底席位**，
             防止一条强线索吃光预算。

        返回 (traces, 按相关度排序的 cell_id 列表)。
        """
        probes = self.decompose(task)[:max_probes]
        borrowed_pool = self._rare_terms(" ".join(probes), limit=12) if expand else []
        traces: List[Trace] = []
        per_probe_best: List[str] = []
        merged: Dict[str, float] = {}

        for p in probes:
            mine = set(self._rare_terms(p, limit=6))
            borrowed = [t for t in borrowed_pool if t not in mine][:6]
            q = p + (" " + " ".join(mine | set(borrowed)) if borrowed else "")
            if two_hop:
                tr, spread_ids = self.route_with_spread(
                    q, top_k=per_probe, allow_external=allow_external)
                for cid in spread_ids:
                    s = self._cell_rerank_score(q, self.cells[cid])
                    merged[cid] = max(merged.get(cid, 0.0), s)
            else:
                tr = self.route(q, top_k=per_probe, allow_external=allow_external,
                                spread=spread)
                for cid, score in tr.hits:
                    # 取最大值而不是累加：累加会让"被多条探针顺带命中"的噪声细胞顶到前面，
                    # 挤掉真正相关的那一个（这个 bug 实测让主基准从 30% 掉到 22%）。
                    merged[cid] = max(merged.get(cid, 0.0), score)
            tr.reason = f"探针「{p[:20]}…」" + ("+扩展 " if borrowed else " ") + tr.reason
            traces.append(tr)
            if tr.hits:
                per_probe_best.append(tr.hits[0][0])
            elif tr.partial:            # 被闸门拦下但确有候选：仍给它一个保底席位
                best = max(
                    ((cid, self._cell_rerank_score(q, self.cells[cid]))
                     for cid in self.cells),
                    key=lambda kv: kv[1],
                )
                per_probe_best.append(best[0])
                merged[best[0]] = merged.get(best[0], 0.0) + best[1]

        ordered = self.rerank(task, list(merged))
        head: List[str] = []
        for cid in per_probe_best:
            if cid not in head:
                head.append(cid)
        return traces, head + [cid for cid in ordered if cid not in head]

    def route_with_spread(self, query: str, *, top_k: int = 6, seed_k: int = 3,
                          spread_terms: int = 8, second_k: int = 4,
                          allow_external: bool = True) -> Tuple[Trace, List[str]]:
        """二跳联想检索（模式补全 / pattern completion）。

        为什么需要它 —— 这是"因果链"问题的算法解法：
            查询「登录后停留 24 分钟就掉线」里，**没有任何一个词**出现在正确的神经元
            `SESSION_TTL_SECONDS = 25 * 60` 里，所以词汇检索无论怎么调都召不回它。
            但第一跳会命中「掉线 / 登录态」相关的神经元，而它们的邻域里含有
            `session`、`TTL`、`过期` 这些**词面上能连到正确神经元**的概念。

        机制：
            ① 第一跳：正常路由取 seed_k 个种子神经元；
            ② 把这些种子里的**罕见词（高 IDF）**当作扩散信号 —— 等价于大脑的
               联想激活：想到"登录态失效"就会顺带激活"session 配置"那一簇；
            ③ 第二跳：用「原查询 + 扩散词」重新打分，但要求候选**与多个扩散词共现**
               （coordination 约束），否则扩散会把整张网都拉亮，等于没筛选。

        返回 (主 trace, 合并排序后的 cell_id 列表)。第一跳的置信门仍然生效：
        种子不可信时不扩散，直接交给外呼。
        """
        t0 = time.perf_counter()
        first = self.route(query, top_k=seed_k, allow_external=allow_external, spread=False)
        if first.stage != "cell":
            return first, [cid for cid, _ in first.hits]

        seeds = [cid for cid, _ in first.hits]
        spread: List[str] = []
        for cid in seeds:
            c = self.cells.get(cid)
            if c:
                spread.extend(self._rare_terms(c.text, limit=spread_terms))
        # 扩散词的 IDF 权重：越罕见越有指示性
        n = max(1, len(self.cells))
        seen: Dict[str, float] = {}
        for t in spread:
            df = self.emb.df_count(t)
            seen[t] = max(0.0, math.log(1 + (n - df + 0.5) / (df + 0.5)))
        top_spread = [t for t, _ in sorted(seen.items(), key=lambda kv: -kv[1])[:spread_terms * 2]]

        hop1 = {cid: self._cell_rerank_score(query, self.cells[cid]) for cid in seeds}
        hop2: Dict[str, float] = {}
        for cid, c in self.cells.items():
            if cid in hop1:
                continue
            hits = [t for t in top_spread if t in c.tf]
            if len(hits) < 2:                      # coordination 约束：至少共现两个扩散词
                continue
            mass = sum(seen[t] for t in hits)
            hop2[cid] = mass * self._cell_rerank_score(query, c)
        ranked2 = [cid for cid, _ in sorted(hop2.items(), key=lambda kv: -kv[1])[:second_k]]

        # 合并：第一跳种子优先，其后是二跳新发现的神经元，再兜底按相关度
        merged: List[str] = list(seeds)
        for cid in ranked2:
            if cid not in merged:
                merged.append(cid)
        for cid in self.rerank(query, list(self.cells)):
            if cid not in merged:
                merged.append(cid)

        first.reason += (f" | 二跳扩散: 种子{len(seeds)} 扩散词{top_spread[:4]} "
                         f"新召回{len(ranked2)}")
        first.latency_ms = (time.perf_counter() - t0) * 1000
        return first, merged            # 不截断：由调用方决定用前几个（便于度量 recall@k）

    def _rare_terms(self, text: str, *, limit: int) -> List[str]:
        """取文本里 IDF 最高（最罕见、最能定位）的词项。"""
        n = max(1, len(self.cells))
        seen: Dict[str, float] = {}
        for t in set(tokenize(text)):
            if t in _STOP:
                continue
            df = self.emb.df_count(t)
            if df == 0:
                continue                      # 库里根本没有的词，借来也没用
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            if idf <= 0:
                continue
            seen[t] = idf
        return [t for t, _ in sorted(seen.items(), key=lambda kv: -kv[1])[:limit]]

    @staticmethod
    def decompose(task: str) -> List[str]:
        """把任务切成"线索"。零成本实现：先剔除代码块与工具输出，再按句读切分。

        真实系统里这一步由小模型或 Jev 这类单次前向的决策模型完成（见 DESIGN.md §6）。
        关键纪律：**只拿人类原话做路由**。把工具输出、生成的代码一起丢进来，
        路由会被 TURN_72_CONST 这种噪声带跑（这个坑本原型真踩过一次）。
        """
        import re
        cleaned: List[str] = []
        skip = False
        for line in task.splitlines():
            s = line.strip()
            if s.startswith("```"):
                skip = not skip
                continue
            if skip or s.startswith("[tool]") or s.startswith("·") or s.startswith("### T"):
                continue
            cleaned.append(line)
        text = "\n".join(cleaned)
        parts = [p.strip() for p in re.split(r"[。；;！!？?\n]+|，(?=[^，]{6,})", text) if p.strip()]
        if not parts:
            return [text or task]
        scored = []
        for p in parts:
            score = 0
            score += 4 * len(re.findall(r"[A-Z][A-Z_]{2,}", p))      # ORDERS_DB_DSN
            score += 3 * len(re.findall(r"[\w/-]*\.(?:py|sql|yml|json|md)", p))
            score += 2 * len(re.findall(r"\d+", p))                  # 24 分钟 / 504 / 3500ms
            score += 1 if re.search(r"[\u4e00-\u9fff]", p) else 0
            scored.append((score, len(p), p))
        scored.sort(key=lambda t: (-t[0], -t[1]))
        picked = [p for _, _, p in scored[:4]]
        if text not in picked and text.strip():
            picked.append(text.strip())                              # 整体任务也作为一个探针
        return picked

    def route(
        self, query: str, *, top_k: Optional[int] = None, allow_external: bool = True,
        spread: bool = True,
    ) -> Trace:
        """一次「脑电波」：匹配神经元 → 置信判定 → 不足则外呼。

        spread=True 会沿着突触把命中神经元的相关邻居一并拉回（模式补全，
        对应海马体的 pattern completion）。
        """
        t0 = time.perf_counter()
        k = top_k or self.top_k
        tr = Trace(query=query)
        self.stats["route_calls"] += 1
        self._sync_index()

        if not self.cells:
            tr.stage, tr.reason = "escalate", "空记忆"
            self._escalate(tr, query, allow_external)
            tr.latency_ms = (time.perf_counter() - t0) * 1000
            return tr

        scored = self._score_all(query)
        dense = {cid: d for cid, _, d, _, _ in scored}
        confs = sorted(((cid, self._combine_any(d, s, cov)) for cid, _, d, s, cov in scored),
                       key=lambda kv: -kv[1])

        # 双路候选合并：稀有词命中（稀疏）负责"精确"，稠密路负责"意思像"。
        # 即便稠密路权重为 0，也仍让它参与候选生成，避免同义改写完全丢失。
        fused: Dict[str, float] = {}
        for rank, (cid, _) in enumerate(confs, 1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank)
        for rank, (cid, _) in enumerate(
            sorted(((cid, d) for cid, d in dense.items()), key=lambda kv: -kv[1]), 1
        ):
            fused[cid] = fused.get(cid, 0.0) + 0.8 / (60 + rank)

        if not confs:
            tr.stage, tr.reason = "escalate", "无候选"
            self._escalate(tr, query, allow_external)
            tr.latency_ms = (time.perf_counter() - t0) * 1000
            return tr

        # 置信度 = 三路融合分；间隔 = 与第二名的差。这是防幻觉的第一道闸。
        top_cid, conf = confs[0]
        margin = conf - confs[1][1] if len(confs) > 1 else conf
        tr.confidence, tr.margin = conf, margin

        if conf < self.conf_threshold:
            tr.stage, tr.reason = "escalate", f"置信不足({conf:.3f}<{self.conf_threshold})"
            tr.partial = bool(confs)
            self._escalate(tr, query, allow_external)
            tr.latency_ms = (time.perf_counter() - t0) * 1000
            return tr
        # 只有"又不够自信、又打平"的时候才因歧义外呼。
        # 长复合问题天然会在多个神经元之间打平，不该因此放弃召回 —— 直接取 top_k 全带上。
        if (conf < self.sure_threshold and len(confs) > 1
                and margin < self.margin_threshold and conf < 0.60):
            tr.stage, tr.reason = "escalate", f"歧义(margin={margin:.3f}<{self.margin_threshold})"
            tr.partial = True
            self._escalate(tr, query, allow_external)
            tr.latency_ms = (time.perf_counter() - t0) * 1000
            return tr

        # 命中后按 RRF 融合名次取 top_k（融合名次比单一相似度更稳）
        by_fusion = sorted(fused.items(), key=lambda kv: -kv[1])
        picked = [cid for cid, _ in by_fusion[:k]]
        if spread:
            extra: List[Tuple[str, float]] = []
            for cid in list(picked):
                for nb, w in self.synapses.get(cid, {}).items():
                    if nb not in picked and w >= 0.5:
                        extra.append((nb, w))
            extra.sort(key=lambda kv: -kv[1])
            picked += [nb for nb, _ in extra[:2]]
        # 重排：候选由融合名次产生，但最终排序用"查询词重量级"，压掉巧合命中的大细胞
        picked = self.rerank(query, picked)
        # 第三层（可选）：让在线 LLM 读一遍候选原文，专门捕捉词面检索看不见的因果关联
        # （"24 分钟掉线" ↔ "session TTL 25 分钟"）。失败静默回退，绝不阻塞路由。
        llm_scores = self._llm_rerank(query, picked)
        if llm_scores:
            picked = sorted(
                picked,
                key=lambda cid: (-llm_scores.get(cid, 0.0),
                                 -self._cell_rerank_score(query, self.cells[cid])),
            )
            tr.reason += f" | LLM重排 {len(llm_scores)} 项"
        tr.hits = [(cid, self._cell_rerank_score(query, self.cells[cid])) for cid in picked]
        tr.stage, tr.reason = "cell", f"命中 {len(picked)} 个神经元"
        self.stats["local_hit"] += 1
        self._reinforce(picked)
        tr.latency_ms = (time.perf_counter() - t0) * 1000
        return tr

    def rerank(self, query: str, cell_ids: Sequence[str]) -> List[str]:
        """词项重量级重排：按 (命中查询词的 IDF 之和 × 覆盖率) / 长度惩罚 排序。

        为什么必须做：召回阶段用的是"名次融合"，一个又长又含通用代码的细胞
        会靠巧合挤进候选；如果直接按名次或按成本打包，真正相关的短细胞会被挤掉。
        真实系统里这一层就是 cross-encoder reranker。
        """
        ids = [cid for cid in cell_ids if cid in self.cells]
        return sorted(ids, key=lambda cid: -self._cell_rerank_score(query, self.cells[cid]))

    def _cell_rerank_score(self, query: str, cell: Cell) -> float:
        terms = self._q_terms(query)
        if not terms:
            return 0.0
        n = max(1, len(self.cells))
        mass = 0.0
        total = 0.0
        for t in terms:
            df = self.emb.df_count(t)
            idf = max(0.0, math.log(1 + (n - df + 0.5) / (df + 0.5)))
            total += idf
            if t in cell.tf:
                mass += idf
        cov = mass / total if total else 0.0
        length_penalty = 1.0 / (1.0 + math.log1p(cell.tokens) / 6.0)
        return mass * cov * length_penalty

    def _reinforce(self, cell_ids: Iterable[str]) -> None:
        """被激活的神经元：权重上升（长时程增强 LTP），并更新最近访问时间。"""
        now = time.time()
        for cid in cell_ids:
            c = self.cells.get(cid)
            if not c:
                continue
            c.access_count += 1
            c.last_access = now
            c.weight = min(3.0, c.weight + 0.18)
            if c.archived:
                c.archived = False  # 重新激活 = 从存档回迁
        # Hebbian：同时被激活的神经元之间加边
        ids = list(cell_ids)
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                self.synapses.setdefault(a, {})[b] = min(1.0, self.synapses.get(a, {}).get(b, 0.0) + 0.15)
                self.synapses.setdefault(b, {})[a] = min(1.0, self.synapses.get(b, {}).get(a, 0.0) + 0.15)

    def _escalate(self, tr: Trace, query: str, allow_external: bool) -> None:
        self.stats["escalated"] += 1
        if not allow_external or self.external_fn is None:
            tr.reason += " | 无外呼通道"
            return
        self.stats["external_calls"] += 1
        tr.external_calls += 1
        result = self.external_fn(query)
        if result:
            cell = self.make_cell([result], turn_start=-1, turn_end=-1,
                                  summary=f"外呼所得：{query[:60]}", source="external",
                                  files=[], pinned=False)
            tr.external_used = True
            tr.hits.append((cell.cell_id, 1.0))
            tr.reason += " | 已固化外呼结果"

    # ---------------------------------------------------------------- 元管理
    def decay(self, *, age_unit_turns: float = 12.0, half_life_units: float = 3.0) -> None:
        """艾宾浩斯式遗忘：R = exp(-Δt / S)，S 随使用次数增长。"""
        now_turn = max((c.turn_end for c in self.cells.values()), default=0)
        for c in list(self.cells.values()):
            if c.pinned:
                continue
            dt = max(0, now_turn - c.turn_end)
            strength = half_life_units * (1 + 1.3 * c.access_count)
            retention = math.exp(-dt / max(1e-6, age_unit_turns * strength))
            c.weight = max(0.0, min(3.0, 0.35 * c.weight + 0.65 * (retention * (1 + 0.5 * c.access_count))))
            if c.weight < self.drop_weight and c.access_count == 0 and c.source != "external":
                self.synapses.pop(c.cell_id, None)
                for s in self.synapses.values():
                    s.pop(c.cell_id, None)
                self.cells.pop(c.cell_id, None)
                self.stats["dropped"] += 1
            elif c.weight < self.archive_weight:
                if not c.archived:
                    self.stats["archived"] += 1
                c.archived = True

    def _topic_overlap(self, a: Cell, b: Cell) -> float:
        """两个神经元在"罕见词"上的 Jaccard 重叠。这是比稠密相似度可靠得多的同主题判据。"""
        ra = {t for t in a.tf if self.emb.df_count(t) <= 3}
        rb = {t for t in b.tf if self.emb.df_count(t) <= 3}
        if not ra or not rb:
            return 0.0
        return len(ra & rb) / len(ra | rb)

    def consolidate(self, *, min_sim: Optional[float] = None, min_topic: float = 0.25) -> List[str]:
        """睡眠期巩固：把**时间相邻 + 互为最近邻 + 同主题**的神经元合并。

        三条约束缺一不可，否则巩固会变成"把所有东西糊成一团"：
          · 时间相邻   —— 跨半个会话的相似内容不该合并（大事记不能压平）
          · 互为最近邻 —— 防止链式合并把整条时间轴吞掉
          · 主题重叠   —— 纯字符稠密相似度会误判，必须让罕见词投票
        合并只改索引，不动任何原文（保真原则）。
        """
        sim_th = self.consolidate_sim if min_sim is None else min_sim
        merged_log: List[str] = []
        changed = True
        while changed:
            changed = False
            cells = sorted(self.cells.values(), key=lambda c: c.turn_start)
            for a, b in zip(cells, cells[1:]):
                if a.pinned != b.pinned:
                    continue
                if self._topic_overlap(a, b) < min_topic:
                    continue
                if self.emb.cos(a.vec, b.vec) < sim_th:
                    continue
                # 互为最近邻：b 之后不得有比 a 更像的细胞（反之亦然）
                if self._nearest(b, cells) != a.cell_id or self._nearest(a, cells) != b.cell_id:
                    continue
                # 容量守护：合并后的神经元不得超过 tau_ceiling。
                # 没有这条约束，巩固会把整条时间轴糊成"巨型神经元"，
                # 一个向量要代表九轮内容 → 检索精度和预算同时崩掉。
                if a.tokens + b.tokens > self.tau_ceiling:
                    continue
                a.text = a.text + "\n\n" + b.text
                a.summary = (a.summary + " || " + b.summary)[:600]
                a.tokens = est_tokens(a.text)
                a.merged_from.append(b.cell_id)
                a.vec = self._doc_vec(a.text)
                for t in tokenize(b.text):
                    a.tf[t] = a.tf.get(t, 0) + 1
                a.files = sorted(set(a.files) | set(b.files))
                a.weight = min(3.0, a.weight + b.weight * 0.5)
                a.turn_end = max(a.turn_end, b.turn_end)
                self.synapses.pop(b.cell_id, None)
                for s in self.synapses.values():
                    s.pop(b.cell_id, None)
                self.cells.pop(b.cell_id, None)
                self.stats["merged"] += 1
                merged_log.append(f"{b.cell_id} -> {a.cell_id}")
                changed = True
                break
        return merged_log

    def _nearest(self, cell: Cell, cells: List[Cell], span: int = 4) -> Optional[str]:
        """在时间邻域内找最相似的邻居（返回 cell_id）。"""
        idx = cells.index(cell)
        best_id, best_sim = None, -2.0
        for other in cells[max(0, idx - span) : idx] + cells[idx + 1 : idx + 1 + span]:
            s = self.emb.cos(cell.vec, other.vec)
            if s > best_sim:
                best_sim, best_id = s, other.cell_id
        return best_id

    # ---------------------------------------------------------------- 统计
    def total_tokens(self) -> int:
        return sum(c.tokens for c in self.cells.values())

    def snapshot(self) -> Dict[str, object]:
        return {
            "cells": len(self.cells), "total_tokens": self.total_tokens(),
            "avg_cell_tokens": int(self.total_tokens() / max(1, len(self.cells))),
            "archived": sum(1 for c in self.cells.values() if c.archived),
            "pinned": sum(1 for c in self.cells.values() if c.pinned),
            "synapses": sum(len(v) for v in self.synapses.values()) // 2,
            "stats": dict(self.stats),
        }

    def save(self, path: str) -> None:
        payload = {
            "config": {
                "tau_floor": self.tau_floor, "tau_ceiling": self.tau_ceiling,
                "top_k": self.top_k, "conf_threshold": self.conf_threshold,
                "margin_threshold": self.margin_threshold,
            },
            "df": self.emb.df, "n_docs": self.emb.n_docs, "seq": self._seq,
            "cells": [asdict(c) for c in self.cells.values()],
            "synapses": self.synapses, "stats": self.stats,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str, **kw) -> "Hub":
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        emb = HashingEmbedder(dim=4096)
        emb.df = {int(k): v for k, v in payload["df"].items()}
        emb.n_docs = payload.get("n_docs", 0)
        hub = cls(emb, **{**payload["config"], **kw})
        hub._seq = payload.get("seq", 0)
        hub.stats.update(payload.get("stats", {}))
        for d in payload["cells"]:
            c = Cell(**d)
            hub.cells[c.cell_id] = c
        hub.synapses = payload.get("synapses", {})
        return hub
