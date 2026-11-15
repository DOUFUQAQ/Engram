"""
embedder.py —— 真嵌入层（在线模型）+ 可选 LLM 重排

设计原则：**接口与 HashingEmbedder 完全一致**，所以 Hub 不需要改一行就能换掉检索层。
    embed(text, learn=False) -> List[float]
    cos(a, b) -> float
    df_count(token) -> int

支持两类在线能力，各解决一个问题：

  ① **真嵌入（EmbeddingProvider）** —— 解决"意思像"。
     任何 OpenAI 兼容的 /v1/embeddings 端点都能接（DeepSeek / SiliconFlow / OpenAI / 本地 vLLM…）。
     ⚠️ 重要事实：DeepSeek 官方 API 目前**只提供 chat 接口，没有 embeddings 端点**。
     所以 .env 里给了两组配置：EMBED_BASE_URL + EMBED_API_KEY 可以指向别的厂商（推荐 BGE-M3），
     而 DEEPSEEK_* 用于 ② 的重排。

  ② **LLM 重排（ChatReranker）** —— 解决"因果链"。
     用 chat 模型给候选神经元打分（0-10），专门补 L3 因果查询的短板：
     词面检索召不回"24 分钟掉线 → session TTL"，但模型一读就懂。
     代价是一次 API 往返，所以只在**候选集有歧义**时才触发（见 Hub 的 rerank 钩子）。
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from textutil import STOP, HashingEmbedder, char_ngrams, stable_hash, tokenize


# ---------------------------------------------------------------------------
# .env 加载（零依赖）
# ---------------------------------------------------------------------------

def load_dotenv(path: str = ".env", *, override: bool = False) -> Dict[str, str]:
    """极简 .env 解析：KEY=VALUE、支持 # 注释、引号可选、允许 = 出现在值里。

    故意不引入 python-dotenv —— 这个项目的卖点之一就是零依赖可跑。
    """
    loaded: Dict[str, str] = {}
    if not os.path.exists(path):
        return loaded
    # encoding="utf-8-sig"：吃掉 Windows 记事本 / PowerShell Out-File 写入的 BOM。
    # 这个坑很隐蔽——BOM 会让**第一个键名**变成 "\ufeffEMBED_BASE_URL"，
    # 于是它静默失效、悄悄回退到默认值（本原型真踩过）。
    with open(path, encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key, val = key.strip().lstrip("\ufeff"), val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                val = val[1:-1]
            if not key:
                continue
            loaded[key] = val
            if override or key not in os.environ:
                os.environ[key] = val
    return loaded


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name) or default)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# HTTP（只用标准库）
# ---------------------------------------------------------------------------

@dataclass
class HttpStats:
    calls: int = 0
    errors: int = 0
    retries: int = 0
    total_ms: float = 0.0
    embed_tokens: int = 0

    def describe(self) -> str:
        if not self.calls:
            return "无调用"
        return (f"{self.calls} 次调用 / {self.errors} 次失败 / {self.retries} 次重试 / "
                f"累计 {self.total_ms / 1000:.1f}s / 平均 {self.total_ms / self.calls:.0f}ms")


def _post_json(url: str, payload: dict, headers: Dict[str, str], timeout: float,
               retries: int = 3) -> Tuple[Optional[dict], str]:
    """带指数退避的 POST。返回 (json, err)。"""
    body = json.dumps(payload).encode("utf-8")
    hdrs = {"Content-Type": "application/json", **headers}
    last_err = ""
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8")), ""
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8")[:300]
            except Exception:
                pass
            last_err = f"HTTP {e.code}: {detail}"
            # 4xx（除 429）不重试：多半是鉴权或参数问题，重试只是浪费配额
            if 400 <= e.code < 500 and e.code != 429:
                return None, last_err
        except Exception as e:                                  # 网络层错误
            last_err = f"{type(e).__name__}: {e}"
        if attempt < retries:
            time.sleep(min(8.0, 1.5 ** attempt))
    return None, last_err


# ---------------------------------------------------------------------------
# ① 真嵌入
# ---------------------------------------------------------------------------

class EmbeddingProvider:
    """OpenAI 兼容嵌入客户端，兼容三种"兼容"程度不同的实现。

    实测遇到三种请求/响应形态（各家"OpenAI 兼容"其实不一样）：
      ① 标准 OpenAI：请求 input=["a","b"]，响应 data=[{embedding:[...]}, ...]
      ② 火山方舟 Ark：请求 input=[{"type":"text","text":"a"}]，
         路径是 /embeddings/multimodal，响应 data={embedding:[...], object:"embedding"}
         —— 注意 data 是**对象**而不是数组。Ark 的文本嵌入模型也走这个 multimodal 端点，
         而 /embeddings 会直接返回 404 "model does not exist"，极易被误判成"没有嵌入能力"。
      ③ 部分中转站：data 是数组但缺 index 字段。

    api_path + 响应结构自动判别，调用方不需要关心。
    """

    def __init__(self, base_url: str, api_key: str, model: str, *,
                 dim: Optional[int] = None, batch: int = 32, timeout: float = 30.0,
                 query_prefix: str = "", doc_prefix: str = "",
                 api_path: str = "/embeddings", input_style: str = "auto"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.batch = batch
        self.timeout = timeout
        # 有些模型（如 bge-m3 的 instruct 变体、E5 系列）要求给查询/文档加前缀
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.api_path = api_path or "/embeddings"
        if input_style == "auto":
            input_style = "multimodal" if "multimodal" in self.api_path else "openai"
        self.input_style = input_style
        self.dim = dim or 0
        self.stats = HttpStats()
        self.df: Dict[int, int] = {}          # 为兼容 HashingEmbedder 的属性
        self.n_docs = 0
        self._df_cache: Dict[str, int] = {}
        self._df_stamp = -1
        self._cells_ref: Optional[Dict[str, Cell]] = None

    # ---------------------------------------------------------------- 属性兼容
    def df_count(self, token: str) -> int:
        cells = self._cells_ref or {}
        if self._df_stamp != len(cells):
            self._df_cache.clear()
            self._df_stamp = len(cells)
        if token in self._df_cache:
            return self._df_cache[token]
        n = sum(1 for c in cells.values() if token in c.tf)
        self._df_cache[token] = n
        return n

    # ---------------------------------------------------------------- 调用
    def _pack_input(self, chunk: List[str]) -> object:
        if self.input_style == "multimodal":
            return [{"type": "text", "text": t} for t in chunk]
        return chunk

    def _unpack(self, data: dict, chunk: List[str]) -> Tuple[Optional[List[List[float]]], str]:
        """把两种响应结构都归一成 List[List[float]]。"""
        raw = data.get("data")
        if raw is None:
            return None, "返回里没有 data 字段"
        if isinstance(raw, dict):                      # Ark multimodal：data 是单个对象
            v = raw.get("embedding")
            if not v:
                return None, "data 对象里没有 embedding"
            return [[float(x) for x in v]], ""
        if not isinstance(raw, list):
            return None, f"data 结构未知: {type(raw).__name__}"
        items = sorted(raw, key=lambda d: d.get("index", 0))
        if len(items) != len(chunk):
            return None, f"返回条数不匹配: 期望 {len(chunk)} 得到 {len(items)}"
        out = []
        for it in items:
            v = it.get("embedding")
            if not v:
                return None, "条目里没有 embedding 字段"
            out.append([float(x) for x in v])
        return out, ""

    def _request(self, texts: Sequence[str]) -> Tuple[Optional[List[List[float]]], str]:
        if not self.api_key:
            return None, "EMBED_API_KEY 未配置"
        url = f"{self.base_url}{self.api_path}"
        vecs: List[List[float]] = []
        for i in range(0, len(texts), self.batch):
            chunk = list(texts[i : i + self.batch])
            payload = {"model": self.model, "input": self._pack_input(chunk)}
            t0 = time.perf_counter()
            data, err = _post_json(url, payload,
                                   {"Authorization": f"Bearer {self.api_key}"}, self.timeout)
            self.stats.calls += 1
            self.stats.total_ms += (time.perf_counter() - t0) * 1000
            if data is None:
                self.stats.errors += 1
                return None, err
            got, perr = self._unpack(data, chunk)
            if got is None:
                self.stats.errors += 1
                return None, perr
            vecs.extend(got)
            usage = (data.get("usage") or {}).get("prompt_tokens") or \
                    (data.get("usage") or {}).get("total_tokens")
            if usage:
                self.stats.embed_tokens += int(usage)
        return vecs, ""

    def embed(self, text: str, learn: bool = False) -> List[float]:
        vecs, err = self._request([self.doc_prefix + text])
        if vecs is None:
            raise RuntimeError(f"嵌入调用失败: {err}")
        v = vecs[0]
        if not self.dim:
            self.dim = len(v)
        if learn:
            self.n_docs += 1
        return v

    def embed_batch(self, texts: Sequence[str], *, prefix: str = "") -> List[List[float]]:
        """批量嵌入（入库用）。比逐条快一个数量级。"""
        if prefix:
            texts = [prefix + t for t in texts]
        vecs, err = self._request(list(texts))
        if vecs is None:
            raise RuntimeError(f"批量嵌入失败: {err}")
        if vecs and not self.dim:
            self.dim = len(vecs[0])
        return vecs

    def embed_query(self, text: str) -> List[float]:
        vecs, err = self._request([self.query_prefix + text])
        if vecs is None:
            raise RuntimeError(f"查询嵌入失败: {err}")
        return vecs[0]

    @staticmethod
    def cos(a: Sequence[float], b: Sequence[float]) -> float:
        """余弦相似度。真嵌入是稠密语义向量，必须用真正的余弦（归一化后点积）。"""
        if not a or not b:
            return 0.0
        dot = na = nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        if na <= 0 or nb <= 0:
            return 0.0
        return dot / math.sqrt(na * nb)

    def precompute(self, cells: Dict[str, Cell]) -> int:
        """入库后一次性把所有神经元重新嵌入（真嵌入不会"边写边学"）。"""
        ids = [cid for cid, c in cells.items() if c.text]
        if not ids:
            return 0
        vecs = self.embed_batch([cells[cid].text for cid in ids], prefix=self.doc_prefix)
        for cid, v in zip(ids, vecs):
            cells[cid].vec = v
        return len(ids)

    def probe(self) -> Tuple[bool, str]:
        """连通性探测：给 .env 一个可执行的健康检查。"""
        try:
            v = self.embed("ping")
            return True, f"OK  dim={len(v)}  model={self.model}"
        except Exception as e:
            return False, str(e)


# ---------------------------------------------------------------------------
# ② LLM 重排（chat 模型）
# ---------------------------------------------------------------------------

_RERANK_SYS = (
    "你是记忆检索的相关性判定器。用户给出一个查询和若干编号的记忆片段。\n"
    "对每个片段判断：**它是否包含回答这个查询所必需的信息？**\n"
    "\n"
    "判定为 yes 的情况：\n"
    "· 片段直接含有查询要的那个事实（端口号、环境变量名、阈值、文件名、前缀……）；\n"
    "· 片段含有解释查询里某个现象的**根因或相关约定**。\n"
    "  特别注意：查询描述的是某个现象的**后果**，而片段里是导致它的配置或决策 —— 这是 yes。\n"
    "  例：查询「用户停留 24 分钟就掉线」，片段里写着「session TTL 设为 25 分钟，前端心跳 20 分钟」→ yes\n"
    "· 片段含有修复该问题需要的操作约定。\n"
    "\n"
    "判定为 no 的情况：\n"
    "· 同一技术领域但对该查询没有帮助（例：查询「超时多少」，片段只讲「缓存前缀」）；\n"
    "· 只是碰巧出现相同词汇但含义无关。\n"
    "\n"
    "只输出 JSON，不要解释，不要 Markdown 代码块。"
)


class ChatReranker:
    """用 chat 模型判定候选神经元与查询的相关性。专治词面检索的因果链断裂。

    设计选择（踩过坑）：
      第一版是"打 0-10 分"。实测不可用 —— 7B~flash 这类模型给的分数几乎全在 8-10，
      分辨力极差，重排等于没排。改成**二分类 yes/no** 后稳定得多：
      模型对"这段是否包含回答所需信息"的判断远比"打几分"可靠。
    """

    def __init__(self, base_url: str, api_key: str, model: str, *,
                 timeout: float = 60.0, max_candidates: int = 12,
                 snippet_chars: int = 600, enabled: bool = True,
                 mode: str = "binary"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_candidates = max_candidates
        self.snippet_chars = snippet_chars
        self.enabled = enabled and bool(api_key)
        self.mode = mode                     # binary | score
        self.stats = HttpStats()
        self.cache: Dict[Tuple[str, str], float] = {}

    def _prompt(self, query: str, cands: List[Tuple[str, str]]) -> str:
        blocks = []
        for i, (cid, text) in enumerate(cands, 1):
            body = text[: self.snippet_chars]
            blocks.append(f"### 片段{i} (id={cid})\n{body}")
        ids = ", ".join(f'"{cid}"' for cid, _ in cands)
        if self.mode == "binary":
            ask = (f"\n\n输出 JSON，键是片段 id，值是 \"yes\" 或 \"no\"，必须包含这些键：{ids}。"
                   f'\n格式示例：{{"N0001": "yes", "N0002": "no"}}')
        else:
            ask = (f"\n\n输出 JSON，键是片段 id，值是 0-10 整数分数，必须包含这些键：{ids}。"
                   f'\n格式示例：{{"N0001": 9, "N0002": 3}}')
        return f"查询：{query}\n\n" + "\n\n".join(blocks) + ask

    @staticmethod
    def _salvage(content: str) -> Optional[dict]:
        """从被截断/带包裹的输出里抢救 JSON。

        模型有时会输出 ```json ... ``` 或先啰嗦一句再给 JSON，
        更常见的是 max_tokens 截断导致末尾缺 }。这里按最外层花括号截取并补齐。
        """
        if not content:
            return None
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
            text = re.sub(r"\s*```\s*$", "", text)
        start = text.find("{")
        if start < 0:
            return None
        body = text[start:]
        for attempt in range(3):
            candidate = body + "}" * attempt
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        # 最后一招：逐对抓 "id": value
        pairs = re.findall(r'"([A-Za-z0-9_]+)"\s*:\s*("?)(yes|no|true|false|\d+(?:\.\d+)?)\2', body)
        return {k: v for k, _, v in pairs} if pairs else None

    def _parse(self, parsed: dict, cid: str) -> Optional[float]:
        v = parsed.get(cid)
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("yes", "true", "y", "是", "相关"):
                return 1.0
            if s in ("no", "false", "n", "否", "不相关"):
                return 0.0
            if re.fullmatch(r"\d+(\.\d+)?", s):
                return float(s)
        return None

    def score(self, query: str, cands: List[Tuple[str, str]]) -> Dict[str, float]:
        """返回 {cell_id: 相关度}（binary 模式下 yes=1.0 / no=0.0）。

        任何失败都返回空字典 —— 调用方回退到词面名次，绝不阻塞路由。
        """
        if not self.enabled or not cands:
            return {}
        cands = cands[: self.max_candidates]
        todo = [(cid, txt) for cid, txt in cands if (query, cid) not in self.cache]
        if todo:
            payload = {
                "model": self.model,
                "messages": [{"role": "system", "content": _RERANK_SYS},
                             {"role": "user", "content": self._prompt(query, todo)}],
                "temperature": 0,
                "response_format": {"type": "json_object"},
                # 实测教训：max_tokens 给小了会把 JSON 截断，报成"调用失败"。
                # 12 个候选的 yes/no JSON 大约 100 token，但模型有时会先啰嗦一段，
                # 给 1500 才稳。截断的 JSON 无法解析 → 整批重排白跑。
                "max_tokens": 1500,
            }
            t0 = time.perf_counter()
            data, err = _post_json(f"{self.base_url}/chat/completions", payload,
                                   {"Authorization": f"Bearer {self.api_key}"}, self.timeout)
            self.stats.calls += 1
            self.stats.total_ms += (time.perf_counter() - t0) * 1000
            if data is None:
                self.stats.errors += 1
                return {}
            parsed = None
            content = ""
            try:
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
            except Exception:
                parsed = self._salvage(content)
            if not isinstance(parsed, dict):
                self.stats.errors += 1
                if len(todo) > 4:
                    # 整批失败时降半重试一次：候选越少越不容易被截断
                    self.stats.retries += 1
                    return self.score(query, cands[: max(4, len(cands) // 2)])
                return {}
            for cid, _ in todo:
                v = self._parse(parsed, cid)
                if v is not None:
                    self.cache[(query, cid)] = v
        return {cid: self.cache[(query, cid)] for cid, _ in cands if (query, cid) in self.cache}


# ---------------------------------------------------------------------------
# 从环境构造
# ---------------------------------------------------------------------------

@dataclass
class Components:
    embedder: object
    reranker: Optional[ChatReranker]
    notes: List[str] = field(default_factory=list)
    strict_ok: bool = False

    def describe(self) -> str:
        lines = []
        emb = self.embedder
        if isinstance(emb, EmbeddingProvider):
            lines.append(f"  嵌入      : 在线 {emb.model} @ {emb.base_url}")
        else:
            lines.append(f"  嵌入      : 离线 HashingEmbedder(dim={getattr(emb, 'dim', '?')})  "
                         f"← 未配置在线嵌入，稠密路判别力弱")
        if self.reranker and self.reranker.enabled:
            lines.append(f"  重排      : 在线 {self.reranker.model} @ {self.reranker.base_url}")
        else:
            lines.append("  重排      : 关闭")
        for n in self.notes:
            lines.append(f"  ! {n}")
        return "\n".join(lines)


def build_components(env_path: str = ".env", *, prefer_online: bool = True,
                     strict: bool = False) -> Components:
    """按 .env 组装嵌入与重排。strict=True 时若在线嵌入不可用直接抛错。

    strict 的用途很重要：不做 strict 的时候，配置写错会静默退化成离线检索，
    于是"换了真嵌入但没效果"这种结论就会被错误地得出。宁可炸掉。
    """
    load_dotenv(env_path)
    notes: List[str] = []

    emb_base = _env("EMBED_BASE_URL")
    emb_key = _env("EMBED_API_KEY")
    emb_model = _env("EMBED_MODEL")
    ds_base = _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    ds_key = _env("DEEPSEEK_API_KEY")
    ds_model = _env("DEEPSEEK_MODEL", "deepseek-chat")

    embedder: object
    if prefer_online and emb_key and emb_model:
        embedder = EmbeddingProvider(
            emb_base or ds_base, emb_key, emb_model,
            dim=int(_env("EMBED_DIM") or 0) or None,
            batch=int(_env("EMBED_BATCH") or 32),
            timeout=_env_float("EMBED_TIMEOUT", 30.0),
            query_prefix=_env("EMBED_QUERY_PREFIX"),
            doc_prefix=_env("EMBED_DOC_PREFIX"),
            api_path=_env("EMBED_API_PATH", "/embeddings"),
            input_style=_env("EMBED_INPUT_STYLE", "auto"),
        )
    else:
        if strict:
            raise RuntimeError(
                "strict 模式要求在线嵌入：请在 .env 里配置 EMBED_BASE_URL / EMBED_API_KEY / EMBED_MODEL")
        notes.append("未配置 EMBED_*，退化为离线哈希嵌入")
        embedder = HashingEmbedder(dim=4096)

    reranker: Optional[ChatReranker] = None
    rerank_on = _env("RERANK_ENABLED", "1") not in ("0", "false", "False", "")
    if rerank_on and ds_key:
        reranker = ChatReranker(ds_base, ds_key, ds_model,
                                timeout=_env_float("RERANK_TIMEOUT", 60.0),
                                max_candidates=int(_env("RERANK_MAX_CANDIDATES") or 12),
                                snippet_chars=int(_env("RERANK_SNIPPET_CHARS") or 600))
    elif rerank_on:
        notes.append("未配置 DEEPSEEK_API_KEY，重排关闭")

    return Components(embedder=embedder, reranker=reranker, notes=notes,
                      strict_ok=isinstance(embedder, EmbeddingProvider))


if __name__ == "__main__":
    import sys
    comp = build_components(strict=False)
    print("=" * 80)
    print("embedder.py 自检")
    print("=" * 80)
    print(comp.describe())
    print()
    if isinstance(comp.embedder, EmbeddingProvider):
        ok, msg = comp.embedder.probe()
        print(f"嵌入连通性: {'✅' if ok else '❌'} {msg}")
        if ok:
            a = comp.embedder.embed_query("session TTL 过期")
            b = comp.embedder.embed_query("登录态失效时间")
            c = comp.embedder.embed_query("库存扣减")
            print(f"  语义近邻检查: 'session TTL 过期' vs '登录态失效时间' = "
                  f"{EmbeddingProvider.cos(a, b):.3f}")
            print(f"                 'session TTL 过期' vs '库存扣减'       = "
                  f"{EmbeddingProvider.cos(a, c):.3f}")
            print("  （前者应显著高于后者；若差不多，说明嵌入模型不适合中文语义匹配）")
    else:
        print("离线模式：无需连通性检查（但检索质量会明显偏低）")
    sys.exit(0)
