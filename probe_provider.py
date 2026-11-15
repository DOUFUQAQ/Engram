"""
probe_provider.py —— 端点与模型体检：在花钱跑实验前，先确认"到底接的是谁"

为什么要这个东西：`.env` 里 base_url 与 api_key 来自不同厂商时，
表现是"401 或模型不存在"，很容易被误读成"真嵌入没用"。
本脚本对若干候选端点 + 候选模型做只读探测，直接把可用组合列出来。

用法：
    python probe_provider.py                 # 用 .env 里的 key 探所有候选
    python probe_provider.py --key sk-xxx    # 用指定 key 探
    python probe_provider.py --list-models   # 列出各端点可用模型（若支持 /models）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

from embedder import load_dotenv, _post_json

# (名称, base_url, 说明)
CANDIDATE_ENDPOINTS: List[Tuple[str, str, str]] = [
    ("DeepSeek 官方", "https://api.deepseek.com/v1", "只有 chat，没有 embeddings"),
    ("DeepSeek 官方(无 /v1)", "https://api.deepseek.com", "旧写法，兼容 OpenAI 路径"),
    ("硅基流动", "https://api.siliconflow.cn/v1", "BGE-M3 等中文嵌入"),
    ("火山方舟 Ark", "https://ark.cn-beijing.volces.com/api/v3", "key 前缀 ark- 对应这里"),
    ("阿里云百炼", "https://dashscope.aliyuncs.com/compatible-mode/v1", "text-embedding-v3"),
    ("月之暗面", "https://api.moonshot.cn/v1", "Moonshot"),
    ("智谱", "https://open.bigmodel.cn/api/paas/v4", "embedding-3"),
]
# 你环境里 ANTHROPIC_BASE_URL 指向的中转站（若可用，常同时提供 chat + embeddings）
if os.environ.get("ANTHROPIC_BASE_URL"):
    host = os.environ["ANTHROPIC_BASE_URL"].rstrip("/")
    CANDIDATE_ENDPOINTS.insert(0, ("环境里的中转站", host, "来自 ANTHROPIC_BASE_URL"))

EMBED_CANDIDATES: List[str] = [
    "BAAI/bge-m3", "BAAI/bge-large-zh-v1.5", "netease-youdao/bce-embedding-base_v1",
    "text-embedding-v3", "text-embedding-3-small", "embedding-3",
    # 火山方舟 Ark（豆包系）
    "doubao-embedding-large-text-250515", "doubao-embedding-large-text-240915",
    "doubao-embedding-text-240715", "doubao-embedding-text-240515",
    "doubao-embedding-vision-251215",
]

CHAT_CANDIDATES: List[str] = [
    "deepseek-chat", "deepseek-reasoner", "deepseek-v4-1-flash-260910",
    "deepseek-v3", "doubao-seed-1-6", "Qwen/Qwen3-8B",
]


def _get_models(base: str, key: str, timeout: float = 12.0) -> Optional[List[str]]:
    req = urllib.request.Request(f"{base}/models",
                                 headers={"Authorization": f"Bearer {key}"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        return [m.get("id", "?") for m in (data.get("data") or [])]
    except Exception:
        return None


def _try_embed(base: str, key: str, model: str, timeout: float = 15.0) -> Tuple[bool, str]:
    data, err = _post_json(f"{base}/embeddings", {"model": model, "input": "ping"},
                           {"Authorization": f"Bearer {key}"}, timeout, retries=0)
    if data is None:
        return False, err[:120]
    items = data.get("data") or []
    if not items or not items[0].get("embedding"):
        return False, "返回里没有 embedding"
    return True, f"dim={len(items[0]['embedding'])}"


def _try_chat(base: str, key: str, model: str, timeout: float = 20.0) -> Tuple[bool, str]:
    data, err = _post_json(f"{base}/chat/completions",
                           {"model": model, "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 4, "temperature": 0},
                           {"Authorization": f"Bearer {key}"}, timeout, retries=0)
    if data is None:
        return False, err[:120]
    try:
        return True, f"回复: {data['choices'][0]['message']['content'][:20]!r}"
    except Exception:
        return False, "返回结构异常"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=None)
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--env", default=".env")
    args = ap.parse_args()

    load_dotenv(args.env)
    key = args.key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("EMBED_API_KEY") or ""
    if not key:
        print("没有可用的 key。用 --key 指定，或在 .env 里填。")
        return
    print(f"探测使用 key: {key[:7]}...{key[-4:]}  (长度 {len(key)})\n")

    print("=" * 92)
    print("第 1 步：端点可达性与鉴权")
    print("=" * 92)
    reachable: List[Tuple[str, str]] = []
    for name, base, note in CANDIDATE_ENDPOINTS:
        models = _get_models(base, key)
        if models is None:
            # /models 不被支持时会返回 None，不能据此判定端点不可用，
            # 所以再用一次 chat 探测做交叉验证
            ok, detail = _try_chat(base, key, "deepseek-chat")
            if ok:
                print(f"  {name:<20} ✅ 可达（/models 不可用，chat 通过）")
                reachable.append((name, base))
            else:
                print(f"  {name:<20} ❌ {detail}")
        else:
            print(f"  {name:<20} ✅ 可达   共 {len(models)} 个模型   {note}")
            reachable.append((name, base))
            if args.list_models:
                for m in models[:25]:
                    print(f"        - {m}")

    if not reachable:
        print("\n没有任何端点接受这个 key。请在对应厂商控制台确认 key 与 base_url 是否配套。")
        return

    print()
    print("=" * 92)
    print("第 2 步：嵌入模型可用性（决定能不能跑真嵌入）")
    print("=" * 92)
    embed_ok: List[Tuple[str, str, str]] = []
    for name, base in reachable:
        hits = []
        for model in EMBED_CANDIDATES:
            ok, detail = _try_embed(base, key, model)
            if ok:
                hits.append((model, detail))
                embed_ok.append((name, base, model))
        if hits:
            print(f"  {name}")
            for model, detail in hits:
                print(f"      ✅ {model:<44} {detail}")
        else:
            print(f"  {name:<20} 无可用嵌入模型")

    print()
    print("=" * 92)
    print("第 3 步：chat 模型可用性（决定能不能跑 LLM 重排）")
    print("=" * 92)
    chat_ok: List[Tuple[str, str, str]] = []
    for name, base in reachable:
        hits = []
        for model in CHAT_CANDIDATES:
            ok, detail = _try_chat(base, key, model)
            if ok:
                hits.append((model, detail))
                chat_ok.append((name, base, model))
        if hits:
            print(f"  {name}")
            for model, detail in hits:
                print(f"      ✅ {model:<44} {detail}")
        else:
            print(f"  {name:<20} 无可用 chat 模型")

    print()
    print("=" * 92)
    print("结论：把下面这段贴进 .env")
    print("=" * 92)
    if embed_ok:
        n, b, m = embed_ok[0]
        print(f"  # 嵌入来自「{n}」")
        print(f"  EMBED_BASE_URL={b}")
        print(f"  EMBED_API_KEY=<你刚才这个 key>")
        print(f"  EMBED_MODEL={m}")
    else:
        print("  # 没找到可用的嵌入端点 —— 真嵌入这条路在这个 key 下走不通。")
        print("  # 可选：① 换一个提供 embeddings 的厂商；② 用中转站（若它转发 embeddings）")
    print()
    if chat_ok:
        n, b, m = chat_ok[0]
        print(f"  # 重排来自「{n}」")
        print(f"  DEEPSEEK_BASE_URL={b}")
        print(f"  DEEPSEEK_API_KEY=<你刚才这个 key>")
        print(f"  DEEPSEEK_MODEL={m}")
    else:
        print("  # 没找到可用的 chat 端点，LLM 重排只能关闭。")


if __name__ == "__main__":
    sys.exit(main())
