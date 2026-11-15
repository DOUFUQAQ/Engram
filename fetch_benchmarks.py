"""
fetch_benchmarks.py —— 下载公开评测集数据

本环境实测：**HuggingFace 官方域名超时，但镜像 hf-mirror.com 可直连**。
所以默认走镜像，并保留官方地址作为回退。

用法：
    python fetch_benchmarks.py                 # 默认下 LongMemEval oracle + s
    python fetch_benchmarks.py --list          # 只列出可用文件与大小
    python fetch_benchmarks.py --include m     # 连 500 会话的 m 版一起下（约 1.5GB）
    python fetch_benchmarks.py --repo locomo   # 下 LoCoMo
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, List, Tuple

socket.setdefaulttimeout(300)
UA = {"User-Agent": "Mozilla/5.0 (memory-bench fetch)"}

LME_MIRRORS = [
    ("hf-mirror.com（本环境实测可用）",
     "https://hf-mirror.com/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/"),
    ("huggingface.co（官方，本环境常超时）",
     "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/"),
]
LME_FILES = {
    "oracle": "longmemeval_oracle.json",        # 只含证据会话，难度基准
    "s": "longmemeval_s_cleaned.json",          # 约 53 会话 / 115k tokens
    "m": "longmemeval_m_cleaned.json",          # 约 500 会话
}
LOCOMO_FILES = [
    ("https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
     "locomo10.json"),
]

DATA_DIR = "bench_data"


def _download(url: str, dest: str) -> Tuple[bool, str]:
    tmp = dest + ".part"
    try:
        t0 = time.time()
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
            total = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
                if total % (20 << 20) < (1 << 20):
                    print(f"      {total / 1e6:7.1f} MB …", end="\r")
        os.replace(tmp, dest)
        return True, f"{total / 1e6:.1f} MB / {time.time() - t0:.0f}s"
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        return False, f"{type(e).__name__}: {str(e)[:80]}"


def fetch_lme(keys: List[str], force: bool = False) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    for key in keys:
        fn = LME_FILES[key]
        dest = os.path.join(DATA_DIR, fn)
        if os.path.exists(dest) and not force:
            print(f"  ✓ {fn} 已存在（{os.path.getsize(dest) / 1e6:.1f} MB），跳过")
            continue
        print(f"  · {fn}")
        for label, base in LME_MIRRORS:
            print(f"      试 {label}")
            ok, msg = _download(base + fn, dest)
            if ok:
                print(f"      ✅ {msg}")
                break
            print(f"      ❌ {msg}")
        else:
            print(f"      ⚠️ 全部镜像失败：{fn}")
            print(f"         请手动下载：{LME_MIRRORS[0][1]}{fn}")


def fetch_locomo(force: bool = False) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    for url, fn in LOCOMO_FILES:
        dest = os.path.join(DATA_DIR, fn)
        if os.path.exists(dest) and not force:
            print(f"  ✓ {fn} 已存在，跳过")
            continue
        print(f"  · {fn}")
        ok, msg = _download(url, dest)
        print(f"      {'✅' if ok else '❌'} {msg}")


def summarize() -> None:
    print("=" * 88)
    print("已就绪的数据")
    print("=" * 88)
    if not os.path.isdir(DATA_DIR):
        print("  （bench_data/ 不存在）")
        return
    for fn in sorted(os.listdir(DATA_DIR)):
        if not fn.endswith(".json"):
            continue
        p = os.path.join(DATA_DIR, fn)
        mb = os.path.getsize(p) / 1e6
        note = ""
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, list) and d:
                from collections import Counter
                if "question_type" in d[0]:
                    c = Counter(x.get("question_type") for x in d)
                    n_sess = sum(len(x.get("haystack_sessions", [])) for x in d)
                    note = (f"{len(d)} 条问题 / 约 {n_sess // max(1, len(d))} 会话每问 / "
                            f"题型 {len(c)} 种")
        except Exception:
            note = "（非 JSON 数组，未解析）"
        print(f"  {fn:<34}{mb:>9.1f} MB   {note}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include", default="", help="额外版本，如 m")
    ap.add_argument("--repo", default="lme", choices=["lme", "locomo", "all"])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("LongMemEval 可用文件:")
        for k, v in LME_FILES.items():
            print(f"  {k:<8} {v}")
        print("LoCoMo 可用文件:")
        for url, fn in LOCOMO_FILES:
            print(f"  {fn:<24} {url}")
        return

    print("=" * 88)
    print("下载公开评测集")
    print("=" * 88)
    if args.repo in ("lme", "all"):
        keys = ["oracle", "s"] + ([k.strip() for k in args.include.split(",") if k.strip()])
        fetch_lme([k for k in keys if k in LME_FILES], force=args.force)
    if args.repo in ("locomo", "all"):
        fetch_locomo(force=args.force)
    print()
    summarize()
    print()
    print("下一步：")
    print("  python eval_longmemeval.py --per-type 5      # 30 条，约 20 分钟")
    print("  python eval_longmemeval.py --per-type 20     # 120 条，正式评测")
    print("\n注：首次运行会调嵌入 API 建立向量缓存（bench_data/.embed_cache.pkl），")
    print("    之后重复运行会命中缓存，快很多。缓存键含模型名，换模型不会读错。")


if __name__ == "__main__":
    sys.exit(main())
