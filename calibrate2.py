"""
calibrate2.py —— 阈值与融合参数的经验标定（用数据决定，而不是拍脑袋）

要回答三个问题：
  Q1 稠密分与稀疏分怎么加权，正负样本才能分层？   → 扫 alpha，看 AUC
  Q2 置信门还该看什么？                          → 加"查询词覆盖率"这一路
  Q3 一个答案被拆到相邻两个神经元里怎么办？        → 测相邻覆盖需求，决定要否邻居扩散
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Tuple

from hub import Hub, tokenize
from sim_session import build_session, format_turn

PROBES: List[Tuple[str, str]] = [
    ("session TTL 设的是多少秒？", "SESSION_TTL_SECONDS = 25 * 60"),
    ("staging 数据库的主机和端口是什么？", "10.42.7.31:5432/orders_staging"),
    ("数据库连接串从哪个环境变量读？", "ORDERS_DB_DSN"),
    ("docker 宿主端口映射到哪个端口？", '"18080:8000"'),
    ("webhook 读超时统一调成多少毫秒？", "3500ms"),
    ("httpx 升级到哪个版本？", "httpx 0.27.0"),
    ("那个索引迁移文件叫什么？", "migrations/0007_index_orders_status.sql"),
    ("连接池上限最后改成多少？", "改成 8"),
    ("库存扣减的原子写法是什么？", "UPDATE ... WHERE stock >= qty"),
    ("灰度先放多少流量？", "5%"),
    ("镜像从 1.2G 瘦身到多少？", "380M"),
    ("队列重试几次后进死信？", "重试 3 次后进 DLQ"),
    ("CSV 导出的分批大小是多少行？", "分批 2000 行"),
    ("订单号前缀是什么？", "ORD-"),
    ("缓存 key 的前缀是什么？", "orders:"),
    ("支付事件幂等用的表叫什么？", "payment_events"),
    ("日志格式里带什么头？", "X-Orders-Token"),
    ("健康检查路径是什么？", "/healthz"),
    ("慢测试用什么 marker 排除？", 'pytest -m "not slow"'),
    ("手机号脱敏放在哪个模块？", "app/users/privacy.py"),
]

NEGATIVES: List[str] = [
    "我们的 Kubernetes 集群用的是什么 CNI 插件？",
    "去年 Q3 的营收数字是多少？",
    "这个项目的 Go 版本是多少？",
    "CDN 用的是哪家厂商？",
    "iOS 客户端最低支持版本是多少？",
    "设计师给的 Figma 链接在哪？",
    "团队每周站会是周几？",
]


def build_hub() -> Hub:
    turns, _, _ = build_session()
    hub = Hub(conf_threshold=0.0, margin_threshold=0.0)
    for t in turns:
        hub.write(format_turn(t), turn_index=t.index, files=t.files)
    hub._sync_index()
    return hub


def score_of(d: float, s: float, cov: float, alpha: float) -> float:
    """alpha 给稠密路，其余给稀疏路（稀疏做 log 压缩防止长细胞刷分），再加上覆盖率。"""
    sparse_term = math.log1p(max(0.0, s)) / math.log1p(30.0)
    return alpha * d + (1 - alpha - 0.15) * sparse_term + 0.15 * cov


def coverage(hub: Hub, cell, qtokens) -> float:
    if not qtokens:
        return 0.0
    hit = sum(1 for t in set(qtokens) if t in cell.tf)
    return hit / len(set(qtokens))


def auc(pos: List[float], neg: List[float]) -> float:
    if not pos or not neg:
        return 0.5
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    hub = build_hub()
    print(f"神经元 {len(hub.cells)} 个，冻结原文 {hub.total_tokens():,} tokens\n")

    # 预计算每题的 (dense, sparse, coverage) 最优值（真值所在细胞的分数）
    pos_feats: List[Tuple[float, float, float]] = []
    truth_ranks: List[int] = []
    for q, expect in PROBES:
        rows = hub.explain(q, top=999)
        found = None
        for i, (cid, d, s, _, _, _) in enumerate(rows, 1):
            if expect in hub.cells[cid].text:
                found = (d, s, coverage(hub, hub.cells[cid], tokenize(q)), i)
                break
        if found is None:
            print(f"  ⚠️ 语料里找不到锚点: {expect}")
            continue
        pos_feats.append(found[:3])
        truth_ranks.append(found[3])

    neg_feats: List[Tuple[float, float, float]] = []
    for q in NEGATIVES:
        rows = hub.explain(q, top=1)
        cid, d, s, _, _, _ = rows[0]
        neg_feats.append((d, s, coverage(hub, hub.cells[cid], tokenize(q))))

    print("=" * 96)
    print("Q1/Q2 扫描 alpha（稠密权重）与覆盖率权重，用 AUC 衡量正负分层能力")
    print("=" * 96)
    print(f"{'dense_w':>8}{'sparse_w':>9}{'cov_w':>7}{'AUC':>8}{'正样本均值':>11}{'负样本均值':>11}{'分离度':>9}")
    best = None
    for dw in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        for cw in [0.0, 0.1, 0.2, 0.3]:
            sw = 1.0 - dw - cw
            if sw < -1e-9:
                continue
            pos = [dw * d + sw * (math.log1p(s) / math.log1p(30)) + cw * c for d, s, c in pos_feats]
            neg = [dw * d + sw * (math.log1p(s) / math.log1p(30)) + cw * c for d, s, c in neg_feats]
            a = auc(pos, neg)
            pm, nm = sum(pos) / len(pos), sum(neg) / len(neg)
            sep = pm - nm
            flag = ""
            if best is None or (a, sep) > (best[0], best[3]):
                best = (a, (dw, sw, cw), None, sep)
                flag = " ★"
            print(f"{dw:>8.2f}{sw:>9.2f}{cw:>7.2f}{a:>8.3f}{pm:>11.3f}{nm:>11.3f}{sep:>9.3f}{flag}")

    print()
    print("=" * 96)
    print("Q3 真值所在神经元的排名分布（无邻居扩散时）")
    print("=" * 96)
    from collections import Counter
    c = Counter(truth_ranks)
    total = len(truth_ranks)
    print(f"  排名=1: {c.get(1,0)}/{total}   排名<=2: {sum(v for k,v in c.items() if k<=2)}/{total}   "
          f"排名<=3: {sum(v for k,v in c.items() if k<=3)}/{total}   最差排名: {max(truth_ranks)}")
    print(f"  逐题排名: {sorted(truth_ranks)}")

    # 答案是否被拆到相邻细胞：查每对相邻细胞之间共享的"罕见词"
    print()
    print("=" * 96)
    print("Q3b 相邻神经元之间的词项重叠（判断是否需要跨神经元扩散召回）")
    print("=" * 96)
    cells = sorted(hub.cells.values(), key=lambda c: c.turn_start)
    overlaps = []
    for a, b in zip(cells, cells[1:]):
        inter = set(a.tf) & set(b.tf)
        # 只统计"罕见词"（df<=3）的重叠，常见词没有信息量
        rare = [t for t in inter if hub.emb.df_count(t) <= 3]
        overlaps.append(len(rare))
    print(f"  相邻细胞罕见词重叠: 均值 {sum(overlaps)/len(overlaps):.1f}, 最大 {max(overlaps)}, 最小 {min(overlaps)}")
    print(f"  含义：重叠高的相邻细胞，答案很可能横跨两者 → 需要 top_k>=2 或邻居扩散")

    best_alpha = best[1]
    print()
    print(f"建议参数: dense={best_alpha[0]:.2f} sparse={best_alpha[1]:.2f} coverage={best_alpha[2]:.2f}, "
          f"AUC={best[0]:.3f}, 分离度={best[3]:.3f}")
    # 给出阈值建议：取负样本最大值与正样本分位数之间的中点
    dw, sw, cw = best_alpha
    pos = sorted(dw * d + sw * (math.log1p(s) / math.log1p(30)) + cw * c for d, s, c in pos_feats)
    neg = sorted(dw * d + sw * (math.log1p(s) / math.log1p(30)) + cw * c for d, s, c in neg_feats)
    p10 = pos[max(0, len(pos) // 10)]
    n90 = neg[min(len(neg) - 1, int(len(neg) * 0.9))]
    print(f"  正样本 10 分位 = {p10:.3f}   负样本 90 分位 = {n90:.3f}   "
          f"→ 阈值取中: {(p10 + n90) / 2:.3f}")


if __name__ == "__main__":
    main()
