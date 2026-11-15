"""
calibrate.py —— 中枢路由标定器（先证明"想得起来"，再谈省钱）

流程：把 71 轮会话原样灌进 Hub，然后用"很久以前的事实"提问，
看中枢能否把正确神经元排到前面、置信度落在什么区间、负样本是否老实外呼。

这是整套设计的地基：路由不准，后面所有"按需调取"都是空话。
"""

from __future__ import annotations

import sys
from typing import List, Tuple

from hub import Hub, est_tokens
from sim_session import build_session, format_turn

# (提问, 期望命中，如下的历史精确串, 事实所在轮次；expect 为空表示负样本)
PROBES: List[Tuple[str, str, int]] = [
    ("session TTL 设的是多少秒？", "SESSION_TTL_SECONDS = 25 * 60", 5),
    ("staging 数据库的主机和端口是什么？", "10.42.7.31:5432/orders_staging", 3),
    ("数据库连接串从哪个环境变量读？", "ORDERS_DB_DSN", 2),
    ("docker 宿主端口映射到哪个端口？", '"18080:8000"', 8),
    ("webhook 读超时统一调成多少毫秒？", "3500ms", 17),
    ("httpx 升级到哪个版本？", "httpx 0.27.0", 46),
    ("那个索引迁移文件叫什么？", "migrations/0007_index_orders_status.sql", 38),
    ("连接池上限最后改成多少？", "改成 8", 62),
    ("库存扣减的原子写法是什么？", "UPDATE ... WHERE stock >= qty", 20),
    ("灰度先放多少流量？", "5%", 54),
    ("镜像从 1.2G 瘦身到多少？", "380M", 70),
    ("队列重试几次后进死信？", "重试 3 次后进 DLQ", 44),
    ("CSV 导出的分批大小是多少行？", "分批 2000 行", 34),
    # 负样本：记忆里根本没有，中枢必须老实举手（否则就是幻觉）
    ("我们的 Kubernetes 集群用的是什么 CNI 插件？", "", -1),
    ("去年 Q3 的营收数字是多少？", "", -1),
    ("这个项目的 Go 版本是多少？", "", -1),
]


def build_hub() -> Hub:
    turns, _, _ = build_session()
    hub = Hub(conf_threshold=0.0, margin_threshold=0.0)  # 标定阶段关掉闸门，只看排序
    for t in turns:
        hub.write(format_turn(t), turn_index=t.index, files=t.files)
    hub._sync_index()
    return hub


def main() -> None:
    hub = build_hub()
    print("=" * 108)
    print(f"神经元: {len(hub.cells)} 个   冻结原文: {hub.total_tokens():,} tokens   "
          f"平均 {hub.total_tokens() // max(1, len(hub.cells)):,} tokens/神经元")
    print("=" * 108)
    print(f"{'提问':<30}{'top1':>7}{'sparse':>8}{'置信':>7}{'间隔':>7}{'真值排名':>8}{'真值置信':>9}   判定")
    print("-" * 108)

    pos_hit = pos_total = 0
    neg_hit = neg_total = 0
    rows_all: List[Tuple[float, str]] = []
    for q, expect, want_turn in PROBES:
        rows = hub.explain(q, top=40)
        top_cid, d, s, conf, ts, te = rows[0]
        second = rows[1][3] if len(rows) > 1 else 0.0
        margin = conf - second
        rank = "-"
        truth_conf = 0.0
        if expect:
            for i, (cid, dd, ss, cc, a, b) in enumerate(rows, 1):
                if expect in hub.cells[cid].text:
                    rank, truth_conf = str(i), cc
                    break
            pos_total += 1
            ok = rank != "-" and (int(rank) <= 3 or truth_conf >= 0.35)
            pos_hit += 1 if ok else 0
            verdict = "✅ 命中" if rank == "1" else (f"🟡 第{rank}位" if rank != "-" else "❌ 未命中")
        else:
            neg_total += 1
            # 负样本正确行为：无任何候选达到阈值（这里用 0.35 作参考线）
            ok = conf < 0.35
            neg_hit += 1 if ok else 0
            verdict = "✅ 老实外呼" if ok else f"❌ 幻觉风险({conf:.2f})"
        print(f"{q[:28]:<30}{d:>7.3f}{s:>8.2f}{conf:>7.3f}{margin:>7.3f}{rank:>8}{truth_conf:>9.3f}   {verdict}")
        rows_all.append((conf, verdict))

    print("-" * 108)
    print(f"正样本 可召回率(top3 或置信>=0.35): {pos_hit}/{pos_total}     负样本 正确拒绝: {neg_hit}/{neg_total}")

    # 给出一个参考阈值：把所有样本的 top1 置信度排开，看正负样本能否分离
    print()
    print("top1 置信度分布（用于选阈值 conf_threshold）:")
    for q, expect, wt in PROBES:
        rows = hub.explain(q, top=1)
        conf = rows[0][3] if rows else 0.0
        tag = "正样本" if expect else "负样本"
        bar = "█" * max(1, int(conf * 40))
        print(f"  {tag} {conf:>5.2f} {bar:<40} {q[:26]}")
    print()
    print("说明：正负样本的置信度若明显分层，阈值门就有意义；")
    print("      真实系统里这一层会用一次 cross-encoder 重排把分层拉得更开。")


if __name__ == "__main__":
    sys.exit(main())
