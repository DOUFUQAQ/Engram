"""
eval_holdout.py —— 留出评测集（hold-out）

纪律：**这个文件里的探针不参与任何参数调节**。调参只看 calibrate*.py 与 bench 的开发集，
最终结论只认这里的数字。否则就是在拟合噪声。

设计原则：每条探针都必须能定位到一个**唯一的**神经元，且分三类难度：
  L1 字面    查询里直接含答案的精确串（例：'那个迁移文件叫什么' → 文件名在神经元里）
  L2 概念    查询用的是同义/上位说法，答案神经元里有对应专业词（例：'登录态多久失效'）
  L3 因果    查询里**没有任何一个词**出现在答案神经元里，必须靠联想跳转（例：24 分钟掉线）
L3 是本架构的真正考题。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class EvalCase:
    query: str
    anchor: str        # 必须被召回的那条精确串
    level: str         # L1 字面 / L2 概念 / L3 因果
    note: str = ""


HOLDOUT: List[EvalCase] = [
    # ---------------- L1 字面：查询词与神经元原文大量重合 ----------------
    EvalCase("那个索引迁移文件名是什么？", "migrations/0007_index_orders_status.sql", "L1"),
    EvalCase("docker 把宿主端口映射到哪个端口？", '"18080:8000"', "L1"),
    EvalCase("健康检查用的路径是什么？", "/healthz", "L1"),
    EvalCase("支付幂等用的是哪张表？", "payment_events", "L1"),
    EvalCase("订单号前缀是怎么定的？", "ORD-", "L1"),
    EvalCase("缓存 key 的前缀是什么？", "orders:", "L1"),
    EvalCase("CI 里怎么排除慢测试？", 'pytest -m "not slow"', "L1"),
    EvalCase("镜像从 1.2G 瘦到了多少？", "380M", "L1"),

    # ---------------- L2 概念：同义/上位说法 ----------------
    EvalCase("登录态多久会失效？", "SESSION_TTL_SECONDS = 25 * 60", "L2",
             "查询说'登录态失效'，原文写的是 SESSION_TTL_SECONDS"),
    EvalCase("数据库连接串从哪个环境变量注入？", "ORDERS_DB_DSN", "L2",
             "查询说'注入'，原文是 os.environ[...]"),
    EvalCase("回调超时留了多少余量？", "3500ms", "L2",
             "查询说'余量'，原文写的是 3500ms + p99 3.1s"),
    EvalCase("连接池上限最后定成多少？", "改成 8", "L2",
             "查询说'上限'，原文是'池子 16 上限偏大…改成 8'"),
    EvalCase("灰度放量的比例是多少？", "5%", "L2"),
    EvalCase("消息消费失败几次进死信？", "重试 3 次后进 DLQ", "L2"),

    # ---------------- L3 因果：查询里没有任何词出现在答案神经元里 ----------------
    EvalCase("用户停留 24 分钟后就掉线，重新登录又好了，根因是什么？",
             "SESSION_TTL_SECONDS = 25 * 60", "L3",
             "『24 分钟/掉线/重新登录』三词在答案神经元里都不存在"),
    EvalCase("支付回调大量 504，该先去看哪个配置？",
             "3500ms", "L3",
             "『504/回调』不在答案神经元里，需从 payment 簇联想过去"),
    EvalCase("日志说数据库连接被强制中断并进入只读，是哪个参数导致的？",
             "DB_STATEMENT_TIMEOUT_MS = 5000", "L3",
             "答案神经元里没有『强制中断/只读』"),
    EvalCase("前端 20 分钟心跳和它有什么关系？",
             "20 分钟", "L3",
             "『心跳』不在答案神经元里"),
    EvalCase("为什么用户会看到『已下单但没扣到库存』的中间态？",
             "UPDATE ... WHERE stock >= qty", "L3",
             "『中间态』不在答案神经元里"),

    # ---------------- 负样本：库里根本没有，必须老实外呼 ----------------
    EvalCase("我们的 CDN 用的是哪家？", "", "NEG"),
    EvalCase("iOS 客户端最低支持版本？", "", "NEG"),
    EvalCase("团队周会定在周几？", "", "NEG"),
]


def levels() -> List[str]:
    return ["L1", "L2", "L3", "NEG"]
