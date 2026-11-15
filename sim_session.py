"""
sim_session.py —— 一个 72 轮"编码 agent 会话"模拟器（真实体量）

为什么要模拟：想量化"40 轮后开始飘忽"，就必须有一个**细节位置可控**的长会话。
这里把真正致命的精确串（DSN、端口、环境变量名、迁移文件、字段名……）明确标成 Anchor，
且刻意埋成"很久之前写下的事实，很久之后才需要用到"——这正是压缩策略必然失败、
而保真存储 + 按需调取才能救回来的场景。

体量对齐真实编码 agent：每轮 400~700 token（讲解 + 代码块 + 工具输出），
72 轮 ≈ 3.5 万 token —— 超过多数模型的"舒适区"，于是必然触发压缩/遗忘。

会话剧情（一个订单服务项目的真实工作流）：
    T1-T8    基建：数据库接入、auth 中间件、Docker 端口映射      ← 最终事故的根因在这里
    T9-T14   下单主流程
    T15-T18  支付集成 + 升级导致的 webhook 超时
    T19-T24  库存并发
    T25-T30  日志、指标、缓存
    T31-T38  用户模块与报表导出
    T39-T44  同步/异步边界、消息队列
    T45-T50  依赖升级、测试与 CI
    T51-T60  部署、灰度、交接文档、性能
    T61-T72  池化参数、日志级别、镜像瘦身……最后 T72 追查线上事故
             —— 要正确修复，必须回忆起 T5 的 session TTL 与 T17 的超时决定
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from metrics import Anchor

# 每轮固定附带的工具输出（确定性，模拟真实噪声体量）
#
# 注意：这里**不能**出现会话里被考的关键标识符（session TTL、DSN、端口映射、池大小…）。
# 一旦它们出现在每一轮的噪声里，所有神经元都会"包含"这个事实，
# 检索就无法区分——那是模拟器的 bug，不是记忆架构的问题。真实系统里也有同样的陷阱：
# 把关键约定到处复制，等于把记忆废掉。
_TOOL_LINES = [
    "app/orders/step: helper imported ok",
    "wrote 128 bytes to .cache/index.json",
    "resolved 4 imports, 0 errors",
    "lint clean (line-length 100)",
    "types ok: no new diagnostics",
    "gc pause 12ms, heap 240MB, rss 410MB",
    "connection reused from pool (age 41s)",
    "trace sampled 1/100, span=orders.create",
    "index rebuilt over 12 files in 0.31s",
    "docker compose: api container healthy",
    "GET /metrics -> 200 (internal only)",
    "redis: cache hit, ttl 287",
    "rabbitmq: publish ok, exchange=orders.events, attempt=1",
    "pytest: 42 passed, 3 deselected, 1.84s",
    "asyncio: 200 concurrent requests finished",
]


def _code_block(idx: int) -> str:
    """每轮附带一段与剧情无关的"真实代码"，用来把上下文撑到真实体量。

    注意：每轮的片段里含有一个只在那一轮出现的常量名，方便做检索定位检查。
    """
    key = f"TURN_{idx:02d}_CONST"
    return f"""```python
# app/orders/step_{idx:02d}.py
from __future__ import annotations
import logging

log = logging.getLogger(__name__)
{key} = {idx * 7}

async def handle_step_{idx:02d}(ctx, payload):
    # 第 {idx} 步的处理入口：失败的路径必须留痕，不要静默吞掉异常。
    if payload is None:
        raise ValueError("payload required for step {idx}")
    log.info("step_{idx:02d} start", extra={{"trace_id": ctx.trace_id}})
    async with ctx.session() as sess:
        result = await sess.execute(
            "SELECT id, status FROM orders WHERE id = :oid FOR UPDATE",
            {{"oid": payload.get("order_id")}},
        )
        row = result.fetchone()
        if row is None:
            return {{"ok": False, "code": "ORDER_NOT_FOUND", "step": {idx}}}
        ctx.state[f"step_{idx:02d}"] = row[1]
    log.info("step_{idx:02d} done", extra={{"trace_id": ctx.trace_id}})
    return {{"ok": True, "step": {idx}, "const": {key}}}
```"""


def _tool_output(idx: int) -> str:
    start = (idx * 3) % len(_TOOL_LINES)
    picked = [_TOOL_LINES[(start + i) % len(_TOOL_LINES)] for i in range(15)]
    return "[tool] 运行输出：\n" + "\n".join(f"  {ln}" for ln in picked)


def _turn(idx: int, role: str, text: str, files: Tuple[str, ...] = ()) -> "Turn":
    body = f"{text}\n\n{_code_block(idx)}\n\n{_tool_output(idx)}"
    return Turn(index=idx, role=role, text=body, files=files, utterance=text)


@dataclass
class Turn:
    index: int
    role: str
    text: str                                   # 该轮完整内容（含代码与工具输出）
    files: Tuple[str, ...] = ()
    utterance: str = ""                         # 人类原话：路由时只能用这个


# ---------------------------------------------------------------------------
# 基建期：所有关键精确串（最后事故的根因就在这里）
# ---------------------------------------------------------------------------

_SEED: List[Tuple[int, str, str, Tuple[str, ...]]] = [
    (1, "user", "我们的订单服务要接一个 PostgreSQL，先配好数据库连接。生产库是只读账号，写操作走主库。", ()),
    (2, "assistant", """先定配置约定，然后建一个配置模块。

```python
# app/config.py
DATABASE_URL = os.environ["ORDERS_DB_DSN"]
DB_POOL_MIN = 4
DB_POOL_MAX = 16
DB_STATEMENT_TIMEOUT_MS = 5000
```

这个模块后面所有地方都要 import，不要各处硬编码连接串。""", ("app/config.py",)),
    (3, "user", """staging 和 prod 的库不一样，别写死。
staging 是 `10.42.7.31:5432/orders_staging`，prod 是 `10.42.7.31:5432/orders_prod`。先只连 staging。""", ()),
    (4, "assistant", """好，用环境区分，不写死主机。

```python
# app/db.py
STAGING_DSN = "postgresql://orders_rw@10.42.7.31:5432/orders_staging"
```
真实 DSN 从 `ORDERS_DB_DSN` 注入，代码里只留默认值便于本地跑。""", ("app/db.py",)),
    (5, "user", "auth 中间件也做掉，session 的 TTL 定 25 分钟，因为前端有 20 分钟的 idle 心跳，留 5 分钟缓冲。", ()),
    (6, "assistant", """写中间件。session TTL 用常量，别散落各处。

```python
# app/auth/middleware.py
SESSION_TTL_SECONDS = 25 * 60      # 与前端 20 分钟 idle 心跳留 5 分钟缓冲
SESSION_COOKIE_NAME = "orders_sid"
AUTH_HEADER = "X-Orders-Token"

async def require_session(request):
    sid = request.cookies.get(SESSION_COOKIE_NAME)
    if not sid or not store.touch(sid, ttl=SESSION_TTL_SECONDS):
        raise HTTPUnauthorized()
```

cookie 是 HttpOnly + SameSite=Lax，不设 domain，避免子域共享。""", ("app/auth/middleware.py",)),
    (7, "user", "docker 端口怎么映射的？我本地 8080 被占了。", ()),
    (8, "assistant", """把宿主端口换成 18080，容器内保持 8000 不动，不要改应用监听端口。

```yaml
# docker-compose.yml
services:
  api:
    ports:
      - "18080:8000"     # 宿主 18080 -> 容器 8000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/healthz"]
```

本地访问是 http://localhost:18080，健康检查在 /healthz。""", ("docker-compose.yml",)),
]

# ---------------------------------------------------------------------------
# 中段：大量与最终事故无关的工作（把 T5/T17 推远）
# ---------------------------------------------------------------------------

_MID: List[Tuple[int, str, str, Tuple[str, ...]]] = [
    (9, "user", "开始做下单主流程：创建订单、扣库存、写流水，三段式，任何一步失败都要回滚。", ("app/orders/service.py",)),
    (10, "assistant", "写成事务性服务，库存扣减先用 `SELECT ... FOR UPDATE` 悲观锁跑通，再优化。", ("app/orders/service.py",)),
    (11, "user", "订单号规则是什么？我记得以前是时间戳前缀。", ()),
    (12, "assistant", "别用时间戳前缀，会泄露业务量。用 `ORD-` + ULID，数据库里再加唯一索引。", ("app/orders/ids.py",)),
    (13, "user", "支付接进来，先用 Stripe 的 test key，别碰生产。", ()),
    (14, "assistant", "支付适配器抽一层，handler 只做编排，SDK 调用收在 gateway 内。", ("app/payments/gateway.py",)),
    (15, "user", "webhook 那边老是重试，你看下幂等怎么做的。", ("app/payments/webhook.py",)),
    (16, "assistant", "现在只按 event id 去重，缺订单维度，补 `payment_events` 唯一约束。", ("app/payments/webhook.py",)),
    (17, "user", "另外 webhook 超时统一调成 3500ms，支付方 p99 是 3.1s，留 400ms 余量。以后回调变慢、出现 504，多半是这个数不够。", ()),
    (18, "assistant", "好，超时统一 3500ms，且必须是**读超时**，连接超时保持 800ms 别混。回调超时升级时优先看这里。", ("app/payments/http.py",)),
    (19, "user", "库存并发压测出现超卖，看下哪里的锁没生效。", ("app/inventory/service.py",)),
    (20, "assistant", "问题是先查后改、不在同一事务加锁。改成 `UPDATE ... WHERE stock >= qty` 的原子扣减。", ("app/inventory/service.py",)),
    (21, "user", "对，别用乐观锁重试，高并发下重试风暴更糟。", ()),
    (22, "assistant", "同意，原子扣减 + 库存不足直接返回 `INSUFFICIENT_STOCK`。", ()),
    (23, "user", "日志加 trace id，要能串起一次下单的全部日志。", ("app/observability/logging.py",)),
    (24, "assistant", "用 contextvar 存 trace id，中间件入口生成，日志 formatter 自动带。", ("app/observability/logging.py",)),
    (25, "user", "指标也加：下单成功率、扣库存耗时、支付回调延迟。", ("app/observability/metrics.py",)),
    (26, "assistant", "暴露在 `/metrics`，只在内网端口，不要挂到公网路由上。", ("app/observability/metrics.py",)),
    (27, "user", "缓存用 Redis，key 前缀统一一下，别和别的服务撞。", ()),
    (28, "assistant", "前缀用 `orders:`，商品详情缓存 300 秒，订单快照不缓存（必须实时）。", ("app/cache/keys.py",)),
    (29, "user", "缓存击穿怎么办？热 key 过期瞬间打爆。", ()),
    (30, "assistant", "单飞 + 逻辑过期：热 key 加 60 秒逻辑过期窗口，后台异步刷新。", ("app/cache/singleflight.py",)),
    (31, "user", "用户模块要加手机号，注意脱敏。", ("app/users/models.py",)),
    (32, "assistant", "手机号存明文 + 单独映射表加密，展示层统一走脱敏函数。", ("app/users/privacy.py",)),
    (33, "user", "导出报表要支持 10 万行，别把内存打爆。", ()),
    (34, "assistant", "用服务端游标流式写 CSV，不要一次 fetchall，分批 2000 行刷出。", ("app/reports/export.py",)),
    (35, "user", "CSV 里的逗号和换行要转义，Excel 打开会错行。", ()),
    (36, "assistant", "字段统一引号包裹 + 内部引号翻倍，BOM 头保留给 Excel 识别 UTF-8。", ("app/reports/export.py",)),
    (37, "user", "这个查询太慢了，看下执行计划。", ()),
    (38, "assistant", "缺复合索引，按 (status, created_at DESC) 建一个，扫行从 20 万降到 800。", ("migrations/0007_index_orders_status.sql",)),
    (39, "user", "把下单接口拆成两段，先落单再异步扣库存？", ()),
    (40, "assistant", "不建议，用户会看到「已下单但没扣到库存」的中间态。保持同步事务，只把通知异步化。", ()),
    (41, "user", "那通知走消息队列，用 RabbitMQ 还是 Kafka？", ()),
    (42, "assistant", "通知这种低吞吐高可靠场景用 RabbitMQ 就够，Kafka 留给埋点流。", ("app/mq/publisher.py",)),
    (43, "user", "消息要保证不丢，消费者挂了怎么办？", ()),
    (44, "assistant", "手动 ack + 死信队列，重试 3 次后进 DLQ，别无限重试。", ("app/mq/consumer.py",)),
    (45, "user", "依赖升一下，之前那个 HTTP 库有 CVE。", ()),
    (46, "assistant", "httpx 升到 0.27.0，注意这版默认超时从 5s 变成 None，必须显式传超时。", ("requirements.txt",)),
    (47, "user", "测试补一下，重点测并发扣库存。", ()),
    (48, "assistant", "用 asyncio.gather 并发 200 个请求打同一个 SKU，断言最终库存不为负。", ("tests/test_inventory_concurrency.py",)),
    (49, "user", "CI 里跑这些测试太慢，怎么办。", ()),
    (50, "assistant", '慢的并发测试放 nightly，PR 只跑快集，用 marker 区分：`pytest -m "not slow"`。', (".github/workflows/ci.yml",)),
    (51, "user", "部署脚本要支持回滚，别一键上线没有退路。", ()),
    (52, "assistant", "每次发布保留最近 5 个镜像 tag，回滚就是改 tag 重新拉取，不用重建。", ("deploy/rollout.sh",)),
    (53, "user", "灰度怎么做？", ()),
    (54, "assistant", "按 cookie 哈希分桶，先 5% 观察 30 分钟，错误率超 1% 自动回退。", ("app/flags/rollout.py",)),
    (55, "user", "feature flag 用第三方还是自建？", ()),
    (56, "assistant", "自建一张表够用，别引入第三方 SDK 把启动路径搞复杂。", ("app/flags/store.py",)),
    (57, "user", "写个交接文档，把关键决策记一下。", ()),
    (58, "assistant", "记录了事务边界、锁策略、超时约定三类决策，放在 docs/decisions.md。", ("docs/decisions.md",)),
    (59, "user", "性能优化一轮，P95 目标 200ms 以内。", ()),
    (60, "assistant", "主要耗时在三次串行查询，改一次 JOIN 后 P95 从 410ms 降到 165ms。", ("app/orders/repository.py",)),
    (61, "user", "连接池参数也调一下，staging 只有 100 个连接上限。", ()),
    (62, "assistant", "池子 16 上限偏大，多实例会打满；改成 8，并加 3 秒获取超时快速失败。", ("app/db.py",)),
    (63, "user", "日志级别生产别开 DEBUG，量太大。", ()),
    (64, "assistant", "生产 INFO，慢查询单独 WARN，采样率 1/100 打点避免抖动。", ()),
    (65, "user", "这个项目最后要交给别人维护，注意可读性。", ()),
    (66, "assistant", "把隐式约定都写成显式常量，特别是超时和 TTL 这类魔法数。", ()),
    (67, "user", "migration 要不要做成自动执行？", ()),
    (68, "assistant", "不要，迁移必须人工确认后执行，自动迁移出过事故。", ()),
    (69, "user", "打包镜像瘦身一下，现在 1.2G。", ()),
    (70, "assistant", "多阶段构建 + 只装运行时依赖，降到 380M。", ("Dockerfile",)),
    (71, "user", "好，先这样。", ()),
]


# ---------------------------------------------------------------------------
# 最后一轮：事故复盘，必须回忆起很久以前的两个决定
# ---------------------------------------------------------------------------

PROBE_TURN = Turn(
    index=72, role="user", utterance="""线上出事了，你看一下：
用户投诉"登录后过一会儿就掉线，重新登录又好了"，同时支付回调开始大量 504。
复现路径：登录 → 停留约 24 分钟 → 点支付 → 掉线。
日志里有 ORDERS_DB_DSN 对应的连接被强制中断的记录，也有 webhook 超时。
请给出根因和修复方案。""",
    text="""线上出事了，你看一下：
用户投诉"登录后过一会儿就掉线，重新登录又好了"，同时支付回调开始大量 504。
复现路径：登录 → 停留约 24 分钟 → 点支付 → 掉线。
日志里有 ORDERS_DB_DSN 对应的连接被强制中断的记录，也有 webhook 超时。
请给出根因和修复方案。""",
)

# 正确回答必须用到的历史精确串（全部来自很早的轮次）
ANCHORS: List[Anchor] = [
    # 基建期 T2-T8：最终事故的根因藏在这里
    Anchor(2, "ORDERS_DB_DSN"),
    Anchor(2, "DB_STATEMENT_TIMEOUT_MS = 5000"),
    Anchor(3, "10.42.7.31:5432/orders_staging"),
    Anchor(5, "SESSION_TTL_SECONDS = 25 * 60"),
    Anchor(5, "20 分钟"),
    Anchor(6, 'SESSION_COOKIE_NAME = "orders_sid"'),
    Anchor(6, "X-Orders-Token"),
    Anchor(6, "HttpOnly"),
    Anchor(8, '"18080:8000"'),
    Anchor(8, "/healthz"),
    # webhook 超时约定 T16-T18
    Anchor(16, "payment_events"),
    Anchor(17, "3500ms"),
    Anchor(18, "连接超时保持 800ms"),
    # 中段关键决定（验证稀疏激活不会把无关内容拖回来）
    Anchor(12, "ORD-"),
    Anchor(20, "UPDATE ... WHERE stock >= qty"),
    Anchor(28, "orders:"),
    Anchor(30, "逻辑过期"),
    Anchor(38, "migrations/0007_index_orders_status.sql"),
    Anchor(46, "httpx 0.27.0"),
    Anchor(50, 'pytest -m "not slow"'),
    Anchor(54, "5%"),
    Anchor(62, "改成 8"),
    Anchor(70, "380M"),
]


def build_session() -> Tuple[List[Turn], Turn, List[Anchor]]:
    turns: List[Turn] = [_turn(i, r, t, f) for i, r, t, f in _SEED]
    turns += [_turn(i, r, t, f) for i, r, t, f in _MID]
    return turns, PROBE_TURN, ANCHORS


# 解决第 72 轮事故**真正需要**的那几条事实（工作集）。
# 主基准里 23 条锚点是"任意事实都能召回吗"的检索能力拷问；
# 而"能不能解决这次事故"只取决于下面这几条能不能进上下文。
# 两个指标必须分开报，否则会用不可能达到的目标去否定一个可用的系统。
PROBE_WORKING_SET: List[str] = [
    "SESSION_TTL_SECONDS = 25 * 60",      # 根因：TTL 25 分钟 vs 前端 20 分钟心跳
    "20 分钟",
    "ORDERS_DB_DSN",                      # 日志里点名的连接
    "3500ms",                             # 504 的直接原因
    "连接超时保持 800ms",
    "payment_events",
]


def format_turn(t: Turn) -> str:
    f = f" files={','.join(t.files)}" if t.files else ""
    return f"### T{t.index} <{t.role}>{f}\n{t.text}"
