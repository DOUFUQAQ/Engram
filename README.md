# memory —— 「神经元 + 神经中枢」记忆架构原型

一个针对"编码 agent 跑到 40 轮就飘忽"的可运行原型。
核心主张：**记忆的存储必须保真（神经元冻结原文），压缩只发生在当前工作区，
且必须能指回原件（指针内存）。**

完整设计、实测数据、以及这个想法的边界 → **[DESIGN.md](DESIGN.md)**

## 快速开始

```bash
cd F:\memory

# 真嵌入 / LLM 重排（可选，见 .env.example）
copy .env.example .env      # Windows   （Linux/mac: cp .env.example .env）
#   然后填入 EMBED_*（推荐 BGE-M3）与 DEEPSEEK_*
python run_experiment.py --check          # 先做组件连通性体检
python run_experiment.py                  # A/B：offline vs 真嵌入 vs +LLM重排

# 离线即可跑（不需要任何 key）
python bench_memory.py                 # 主实验：四种上下文策略对比
python bench_memory.py --only cells --verbose   # 只看本方案，并打印最终打包内容
python bench_memory.py --recall-loop   # 记忆工具闭环（收益最大的一步）
python bench_memory.py --sweep-cold    # 冷区数量的成本-收益曲线
python evaluate.py                     # 留出集：四条管线的排名对比（L1/L2/L3/负样本）
python evaluate.py -v                  # 逐条看命中情况
python calibrate.py                    # 路由标定：正负样本能否分层
python calibrate2.py                   # 参数扫描：稠密/稀疏/覆盖率权重与阈值
```

`.env` 已在 `.gitignore` 里，密钥不会被提交。

## 实测结果

### 主基准（72 轮会话，23 条精确细节陷阱）

| 策略 | 神经元级召回 | 精确串存活 | 上下文 tokens | 中枢延迟 |
|---|---|---|---|---|
| `full` 全量原文 | 100% | 96% | 31,541 | – |
| `compaction` 压缩摘要 | – | **4%** | 1,903 | – |
| `recency` 只留最近 6 轮 | – | **4%** | 2,597 | – |
| `cells` 被动打包 | 30% | **35%** | 6,528 | **3 ms** |
| `cells` + 记忆工具 | 57% | **74%** | 20,867 | 3 ms |

### 事故工作集（解决第 72 轮事故真正需要的 6 条事实）

| 管线 | 工作集命中 | 上下文 tokens |
|---|---|---|
| 被动打包（单次融合查询） | **3/6** | 8,894 |
| 被动打包 + 模型主动 recall 三次 | **6/6** | 20,867 |

> **这是整份文档最重要的数字**：一次融合查询召回不全，让模型分三次聚焦提问就全中。

### 留出集（22 条探针，不参与调参，看正确神经元排第几）

| 管线 | L1 字面 @6 | L2 概念 @6 | L3 因果 @6 | MRR | NEG 拒绝 |
|---|---|---|---|---|---|
| `single` 单指针 | 75% | 100% | 100% | 0.44 | 3/3 |
| `multi` 多路脑电波 | 100% | 100% | 100% | 0.55 | 3/3 |
| `multi + 二跳联想` | 100% | 100% | 100% | **0.62** | 3/3 |
| `multi + 二跳 + 扩展` | 100% | 100% | 100% | 0.64 | **2/3** |

查询扩展 MRR 只 +0.02（在噪声范围内），而负样本拒绝率在多次运行中不稳定
（开发集上观察到 2/3 ~ 3/3 波动）——**收益不显著 + 安全指标可能回退，所以默认关闭。**

## 三级降级策略

| 级别 | 动作 | 实测成本 | 何时用 |
|---|---|---|---|
| L1 被动打包 | 热区原文 + 按需调取 4 个神经元 | ~6.5k token | 90% 的轮次 |
| L2 记忆工具 | 模型主动 `recall` 1~3 次 | +4k/次 | 需要很久之前的事实，或模型自述信息不足 |
| L3 外呼 | 联网 / 重新推理 / 问用户 | 一次往返 | 置信门判定库里没有 |

## 文件地图

| 文件 | 职责 |
|---|---|
| [hub.py](hub.py) | **核心**：`Cell`（神经元）、`Hub`（神经中枢）、置信门、外呼、突触、巩固、遗忘、二跳扩散 |
| [context_packer.py](context_packer.py) | 工作区打包（热区+冷区+指针目录）、**记忆工具 `recall`**、四种策略同接口 |
| [metrics.py](metrics.py) | 指标：精确串存活率 |
| [sim_session.py](sim_session.py) | 72 轮模拟编码会话 + 事故工作集定义 |
| [bench_memory.py](bench_memory.py) | 主实验 / 记忆工具闭环 / 冷区扫描 |
| [eval_holdout.py](eval_holdout.py) | 留出探针（L1 字面 / L2 概念 / L3 因果 / 负样本） |
| [evaluate.py](evaluate.py) | 留出集评测：四条管线的排名对比 |
| [calibrate.py](calibrate.py) [calibrate2.py](calibrate2.py) | 路由标定与参数扫描 |
| [DESIGN.md](DESIGN.md) | 设计图纸、机制归因、边界与落地建议 |

## 最小用法

```python
from hub import Hub
from context_packer import Packer

hub = Hub(external_fn=my_web_search)          # 置信不足时自动外呼
writer = Packer(hub, budget_tokens=9000, hot_turns=6, max_cold=4)

hub.write(turn_text, turn_index=i, files=["app/db.py"])   # 每轮：保真冻结
hub.consolidate(); hub.decay()                            # 定时：巩固 + 遗忘

packed = writer.cells(current_task, current_turn=i)       # L1 被动打包
print(packed.trace.describe())                            # 命中谁 / 置信 / 是否外呼 / 耗时

rp = writer.recall("session TTL 过期时间是多少？")          # L2 记忆工具（暴露给模型）
```

## 四个已踩过的坑（别重复踩）

1. **噪声污染神经元**：关键约定出现在每一轮的工具输出里 → 所有神经元"都包含"它 → 检索失去分辨力。
2. **巩固过度合并**：只按相似度合并会把 34 个神经元糊成 5 个，检索粒度归零。
   必须"时间相邻 + 互为最近邻 + 罕见词主题重叠 + 合并后不超容量"。
3. **热区吃光预算**：热区上限必须留出至少 40% 给按需调取，否则召回率被预算限制而非算法限制。
4. **在弱信号上反复调阈值 / 用错评测口径**：当前打分函数分辨率不足（最优 AUC 0.75–0.84，
   稠密路权重应为 0）。更严重的是第一版评测里"闸门拦下就全库重排兜底"，让基线虚高到 19/22 ——
   **评测口径错了，优化方向必错。** 改成度量排名后谁都无法刷分。
