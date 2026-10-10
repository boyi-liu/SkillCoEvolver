# SkillCoEvolver

<p align="center">
  <a href="README.md">🇬🇧 English</a>
  &nbsp;•&nbsp;
  <a href="README_zh.md">🇨🇳 中文</a>
</p>

## 工作流程

```text
任务 + 当前技能
        |
        v
设备端 LLM 选择环境动作
        |
        v
环境返回下一条观测
        |
        v
轨迹被保存，并合并到前缀树中
        |
        v
云端 LLM 诊断失败原因并更新技能树
        |
        v
更新后的技能被持久化，用于后续任务
```

## 包结构

运行时将可复用的基础行为与具体方法实现分离：

```text
src/coskill/
├── framework/     # 契约、默认运行时以及共享控制循环
├── methods/
│   └── coskill/   # CoSkill 边缘端、云端、技能、提示词和流水线
├── utils/         # 共享数据模型和运行产物记录
├── llm/           # 与模型提供商无关的 LLM 客户端接口
└── integrations/  # 可选环境适配器，例如 ALFWorld
```

## 通用框架与 CoSkill 实例化

`framework/` 定义运行时契约、默认边缘智能体/运行时、具备失败感知能力的轨迹调度器、扁平技能云端运行时、结构校验以及共享控制循环。其默认流水线会上传原始轨迹，并重写一条扁平指令。

`methods/coskill/` 在这些框架基类之上实现纠错动作重试、跨任务检索、轨迹前缀树、渐进式技能树、持久化存储以及基于证据的树补丁。方法代码只依赖 `framework/`，不会依赖其他方法包。

`CoSkill` 将上述框架与 `CoSkillCloud` 组合起来，通过跨任务技能检索、前缀树轨迹压缩以及基于证据的渐进式树补丁，改进选定的操作符。新增算法时，只需添加另一个 `methods/<name>/` 包，无需修改共享控制循环，也不需要将实现分散到各层目录中。

### 参数归属

全局运行控制参数在 `cli.py` 中组装，并直接传递给流水线，包括最大 episode 步数、是否启用演化以及运行产物输出位置。方法专属的超参数与具体流水线放在一起，由 `methods/coskill/config.py` 管理动作重试次数、上传触发阈值、前缀树渲染限制以及云端演化操作的最大数量。这样可以将基准/运行时设置与算法设置分离。

扩展实现时，请使用方法作用域内的导入，例如：

```python
from coskill.methods.coskill import EdgeAgent, CloudSkillEvolver
```

包级公开 API（`from coskill import ...`）仍可供应用使用。

### 算法选择

两个基准命令都支持 `--alg`。省略该参数（或传入空字符串）时运行 `BaseSkillEvolutionFramework`；传入 `--alg coskill` 时运行 CoSkill 专用实现。`--db` 选项仅供 CoSkill 使用。

## 在 ALFWorld 上运行

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[alfworld]'
alfworld-download
```

配置兼容 OpenAI API 的设备端和云端地址：

```bash
export COSKILL_DEVICE_BASE_URL=http://localhost:8123/v1
export COSKILL_DEVICE_API_KEY=EMPTY
export COSKILL_DEVICE_MODEL=Qwen/Qwen2.5-3B-Instruct

export COSKILL_CLOUD_BASE_URL=https://api.openai.com/v1
export COSKILL_CLOUD_API_KEY=...
export COSKILL_CLOUD_MODEL=...
```

在训练集上演化技能：

```bash
coskill alfworld --alg coskill --split train --episodes 100 \
  --db runs/alfworld-skills.db \
  --output runs/train.jsonl \
  --artifacts-dir runs/train-artifacts
```

设置 `--artifacts-dir` 后，每个 episode 结束时都会刷新产物：

- `trajectories.jsonl`：完整轨迹，包括状态、动作和观测。
- `skill_trees.jsonl`：按时间顺序保存的技能树快照。
- `skills/<task_type>.json`：每种任务类型最新的可读技能树。

在不继续演化的情况下评估已持久化的技能：

```bash
coskill alfworld --alg coskill --split eval_out_of_distribution --episodes 100 \
  --db runs/alfworld-skills.db \
  --output runs/eval.jsonl \
  --no-evolution
```

## 在 SearchQA 上运行

SearchQA 集成从本地 JSON 或 JSONL 文件读取数据，不会额外引入数据集运行时依赖。它支持原始字段（`question`、`answer` 和 `search_results.snippets`）、Hugging Face JSON 导出格式以及 MRQA 风格的 `context`/`answers` 示例。JSON 文件可以是示例列表，也可以是将 split 名称映射到示例列表的字典。

```bash
coskill searchqa --alg coskill --data data/searchqa.json \
  --split train \
  --episodes 100 \
  --db runs/searchqa-skills.db \
  --output runs/searchqa-train.jsonl \
  --artifacts-dir runs/searchqa-artifacts
```

每个问题都是一个单步 episode：边缘模型接收问题和最多 50 条检索片段，然后将最短的最终答案作为开放环境动作返回。运行结果会报告归一化 exact match 和 token F1。评估器会保留标准答案，但不会将其放入模型提示词或技能演化轨迹中。使用 `--max-snippets` 和 `--max-context-chars` 可以限制提示词上下文；使用 `--shuffle --seed 42` 可以进行确定性采样。

对于重复的 Look 任务实验，每轮使用 20 条训练轨迹，并在每 20 条轨迹之后更新技能。默认运行 5 轮；所有轮次复用同一个技能数据库：

```bash
scripts/remote_look_sweep.sh PROJECT_DIR ALFWORLD_DATA 5 look_train_sweep 20 18 CONFIG_PATH
```

因此，每一轮都会先进行一次固定规模的技能更新，再在包含 18 个任务的 `valid_unseen` Look 测试集上评估更新后的技能。测试结果会写入 `runs/<run_name>_eval_round<N>.jsonl`，并在主日志中以 `TEST_SUMMARY` 的形式汇总。评估会使用数据库副本，因此测试阶段的使用统计不会影响下一轮训练。

如需依次在后台运行全部六类 ALFWorld 任务，请使用：

```bash
nohup bash scripts/remote_all_task_sweeps.sh \
  /home/liuboyi/CoSkillEvo-Code \
  /home/liuboyi/CoSkillEvo-Code/data/alfworld \
  5 20 alfworld_all_tasks \
  > /home/liuboyi/CoSkillEvo-Code/runs/alfworld_all_tasks_launcher.log 2>&1 &
```

批处理脚本会为 Pick、Look、Clean、Heat、Cool 和 Pick-2 分别创建独立的数据库和日志；如果任意任务类型失败，脚本会停止。

