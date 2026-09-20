# 实验 01 — 程序阶段探测（Procedural Stage Probe）

## 问题

自然语言 Agent 技能（Agent Skill）能否诱导出模型对**当前程序阶段**的内部表征，且这种表征无法仅由轨迹位置或下一工具身份解释？

这是一个表征发现实验。本实验**不**进行激活操控（activation steering）。

## 仅限本地模型

请将一个或多个 Hugging Face 格式的模型目录放在仓库根目录下：

```text
skills4s/
└── models/
    └── Qwen2.5-Coder-7B-Instruct/
        ├── config.json
        ├── tokenizer_config.json
        ├── tokenizer.json
        ├── model-00001-of-....safetensors
        └── ...
```

不使用 Hugging Face 下载。脚本设置了离线模式，调用
`from_pretrained(..., local_files_only=True)`。

若仅找到一个本地模型，直接运行：

```bash
python experiments/exp01_stage_probe/run.py
```

若有多个本地模型：

```bash
python experiments/exp01_stage_probe/run.py \
  --model models/Qwen2.5-Coder-7B-Instruct
```

首次正式运行推荐：

```bash
python experiments/exp01_stage_probe/run.py --num-tasks 12
```

## 实验条件

对于每个任务/阶段组合，模型看到相同任务和相同的教师强制（teacher-forced）轨迹历史，分为三个条件：

1. `no_skill`（无技能）
2. `correct_skill`（正确技能）
3. `shuffled_skill`（顺序打乱技能）

`shuffled_skill` 包含与正确技能相同的程序块，但顺序被固定打乱。因此主要对照为：

```text
Δ_order = h(correct_skill) - h(shuffled_skill)
```

这比仅使用 `Skill - NoSkill` 更强，因为它减少了上下文长度/内容的混淆变量。

## 控制条件

六个程序阶段为：

```text
REPRODUCE
INSPECT_TEST
INSPECT_IMPLEMENTATION
REPAIR
TARGET_VERIFY
REGRESSION_VERIFY
```

同工具对照：

```text
INSPECT_TEST           vs INSPECT_IMPLEMENTATION  -> read_file vs read_file
TARGET_VERIFY          vs REGRESSION_VERIFY       -> run_tests vs run_tests
```

历史中插入了无关工具调用，因此程序阶段不等同于原始工具调用索引。

## 输出

```text
outputs/exp01_stage_probe/
├── activations.npz
├── metadata.csv
├── layer_probe.csv
├── position_baseline.csv
├── same_tool_results.csv
├── confusion_matrix_best_layer.csv
├── layer_probe.png
└── summary.json
```

做出首次科学判断时应使用：

- `best_order_delta_macro_f1`
- `best_layer`
- `position_baseline_macro_f1`
- 两组同工具对照的 F1 值
- 逐层曲线

六分类阶段基线准确率约为 `0.167`。