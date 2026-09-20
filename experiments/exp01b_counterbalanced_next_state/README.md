# EXP01b — 反平衡技能受控的下一步状态解码

## 目标

检验仅改变技能（Skill）中规定的程序，是否会改变模型对**规定下一步程序状态**的内部表征。

EXP01 发现阶段可解码性很强，但「无技能」条件同样高度可解码。因此 EXP01b 将任务/历史/位置/工具身份保持恒定，仅改变技能步骤顺序。

## 确证层面（Confirmatory layer）

隐藏状态索引 **19**，由 EXP01 独立选定。全层扫描为探索性分析。

## 对照

在失败被复现后立即评估：

- 测试优先（TEST-first）技能：下一步状态 = `INSPECT_TEST`
- 实现优先（IMPLEMENTATION-first）技能：下一步状态 = `INSPECT_IMPLEMENTATION`

两种下一步动作均使用 `read_file`；仅文件参数不同。

使用两种措辞族：

- `canonical`：显式阶段标签
- `paraphrase`：不含这些标签的独立措辞

关键做法：探针在一个措辞族上训练，在另一个措辞族上测试，并整体留出整任务（hold out whole tasks）。

## 运行方式

在仓库根目录执行：

```bash
python experiments/exp01b_counterbalanced_next_state/run.py --num-tasks 48
```

脚本仅使用 `./models/` 下的本地 Hugging Face 格式权重。

## 输出

`outputs/exp01b_counterbalanced_next_state/` 包含：

- `summary.json`
- `run_manifest.json`
- `metadata.csv`
- `behavior.csv`
- `no_skill_behavior.csv`
- `layer_probe.csv`
- `control_probe.csv`
- `confirmatory_confusion_matrix.csv`
- `layer_probe.png`
- `activations.npz`
- `stimuli.json`

## 主要成功模式

1. 层面 19 的跨措辞 Macro-F1 显著高于 0.5。
2. 随机任务特定映射控制接近随机水平。
3. 技能顺序按规定方向反转下一动作的对数概率。
4. 留出探针得分与留出动作偏好相关。