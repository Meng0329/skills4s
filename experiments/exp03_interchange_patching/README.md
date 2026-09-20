# EXP03 — 配对互换修补（Paired Interchange Patching）

## 动机

EXP01b 发现了技能敏感的程序状态表征：

- 隐藏状态索引 19 处跨措辞 Macro-F1 = 0.709；
- 行为准确率 = 0.651。

EXP02 随后发现了**线性均值方向操控**的空结果：

- 真实对称效应 ≈ +0.00086；
- 相比同状态/随机控制无优势；
- 无有意义的行为翻转；
- 剂量反应可忽略。

因此 EXP02 拒绝了简单假设：

> 隐藏状态索引 19 处的全局线性方向足以控制下一个程序动作。

它**未**拒绝更强的可能性：程序状态是上下文依赖的，或以非线性方式嵌入。

## EXP03 问题

来自反技能条件的实际配对隐藏状态，插入同一任务后能否因果转移下一次动作偏好？

这移除了线性向量假设。

对于每个任务和措辞族：

```text
TEST-first 提示词  <----配对---->  IMPLEMENTATION-first 提示词
```

任务、历史、相关文件和工具类型完全相同。

仅技能顺序不同。

## 干预方式

对于接收者（recipient）提示词，取其同任务反技能供体（donor）的精确最后一个提示词 token 残差激活，替换接收者的激活：

```text
h_recipient^(l) <- h_donor^(l)
```

然后继续前向传播并重新评分：

```text
read_file('tests/...')
read_file('src/...')
```

这是一种互换干预（interchange intervention）/ 激活修补（activation patch）。

## 主要终点

对每次修补：

```text
margin = logP(src action) - logP(test action)
```

有符号转移效应为：

```text
+ [patched - baseline]   当供体为 IMPLEMENTATION-first
- [patched - baseline]   当供体为 TEST-first
```

正值表示接收者被转向供体规定的程序状态。

## 确证位点

EXP01/EXP01b 独立选定：

```text
隐藏状态索引 19
```

对应于：

```text
decoder block 18 输出
```

索引 19 处的效应为确证性检验。

全层扫描为探索性分析。

## 自检控制（Sanity control）

在隐藏状态索引 19 处，EXP03 还执行自修补：

```text
h_recipient <- h_recipient
```

这应产生近似为零的数值变化。非零自修补表明存在实现缺陷（bug）。

## 运行方式

在仓库根目录执行：

```bash
python experiments/exp03_interchange_patching/run.py
```

推荐：

```bash
python experiments/exp03_interchange_patching/run.py --batch-size 4
```

若显存受限：

```bash
python experiments/exp03_interchange_patching/run.py --batch-size 1
```

## 所需前置文件

```text
outputs/exp01b_counterbalanced_next_state/
├── activations.npz
├── metadata.csv
├── behavior.csv
└── run_manifest.json
```

以及：

```text
experiments/exp01b_counterbalanced_next_state/run.py
```

## 输出

```text
outputs/exp03_interchange_patching/
├── patch_results.csv
├── layer_effects.csv
├── task_effects_confirmatory.csv
├── layer_effects.png
├── run_manifest.json
└── summary.json
```

## 解释

### 若精确互换有效但 EXP02 线性操控为空

这支持：

```text
线性可解码
+
上下文依赖 / 非线性因果状态
```

而非全局可操控的线性向量。

### 若精确互换也为空

则最后一个提示词 token 的残差状态可能不是充分的中介。下一实验应检验分布式机制：

- 多个 token 位置；
- 多个相邻层；
- 注意力/MLP 通路修补；
- 跨技能处理轨迹的顺序修补。