# EXP02 — 因果程序状态操控（Causal Procedural-State Steering）

## 研究问题

EXP01b 中识别的程序状态方向是否**因果控制**模型的下一次动作偏好？

EXP01b 已表明技能规定的下一步状态：

- 可在隐藏状态索引 19 处进行跨任务/跨措辞解码；
- 与下一次动作偏好相关联。

EXP02 从关联（association）走向干预（intervention）。

## 重要的层映射

Hugging Face `hidden_states` 包含：

```text
hidden_states[0]  = 嵌入输出
hidden_states[1]  = decoder block 0 之后
...
hidden_states[19] = decoder block 18 之后
```

因此预注册的 EXP01/EXP01b 隐藏状态索引 19 映射到：

```text
model.model.layers[18]
```

EXP02 显式记录两个索引，以防差一（off-by-one）错误。

## 因果方向

对于每个留出任务折，方向**仅使用 EXP01b 中的训练任务**计算：

```text
v = mean(h_impl) - mean(h_test)
```

位于隐藏状态索引 19 处。

然后将该方向注入 decoder block 18 的最后一个提示词 token。

## 主要因果终点

对每个提示词：

```text
margin =
mean_logP(read_file(src/...))
-
mean_logP(read_file(tests/...))
```

正值表示偏向实现（implementation）检查。

预注册主效应：

```text
E_real =
mean( margin(alpha=+1) - margin(alpha=-1) ) / 2
```

预期：

```text
E_real > 0
```

且显著大于匹配的控制条件。

## 控制条件

1. **正交随机方向**
   - 随机向量；
   - 显式与真实方向正交化；
   - 范数与真实方向匹配。

2. **同状态方向**
   - 由实现（IMPLEMENTATION）状态训练样本的两半构造；
   - 范数与真实方向匹配；
   - 捕捉状态内任意的激活变异。

3. **留出任务**
   - 方向仅从训练任务学习；
   - 干预仅在留出任务上评估。

4. **跨措辞**
   - 规范（canonical）与改述（paraphrase）技能均被评估。

5. **剂量反应**
   - 真实方向：alpha = -2, -1, -0.5, +0.5, +1, +2；
   - alpha = ±1 是预注册的主要比较；
   - 更大/更小的系数为探索性分析。

6. **多重随机控制**
   - 默认五个独立随机方向。

## 为什么使用原生 PyTorch hooks？

本实验使用 Qwen 解码器块输出的标准前向钩子（forward hook），而不是引入可解释性框架依赖。

它精确实现了我们需要的干预：

```text
hidden[last_prompt_token] += alpha * direction
```

如果 EXP02 成功，后续实验可采纳 NNsight/pyvene 进行更复杂的多组件修补。

## 所需前置输出

EXP02 读取：

```text
outputs/exp01b_counterbalanced_next_state/
├── activations.npz
├── metadata.csv
└── run_manifest.json
```

原始的 EXP01b 激活仅用于估计方向。不使用任何 Hugging Face 网络访问。

## 运行方式

在仓库根目录执行：

```bash
python experiments/exp02_causal_state_steering/run.py
```

推荐：

```bash
python experiments/exp02_causal_state_steering/run.py \
  --batch-size 4 \
  --num-random-controls 5
```

若 GPU 显存紧张：

```bash
python experiments/exp02_causal_state_steering/run.py --batch-size 1
```

## 输出

```text
outputs/exp02_causal_state_steering/
├── intervention_results.csv
├── fold_directions.csv
├── primary_effects_by_task.csv
├── random_control_effects.csv
├── dose_response.csv
├── dose_response.png
├── primary_effects.png
├── run_manifest.json
└── summary.json
```

## 进行 / 停止（Go / No-Go）标准

强支持需要满足以下全部条件：

1. 真实方向主效应具有预测的符号；
2. 任务自助法 95% CI 排除零；
3. 真实效应超过同状态控制；
4. 真实效应超过正交随机控制分布；
5. 剂量反应在 alpha = 0 附近大致单调；
6. 规范与改述技能中均出现该效应。

若这些成立，则项目拥有因果证据，表明线性识别的程序状态方向参与控制下一次动作偏好。