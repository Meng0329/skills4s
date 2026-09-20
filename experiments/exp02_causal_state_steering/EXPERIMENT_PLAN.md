# EXP02 — 预注册实验计划

## 状态
已规划（PLANNED）

## 日期
2026-09-20

## 动机
EXP01b 支持隐藏状态索引 19 处存在对规定下一程序状态具有技能敏感性的表征，并显示较弱但与行为正相关的关联。仅可解码性本身不能确立因果用途。

## 假设
在预注册表征位点处，区分 `INSPECT_IMPLEMENTATION` 与 `INSPECT_TEST` 的方向能因果地改变模型的下一次动作偏好。

## 预注册表征位点
EXP01/EXP01b 的隐藏状态索引：

```text
19
```

对应的 Qwen 解码器模块：

```text
model.model.layers[18]
```

因为 Hugging Face 隐藏状态索引 0 是嵌入输出。

## 方向
仅在每个训练折内估计：

```text
v = mean(h_impl) - mean(h_test)
```

## 主要结局
候选动作平均对数概率边际：

```text
M = logP(src action) - logP(test action)
```

alpha = 1 时的主对称操控效应：

```text
E = [M(+1) - M(-1)] / 2
```

## 主要控制条件
- 与 `v` 正交的等范数随机方向；
- 等范数的同状态方向；
- 留出任务评估；
- 规范（canonical）与改述（paraphrase）技能。

## 探索性分析
剂量反应（dose response）：

```text
alpha ∈ {-2, -1, -0.5, +0.5, +1, +2}
```

## 成功标准
- `E_real > 0`；
- 95% 任务自助法（task-bootstrap）CI 排除 0；
- `E_real > E_same_state`；
- `E_real` 超过随机控制分布；
- 规范与改述条件下方向一致；
- 围绕零点的剂量反应大致单调。

## 解释边界
阳性结果支持已识别的残差流方向对下次动作偏好具有因果影响。它尚不能确立该方向是程序状态的完整或唯一实现。