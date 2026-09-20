# EXP01b — 实验记录（预注册计划）

## 状态
已规划（PLANNED）

## 动机
EXP01 发现程序阶段信息高度可解码，但「无技能」（No-Skill）条件下的 F1 也达到 0.956。因此 EXP01 未能分离出由技能（Skill）引发的程序状态（procedural state）。

## 研究问题
当任务、历史、位置和工具身份保持恒定时，技能内的程序步骤顺序是否改变模型对**规定下一步程序状态**的内部表征？

## 预注册的主层面
隐藏状态索引 **19**，由 EXP01 独立选定。

## 主对照
`INSPECT_TEST` vs `INSPECT_IMPLEMENTATION`；两者均使用 `read_file`。

## 控制条件
- 每一对比较中任务/历史完全相同
- 相关文件顺序进行反平衡（counterbalanced）
- 内容完全相同但顺序打乱的规范技能（canonical Skills）
- 独立的改述技能族（paraphrased Skill family）
- 跨任务 + 跨措辞评估
- 随机化任务特定标签映射控制
- 候选下一步动作的对数概率（log-probabilities）

## 下一步决策
如果表征与行为关联测试均为阳性，则进入因果状态操控（causal state steering）/ 激活修补（activation patching）。