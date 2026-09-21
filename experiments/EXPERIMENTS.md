# skills4s — 实验日志

本文件是项目级的科学记录。

原则：

- 在解释结果**之前**记录假设。
- 区分观察、解释与因果主张。
- 保留空结果与阴性结果。
- 将每个已完成的实验与其 Git 提交关联。
- 不将主张升级到超出干预实际所确立的范围。
- 将探索性层扫描与预注册/确证性检验分开对待。

---

## 实验索引

| 实验 | 核心问题 | 状态 | 关键结果 | Commit |
|---|---|---|---|---|
| EXP01 | 程序阶段是否可解码？ | 完成 | H19 `order_delta` F1=1.000，但 No-Skill F1=0.956 | `ed1de55` |
| EXP01b | Skill 规定的下一状态是否跨措辞可解码？ | 完成 | cross-wording F1=0.709；行为准确率=0.651 | `62aee2c` |
| EXP02 | H19 全局线性方向是否可因果 steering？ | 完成，NULL | effect≈+0.00086；不优于控制 | `20dba87` |
| EXP03 | 单 token exact interchange 是否可转移状态？ | 完成，NULL/负 | H19 弱；深层负效应；部分深层结果受 capture/patch 位置错位影响 | `e98dade` |
| EXP04 | 多 token × 多层一致恢复是否恢复因果效应？ | 完成，POSITIVE | `common_H19_H28=+0.0579 [0.0523,0.0636]`；`H15_H28=+0.0796`；控制≈0 | `bbabe94` 等 |
| EXP05 | H15–H20 是否作为 selector，重配置后续 attention/MLP？ | 完成，POSITIVE | `early_H15_H20=+0.0784 [0.0717,0.0849]`；selector=98.4% 全效应；late≈0；**MLP recovery 0.875 >> attn 0.700** | `8e70ce0` |
| EXP06 | 选中的 H21–H28 MLP 神经元组是否因果中介？ | 完成，NEGATIVE/反转 | top256 suff/nec 均 ≈ −0.0014（负）；全 MLP suff −0.0141；K 单调走负；**MLP 神经元既非充分也非必要** | `4b4b47c` |
| EXP07 | H21–H28 注意力头输出是否因果中介？ | 完成，NEGATIVE/部分正 | AllAttention suff/nec −0.015/−0.016（负）；Top16 suff≈0，nec +0.0018（弱正）；K≥32 负；**注意力头输出不构成可移植的充分性中介** | `5ab0009` |

---

# EXP01 — 程序阶段探测

## 日期
2026-09-20

## 提交
`ed1de55`

## 状态
已完成

## 研究问题

在技能条件化的编码智能体（coding-agent）场景中，当前程序阶段能否从隐藏表征中被线性解码？

## 假设

技能条件化的隐藏状态包含关于当前程序阶段的线性可解码信息，且超越轨迹位置与工具身份。

## 设计

条件：

- 无技能（No Skill）
- 正确技能（Correct Skill）
- 顺序打乱技能（Shuffled Skill）

六个阶段：

1. `REPRODUCE`
2. `INSPECT_TEST`
3. `INSPECT_IMPLEMENTATION`
4. `REPAIR`
5. `TARGET_VERIFY`
6. `REGRESSION_VERIFY`

控制条件包括：

- 按任务分组交叉验证；
- 轨迹位置基线；
- 同工具阶段对。

## 主要结果

- 最佳隐藏状态索引：**19**
- `correct_skill - shuffled_skill` Macro-F1：**1.000**
- 正确技能 F1：**0.978**
- 打乱技能 F1：**0.978**
- 无技能 F1：**0.956**
- 位置基线：**0.422**

同工具对照：

- `INSPECT_TEST` vs `INSPECT_IMPLEMENTATION`：**1.000**
- `TARGET_VERIFY` vs `REGRESSION_VERIFY`：**1.000**

## 观察

程序阶段信息被极其强烈地线性解码，尤其是在 H19 附近。

## 关键混淆变量

无技能表征已经达到：

```text
Macro-F1 = 0.956
```

因此 EXP01 **不能**确立该表征是由技能诱导的。

教师强制（teacher-forced）执行历史本身携带关于当前程序阶段的强语义证据。

## 已支持的主张

- 程序阶段被强烈表征。
- 该表征在留出的合成任务上泛化。
- 它不能还原为下一工具身份。
- 原始位置特征不能解释全部信号。

## 未确立

- 技能因果地创建该表征。
- 技能顺序因果地控制该表征。
- H19 是因果控制位点。
- 该表征对行为是必要的。

## 决策

进入反平衡设计：任务/历史/工具身份保持固定，仅技能规定的下一状态改变。

---

# EXP01b — 反平衡技能受控的下一步状态解码

## 日期
2026-09-20

## 提交
`62aee2c`

## 状态
已完成

## 动机

EXP01 显示阶段可解码性很强，但也存在很强的无技能信号。

EXP01b 通过保持恒定来消除此混淆：

- 任务；
- 失败观察；
- 执行历史；
- 当前位置；
- 相关文件；
- 下一工具类型。

仅技能定义的程序顺序改变。

## 主对照

```text
INSPECT_TEST
vs
INSPECT_IMPLEMENTATION
```

两者均要求：

```text
read_file(...)
```

仅文件目标不同：

```text
tests/...
vs
src/...
```

## 跨措辞设计

两种措辞族：

- 规范（canonical）；
- 改述（paraphrased）。

主要评估跨两者：

- 留出任务；
- 留出措辞族。

基于 EXP01，H19 被作为确证性表征位点。

## 主要结果

### 表征

跨措辞 Macro-F1：

```text
0.709
```

随机基线：

```text
0.500
```

### 随机任务特定映射控制

平均 Macro-F1：

```text
0.458
```

这保持在随机水平附近，低于真实语义标签探针。

### 行为偏好

总体下一次动作准确率：

```text
0.651
```

规范措辞：

```text
0.719
```

表征–行为皮尔逊相关：

```text
r = 0.345
```

### 无技能先验

无技能时，模型在：

```text
97.9%
```

的情况下选择实现/源码检查。

这是一个很强的固有「实现优先」先验。

## 解释

EXP01b 支持一种技能敏感的对规定下一程序状态的表征，该表征跨任务与措辞泛化。

它也提供了行为层面的证据：技能顺序会改变动作偏好。

然而：

```text
可解码表征
!=
因果控制
```

表征层面证据强于行为层面效应。

## 已支持的主张

1. 技能规定的下一状态是线性可解码的。
2. 该表征在留出任务上泛化。
3. 它在显著不同的技能措辞上泛化。
4. 该效应不能由工具身份解释。
5. 任意的任务特定标签映射不能类似地泛化。
6. 技能顺序可测地改变下一次动作偏好。

## 未确立

- H19 是必要的。
- 解码方向被因果使用。
- 该表征足以重定向行为。
- 单一线性状态变量实现了该程序。

## 决策

进入直接因果干预。

---

# EXP02 — 因果程序状态操控

## 日期
2026-09-20

## 提交
`20dba87`

## 状态
已完成 — 空结果

## 研究问题

线性识别的 H19 程序状态方向是否足以因果地改变模型的下一次动作偏好？

## 重要的层映射

Hugging Face：

```text
hidden_states[19]
=
decoder block 18 的输出
```

因为：

```text
hidden_states[0]
=
嵌入输出
```

## 干预

在每个训练折内：

```text
v =
mean(h_IMPLEMENTATION)
-
mean(h_TEST)
```

在留出任务上：

```text
h' = h + alpha * v
```

主要行为指标：

```text
M =
mean_logP(src action)
-
mean_logP(test action)
```

主对称操控效应：

```text
E =
[M(+1) - M(-1)] / 2
```

## 控制条件

- 同状态方向；
- 等范数正交随机方向；
- 留出任务构造；
- 规范/改述措辞；
- 正/负操控；
- 剂量反应。

## 主要结果

| 标准 | 预期 | 观察 | 结果 |
|---|---:|---:|---|
| `E_real > 0` | 明显为正 | `+0.00086` | 可忽略 |
| 真实 > 同状态 | 明显更强 | 真实减同状态 ≈ `-0.00030` | 失败 |
| 真实 > 随机 | 明显更强 | 经验 `p = 0.167` | 失败 |
| 行为反转 | ±alpha 下相反 | `83.9% vs 84.4%` | 失败 |
| 剂量反应 | 有意义的单调移动 | 总跨度 ≈ `0.004` | 可忽略 |

## 解释

EXP02 拒绝了简单因果假设：

> H19 程序状态信息被实现为一个全局的、任务无关的线性操控方向。

该结果与一般的可解释性警示一致：

```text
可解码
!=
全局线性可操控
```

## EXP02 未显示的内容

空结果并不确立技能敏感表征在因果上未被使用。

替代解释仍然存在：

- 非线性/上下文条件化表征；
- 多个交互的中介；
- 分布在 token 位置；
- 分布在层；
- H19 可能是一个可读的下游摘要，而非因果来源。

## 决策

不要简单地通过增大 alpha 或尝试许多新的线性分类器来「挽救」该假设。

进入精确配对激活互换。

---

# EXP03 — 配对互换修补

## 日期
2026-09-20

## 提交
`e98dade`

## 状态
已完成 — 空 / 负结果

## 研究问题

如果移除全局线性方向假设，来自同任务反技能条件的精确隐藏状态能否转移供体的下一次动作偏好？

## 干预

对于配对的提示词，其中：

- 相同任务；
- 相同历史；
- 相同措辞族；
- 反技能规定的下一状态；

用供体的精确激活替换接收者的最终提示词 token 隐藏状态：

```text
h_recipient^(l)
<-
h_donor^(l)
```

然后重新评分：

```text
read_file(tests/...)
vs
read_file(src/...)
```

## 确证位点

```text
H19
=
decoder block 18 输出
```

全层扫描为探索性分析。

## 主要结果

确证性 H19 效应最多是边际的，未确立因果转移。

更显著的是，H21–H28 附近的更深隐藏状态显示出更强的**负**有符号转移效应，量级约为：

```text
-0.011 至 -0.017
```

其幅度远大于 H19 的弱正效应。

## 初步观察

精确单一位点供体激活的行为不像一个可移植的程序状态变量。

深层激活替换可以主动将行为推向错误方向。

## 解释边界

该结果支持：

```text
单 token 精确残差状态互换
不足够
```

它**尚不证明**：

```text
完整机制是分布式电路
```

因为存在一个更简单的方法论解释。

## 关键替代解释：流形外嵌合体（Off-Manifold Chimera）

EXP03 仅替换一个 token 位置，同时保持：

- 相邻 token 状态；
- 前层上下文；
- 后续交互状态；

来自接收者不变。

因此该修补可能创建不一致的混合：

```text
供体状态
+
接收者周围的计算
```

深层强负效应可能反映由这种状态不匹配引起的破坏，而非特定分布式机制的证据。

## EXP03 后的竞争性解释

### A. 单一位点嵌合体

一致的供体状态需要恢复额外的对齐 token/层上下文。

预测：

```text
单一位点修补          失败
一致分布式修补        成功
```

### B. 分布式 / 交互中介

程序控制由多个 token 位置和/或层共同承载。

预测：

```text
multi-token 和/或 multi-layer 恢复
优于单一位点修补
```

### C. 仅残差状态读出

残差流包含可读的技能信息，但真正的因果机制存在于通路选择中，例如注意力/MLP 路由。

预测：

```text
即使一致的残差恢复也为空
```

## 决策

在宣称分布式电路之前，运行判别性的 multi-token × multi-layer 恢复实验。

---

# EXP04 — 分布式程序状态恢复

## 日期
2026-09-20

## 提交
74155e8

## 状态
已完成 — 阳性，self-patch 通过，所有控制条件满足

## 主要结果（common_H19_H28，48 个任务）

| 指标 | 值 | 95% CI |
|--------|-------|--------|
| 有符号转移效应 | +0.0579 | [0.0523, 0.0636] |
| 自修补控制 | −0.00027 | [−0.00057, +0.000008] |
| 同状态跨措辞 | −0.000024 | [−0.00099, +0.00091] |
| 反状态跨措辞 | +0.0571 | [0.0514, 0.0627] |

按接收者措辞：
- 规范（canonical）：+0.037
- 改述（paraphrase）：+0.079

## 动机

EXP03 排除了可移植的单 token 精确状态，但留下了一个主要歧义：

```text
单一位点修补失败
=
流形外嵌合体？
或
分布式中介？
```

EXP04 专门设计用于区分这些解释。

## 研究问题

跨越多个对齐提示词 token 和/或多个相邻层的供体激活一致集合，能否因果转移供体技能规定的下一次动作偏好？

## Token 对齐原则

**不要**修补未对齐的技能 token。

对每个供体/接收者对，计算精确的最长公共 token 后缀。

该后缀包含 token ID 完全相同的下游提示词内容，包括：

- 执行历史；
- 最终指令；
- 生成边界。

仅修补对齐位置。

## 定位配置

### Token 分布检验

```text
last1_H19
common_H19
```

解释：

```text
last1_H19 ~ 0
common_H19 > 0
```

将支持多 token 中介。

### 层分布检验

使用对齐的公共后缀：

```text
common_H19
common_H19_H24
common_H19_H28
common_H21_H28
common_H15_H28
```

## 预注册主配置

```text
供体：
同任务
同措辞
反规定状态

token 跨度：
完整精确公共后缀

隐藏状态范围：
H19-H28
```

主指标：

```text
M =
mean_logP(src action)
-
mean_logP(test action)
```

有符号转移：

```text
T =
donor_sign * (M_patched - M_baseline)
```

正值表示接收者移向供体的规定状态。

## 控制条件

### 自修补（Self Patch）

```text
recipient <- recipient
```

预期：

```text
~ 0
```

### 同状态跨措辞

示例：

```text
canonical TEST-first
<-
paraphrase TEST-first
```

改变措辞而不改变程序状态。

预期的方向性状态转移效应：

```text
~ 0
```

### 反状态跨措辞

示例：

```text
canonical TEST-first
<-
paraphrase IMPLEMENTATION-first
```

阳性结果将显示跨技能表面措辞的因果转移。

## 成功标准

对分布式残差状态中介的支持要求：

1. 主平均有符号转移 > 0；
2. 任务自助法 95% CI 排除零；
3. 同状态控制明显更小；
4. 自修补近似为零；
5. 规范与改述接收者条件符号一致；
6. 反状态跨措辞转移也为正。

## 计划性解释

### 结果 A — 多 token 挽救

```text
last1_H19 ~ 0
common_H19 > 0
```

解释：

> 程序中介分布在多个下游提示词位置。

### 结果 B — 多层挽救

```text
common_H19 ~ 0
common_H19_H28 > 0
```

解释：

> 程序中介依赖于跨层的协调状态。

### 结果 C — 完全分布式挽救

主检验与跨措辞反状态转移均为正，而自修补与同状态控制保持在零附近。

解释：

> 一致的分布式残差流状态足以转移部分技能条件化动作偏好。

### 结果 D — 所有残差恢复仍为空

解释：

> 停止将程序控制视为可移植的残差状态变量。

下一个目标：

- 注意力输出通路；
- MLP 输出通路；
- 头级路由；
- 技能 token → 动作 token 因果路径；
- 电路选择机制。

## 定位效应（有符号转移，各 48 个任务）

| 配置 | 跨度 | 层 | 效应 | 95% CI |
|--------|------|--------|--------|--------|
| last1_H19 | last1 | 19 | +0.0011 | [0.0003, 0.0021] |
| common_H19 | common | 19 | +0.0305 | [0.0266, 0.0345] |
| common_H19_H24 | common | 19–24 | +0.0558 | [0.0502, 0.0615] |
| **common_H19_H28** | **common** | **19–28** | **+0.0579** | **[0.0523, 0.0636]** |
| common_H21_H28 | common | 21–28 | −0.0002 | [−0.0038, 0.0033] |
| common_H15_H28 | common | 15–28 | +0.0796 | [0.0731, 0.0863] |

模式：
- 单独最后一个 token：空；
- H19 处完整公共后缀：为正 → 确认多 token 分布；
- 加入 H19–H20 是必要的（不含它们的 H21–H28 → 空）；
- 扩展到 H15 增加效应。

## 解释结果 — 结果 C（完全分布式挽救）

```text
Token：     1 个 token → 0.001     公共后缀 → 0.058
层：         19 单独 → 0.031     19–28 → 0.058     21–28 → ~0
控制：      self −0.00027     同状态跨措辞 −0.00002
```

> 跨越完整共享提示词后缀和 H19–H28 块（关键包含 H19–H20）的一致分布式残差流状态，足以转移技能条件化动作偏好，控制条件处于机器精度零。

## 方法说明 — 最终归一化捕获缺陷（已修复）

首次 EXP04 运行未通过自修补自检（均值 ≈ +0.197 而非 ~0）。根本原因：在 `outputs.hidden_states` 中，索引等于 `num_layers` 的元素（此处为 `hidden_states[28]`）是**最终 RMSNorm 之后**的激活——而非 `layers[27]` 的原始输出。将该值修补回 `layers[27]` 的归一化前输出，向残差流注入了错误缩放的向量（经验证最大 |diff| ≈ 735 → 产生 0.196 的自修补噪声）。

修复：`capture_prompt_states` 现在通过 `layers[hidx-1]` 上的前向钩子（pre-norm）获取隐藏状态，与 `patch_hook` 替换的位置完全一致。此外，将 `score_one` 改为逐候选 batch-1 前向传播，以消除右填充不对称。自修补降至 −0.00027。

对早期实验的含义：任何使用 `hidden_states[len(layers)]` 的位置（例如 EXP03 的 L27/L28 修补条件）共享此捕获/修补错位，因此 EXP03 中这些特定层在被修正的捕获方式重新运行之前应视为未经验证。

## 决策

进入 EXP05：测试 H15–H20 是否充当 selector 状态，以及该状态是否会在未修补的 H21–H28 中重配置 attention/MLP 输出。

---

# EXP05 — Selector → 通路重配置（Selector → Pathway Reconfiguration）

## 状态
已完成 — 阳性，MLP-dominant selector，所有控制条件满足

## 日期
2026-09-21

## 提交
`8e70ce0`

## 科学动机

EXP04 给出：

```text
H15-H28 = 强正效应
H21-H28 = null
```

这产生一个可检验的 selector 假说：

> H15–H20 可能负责建立任务/程序控制状态；一旦该状态被恢复，后续 H21–H28 可以在不直接修补的情况下自行切换到供体式计算。

这与「持续携带供体激活到输出端」的 carrier 模型不同。

## 主假设

H5：

```text
在 H15-H20 恢复供体公共后缀
           ↓
未修补的 H21-H28 attention/MLP 输出
移向供体式计算
           ↓
动作偏好移向供体技能状态
```

## 主行为实验

比较：

```text
early_H15_H20
late_H21_H28
full_H15_H28
```

预期的 selector 模式：

```text
early_H15_H20 > 0
late_H21_H28 ~ 0
full_H15_H28 > 0
```

## 通路读出（Pathway Readout）

仅修补 H15–H20。

随后不修补 H21–H28，而是测量：

- attention 分支输出；
- MLP 分支输出；
- decoder block 输出。

定义：

```text
projection recovery =
((patched-recipient) dot (donor-recipient))
/
||donor-recipient||^2
```

## 控制

- self H15-H20 patch；
- same-state cross-wording H15-H20；
- opposite-state cross-wording H15-H20；
- late H21-H28 阴性对照；
- full H15-H28 阳性对照。

## 成功标准

支持 selector-like 机制需要同时看到：

1. `early_H15_H20` 任务自助法 95% CI > 0；
2. `late_H21_H28` 仍近 0；
3. H21–H28 至少一个分支的 recovery > 0；
4. self / same-state 控制近 0；
5. opposite-state cross-wording 同方向；
6. 下游 recovery 与行为效应至少呈正相关趋势。

## 主要结果（48 个任务，任务自助法 95% CI）

### 行为效应（same-task, same-wording, opposite-state donor）

| 配置 | 角色 | 效应 | 95% CI |
|--------|--------|--------|--------|
| **early_H15_H20** | primary | **+0.0784** | [0.0717, 0.0849] |
| late_H21_H28 | 阴性对照 | +0.00002 | [−0.0034, +0.0035] |
| full_H15_H28 | 阳性对照 | +0.0797 | [0.0731, 0.0864] |

**selector_fraction = 0.984**：仅恢复 H15–H20 即捕获完整效应的 98.4%。

### 控制条件（early_H15_H20 修补下）

| donor 类型 | 效应 | 95% CI |
|--------|--------|--------|
| self（自修补） | −0.00008 | [−0.0005, +0.0003] |
| same-state cross-wording | +0.0021 | [0.0011, 0.0032] |
| opposite-state cross-wording | +0.0831 | [0.0773, 0.0889] |

### 通路读出（仅 patch H15–H20，H21–H28 未干预）

| 组件 | Projection Recovery | 95% CI | Distance Recovery |
|--------|--------|--------|--------|
| attn | 0.700 | [0.690, 0.711] | 0.437 |
| **mlp** | **0.875** | [0.870, 0.880] | **0.649** |
| residual | 0.872 | [0.868, 0.877] | 0.658 |

分层模式：MLP recovery 在 H21 处 0.962 逐层衰减至 H28 处 0.835；attention 在 0.64–0.75 波动。

Per-task recovery × 行为效应 Pearson 相关：

```text
attn      r = 0.126
mlp       r = 0.251   ← 最高
residual  r = 0.229
```

## 解释结果 — Pattern D（MLP-dominant recovery，因果性已被 EXP06 证伪）

```text
行为：      early_H15_H20  +0.0784（= 98.4% full）   late_H21_H28 ≈ 0
通路：      MLP recovery 0.875 >> attention recovery 0.700
控制：      self −0.00008      same-state +0.0021
            opposite-cross-wording +0.0831（同号正）
```

> H15–H20 的早期分布式残差状态充当 selector：恢复它之后，未修补的 H21–H28 下游计算自发放射向供体，其中 MLP 分支的重配置程度显著高于 attention 分支。支持 selector-like 机制。

**重要限制（EXP06 更新）**：EXP05 的分支不对称性（MLP recovery 0.875 > attention 0.700）是 select 恢复后下游计算的**伴随表现（correlational）**，而非因果中介本身。EXP06 通过稀疏神经元充分性/必要性干预直接检验后，否定了「该重配置由 H21–H28 MLP 神经元承载」的主张（详见下方 EXP06 章节）。本节的 Pattern D 应降级为读出现象，不作为机制结论引用。

## 方法说明

- 捕获沿用 EXP04 修复：通过 `layers[hidx-1]` 前向钩子取 pre-norm 激活，float32 存储。
- 通路读出在仅恢复 H15–H20 的前向中，对 H21–H28 的 attention/MLP/decoder-block 输出进行未干预测量（branch output 从未被 patch）。
- 修复了与 EXP01b 接口的 3 处命名不匹配（`build_history`→`history`、`build_messages`→`messages`、`CONDITIONS`→`CONDS`）。

## 下一步

MLP recovery（0.875）明显占优 → 进入 **EXP06：MLP feature / neuron-group causal analysis**：

```text
EXP06 -> 定位 H21-H28 中被 selector 状态条件化重配置的
         MLP 神经元/特征组合，
         验证其对动作偏好的充分性与必要性。
```

参考路线：组合神经元特征的 causal steering（ACL 2026, "Constructing Interpretable Features from Compositional Neuron Groups"）。

**EXP06 结果更新**：该假设已被否定——MLP 神经元组（无论稀疏还是全量）均无充分性与必要性。后续不沿 MLP 神经元收缩路径推进。

---

# EXP06 — MLP 神经元组因果中介（MLP Neuron-Group Causal Mediation）

## 状态
已完成 — 阴性/反转，所有神经元级干预为负效应

## 日期
2026-09-21

## 提交
`4b4b47c`

## 科学动机

EXP05 将最强下游重配置定位到 MLP 分支：

```text
projection recovery:
MLP      = 0.875
Residual = 0.872
Attention= 0.700
```

H15–H20 selector 窗口捕获完整 H15–H28 行为效应的 98.4%。

EXP06 从分支级定位推进到 MLP 中间神经元组，检验因果充分性与必要性。

## 研究问题

H21–H28 中，哪些 MLP 神经元组合真正把 H15–H20 selector 状态转换为下一次动作策略？

## 因果标准

候选神经元组必须同时通过两关（train 选组，held-out 干预）：

```text
sufficiency:
donor 神经元组移植 -> 行为移向 donor

necessity:
selector 恢复 + 钳回组到 recipient -> selector 效应下降
```

## 发现统计量（仅 train tasks）

```text
R = baseline recipient 神经元激活
D = baseline donor 神经元激活
P = H15-H20 selector 恢复后的 recipient 激活

aligned = E[(P-R)(D-R)]
energy  = E[(D-R)^2]
recovery_ratio = aligned / (energy + eps)
impact_score = max(recovery,0) * sqrt(energy) * ||W^down[:,j]||
```

介入点：`mlp.down_proj` 的 `forward_pre_hook`，直接捕获/编辑中间激活 `z`（Qwen2：`z = SiLU(gate_proj(x)) * up_proj(x)`，`y = down_proj(z)`）。

## 主配置

- 主组大小：**K = 256**（层-神经元对）；探索性 K = 64 / 1024；
- 4-fold GroupKFold：36 train / 12 held-out；
- 5 个按层 count-matched 随机组（每组各层数量与 top-K 完全一致）；
- cross-wording 检验（K=256）：canonical TEST recipient ← paraphrase IMPLEMENTATION donor；
- 48 个任务，任务自助法 95% CI。

## 结果（48 个任务）

### 预注册主配置（K=256）

| 指标 | 期望 | 实际 | 95% CI | 判定 |
|---|---:|---:|---:|---|
| early_selector_effect（内部一致性） | +0.078 | +0.0784 | [0.0718, 0.0850] | ✓ 复现 |
| top256_sufficiency | >0 | −0.0014 | [−0.0022, −0.0006] | 负 |
| top256_necessity_loss | >0 | −0.0014 | [−0.0023, −0.0005] | 负 |
| top256_suff − matched_random | >0 | −0.0010 | [−0.0018, −0.0003] | 负 |
| top256_nec − matched_random | >0 | −0.0014 | [−0.0020, −0.0008] | 负 |

### 剂量-反应（K 单调走负）

```text
K=64     suff −0.0000    nec −0.0003   （≈0）
K=256    suff −0.0014    nec −0.0014
K=1024   suff −0.0070    nec −0.0116
全 MLP   suff −0.0141    nec −0.0145
```

越大的组负得越多——无正向剂量-反应，而是随注入量增大的残差流破坏。

### 其他

```text
random256_sufficiency   −0.0004（matched-random 略负）
top256_cross_wording    suff −0.0014  nec −0.0013（同号负）
all_mlp_sufficiency     −0.0141（full-MLP 上界也为负）
```

### 数据质量核验

- early_selector_effect = +0.0784 与 EXP05 逐位一致 → pipeline/hook/scoring 正确；
- K=256 recovery_ratio 均值 0.91（min 0.35 / max 1.32），选择机制正常；
- 层分布跨 4-fold 稳定（H24/H28 偏多），matched-random 构造正确；
- 正负分布：top256 15/48 正 vs 33/48 负；全 MLP 5/43 → 系统性负效应。

## 解释结果 — 反转的因果证据

> H21–H28 的 MLP 中间神经元（稀疏组与全 MLP 皆然）既非充分也非必要。

- 移植 donor MLP 神经元值**不转移**行为（sufficiency 负），反而轻微推离 donor；
- 钳回选中神经元**不削弱** selector 效应（necessity loss 负）；
- full-MLP 上界为负（−0.014），排除「效应宽泛分布于整个 MLP」的替代解释。

EXP05 的 MLP recovery 0.875 是 **donor 状态沿残差流传播的伴随表现（correlational）**，不是因果开关。直接注入 donor z 破坏残差流一致性，负效应与 EXP03 深层 off-manifold chimera 同量级（−0.011 ~ −0.017）。

## 对证据链的影响

```text
Skill
 ↓
distributed H15-H20 selector   ← 仍成立（+0.0784，98.4%）
 ↓
H21-H28 MLP recovery 0.875     ← 降级为伴随现象，非因果
 ↓
specific MLP neuron group      ← 否定：无充分性、无必要性
 ↓
policy shift
```

「representation → branch → neurons → behavior」的神经层级收缩假说被证伪。行为效应的因果载体仍是早期 H15–H20 的分布式残差状态本身，其下游机制不浓缩于 MLP 神经元。

## 下一步（预注册决策触发）

1. 拒绝「compact causal MLP group」与「broad MLP mediation」两个分支；
2. 优先考察 **attention 通路 / head-level**（EXP05 attention recovery 0.700 虽低但为正）；
3. 若 attention 也失败，接受「残差流分布式状态本身即因果载体」，转向高维 multi-component decomposition（含 SAE/SNMF）或跨 Skill 泛化验证。

---

# EXP07 — 注意力头因果中介（Attention-Head Causal Mediation）

## 状态
已完成 — 阴性/部分弱正，AllAttention 上界为负

## 日期
2026-09-21

## 提交
`5ab0009`

## 科学动机

EXP06 证伪了 H21–H28 MLP 中间激活作为因果中介。剩余的主要下游分支是自注意力通路（EXP05 attention recovery 0.700 虽低于 MLP 但为正）。

EXP07 检验：

> H21–H28 的注意力头输出是否因果地中介 H15–H20 分布式 selector 状态的行为效应？

## 干预位点

Qwen2 自注意力将 per-head 输出拼接后经 `o_proj` 投影。EXP07 在 `self_attn.o_proj` 的 `forward_pre_hook` 处捕获/编辑拼接后的输入 `[batch, seq, 28, 128]`（28 头 × 128 head_dim），相当于直接操作头输出向量，绕开 Q/K/V 内部结构。

## 因果标准（与 EXP06 相同双关）

```text
sufficiency:
donor 头组移植（不恢复 selector）-> 行为移向 donor

necessity:
H15-H20 selector 恢复 + 将头组钳回 recipient -> selector 效应下降
```

特征选择仅用 train tasks（4-fold GroupKFold，36 train / 12 held-out）；干预在 held-out tasks 评估。

## 发现统计量（仅 train）

```text
R = baseline recipient 头输出
D = baseline donor 头输出
P = H15-H20 selector 恢复后的 recipient 头输出

aligned = E[(P-R)(D-R)]     （按 token 位置与 head_dim 求和）
energy  = E[(D-R)^2]
recovery_ratio = aligned / (energy + eps)
impact_score = max(recovery,0) * sqrt(energy) * ||W_o^(h)||_F
```

`||W_o^(h)||_F` 为 `o_proj` 中该头对应权重块的 Frobenius 范数（每头 128 列）。

## 主配置

- 主组大小：**K = 16**（层-头对）；探索性 K = 4 / 32 / 64；
- AllAttention 上界：全部 8 层 × 28 头 = 224 个层-头组件一次性 donor 移植/recipient 钳回；
- 5 个按层 count-matched 随机组（每层头数与 top16 完全一致，池中剔除选中头）；
- cross-wording 检验（K=16）：canonical TEST recipient ← paraphrase IMPLEMENTATION donor；
- 48 个任务，任务自助法 95% CI。

## 结果（48 个任务）

### 预注册主配置

| 指标 | 期望 | 实际 | 95% CI | 判定 |
|---|---:|---:|---:|---|
| early_selector_effect（内部一致性） | +0.078 | +0.0784 | [0.0718, 0.0851] | ✓ 复现 |
| all_attention_sufficiency | >0 | −0.0145 | [−0.0161, −0.0128] | 负 |
| all_attention_necessity_loss | >0 | −0.0163 | [−0.0179, −0.0146] | 负 |
| top16_sufficiency | >0 | +0.0002 | [−0.0007, +0.0011] | ≈0（null） |
| top16_necessity_loss | >0 | +0.0018 | [+0.0011, +0.0025] | 弱正 |
| top16_suff − matched_random | >0 | +0.0005 | [−0.0003, +0.0012] | CI 含 0 |
| top16_nec − matched_random | >0 | +0.0017 | [+0.0011, +0.0023] | 正 |

### 剂量-反应（非单调，K=16 是 peak）

```text
K=4     suff −0.0005   nec −0.0006   （≈0）
K=16    suff +0.0002   nec +0.0018   ← 唯一正信号（necessity）
K=32    suff −0.0055   nec −0.0059   （转负）
K=64    suff −0.0025   nec −0.0037   （负）
```

与 EXP06 的单调走负不同，EXP07 呈非单调：K=16 的 necessity 是唯一显著正信号，K≥32 后随注入量增大而转负（残差流破坏占主导）。

### 其他

```text
cross_wording_top16_suff  +0.0005（同号，弱）
cross_wording_top16_nec   +0.0009（同号，弱）
```

## 数据质量核验

- early_selector_effect = +0.0784 与 EXP05/EXP06 逐位一致 → pipeline/hook/scoring 正确；
- intervention_results.csv 共 4416 行 = 48 任务 × 4 条目 × (21 same-wording + 2 cross-wording)，无缺失；
- 首次运行仅 post-processing 一行绘图 bug（`counts.groupby("hidden_state_index").count.mean()` 的 `.count` 被 DataFrame 方法遮蔽）导致 exit 1；数据收集阶段 4 折全部完成，已用修复脚本由 CSV 重放生成 summary.json/图，数值未重新计算。

## 解释结果 — 注意力头不构成可移植的充分性中介

> H21–H28 的注意力头输出（o_proj 输入层面的层-头组件）既不充分也不必要（作为整体或稀疏组）。

- **AllAttention 上界为负**（suff −0.0145 / nec −0.0163）：把全部 224 个头输出整体移植/钳回，行为被推离 donor 方向，与 EXP06 全 MLP 干预同量级（−0.014 ~ −0.016）——深层头输出携带的不只是可移植状态，还有残差流一致性约束；
- **Top16 sufficiency ≈ 0**（CI 含 0）：稀疏头组移植不能把行为移向 donor → 不充分；
- **Top16 necessity 弱正**（+0.0018，约 early effect 的 2.3%）：钳回选中头确实微弱削弱 selector 效应，且胜过 matched-random（+0.0017）——存在微弱的必要性贡献，但幅度远不足以构成"主要因果载体"；
- **K≥32 转负**：非单调说明"更多头"不产生更多中介，而是产生更多 off-manifold 破坏。

与 EXP06 合并判断：**下游两大分支（MLP 中间激活、注意力头输出）均未通过中介检验**。EXP05 观察到的下游 recovery 不对称（MLP 0.875 > attn 0.700）在两个独立因果干预下都被证伪为伴随表现。

## 决策分支触发

按预注册决策树：

```text
AllAttention > 0 & Top16 > 0   -> EXP08 Q/K/V + Head Path Patching   ✗ 未满足
AllAttention > 0 & Top16 ≈ 0   -> broad distributed routing          ✗ 未满足
AllAttention ≈ 0 / 负          -> 拒绝可移植 post-attention-output 中介 ✓ 命中
```

证据收敛于第三个分支：**H15–H20 selector 效应的下游因果机制不浓缩于 H21–H28 的注意力头输出，也不浓缩于 MLP 神经元**。行为效应的因果载体仍是早期分布式残差状态本身；下游计算（无论 attention 还是 MLP）是该状态的"joint computation condition"，其重配置是伴随表现，而非可移植中介。

下一步候选（不再沿下游通路收缩路径推进）：

1. H15–H20 内部边界定位（哪些层/token/维度是 selector 的最小因果成分）；
2. 高维 multi-component decomposition（SAE/SNMF）刻画 selector 状态本身；
3. 跨模型复现、跨 Skill / 任务泛化。

---

# 当前证据总结

EXP01–EXP04 逐步建立：

> Agent 技能程序信息可以从残差流中可靠读取，并跨任务与措辞泛化，但在所检验的干预下，无论是全局线性方向还是单 token 精确残差状态互换，都不足以重定向行为。

当前证据链：

```text
技能程序信息
        |
        v
线性可解码                         是
        |
        v
跨措辞语义泛化                     是
        |
        v
与动作偏好关联                     是，中等
        |
        v
全局线性操控                       否
        |
        v
单 token 精确互换                  否
        |
        v
分布式因果中介                     是（EXP04）
        |
        v
早期状态作 selector                是（EXP05）
        |
        v
下游 MLP recovery 更高             是（EXP05，读出）
        |
        v
MLP 神经元因果中介                 否（EXP06 证伪：无充分性/必要性）
        |
        v
注意力头输出因果中介               否（EXP07 证伪：AllAttention 负，Top16 suff≈0 / nec 弱正）
```

## 当前主张边界

EXP07 之后，EXP05–06–07 的三角证据完整闭合：

> Agent 技能程序信息由早期（H15–H20）的分布式残差流状态承载。恢复该早期状态（仅 H15–H20），无需直接干预 H21–H28，即可因果转移技能条件化动作偏好（+0.0784，为完整 H15–H28 恢复效应的 98.4%）。同时下游未修补计算自发朝供体移动（MLP recovery 0.875，attention recovery 0.700）——**但该下游移动的两大主要下游分支（MLP 中间神经元、注意力头输出）均已被独立因果干预证伪为可移植中介**：MLP 神经元既非充分也非必要（EXP06，full-MLP suff −0.014），注意力头输出同样既不充分（all attention suff −0.015）也不构成有效的稀疏中介组（Top16 suff ≈ 0，nec 仅 +0.002 即 early effect 的 2.3%）。下游移动是伴随表现（correlational），不是因果载体。

尚不可声明：

> 「真正的因果机制完全是一个分布式电路。」或「行为转移经由某个特定的下游神经元/通路组件承载。」

已验证的排除项（EXP01–EXP07）：

1. 全局线性操控方向（EXP02，否）；
2. 单 token 精确残差互换（EXP03，否）；
3. H21–H28 稀疏 MLP 神经元组作为因果中介（EXP06，无充分性/必要性）；
4. H21–H28 全 MLP 中间激活作为因果载体（EXP06，full-MLP 干预为负）；
5. H21–H28 注意力头输出作为因果中介（EXP07，AllAttention 为负；Top16 sufficiency null，necessity 仅 +2.3%）。

尚未完成的验证：

1. H15–H20 内部边界定位与 selector 状态的最小成分；
2. 高维 multi-component decomposition（SAE/SNMF）刻画 selector 状态本身；
3. 跨模型复现；
4. 跨 Skill / 任务泛化。

> 「证据排除了五个假设——全局线性操控、单 token 精确互换、MLP 神经元级中介、MLP 全量中介、注意力头输出中介——支持早期分布式残差 selector 的可移植性，但其下游因果机制尚未定位，两条主要下游通路（MLP / attention）均已独立证伪。」

此措辞应保留，直到后续实验定位到行为转移的实际下游通路，或转向 selector 状态本身的分解（SAE/SNMF）。