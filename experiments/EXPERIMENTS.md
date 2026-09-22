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
| EXP07 | H21–H28 注意力头输出是否因果中介？ | 完成，NEGATIVE/部分正 | AllAttention suff/nec −0.015/−0.016（负）；Top16 suff≈0，nec +0.0018（弱正）；K≥32 负；**注意力头输出不构成可移植的充分性中介** | `0281b2f` |
| EXP08 | H15–H20 内部哪里是 causal handoff？ | 完成，紧凑多层核心 | H20=+0.049（62% full）、H15:H19=+0.052（66%）；full−H20=+0.030、full−H15:H19=+0.027（均显著）；**H18–H20 即 97% 全效应，H15–H17 可移除**；既非 H20 单层 handoff 亦非全宽均匀累积 | `186b370` |
| EXP09 | H18–H20 残差经 Q/K/V 投影状态传递？ | 完成，POSITIVE | H20 KV suff=+0.046（94%）/nec=+0.044（91%）；**V 单通道即 89%**，K≈0、Q 为负；chain_KV=+0.066（full 的 84%）；**KV>Q 确认**（+0.051/+0.050）但机制为 V 主导 | `62eb9a9` |
| EXP10 | GQA V-head × token 位置定位？ | 完成，稀疏收敛 | **KV0 单头承载全部 V 效应**（direct +0.047 ≈ full +0.043）；**B5（最末端指令区）即 99.9% full**；KV0×B5 单 cell（suff +0.0448, q=0.0006 / nec +0.0445, q=0.0006）≈ full_V 的 103%；KV1≈0、KV2/3 微负；B0–B4 无效 | 待 commit |

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

证据收敛于第三个分支：**H15–H20 selector 效应的下游因果机制不浓缩于 H21–H28 的注意力头输出，也不浓缩于 MLP 神经元**。可拒绝"可移植的 post-attention head output 是主要中介"这一假设。**注意：尚不能直接断言「H15–H20 residual 本身就是最终因果载体」**——Qwen2 每个 decoder block 均为「输入 residual → RMSNorm → Attention → residual add → RMSNorm → MLP → residual add」的联合计算，上一层输出直接作为下一层整体计算的输入条件，因此 H15–H20 内部究竟哪一层真正完成 causal handoff 仍属未知（由 EXP08 定位）；下游计算（无论 attention 还是 MLP）的重配置是伴随表现，而非可移植中介。

下一步候选（EXP08 正在运行）：

1. **EXP08 H15–H20 内部边界定位**（single / prefix / suffix 扫描，判定 H20 handoff vs depth-distributed）；
2. 高维 multi-component decomposition（SAE/SNMF）刻画 selector 状态本身；
3. 跨模型复现、跨 Skill / 任务泛化。

---

# EXP08 — Selector 边界 / Handoff 定位（Selector Boundary / Handoff Localization）

## 状态
已完成 — 紧凑多层核心（H18–H20），既非 H20 单层 handoff 亦非 H15–H20 全宽累积

## 日期
2026-09-21

## 提交
`186b370`

## 科学动机

EXP06/07 证伪了 H21–H28 两条下游通路（MLP 中间激活、注意力头输出）作为可移植中介。但"早期（H15–H20）分布式 selector"尚未在层深度上被证明真的是 distributed——它可能实际是某个单层的 handoff state，也可能是 H15→H20 的深度累积。

Qwen2 每个 decoder block 为「输入 residual → RMSNorm → Attention → residual add → RMSNorm → MLP → residual add」的联合计算，上一层输出直接作为下一层整体计算的输入条件，因此该问题必须用实验定位而非假设。

## 研究问题

H15–H20 窗口内，行为效应的 causal handoff 究竟在哪里？

## 设计（条件与 EXP04–EXP07 完全一致）

- same task / same wording / opposite state donor / exact common suffix 对齐；
- 48 个任务，任务级 paired bootstrap 95% CI（seed 4808）。

配置族：

```text
single:        H15, H16, H17, H18, H19, H20
prefix:        H15-H16, H15-H17, H15-H18, H15-H19, full H15-H20
suffix:        H16-H20, H17-H20, H18-H20, H19-H20
```

控制（仅对 single_H20 与 full_H15_H20）：

```text
self                     预期 ≈ 0
same-state cross-wording 预期 小
opposite cross-wording   预期 同号正
```

预注册判定规则：

```text
H20-handoff 解释：
H20-only 强正且接近 full，而 H15-H19（无 H20）远弱

depth-distributed 解释：
prefix 在 H20 前累积出可观效应，且无单层接近 full
```

## 主要结果（48 个任务）

### 三个关键数

| 配置 | mean | 95% CI | fraction of full |
|---|---:|---:|---:|
| `single_H20` | **+0.0488** | [0.0440, 0.0536] | **0.622** |
| `prefix_H15_H19` | **+0.0516** | [0.0462, 0.0571] | **0.659** |
| `full_H15_H20` | **+0.0784** | [0.0718, 0.0852] | 1.000 |

### Paired contrasts（task-level paired bootstrap）

```text
full − H20        +0.0296  [+0.0266, +0.0327]   显著正
full − H15:H19    +0.0268  [+0.0239, +0.0297]   显著正
H20 − H15:H19     −0.0028  [−0.0071, +0.0014]   H20 ≈ H15:H19（无差异）
```

### 最小窗口定位（full 相对各 suffix 的 paired 差）

```text
full − suffix_H16_H20   +0.0002  [−0.0005, +0.0009]   不可区分 → H16-H20 ≈ full，H15 可移除
full − suffix_H17_H20   +0.0026  [+0.0019, +0.0033]   仅 +0.0026
full − suffix_H18_H20   +0.0024  [+0.0015, +0.0032]   H18-H20 捕获 97% 全效应
full − suffix_H19_H20   +0.0211  [+0.0187, +0.0235]   仅 H19-H20 显著不足
```

### 单层与累积曲线

```text
single:   H15 −0.016  H16 −0.010  H17 +0.006  H18 +0.042  H19 +0.030  H20 +0.049
prefix:   pre16 −0.010  pre17 +0.008  pre18 +0.044  pre19 +0.052  full +0.078
suffix:   s16_H20 +0.078  s17_H20 +0.076  s18_H20 +0.076  s19_H20 +0.057
```

- Single 层：H18（+0.042）与 H20（+0.049）是两个最强单层，H15/H16 为负；
- Prefix：H20 是最大单步增量（pre19→full +0.027），但 pre18 已 +0.044；
- Suffix：H16-H20 ≈ full；H18-H20 即 97%；仅剩 H19-H20 时掉到 +0.057。

### 控制（全部通过）

```text
single_H20 self        −0.0002   full self        −0.0001
single_H20 same-state  +0.0052   full same-state  +0.0021
single_H20 opp-cross   +0.0462   full opp-cross   +0.0831
```

## 数据质量核验

- `full_H15_H20` = +0.0784 与 EXP05/06/07 的 early_selector_effect 逐位一致 → pipeline/hook/scoring 正确；
- 48 任务 × 4 条目全量记录，无缺失；exit 0 一次通过（无 EXP06/07 的 bug 复发）。

## 解释结果 — 紧凑多层核心（Pattern：neither 纯 handoff 亦非全宽累积）

> **效应集中在 H18–H20 三层**（suffix 0.076 ≈ full，full−H18:H20 仅 +0.0024）；**H20 是单层最强（62% full）但远非充分**；H15–H17 基本可移除（H16–H20 与 full 统计不可区分）。

两个备选故事均被拒：

```text
H20 handoff 假说（H20≈+0.075, H15:H19≈0）：拒
   → H20 仅 +0.049（62%），H15:H19 为 +0.052（66%）绝非 0

H15→H20 深度均匀累积假说（H20 << full）：拒
   → prefix 在 H17 之前 ≈ 0/负，H15–H16 单层为负、可移除
```

实测形态是第三种：**紧凑多层核心 H18–H20，H20 主导但需与 H18/H19 一致恢复才达到 full**。行为效应的 causal carrier 是一个约 3 层宽的分布式窗口，而非单层 handoff，也非 EXP05 窗口名暗示的 H15–H20 全宽。

## 对证据链的影响

- "H15–H20 selector"的窗口名应更新为 **"H18–H20 核心（H20 主导）"**；EXP05/08 中 H15–H17 的贡献接近可忽略/负；
- 与 ACL 2026 *Patches of Nonlinearity* 的对照更加精确：不是"某层形成可搬运 instruction vector"，而是 **连续 2–3 层 joint computation 累积出的条件化状态**（与 Qwen2 block 结构一致：前层输出是后层联合计算的输入条件）；
- EXP06/07 的排除结论不受影响：H21–H28 两条下游通路仍被证伪；EXP08 进一步把"早期窗口"从 6 层收窄到 3 层。

## 下一步（决策触发）

按预注册决策树，结果最接近 **depth-distributed（需修正为 H18–H20 核心）** 分支：

```text
H20-handoff（H20≈full, H15:H19≈0）          ✗
depth-distributed / 紧凑多层核心              ✓（H18-H20，H20 主导）
```

因此 **EXP09 不做单 block 20 的 Q/K/V**（H20 仅 62%，sufficiency 不足），而应做：

> **EXP09：H18–H20 多层 consumer-side path localization** —— 逐层分解 Q/K/V → attention routing → post-attention residual → MLP 的因果贡献，定位"哪一层（或哪几层）的哪个内部投影真正承载 handoff"。

---

# EXP09 — Residual → Q/K/V Routing Mediation（残差 → Q/K/V 路由中介）

## 状态
已完成 — **H20 层 QKV（实际由 V 主导）即消费接口：suff 84% / nec 81%，KV≈V，K≈0，Q 为负**；三层 chain 恢复 full 的 84%。

## 日期
2026-09-22

## 提交
`62eb9a9`

## 科学动机

EXP08 定位出 H18–H20 紧凑因果核心，但"前层输出是后层联合计算的输入条件"（Qwen2 block：input residual → RMSNorm → Attention → residual add → RMSNorm → MLP → residual add），H18–H20 内部哪一层完成 causal handoff 仍未知。EXP06/07 已证伪 H21–H28 的下游 MLP 与注意力头输出作为可移植中介——但那只覆盖了"远离核心的末端通路"。EXP09 检查紧邻核心的**消费接口**：block 18/19/20 的 Q/K/V 投影状态（pre-RoPE）。

## 研究问题

H18/H19/H20 残差状态的行为效应，是否通过各自消费 block 的 Q/K/V（KV/QKV）投影状态传递？

## 设计

- 与 EXP04–EXP08 完全一致：same task / same wording / opposite state donor / exact common suffix 对齐；48 任务，任务级 paired bootstrap 95% CI（seed 4909）。
- 干预位点：`layers[h].self_attn.{q,k,v}_proj` **输出（pre-RoPE，整行替换）**；源残差 `layers[h-1]` 输出（= H_h）。
- 投影组合：`Q / K / V / KV / QKV`；固定 5 投影。
- 充分性（suff）：仅移植 donor 投影状态（无残差 patch），`E_suff = s·(M_patched − M_base)`。
- 必要性（nec）：移植 donor 残差 + clamp 接收方投影为 donor 状态，`E_nec = E_res − s·(M_res+clamp − M_base)`（即"去掉投影能传递的贡献后残差还剩多少"；E_nec 大 = 投影状态对该残差效应是必要的）。
- Chain（三层 consumer 同时干预，投影 donor 状态来自对应层残差）：
  - `chain_KV_sufficiency`、`chain_QKV_sufficiency`。
- 控制（每层 QKV suff + opposite 组 nec）：
  - `self`（预期 ≈ 0）；`same-state cross-wording`（预期 小/≈0）；`opposite cross-wording`（预期 同号正）。
- Paired contrasts（task-level paired bootstrap）：
  - `H20_KV_suff_minus_Q_suff`、`H20_KV_nec_minus_Q_nec`（预注册 **KV > Q** 检验）；
  - `H20_QKV_suff_minus_KV_suff`、`H20_QKV_nec_minus_KV_nec`（Q 是否必要）。

## 预注册假设

> **KV > Q**：prompt 侧 K 决定后续 token 如何匹配 prompt 状态，V 决定读取内容；Q 主要作用于自身读取先验上下文。故消费接口的充分性/必要性应以 KV 为主，Q 贡献小甚至为负。

## 主要结果（48 个任务）

### 用户指定关键数（H20，confirmatory）

| 指标 | mean | 95% CI | fraction of H20 residual |
|---|---:|---:|---:|
| `H20_residual_reference`（=EXP08 single_H20 复现） | **+0.0488** | [0.0440, 0.0536] | 1.000 |
| `H20_KV_sufficiency` | **+0.0460** | [0.0427, 0.0494] | **0.944** |
| `H20_KV_necessity_loss` | **+0.0443** | [0.0409, 0.0479] | **0.909** |
| `H20_QKV_sufficiency` | **+0.0408** | [0.0374, 0.0440] | 0.836 |
| `H20_QKV_necessity_loss` | **+0.0393** | [0.0357, 0.0430] | 0.806 |
| `chain_KV_sufficiency`（H18+H19+H20 同时） | **+0.0661** | [0.0604, 0.0720] | —（full_H15_H20 的 **84%**） |
| `chain_QKV_sufficiency` | **+0.0645** | [0.0587, 0.0704] | —（full 的 82%） |

- H20 residual 复现 +0.0488 与 EXP08 `single_H20` 逐位一致 → 本实验 pipeline/hook 正确。
- **suff 与 nec 双双强正、CI 排除 0** → 按预注册决策树进入 **causal routing interface** 分支。

### Paired contrasts（H20）

```text
KV_suff − Q_suff    +0.0506  [+0.0467, +0.0547]   显著正 → KV >> Q（预注册假设确认）
KV_nec  − Q_nec     +0.0500  [+0.0463, +0.0539]   显著正 → KV >> Q
QKV_suff − KV_suff  −0.0053  [−0.0067, −0.0039]   QKV < KV（加 Q 反而略降）
QKV_nec  − KV_nec   −0.0050  [−0.0064, −0.0037]   QKV < KV
```

### 子分量分解（routing_profile，fraction of 该层 residual）

```text
H20:  Q  −0.094 (负)   K  +0.002 (≈0)   V  +0.890   KV  +0.944   QKV  +0.836
      nec  −0.117       +0.025           +0.897      +0.909      +0.806
H19:  Q  +0.270        K  +0.006/0.203  V  +0.315/0.459   KV  +0.464/0.403   QKV  +0.656/0.639
H18:  Q  −0.011 (≈0)   K  +0.135        V  +0.284        KV  +0.384/0.345    QKV  +0.373/0.350
```

- **H20（confirmatory 层）：V 投影状态即主载体**（suff 89.0% / nec 89.7%），KV≈V（+0.0460 vs +0.0434，差值 ~+0.003）；**K≈0、Q 显著为负（−9.4%）**——加 Q 会压低效应（QKV < KV 的配对差来源）。
- **H19（复制层）**：QKV 最强（66%），V 次之（32%），Q 在这里为正（27%）——中间层 Q 仍有用。
- **H18（复制层）**：KV≈QKV≈38%，V 28%，K 13.5%，Q≈0。
- **随深度递增**：QV-KV 接口的因果占比 H18 38% → H19 46–66% → H20 94%；越靠近最终输出层，block 内 Q/K/V 投影状态越完整地承载残差效应。

### 控制（全部通过）

```text
H20_QKV self                              −0.0002  （≈0 ✓）
H20_QKV same-state cross-wording          −0.0022  （≈0 ✓）
H20_QKV opposite cross-wording suff       +0.0397  （同号正 ✓）
H20_QKV opposite cross-wording nec        +0.0345  （同号正 ✓）
```

## 数据质量核验

- `H20_residual_reference` +0.0488 与 EXP08 `single_H20` 逐位一致（float32）→ 干预管线无系统性偏移；
- chain 值（+0.066/+0.064）介于 single_H20（+0.049）与 EXP08 full（+0.078）之间，且方向单调 → 三层 chain 恢复 full 的 84%，与"紧凑多层核心"一致；
- 48 任务 × 4 条目全量记录；exit 0 一次通过；seed 4909。

## 解释结果 — H20 的 V-projection 是残差→行为的主要消费接口

> **H20 残差效应主要通过 block 20 的 V（及其 KV 组合）投影状态传递**：suff 94%/nec 91%，Q 贡献为负、K≈0，加 Q 反而略损。三层 chain（H18+H19+H20 的 QKV/KV 同时移植）恢复 full_H15_H20 的 82–84%——consumer-side 接口随深度逐层接管：H18 38% → H19 66% → H20 94%。

- 预注册 **KV > Q** 假设**确认**（配对差 +0.0506/+0.0500，均显著正），但机制细节与假设动机不同：不是"K 决定匹配、V 决定读取"的 K+V 组合，而是 **V 单通道近乎完整承载**（89–90%），K 在 H20 ≈ 0；
- Q 在 H20 为负：强制把 H20 的 Q 换成 donor 的 Q 会略微破坏行为（Q 状态含 position-sensitive 读取先验，跨措辞移植时干扰）；
- 必要性 loss（91%）≈ 充分性（94%）：H20 的 KV 投影状态既充分又几乎必要 → 是真正的 causal routing interface，而非"残差另走 MLP 旁路"（若残差可绕开投影走 MLP，necessity loss 应远低于 94%）；
- EXP06/07 的排除不受影响：那里干预的是 H21–H28 内部（更下游），这里干预的是 H18–H20 自身消费块入口。

## 对证据链的影响

- 证据链补上关键一跳：**H18–H20 残差 → 各层 block 的 V/KV 投影状态（consumer 接口）→ 行为**；
- 与 Qwen2 block joint computation 的结构吻合：前层残差不直接"搬运"，而是通过本层 attention 的 V-projection 读取进入后续计算——H20 层此接口几乎饱和（94%）；
- "便携状态"的含义进一步收窄：可移植的因果载体是 **H20 消费块的 V 投影内容状态**（与 EXP07 的 post-attention head output 不同——那在 H21–H28，且被证伪；这里是 H20 自身投影，suff/nec 双高）。

## 下一步（决策触发）

按预注册决策树：

```text
KV/QKV suff+nec 均 +/CI 排除 0  →  causal routing interface     ✓（H20 KV suff 94% / nec 91%）
necessity-only                 →  context-conditioned routing   ✗
null / negative                →  拒绝便携 QKV 中介，转归一化/残差交互研究  ✗
```

进入 **EXP10**：

> **GQA KV-head × token-position 路径定位** —— 既然载体是 V-projection 状态（Qwen2 仅 4 个 KV head，GQA），进一步问：(a) 4 个 KV-head 中哪几个承载（head 级 suff/nec）；(b) 状态在 prompt 哪些 token 位置（内容 token vs skill token vs 分隔符）最强；(c) 位置 × head 的交互矩阵。若 head 或位置高度集中 → 可移植中介收敛为极稀疏电路；若均匀 → 修正为宽路径路由。

---

# EXP10 — GQA Value-Head × Token-Position Causal Localization（GQA V-head × token 位置因果定位）

## 状态
已完成 — **单一 KV head（KV0）× 最靠近 generation boundary 的 bin（B5）** 承载全部 V 效应：KV0×B5 suff +0.0448（q=0.0006）/ nec +0.0445（q=0.0006），即 full_V 的 ~103%。

## 日期
2026-09-22

## 提交
待 commit

## 科学动机

EXP09 把 H20 V/KV 投影状态确认为第一个同时满足强充分性（94%）与强必要性（91%）的便携下游接口，且 V 单通道即承载 89%。下一刀：**这个 V-routing 接口到底落在 4 个 GQA KV head 中的哪几个、以及对齐 prompt 的哪些 token 位置**。Qwen2.5-Coder-7B-Instruct 官方配置 28 query heads / 4 KV heads，`repeat_kv` 将 4 个 KV head 扩展给 28 个 query heads（每 KV head 对应 7 个 query heads）——因此 KV-head 级定位是架构上明确的下一步。

## 研究问题

1. 4 个 GQA value heads 中，H20 V 效应由谁承载？（direct suff/nec + leave-one-out marginal）
2. exact common suffix 内，效应写在哪些 token 位置？（6 个归一化连续 bin B0–B5）
3. 4 heads × 6 bins = 24 cells 的交互矩阵中，是否有 cell 同时满足 suff 与 nec 的 FDR 显著性？

## 范围边界（用户修正）

位置定位仅限于 **exact common suffix**——donor/recipient 在 Skill 措辞/顺序文本已经发生差异之后、重新变得 token-identical 的共享下游上下文。因此 EXP10 定位的是 **Skill-induced procedural state 被写入共享 downstream context 的什么位置**，而不是 Skill tokens 本身。

## 设计

- 与 EXP04–EXP09 完全一致：same task / same wording / opposite state donor / exact common suffix 对齐；48 任务，任务级 paired bootstrap 95% CI（seed 5010）。
- 干预位点：block 20 `v_proj` 输出（pre-RoPE），reshape 为 `[batch, seq, 4 KV heads, 128]`，按 head × token 位置编辑。
- Part A（KV-head 定位）：`head{h}_sufficiency` / `head{h}_necessity_loss`（h=0..3）+ `all_except_head{h}_suff/nec` → direct 与 leave-one-out marginal 效应。
- Part B（位置定位）：`suffix_bins()` 用 `np.array_split` 将每个 common suffix 切成 6 个连续归一化区域 B0（最早）…B5（最靠近 generation boundary）；全 4 heads 只 patch 一个 bin → `bin{b}_allheads_suff/nec`。
- Part C（交互矩阵）：24 cells 全标记 **exploratory**，各做 suff + nec；对 suff 与 nec **分别**做 task-level sign-flip test + **Benjamini-Hochberg FDR**；仅当同一 cell 的 suff q<0.05 **且** nec q<0.05 才进入下一轮独立 replication。
- 控制：full-V self / same-state cross-wording / opposite-state cross-wording；另复现 H20 residual reference。
- Token audit：`token_position_map.csv` 记录 task/wording/suffix_index/offset_from_generation_boundary/bin_index/token_id/decoded_token。

## 主要结果（48 个任务）

### 复现核验（先看）

| 指标 | mean | 95% CI | 与 EXP09 一致性 |
|---|---:|---:|---|
| `H20_residual_reference` | **+0.0488** | [0.0440, 0.0537] | = EXP08/EXP09 逐位一致 ✓ |
| `full_V_sufficiency` | **+0.0434** | [0.0406, 0.0464] | = EXP09 V suff 0.0434 逐位一致 ✓ |
| `full_V_necessity` | **+0.0438** | [0.0406, 0.0471] | = EXP09 V nec 0.0438 逐位一致 ✓ |

控制：self +0.00004；same-state cross-wording −0.0012；opposite cross-wording suff +0.0439 / nec +0.0417（同号正）。全部通过。

### Part A — KV-head profile（决定性）

| head | direct suff | direct nec | leave-one-out marginal suff | marginal nec |
|---|---:|---:|---:|---:|
| **KV0** | **+0.0471** [0.0441, 0.0501] | **+0.0472** [0.0443, 0.0503] | **+0.0482** | **+0.0473** |
| KV1 | −0.0004 [−0.0010, +0.0003] | +0.00004 [−0.0006, +0.0007] | +0.0001 | +0.0002 |
| KV2 | −0.0013 [−0.0020, −0.0006] | −0.0009 [−0.0014, −0.0003] | −0.0005 | −0.0008 |
| KV3 | −0.0040 [−0.0049, −0.0031] | −0.0029 [−0.0037, −0.0020] | −0.0033 | −0.0026 |

- **KV0 单头即承载全部 V 效应**：direct suff +0.0471 甚至略高于 full_V +0.0434（其余头拖累）；leave-one-out marginal +0.0482 ≈ full_V + 0.005。
- KV1 ≈ 0；**KV2/KV3 为负**（移除它们反而增强，轻微抑制性）。

### Part B — position profile（决定性）

| bin | suff mean | nec mean | suff fraction of full | nec fraction of full |
|---|---:|---:|---:|---:|
| B0（最早） | −0.0017 | −0.0007 | −0.039 | −0.017 |
| B1 | −0.0011 | −0.0001 | −0.025 | −0.003 |
| B2 | +0.0001 | +0.0002 | +0.002 | +0.005 |
| B3 | −0.0003 | −0.0002 | −0.006 | −0.005 |
| B4 | +0.0011 | +0.0012 | +0.025 | +0.028 |
| **B5（最靠近 generation）** | **+0.0434** | **+0.0425** | **0.999** | **0.971** |

- **B5 单独即 full_V 的 99.9%（suff）/ 97.1%（nec）**；B0–B4 全部 ≈ 0 或微负。

### Part C — head × position 矩阵（24 cells，全部 exploratory）

两族 FDR 后同时 q<0.05 的 cell（`head_position_cells_fdr_lt_0_05_in_both`）：

```text
            B0   B1   B2   B3   B4   B5
KV0         .    .    .    .    +    ++      ← KV0×B5: suff +0.0448 (q=0.0006) / nec +0.0445 (q=0.0006)
KV1         .    .    .    .    .    .
KV2         .    .    .    .    .    .
KV3         .    .    .    -    .    -
```

- **唯一正效应 cell：KV0×B5**（suff +0.0448 / nec +0.0445，q 均 0.0006）——即 full_V 的 ~103%，**单一 GQA value head × 单一最末端位置 bin**；
- KV0×B4 小正（+0.0012/+0.0018，q 临界）但量级微不足道（full 的 2.5%）；
- KV3×B3 / KV3×B5 为小负显著（−0.0011/−0.0016）——量级微小，且与 KV3 的抑制性一致；
- 其余 20 cells 均不显著。

### Token 语义（B5 对回文本）

B5（~17 token，offset −17..−1）固定包含：

```text
...icts the expected behavior.\n\n Choose the single next action now.\n
<|im_end|>\n <|im_start|>assistant\n
```

即 **「Choose the single next action now.」最终指令句 + chat template 的 assistant turn 开始边界**——Skill 决定的状态被写入**紧邻 generation boundary 的最终指令上下文**，而不是技能叙述/代码片段（B0–B4 均无效）。

## 数据质量核验

- H20 residual / full_V suff / full_V nec 三项与 EXP09 **逐位一致**（float32）→ 干预管线零漂移；
- 48 任务 × 4 条目全量记录；exit 0 一次通过；seed 5010；
- sign-flip p 使用 20000 置换分块计算，BH 仅在 exploratory head×position family 内、suff/nec 分别校正。

## 解释结果 — 稀疏 head×position 收敛：KV0 × B5 为唯一因果 cell

> **H20 V 效应 = KV0 value head 在 B5（最末端 final-instruction/assistant-boundary 区域）写入的内容状态。** KV0×B5 一个 cell（suff +0.0448 ≈ full_V 103%）即完整复现 full-V 效应；KV1 中性、KV2/KV3 轻微抑制；B0–B4 全部无效。这是本链条上第一次把 causal 载体收缩到 **单一 GQA value head × 单一 token 区域**。

- 与 GQA 结构吻合：4 个 KV head 中 1 个承担全部内容路由，其余 3 个近乎空载/微抑制——**head 高度不对称**；
- 位置高度不对称：**全部写在共享上下文最末端**（final instruction + assistant 起始），早期技能叙述/代码文本（B0–B4）不承载可移植效应；
- 保守性：24 cells 全 exploratory + 双族 FDR，仅 KV0×B5 是量级可观的正 cell；KV0×B4、KV3×B3/B5 虽 q<0.05 但量级 <2.5% full，不进入优先 replication。

## 对证据链的影响

- 证据链收束为：**Skill → H18–H20 核心 → H20 V-projection → KV0 value head → B5 末端上下文**；
- 与 ACL 2026 *Patches of Nonlinearity* 对照进一步精确：不是"某层某头是通用 instruction vector"，而是 **KV0 在 generation-boundary 前把程序条件状态写入最终指令区**；
- "稀疏电路"从 6 层核心收窄到 **1 head × 1 区域**，为 EXP11 的精确 token-offset 与 query-head 归因提供了明确锚点。

## 下一步（决策触发）

按预注册决策树：

```text
稀疏 head × 稀疏位置（少数 cell）  ✓（KV0 × B5）
均匀分布                          ✗（KV1-3 ≈ 0/负，B0-B4 ≈ 0）
```

进入 **EXP11**（不再大范围扫描）：

> **causal KV head（KV0）× causal coarse bin（B5）→ exact token-offset 定位 → 对应 7 个 query heads → query→value path patching。** B5 内逐 token offset 扫描 KV0 的 suff/nec，随后定位读取 KV0 的 7 个 query heads（GQA: 4 KV heads × 7 query heads/组）中哪几个在 B5 位置读出该状态。

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
        |
        v
早期窗口内部 handoff 定位          H18-H20 紧凑核心（EXP08：H20 单层 62%，H15:H19 66%，
                                   H18-H20 即 97%；H15-H17 可移除）
        |
        v
残差 → Q/K/V 消费接口             是（EXP09：H20 KV suff 94%/nec 91%；V 单通道即 89%，
                                   K≈0、Q 负；chain_KV 恢复 full 的 84%；KV>Q 确认但为 V 主导）
        |
        v
GQA V-head × token-position        稀疏收敛（EXP10：KV0 单头承载全部 V 效应；B5 即 99.9%；
                                   KV0×B5 一个 cell = full_V 103%，q<0.001 双族；KV1-3 ≈ 0/负）
```

## 当前主张边界

EXP10 之后，可辩护的项目级声明：

> **可恢复的干预位点**：早期窗口内行为效应的因果载体集中在 **H18–H20 三层**（full_H15_H20 = +0.0784；suffix_H18_H20 = +0.0760，即 97% 全效应；H20 单层 +0.0488 为最强单层，但仅 62%，far from sufficient；H15–H17 可移除）。该恢复不需要干预 H21–H28，即可因果转移技能条件化动作偏好。下游两条通路（MLP 中间神经元 EXP06、注意力头输出 EXP07）均已被独立证伪为可移植中介，其 recovery 不对称是伴随表现。

> **消费接口**：H20 残差效应主要通过 block 20 的 **V（KV）投影状态**传递——suff +0.0460（94% 残差）/ nec +0.0443（91%），CI 均排除 0；V 单通道即 89%，K≈0，Q 为负；三层 chain（H18+H19+H20 的 KV/QKV 同时移植）恢复 full 的 82–84%。预注册 KV > Q 假设确认（配对差 +0.051/+0.050 均显著正），但机制细节是 **V 单通道主导**而非 K+V 组合。

> **稀疏 head×position 收敛**：H20 V 效应集中于 **单一 GQA value head（KV0）× 最末端 token 区域（B5，"Choose the single next action now" + assistant turn 边界）**——KV0×B5 一个 cell（suff +0.0448 / nec +0.0445，q 均 <0.001）即完整复现 full-V 效应（~103%）；KV1 近空载，KV2/KV3 轻微抑制；B0–B4 全部无效。procedural state 的路由接口是 **1 head × ~17 token 最末端区域**。

> **不能声称的**：①「H20 单层是 handoff state」——H20 仅 62% 且 full−H20 = +0.030 显著正；②「H15–H20 全宽均匀分布式」——H15–H17 单独为负、可移除；③「H15–H20 residual 本身就是最终因果载体」——Qwen2 block 为 joint computation，前层输出是后层整体计算的输入条件，H18–H20 内部各层各自的 causal contribution 仍未逐层定位；④「Q/K/V 三通道共同负载路由」——H20 上 K≈0、Q 为负，实际是 V 近单通道；⑤「V 投影状态就是最末端载体」——V 状态仍需进入 attention 加权聚合→MLP，其后各步是否可进一步归因仍未实验；⑥「多个 GQA KV head 协同负载」——H20 上 4 个 head 高度不对称，KV0 承载全部正效应；⑦「effect 广泛分布于 prompt 前段内容」——B0–B4 全无效，写入仅发生在 generation boundary 前的最终指令区域。

尚不可声明：

> 「真正的因果机制完全是一个分布式电路。」——已不成立；EXP10 证伪了分布式假设，路由接口是高度稀疏的。
> 「已定位到单个 GQA KV head / 单个 token 位置的完整路径。」——KV0×B5 定位完成，但 B5 仍有约 17 token 宽，且尚未归因到具体 query heads。

已验证的排除项（EXP01–EXP10）：

1. 全局线性操控方向（EXP02，否）；
2. 单 token 精确残差互换（EXP03，否）；
3. H21–H28 稀疏 MLP 神经元组作为因果中介（EXP06，无充分性/必要性）；
4. H21–H28 全 MLP 中间激活作为因果载体（EXP06，full-MLP 干预为负）；
5. H21–H28 注意力头输出作为因果中介（EXP07，AllAttention 为负；Top16 sufficiency null，necessity 仅 +2.3%）；
6. H20 单层作为独立 handoff state（EXP08，仅 62% full，full−H20 显著正）；
7. H15–H17 对早期窗口的必要性（EXP08，单层为负/≈0，H16–H20 与 full 不可区分）；
8. H20 Q 投影作为中介（EXP09，Q suff/nec 均为负，加 Q 反而压低 QKV<KV）；
9. H20 K 投影作为中介（EXP09，suff +0.0001 ≈ 0，nec 仅 +0.025）；
10. 多 GQA KV head 协同负载假设（EXP10：KV1 ≈ 0、KV2/KV3 轻微抑制，仅 KV0 承载正效应）；
11. Prompt 位置均匀/前段分布假设（EXP10：B0–B4 全无效，100% 集中于 B5 末端）。

尚未完成的验证：

1. B5 内逐 token offset 精确归因（EXP11 计划：KV0 × B5 → exact token offsets）；
2. 读取 KV0 的 7 个 query heads 中哪几个执行该 B5 位置的路由（EXP11 计划：Q→KV0 query→value path patching）；
3. V 投影状态之后 attention 加权聚合→MLP 的剩余归因；
4. H18–H20 状态的维度分解（SAE/SNMF）；
5. 跨模型复现；
6. 跨 Skill / 任务泛化。

> 「证据排除了十一个假设——全局线性操控、单 token 精确互换、MLP 神经元级中介、MLP 全量中介、注意力头输出中介、H20 单层 handoff、H15–H17 必要性、H20 Q 投影、H20 K 投影、多 KV head 协同负载、位置均匀分布——效应收缩为：H20 残差 → block 20 V-projection → KV0 value head → B5（最末端 17 token 指令区+assistant 边界）；下一步 EXP11 精确 token offset + 对应 7 个 query heads 归因。」

此措辞应保留，直到 EXP09 定位到各层内部投影（Q/K/V/MLP）的 causal handoff 结构。