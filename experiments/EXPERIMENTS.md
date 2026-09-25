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
| EXP10 | GQA V-head × token 位置定位？ | 完成，稀疏收敛 | **KV0 单头承载全部 V 效应**（direct +0.047 ≈ full +0.043）；**B5（最末端指令区）即 99.9% full**；KV0×B5 单 cell（suff +0.0448, q=0.0006 / nec +0.0445, q=0.0006）≈ full_V 的 103%；KV1≈0、KV2/3 微负；B0–B4 无效 | `a3ed568` |
| EXP11 | B5 内精确 offset + KV0 reader heads？ | 完成，收敛到边界标记 | **offset -5（`<|im_end|>`）即 56% full、-1（assistant 起始）21%**；reader heads **Q0 73% / Q3 37% / Q5 14%**（Q2/4/6 负）；sanity 全过（Q7–Q27 delta=0、leakage ratio=0、all7 重建=full 逐位）；**1 KV head × 2 边界 token × 3 query heads** | `6b48aee` |
| EXP12 | 冻结电路在新 family 独立复制？ | 完成，分层（confirmatory=FALSE） | **reader register 复制成功**：frozen readers {Q0,Q3,Q5} 4/4 新 family 显著正（suff +0.076/nec +0.062）、negative readers {Q2,Q4,Q6} 显著负、contrast +0.16/+0.17、措辞稳定；**writer 层未按冻结位点复制**：frozen offsets V suff −0.013（方向反）、负对照为正、cross 负、措辞间翻转；**非 absolute-position artifact**（同 token 同位置符号随 context 翻转）；3 硬 sanity 全 bit-exact；specificity：skill−direct 全正（procedural specificity） | `0d5d6c8` |
| EXP13 | 上下文条件化 schema 写点能否重映射？ | 完成，confirmatory=TRUE（Pattern A） | **writer 高度 schema-stable：USER_END 7/8 strata 选中**（唯一例外 config_command::canonical=FINAL_INSTRUCTION_END，其 runner-up 即 USER_END）；held-out 双向 V suff/nec 全正（+0.0473/+0.0465，CI>0）、selected−old_absolute +0.061/+0.066、selected−runner_up +0.028/+0.031、cross-wording +0.044/+0.049、target reader 0.067 vs negative −0.028（contrast +0.095/+0.098）；old absolute 再次为负（−0.014）；sanity 全过（all7=selected=verified 精确相等、Q7..Q27 泄漏 0、non-KV0 0）；label0/label1 均正 → bidirectional_writer_pass | `d706328` |
| EXP14 | 跨模型功能同构复现（冻结机制定义、不冻结索引）？ | 完成，**PARTIAL SUCCESS** | Model B=Qwen2-7B-Instruct（同 Qwen2 架构、独立权重、通用域）。discovery 冻结 **L=20,KV0**（自由搜索下与 Model A 的 H20/KV0 **逐索引一致**）、reader **{1,3,5}+/{2,4,6}−**（负集合与 Model A 完全一致）；held-out：selected V suff/nec +0.025/+0.027（CI>0）、label0/1 双向正、cross-wording +0.019、reader pos +0.038 vs neg −0.012（contrast +0.050）、non-set≈0、all28==verified 精确、skill≫direct specificity、old-absolute 仍负；**但 same-state 控制 −0.012 显著负（Model A=0）、leakage 0.153（Model A=0）、selected−runner_up nec −0.108 反常** → 结构同构复现、保真度降低 | `79388ea` |
| EXP15 | 跨架构功能同构复现（冻结机制定义、不冻结索引）？ | 完成，**PARTIAL / ALGORITHMIC HOMOLOG** | Model C=Mistral-7B-Instruct-v0.3（Mistral 架构：32L/32Q/**8KV**/128，与 Qwen2 GQA 不同族）。闸门 PASSED（pooled +1.230，4 family/双 label/双措辞全正）；discovery 冻结 **L=30,KV4、GENERATION_BOUNDARY（8/8 strata，=提示词尾 `action now . [/INST]`）、reader {0,16,18}+/{1,17,19}−（活跃 16+/19−）**——与 Model A/B 的 L20/KV0/USER_END/{0,3,5}/{2,4,6} **索引实现全部不同**（功能冻结、索引不冻结）；held-out：selected V suff/nec **+0.140/+0.151**、cross-wording +0.141（=same-wording，完全迁移）、reader pos +0.165 vs neg −0.020/−0.025（contrast +0.185）、non-set=0、leakage **0.0**、all32==verified 精确、skill≫direct（6.2×）、matched-negative 干净 null、label0/1 双向正、4 family 全正；**但 same-state +0.002 CI 不含 0（1.4% 污染）、runner_up nec +2.37 异常（非边界位点残差通道主导 → V 接口边界特异）** → **功能组织跨架构复制成功、索引实现不同、保真度轻微降级** | `3c19971` |

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
`a3ed568`

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

# EXP11 — Exact KV0×Token → Query-Head Value Path（精确 KV0×token → query-head value path）

## 状态
已完成 — **B5 内部收敛到 4 个因果 offset（-13/-5/-3/-1，其中 -5=final instruction 句号后段 55%、-1=assistant 起始 21%）；KV0 的 7 个 reader query heads 中 Q0/Q3/Q5 承载全部 V 效应（Q0 suff +0.0326 = 73%），Q2/Q4/Q6 为负；三条 sanity 全部通过（leakage ratio = 0.0）**。

## 日期
2026-09-22

## 提交
`6b48aee`

## 科学动机

EXP10 把 V 效应收敛到 **KV0 × B5**（B5 = 最末端 17 token 的 final-instruction + assistant 边界区域）。本轮不再大范围扫描，沿已收敛的路径做两步精确定位：

1. **B5 内逐 token-offset 扫描**：`single-token suff / nec` + `leave-one-token-out loss`，逐 endpoint 做 task-level bootstrap CI + sign-flip p + BH-FDR；
2. **KV0 的 7 个 reader query heads**：Qwen2.5-Coder-7B-Instruct 为 28 Q heads / 4 KV heads，`repeat_kv` 连续重复 7 次 → **KV0 → Q0–Q6**（实现决定的映射，非假设）。

关键方法学改进（区别于 EXP07 直接搬 donor post-attention output）：对每个 candidate 分别计算 **baseline 与 KV0×B5 V-patch 两次 forward 的 block20 o_proj input delta**，然后只传递某个 reader head 的真实 delta（path sufficiency：baseline 上游 + Qh 的 patched delta；path necessity：V patch 上游 + 把 Qh 钳回 baseline）。因为上游只改 V、Q/K 不变，观测到的 per-query-head delta 是 KV0×B5 V intervention 实际造成的下游响应——真正的 path patching。

## 设计

- 与 EXP04–EXP10 完全一致：same task / same wording / opposite state donor / exact common suffix；48 任务，任务级 paired bootstrap 95% CI（seed 5111）。
- 干预位点：block20 `v_proj` 输出（KV0 head × 指定 token offset，pre-RoPE）+ block20 `o_proj` **输入**（28 query-head 拼接，head 切片）。
- Part A offsets：B5 内每个 offset（-1..-20）单 patch single-token suff / nec / LOO；offset 全标记 exploratory，逐 endpoint BH-FDR。
- Part B readers：对 Q0–Q6 各做 path suff / necessity_loss；含 3 条硬性 sanity：
  - `all7 reconstruction`（Q0–Q6 all7 delta 一起传 ≈ KV0×B5 full V effect）；
  - `all7 removal`（Q0–Q6 all7 钳回 baseline ≈ 移除 KV0×B5 full V effect）；
  - `Q7–Q27 induced delta` 应理论上 ≈ 0（KV0 V 只进入 v-states 0–6），`non_kv0_delta_ratio` 度量泄漏。

## 主要结果（48 个任务）

### Sanity checks（全部严格通过）

| check | 值 | 预期 | 结果 |
|---|---:|---:|---|
| `verified_KV0_B5_effect_mean`（reader 通道复现 EXP10） | **+0.0448** [0.0421, 0.0475] | ≈ EXP10 KV0×B5 +0.0448 | ✓ 逐位一致 |
| `all7_reconstruction_sufficiency_mean` | **+0.0448** [0.0420, 0.0476] | ≈ full V effect | ✓ 逐位等于 |
| `all7_removal_necessity_mean` | **+0.0448** [0.0421, 0.0476] | ≈ 移除 full V effect | ✓ 完全移除 |
| `Q7_Q27_control_sufficiency_mean` | **0.0000** | ≈ 0 | ✓ 精确 0 |
| `Q7_Q27_control_necessity_mean` | **0.0000** | ≈ 0 | ✓ 精确 0 |
| `mean_non_KV0_reader_delta_ratio` | **0.0** | ≈ 0 | ✓ 零泄漏 |

**三条硬性 sanity 全部成立** → 单 Q head 结果可解释；GQA 架构映射（KV0→Q0–Q6）被实证确认。

### Part A — B5 内 exact offsets 定位（`offsets_fdr_positive_in_both`）

仅以下 offset 同时满足 single-token suff q<0.05、nec q<0.05、mean>0：

| offset | modal token | suff mean | suff q | nec mean | nec q |
|---|---:|---|---:|---|---:|
| -13 | `.\n\n`（"next action now."句号） | +0.0075 | 0.0003 | +0.0068 | 0.0003 |
| -9 | ` next`（"Choose the single..."） | +0.0009 | 0.048 | +0.0011 | 0.003 |
| -5 | `<|im_end|>` | **+0.0252** | 0.0003 | **+0.0263** | 0.0003 |
| -3 | `<|im_start|>` | +0.0016 | 0.011 | +0.0015 | 0.0007 |
| -1 | `\n`（assistant 起始） | **+0.0095** | 0.0003 | **+0.0096** | 0.0003 |

- 其余 offset（-20..-6, -4, -2）全部 ≈0 或微负、FDR 不显著；
- **单个 offset -5（`<|im_end|>`）即 suff +0.0252 = full KV0×B5 的 56%**；-1（assistant 起始换行）再 +0.0095（21%）→ 两者合计 77%；
- LOO（leave-one-out）与 single-token 一致：-5 与 -1 的 LOO loss 最大——B5 效应不是均匀地铺开，而是写死在 **chat template 边界标记**（`<|im_end|>`、`<|im_start|>`、`\n`）及紧邻的指令尾部。

### Part B — KV0 reader query-heads 定位（`reader_heads_fdr_positive_in_both`）

| Q head | path suff mean | path suff q | path nec mean | nec q |
|---|---:|---|---:|---|
| **Q0** | **+0.0326** | 0.00007 | **+0.0319** | 0.00006 |
| Q1 | +0.0011 | 0.005 | +0.0019 | 0.0001 |
| **Q3** | **+0.0167** | 0.00007 | **+0.0167** | 0.00006 |
| **Q5** | **+0.0062** | 0.00007 | **+0.0060** | 0.00006 |
| Q2 | −0.0021 | — | −0.0028 | — |
| Q4 | −0.0061 | — | −0.0066 | — |
| Q6 | −0.0036 | — | −0.0040 | — |

- **Q0 单 head 即 73% full V（suff +0.0326 vs full +0.0448）**；Q3（37%）、Q5（14%）为正；Q1 微小正；
- Q2/Q4/Q6 为负（转移其 delta 反而干扰）；
- 正贡献 Q0+Q1+Q3+Q5 ≈ +0.057，减去 Q2/Q4/Q6 的负贡献后 ≈ full +0.045——**reader 头部内 delta 近似线性叠加**；
- **理想收敛形态部分达成**：1 个 KV head（KV0）× 4 个 causal offset（-13/-5/-3/-1，核心 -5+−1）× 3 个正 reader heads（Q0/Q3/Q5，Q0 主导 73%）。

## 数据质量核验

- `verified_KV0_B5_effect` = +0.0448 = EXP10 KV0×B5 +0.0448 逐位一致（float32）；
- `all7` 与 `full_KV0_B5` 全部逐位等于——delta 采集/重注入零误差；
- Q7–Q27 delta = 0.0、leakage ratio = 0.0——架构 sanity 实证通过，无 head 映射串位；
- 48 任务 × 4 条目全量记录；exit 0 一次通过；seed 5111。

## 解释结果 — B5 边界标记是 KV0 V 状态的载体，Q0/Q3/Q5 是读取者

> **KV0 在 B5 写入的 V 状态，被 block20 的 Q0（主导 73%）、Q3、Q5 三个 query heads 读取。** 写入位置不是均匀铺满 B5，而是集中在 chat template 边界：`<|im_end|>`（-5，56%）与 assistant 起始 `\n`（-1，21%）——即「用户消息结束 → assistant 开始生成」的边界区间。

- 与 GQA 架构一致：KV0 → v-states 0–6 → 仅有 Q0/Q3/Q5 真正读取该状态（Q1/微，Q2/Q4/Q6 抑制）；
- 行为链路进一步收束：**Skill → H18–H20 → H20 V(KV0) → B5 边界 token → Q0/Q3/Q5 读取 → o_proj → 动作偏好**；
- Q2/Q4/Q6 为负：这些 head 的 patched delta 在别处是正常上下文计算的一部分，强行传递反而破坏行为（与 EXP10 KV2/KV3 轻微抑制类似）。

## 对证据链的影响

- 证据链核心机理解释力首次被压缩到：**1 KV head × 2 个边界 token（-5/-1）× 3 个 query heads（Q0/Q3/Q5）**；
- 支持「Skill 状态被写进 generation-boundary 前的 chat-format 边界标记」——与 chat template 的结构性作用（决定回复以何种身份/格式生成）一致；
- EXP07 的排除不受影响：这里干预的是 block20 内部 V→Q path（读侧），EXP07 干预的是 H21–H28 下游 attention head output（写侧下游）。

## 下一步（决策触发）

EXP11 已把路径收敛到很小规模。按预注册：

> **即使 EXP11 很漂亮，也暂不宣称「完整 circuit 已证明」。** 下一轮（EXP12）应把 EXP11 发现的 **exact offsets（-13/-5/-3/-1）+ query heads（Q0/Q3/Q5）冻结为预注册目标**，在**新的 Skill/task family** 上做独立 replication——比在同一 48 个 synthetic tasks 上继续往更细层级钻，更接近真正可支撑 CCF-A 的证据。若 replication 通过，则可正式提出 candidate circuit：

> **Skill-conditioned procedural state is written into a small set of generation-boundary tokens (im_end→assistant start), stored in H20 KV0 value states, and read by a small subset (Q0/Q3/Q5) of the seven GQA query heads.**

---

# EXP12 — Frozen-Circuit Independent Replication（冻结电路的独立复制）

## 状态
已完成 — **confirmatory_replication_pass = FALSE（分层结果）**。冻结 **reader register 层独立复制成功**：frozen readers {Q0,Q3,Q5} 在全部 4 个新 family 显著为正（suff +0.076 / nec +0.062），negative readers {Q2,Q4,Q6} 显著为负，frozen−negative reader contrast +0.16/+0.17，且措辞稳定；但冻结 **positional 编码层未复制**：frozen offsets {-13,-5,-3,-1} 的 V suff/nec 均显著为负（-0.013/-0.015，方向反转），内容 token 负对照 {-12,-10,-8,-6} 反而为正（+0.011/+0.012），frozen−negative offset contrast 显著为负，cross-wording V 亦为负。三条硬 sanity 全部 bit-exact 通过。specificity falsification：冻结路径在 procedural Skill 下显著强于 direct action instruction（skill−direct 全部 5 指标 × 4 family 显著为正）→ 支持 procedural specificity。

## 日期
2026-09-22

## 提交
`0d5d6c8`

## 科学动机

EXP11 在**同一批 48 synthetic tasks** 上发现「1 KV head × 4 边界 offset（-13/-5/-3/-1）× 3 reader heads（Q0/Q3/Q5）」。按 EXP11 预注册，EXP12 **不做任何 circuit discovery**：把 EXP11 的发现完全冻结为预注册目标，在**全新任务/全新 Skill family/全新 action 词表**上做独立复制，检验这一稀疏路径是否任务一般（task-general），并加一个对论文至关重要的 falsification——procedural Skill vs direct action instruction。

## 设计

- **冻结电路**：H20 → block20 V → KV0 → offsets {-13,-5,-3,-1} → readers {Q0,Q3,Q5} → action preference。无发现性扫描。
- **独立数据**：64 个全新任务 = 4 个新 Skill family（test_edit / search_edit / config_command / docs_code）× 16；action 词表全部更换（run_tests / open_file / search_code / inspect_config / run_command / search_docs 等）；SKILL_TEXT 4 family × 2 label × 2 wording（canonical / paraphrase）。
- **预注册双负对照**（事后不可改）：Frozen offsets {-13,-5,-3,-1}（边界结构 token）vs Negative offsets {-12,-10,-8,-6}（普通内容 token）；Frozen readers {Q0,Q3,Q5} vs Negative readers {Q2,Q4,Q6}。
- **confirmatory 判定**（全部 10 项需同时成立，全 task-level paired bootstrap 95% CI）：4 endpoints（frozen V suff>0、frozen V nec>0、target reader suff>0、target reader nec>0）+ 4 paired contrasts（frozen−negative offsets suff/nec、frozen−negative readers suff/nec）+ 2 cross-wording flags（cross frozen V suff/nec>0）。
- **硬 anchor 断言**：模型加载前验证 offset -5 解码 == `<|im_end|>`、-3 == `<|im_start|>`，不满足立即停止（防 tokenizer 结构漂移）。
- **specificity falsification**：同一批任务以 direct action instruction（"The required next action is ..."）重跑，用完全相同的冻结电路测；skill−direct 差异决定能否叫 procedural Skill circuit 还是只能叫 instruction-conditioned action-selection boundary circuit。
- 保留 EXP11 三条硬 sanity：all7 重健=full V、all7 移除=full、Q7–Q27 induced delta≈0 + leakage ratio 度量；o_proj-input delta path patching 原样沿用。
- 架构断言：28 Q / 4 KV heads、ratio 7、block20 v_proj 几何，不符即停。seed 5212，phase=all（双 cohort）。

## 主要结果（64 个新任务）

### Anchor 预审计（模型加载前硬断言，3072 行全覆盖）

| offset | decoded token | 角色 |
|---|---:|---|
| -13 | `")\n\n` | 边界结构（assistant tool-call 收尾）✓ |
| **-5** | **`<|im_end|>`** | 边界（硬断言通过，151645）✓ |
| **-3** | **`<|im_start|>`** | 边界（硬断言通过，151644）✓ |
| -1 | `\n`（assistant 起始） | 边界 ✓ |
| -12 / -10 / -8 / -6 | `Choose` / ` single` / ` action` / `.` | 普通内容 token（负对照语义干净）✓ |

### Sanity checks（全部 bit-exact 通过）

| check | 值 | 预期 | 结果 |
|---|---:|---:|---|
| `all7_minus_verified_suff_mean` | **0.0** | ≈ 0 | ✓ |
| `all7_minus_verified_nec_mean` | **0.0** | ≈ 0 | ✓ |
| `non_kv0_reader_sufficiency_mean` | **0.0** | ≈ 0 | ✓ |
| `non_kv0_reader_necessity_mean` | **0.0** | ≈ 0 | ✓ |
| `mean_reader_leakage_ratio` | **0.0** | ≈ 0 | ✓ 零泄漏 |
| `max_abs_reader_baseline_margin_diff` / `max_abs_reader_v_effect_diff` | **0.0** | ≈ 0 | ✓ o_proj 管线与 logit 管线逐位一致 |

→ 测量装置在新 family 上完好（Q0–Q6 all7 精确重建 V 效应、Q7–Q27 精确零 delta），reader 分解数字可信。

### 预注册 confirmatory 判定（全部 task-level paired bootstrap）

| 判据（n=64） | mean | 95% CI | 通过? |
|---|---:|---|---|
| frozen_V_sufficiency | **-0.0131** | [-0.0194, -0.0071] | ❌ 显著为负 |
| frozen_V_necessity_loss | **-0.0150** | [-0.0211, -0.0091] | ❌ 显著为负 |
| target_reader_sufficiency | **+0.0763** | [+0.0726, +0.0801] | ✅ |
| target_reader_necessity | **+0.0620** | [+0.0597, +0.0642] | ✅ |
| frozen_offsets−negative_offsets suff | **-0.0240** | [-0.0308, -0.0172] | ❌ frozen 输 |
| frozen_offsets−negative_offsets nec | **-0.0274** | [-0.0346, -0.0204] | ❌ frozen 输 |
| frozen_readers−negative_readers suff | **+0.1609** | [+0.1519, +0.1697] | ✅ |
| frozen_readers−negative_readers nec | **+0.1672** | [+0.1585, +0.1760] | ✅ |
| cross_frozen_V_sufficiency | -0.0130 | [-0.0193, -0.0067] | ❌ 显著为负 |
| cross_frozen_V_necessity_loss | -0.0128 | [-0.0197, -0.0060] | ❌ 显著为负 |

**`confirmatory_replication_pass = FALSE`（10 项中 5 过 5 败：reader 4 项全过；positional-V 4 项 + cross 2 项全败）**

### Family 级复制

| family | frozen_V_suff | frozen_V_nec | reader_suff | reader_nec |
|---|---:|---:|---:|---:|
| test_edit | **+0.0044** ✅ | **+0.0060** ✅ | +0.0937 ✅ | +0.0693 ✅ |
| search_edit | **-0.0153** ❌ | **-0.0134** ❌ | +0.0760 ✅ | +0.0672 ✅ |
| config_command | +0.0104 ✅ | +0.0014（CI 过 0）~ | +0.0592 ✅ | +0.0530 ✅ |
| docs_code | **-0.0518** ❌ | **-0.0540** ❌ | +0.0764 ✅ | +0.0584 ✅ |

→ **frozen readers {Q0,Q3,Q5} 4/4 family 同向显著正**；frozen-offset V 编码方向随 family 翻转（2 正 2 负）。

### Specificity falsification（procedural Skill[canonical] vs direct action instruction）

| metric | skill_mean | direct_mean | skill−direct [95% CI] |
|---|---:|---:|---|
| H20_residual_reference | 1.2919 | 1.1577 | **+0.1342** [+0.0635, +0.2039] |
| frozen_V_sufficiency | +0.0073 | **-0.0644** | **+0.0717** [+0.0637, +0.0791] |
| frozen_V_necessity_loss | +0.0027 (不显著) | -0.0309 | +0.0336 [+0.0278, +0.0392] |
| target_reader_sufficiency | +0.0998 | +0.0320 | +0.0678 [+0.0629, +0.0727] |
| target_reader_necessity | +0.0832 | +0.0236 | +0.0596 [+0.0549, +0.0644] |

→ **skill−direct 全部 5 指标 × 4 family 显著为正**（含 docs_code V suff +0.0721）。direct 指令下冻结 V 效应显著为负（-0.0644）→ 冻结 reader 路径是 **procedural-Skill 条件化**的，而非通用的指令动作选择边界。direct 基线行为准确率 100%、skill 94.9%（任务行为层完全可用）。

### 关键伴随发现

- **H20 residual 转移 +1.164 [1.109, 1.220]**——新 family 上比旧 48 任务（+0.049）强 ~24 倍，且 block20 KV0 V 分支只承担其中 ~1% → 新 family 上动作选择转移的因果主通道在残差的其他分支。
- **negative offsets（内容 token）V suff +0.0109 / nec +0.0124 显著为正**，与 frozen 符号相反。
- **措辞不稳定性（新发现）**：canonical-only skill frozen_V_suff +0.0073 vs 全措辞（canonical+paraphrase）-0.0131 → paraphrase 条目约 -0.034；**边界-V 方向在语义等价的两种措辞间翻转**。reader 层措辞稳定（canonical-only +0.0998 vs 全措辞 +0.0763，同号）。
- **same_state_frozen_V_control -0.0065 [-0.0084, -0.0047]**（同态对照轻微负，语义控制杂质可控）。
- **机械解释**：块20 对 frozen-V patch 的实际下游响应 = Q0/Q3/Q5（促 donor）+0.076 ⊕ Q2/Q4/Q6（推 anti-donor）-0.085 ≈ 净 -0.013 → 新 family 上正负 reader **近对消并把净方向翻负**；EXP11 中 Q2/Q4/Q6 仅轻微负。
- baseline_behavior_accuracy（skill）= 94.9%、baseline_signed_preference = +0.633 [0.606, 0.661]。

## 数据质量核验

- replication_results 5120 行（64 task × 4 entries × 20 metrics）+ specificity_results 1536 行（64 × 2 direct × 12），64 task 全覆盖；
- 3 条硬 sanity 全部 bit-exact（all7−verified=0.0、non_kv0=0.0、leakage=0.0、margin/effect diff=0.0）；
- anchor 预审计 3072 行，硬断言 -5==`<|im_end|>`、-3==`<|im_start|>` 模型加载前通过；
- exit 0 一次通过；seed 5212；双 A6000；run_manifest 记录 git_commit=9544113（运行基线）、script/model config sha256。

## 解释结果 — 分层复制：reader register 稳健，writer 编码上下文条件化（非 absolute-position artifact）

> **EXP12 把 EXP11 的发现拆成两个复制状态不同的结论：** ①**reader register 层（哪些 query heads 读 KV0 的 value 输出）——任务一般性成立**：Q0/Q3/Q5 在 4 个全新 family、全新词表、两种措辞下全部显著促 donor，Q2/Q4/Q6 显著推 anti-donor；②**writer 层（V 状态写在哪里、以何种极性）——未按冻结位点复制**：在新 family 上方向反转（V 效应显著为负）、内容 token 负对照反而为正、cross-wording 均负、措辞间符号翻转。

> **重要修正（2026-09-23 用户实际核验 outputs 后）——不能解释成单纯的 absolute-position artifact**：`token_anchor_audit.csv` 显示 frozen offsets {-13,-5,-3,-1} 在 4 个 family、canonical/paraphrase、两个 label 下解码出的 token 完全一致（-5 恒为 `<|im_end|>`、-3 恒为 `<|im_start|>`、-1 恒为 `\n`、-13 恒为 `")\n\n`），但冻结 V 的因果方向仍强烈依赖上下文。例如 same-wording frozen-V suff：test_edit canonical +0.0115 vs paraphrase −0.0027；config_command +0.0369 vs −0.0161；docs_code −0.0280 vs −0.0756。且 action direction 不对称：test_edit 中 recipient label0 frozen V suff = −0.0480、label1 = +0.0567。同 token 同位置在不同 instruction context 可携带不同甚至相反的因果作用 → 这不是"换个 offset 就能解决"的现象。

- reader 注册表（KV0→Q0/Q3/Q5 读、Q2/Q4/Q6 抑）在 GQA 架构层面稳定：新 family 上 Q2/Q4/Q6 由 EXP11 的"轻微负"变为强负，正负头近乎对消，把净 V 方向翻负——**读侧选择性稳定，正负平衡随任务族变化**；
- 写侧机制的最可辩护表述改为 **context-conditioned writer/code → stable reader register**：write site 与局部 value code 随 family×wording×label 上下文条件化，而非可移植的固定位点；
- specificity 结果守护 reader 层：冻结路径在 procedural Skill 下显著强于 direct 指令（skill−direct 全正）→ 即使 writer 层未按冻结位点复制，reader 路径的行为耦合仍具 procedural 特异性。

## 对证据链的影响

- 证据链**不升级**为「task-general frozen sparse circuit」（confirmatory=FALSE）；
- 可辩护的最强声明修正为：**task-general reader register（{Q0,Q3,Q5} 正读 / {Q2,Q4,Q6} 抑读；procedural 条件化；跨 family / 词表 / 措辞复制成功）＋ context-conditioned writer（write site 与局部 V 极性随 family×wording×label 上下文变化，非 absolute-position 可移植载体）**；
- reader 层证据强度从 discovered（EXP11）升级为同模型跨任务族 replicated（EXP12）；writer 层停留 discovered 且未按固定位点复制——触发 EXP13 语义 anchor 重映射检验。

## 下一步（决策触发）

按 EXP11 的预注册约定，EXP12 未通过 confirmatory → **证据链不升级**。用户实际核验 outputs 后修正方向（2026-09-23）：**不做简单的 offset 重扫**——同 token 同位置的符号翻转说明这是 context-conditioned writer 而非可移植位点问题。下一轮 **EXP13 — Context-Conditioned Schema Write-Site Remapping**（预注册）：在 4 family × 2 wording = 8 strata 内对 9 个语义 anchor（SKILL_END/SYSTEM_END/ISSUE_END/DETAIL_END/ACTION0_END/ACTION1_END/FINAL_INSTRUCTION_END/USER_END/GENERATION_BOUNDARY，各 6-token window）做双向 writer discovery（score = min(S0,S1,N0,N1)），8/8 任务 held-out 确认，reader register {Q0,Q3,Q5} 全冻结不允许重选。判定：writer+reader 双过且 strata 选点不同 → context/schema-dependent writer → stable reader register（强于 EXP11 fixed-token circuit）；writer 败 reader 过 → 永久停止 token-position scan，转向"Q0/Q3/Q5 读取的功能性隐变量"识别。

---

# EXP13 — Context-Conditioned Schema Write-Site Remapping（上下文条件化 schema 写点重映射）

## 状态
PLANNED — **基于 EXP12 outputs 实际核验修正设计**：EXP12 不能解释为 absolute-position artifact（同 token 同位置符号随 family/wording/label 翻转），故不再做 offset 重扫，改为检验 **context-conditioned writer → stable reader register** 假设。

## 日期
2026-09-23

## 提交
d706328

## 科学动机（EXP12_OUTPUT_AUDIT 摘要）

EXP12 实际 outputs 核验（summary.json / replication_summary.csv / replication_contrasts.csv / family_summary.csv / specificity_comparison.csv / replication_results.csv / token_anchor_audit.csv / run_manifest.json）得出：

1. `frozen_V_sufficiency = -0.013089`、`frozen_V_necessity = -0.014992`（负）；
2. 负对照（内容 token）反而为正（+0.010903 / +0.012403）；
3. target readers 保持正（+0.076307 / +0.061981），negative readers 强负（-0.084568 / -0.105234）；
4. target−negative reader contrasts = +0.160875 / +0.167215；
5. all7 重建/移除 exact、non-KV0 leakage 0；
6. 冻结 offset token 在全部 family/wording 条件下完全相同（-13 `")\n\n`、-5 `<|im_end|>`、-3 `<|im_start|>`、-1 `\n`）；
7. canonical−paraphrase frozen-V suff 系统性差（config_command 约 +0.053、docs_code +0.048、search_edit +0.048、test_edit +0.014）；
8. 多个 family×wording×label cell 符号反转 → frozen 写点下 writer 不可双向移植；
9. reader register 在所有 family 与两种措辞下保持正，但 paraphrase 下幅度更弱。

→ 拒绝简单 absolute-position 解释，驱动 EXP13 的 family×wording writer discovery + 显式双向性约束。

## 设计（预注册）

- **冻结组件**：H20 → block20 V → KV0 → readers {Q0,Q3,Q5}（target）/ {Q2,Q4,Q6}（negative）/ all7 / non-KV0；reader 与 layer/head 全部冻结，不允许重选。
- **数据集**：精确复用 EXP12 64 任务（import `experiments/exp12_independent_replication/run.py`，importlib 加载，len==64 硬断言，无任务模板复制）。
- **Split**：每 family 16 任务确定性 seeded shuffle（split seed 5313，per-family RNG = 5313 + family index）→ 8 discovery / 8 confirmation（32 held-out 任务）。
- **9 个语义 anchor**（各固定 6-token window，终止于该语义边界）：SKILL_END、SYSTEM_END、ISSUE_END、DETAIL_END、ACTION0_END、ACTION1_END、FINAL_INSTRUCTION_END、USER_END、GENERATION_BOUNDARY。SKILL_END/SYSTEM_END 允许 writer 落在 Skill/system 区（此前只搜公共后缀根本找不到）。
- **8 个 discovery strata**：4 family × 2 wording（canonical/paraphrase）。label 不作 strata（用于双向得分）。
- **双向选择得分**：`score(a) = min(S0(a), S1(a), N0(a), N1(a))`（S/N = discovery 任务 suff/nec 均值，0/1 = recipient label）。选最高分，tie-break 用固定 anchor 序；同时冻结 runner-up。
- **necessity 局部化**：每个 anchor 单独检验其自身 H20 → KV0-V 局部 mediation：residual_eff = donor H20 residual 在 anchor window 局部恢复；suff = donor KV0 V 在该 window 单独传递；nec = residual_eff − retained（retained = donor H20 局部 residual + 该位置 KV0 V clamp 回 recipient baseline）。不再用"公共后缀整体 residual patch"去解释可能位于 system/Skill 区的 writer。
- **Held-out confirmation**（每 strata）：selected V suff/nec、OLD_ABSOLUTE {-13,-5,-3,-1}（降级为 historical diagnostic，4 稀疏 token 不作 6-token window 的主控制）、runner-up（matched-width control）、label0/1 分别、cross-wording（recipient 用自己 wording 的 selected anchor、donor 用自己 wording 的 selected anchor = 功能位置映射）、same-state 控制、reader {Q0,Q3,Q5} vs {Q2,Q4,Q6}（全冻结）、all7 重建/移除 + Q7–Q27 零泄漏。
- **判定**：writer_pass = selected V suff/nec CI>0 + cross suff/nec CI>0 + selected−runner_up suff/nec CI>0 + bidir_ok（label0/1 均 CI>0）；reader_pass = target suff/nec CI>0 + target−negative reader suff/nec CI>0；sanity_pass = all7−verified≈0、non_kv0≈0、leakage<1e-6、baseline/verified diff<5e-5；confirmatory_pass = 三者全过。机制 pattern：strata 选点唯一 → schema_stable_writer_plus_stable_reader；>1 → contextual_writer_plus_stable_reader；writer 败 reader 过 → portable_writer_not_confirmed_stable_reader_survives（→ 永久停止 token-position scan，转向功能性隐变量识别）。

## 输出

`outputs/exp13_contextual_write_remap/`：split.json、schema_audit.csv、discovery_results.csv、discovery_summary.csv、selected_schema.json、confirmation_results.csv、confirmation_summary.csv、directional_confirmation.csv、family_wording_confirmation.csv、paired_contrasts.csv、summary.json、run_manifest.json（+ discovery_profile.png / confirmation_profile.png）。

## 运行

```bash
python experiments/exp13_contextual_write_remap/run.py --phase discovery   # 先 discovery
python experiments/exp13_contextual_write_remap/run.py --phase confirmation  # 不人工改 selected_schema.json
```

---

# EXP14 — Cross-Model Functional Homolog Replication（跨模型功能同构复现）

## 状态
PREREGISTERED（2026-09-25，设计先行）→ **已执行，判定 PARTIAL SUCCESS**。本机只有 Qwen2.5-Coder-7B-Instruct 与 Qwen2-7B-Instruct 两个模型（无跨架构模型，且约束离线不下载），故"跨模型"= 同 Qwen2 架构、独立权重、不同预训练域（通用 instruct vs Coder）的**第二个 checkpoint**。跨架构（Llama/Mistral/Gemma）复现留作未来工作（需额外模型）。

## 动机
EXP12/13 在同模型上把机制收敛为「schema-anchored writer（USER_END 族）→ 单 block 的 V(KV) 接口 → stable reader register（正/抑）→ 行为」。尚未验证的最大 generalization gap 是**跨模型**：该功能结构是否存在于独立训练、不同域的模型中。EXP14 冻结机制定义、不冻结数字索引，在 Model B 的 discovery split 上搜索功能同构组件，在 held-out confirmation 上冻结验证。

## 冻结的机制定义 M（不允许改）
在 procedural skill 任务中动作偏好由以下功能结构控制：
- **W1 writer**：recipient 上下文中的 schema 锚定 token 窗（语义边界；Model A 中 USER_END 族）在残差流中携带 donor 条件化状态；
- **W2 causal interface**：该状态经**单个 decoder block** 的 V 投影、由**单一高效 KV head** 写入（Model A：H20 残差 → block20 V-proj → KV0）；
- **W3 reader register**：少量 query heads **正读**（suff>0）、少量**抑读**（suff<0），其余 ≈0（Model A：3 正 + 3 负）；
- **W4 behavior**：V 状态移动 mean candidate-token log-prob margin 至 donor 偏好；procedural skill 条件强于 direct 指令（specificity）。

## 不冻结的（在 Model B discovery split 上搜索；选点后于 confirmation 全部冻结）
- source block **L ∈ {16..27}**（deep band，12 层）；
- KV head **k ∈ {0..3}**（4 头全搜）；
- 每 strata 的 **writer anchor**（9 个候选，同 EXP13 选择式）；
- **reader register**（28 个 query heads 中按 discovery suff 取 top-3 正 / bottom-3 负；集合尺寸 3+3 由 Model A 结构先验预注册，不按数据优化）。

## Split
精确复用 EXP12 64 任务与 EXP13 split（seed 5313，per-family RNG = 5313 + family index）→ **32 discovery / 32 confirmation**。task 为统计推断单位（bootstrap 在 task 级）。

## 阶段与选择规则（全部只使用 discovery split；无 post-hoc 挑选冒充 confirmatory）
- **Phase 0 gate（参与度门）**：全 64 任务 skill entries 的 signed baseline margin（label=1 取 +、label=0 取 −）pooled bootstrap CI 排除 0 且 4 family 均值同号 → 通过才继续。**gate FAIL → 报告 null**（Model B 不参与该任务族 → 机制在此模型上不可判定，如实报告，不换指标救）。
- **Phase A（L,k）粗选**：锚定 **{USER_END, FINAL_INSTRUCTION_END}**（Model A 指令边界功能族，两 anchor 均存在于全部 prompt），donor = opposite_same_wording；`score(L,k) = max_{a∈2} min(mean_suff_label0, mean_suff_label1)`（128 discovery entries）；选最大 → (L*,k*)；tie-break 按 (L 升序, k 升序)。
- **Phase B anchor 细选**：冻结 (L*,k*)；每 strata（family×wording）9 anchor 的 `score(a) = min(S0,S1,N0,N1)`（EXP13 同式，S/N = 该 strata discovery entries 的 suff/nec 均值）；选最高 + runner-up（双双冻结）。
- **Phase C reader 搜索**：冻结 (L*,k*) + 各 strata anchor；对 discovery entries 捕获 base/patched oproj 输入，逐 head 替换测量 signed suff；**pos_set = suff 最大 3 头，neg_set = 最小 3 头**。

## Held-out confirmation（32 任务，全部冻结）
- selected V suff/nec（pooled CI>0）；
- label0 / label1 分开（双向要求）；
- cross-wording（recipient 与 donor 各自用自己 wording 的 selected anchor = 功能位置映射）；
- same-state 控制（同 label 跨 wording donor，期望 ≈0）；
- matched-negative 控制（ISSUE_END 内容窗，diagnostic）；
- old-absolute {-13,-5,-3,-1}（historical diagnostic，Model A 上为负，预期仍负/≈0）；
- reader register：pos_set suff/nec CI>0、neg_set suff/nec CI<0、pos−neg contrast CI>0、all28 suff == verified（精确）、non-set ≈0（diagnostic）、leakage ≈0；
- procedural specificity：direct 条件 V suff（USER_END）与 skill 条件 V suff 的 contrast（diagnostic）。

## 判定（预注册）
- **SUCCESS**：gate 过；Phase A/B/C 全部产生非退化选择；confirmation 硬端点全过（selected V suff/nec CI>0、cross-wording CI>0、pos−neg contrast CI>0、same-state≈0、all28==verified 精确、label0/label1 均正）；且 writer anchor 属于**同一功能族**（指令边界：USER_END / FINAL_INSTRUCTION_END / SYSTEM_END / GENERATION_BOUNDARY 之一，而非内容位置）→ 跨模型功能同构复现成功，证据链升级为 model-general functional mechanism（同架构域内）。
- **PARTIAL**：reader register 复现（pos>0/neg<0/contrast）而 writer V 单向、或 anchor 偏离功能族、或 cross-wording 失败、或 specificity 翻转 → 部分同构；区分哪些组件泛化。
- **FAIL**：gate 不过、或 Phase A 最佳 (L,k) 无正效应、或 confirmation 硬端点半数不过 → 机制为 Model-A（Coder 域）特异；**修正机制模型**为「coder-family-specialized writer protocol」；reader register 的跨模型泛化性单独裁定（可能仍复现）。

## Exploratory（明确标注非预注册）
- 相邻 (L,k) 的 effect 剖面（确认非孤立点）；
- 各 anchor family 的得分排名（了解 writer 功能族的域转移）；
- family × wording × label 分层细表；
- 若 confirmation 失败，以 (L*,k*) 邻近选择做 exploratory 再检（不构成 confirmatory）。

## 输出
`outputs/exp14_cross_model_homolog/`：gate_report.json、discovery_results.csv、discovery_summary.csv、selected_homolog.json、confirmation_results.csv、confirmation_summary.csv、paired_contrasts.csv、directional_confirmation.csv、family_wording_confirmation.csv、summary.json、run_manifest.json。

## 运行
```bash
CUDA_VISIBLE_DEVICES=0 python experiments/exp14_cross_model_homolog/run.py \
  --model_dir /data/mzb/ar2_scratch/models/Qwen2-7B-Instruct --phase gate
CUDA_VISIBLE_DEVICES=0 python experiments/exp14_cross_model_homolog/run.py \
  --model_dir /data/mzb/ar2_scratch/models/Qwen2-7B-Instruct --phase discovery
CUDA_VISIBLE_DEVICES=0 python experiments/exp14_cross_model_homolog/run.py \
  --model_dir /data/mzb/ar2_scratch/models/Qwen2-7B-Instruct --phase confirmation
```

## 实验结果（2026-09-25，PARTIAL SUCCESS）

### Hypothesis
EXP13 在 Coder 域上确认的功能结构 M（schema-anchored writer → 单 block V(KV) 接口 → 正/抑 reader register → 行为）是否存在于独立训练、通用域的 Qwen2-7B-Instruct 上。

### Design（按预注册执行）
- 64 EXP12 任务，seed 5313 split → 32 discovery / 32 held-out；task 为推断单位（task 级 paired bootstrap CI，5000 draws）。
- Phase 0 gate：**通过**（pooled signed baseline +1.50，CI [1.40, 1.60]；4 family 均值全正；label0 +2.03 / label1 +0.98 均正）——通用模型完全参与任务。
- Phase A（12 层 × 4 KV × {USER_END, FINAL_INSTRUCTION_END}，48 组合）：**唯一显著正分 L=20,k=0 = +0.0157**（次位 L=19,k=3 = +0.0154；其余 44 组合 ≈0/负）→ **冻结 (20,0)**。与 Model A 的 H20/KV0 **逐索引一致**，尽管搜索完全自由。
- Phase B（每 strata 9 anchor，双向 min 得分）：config_command::canonical→USER_END (0.0258)、paraphrase→FINAL_INSTRUCTION_END (0.0198)、docs_code::paraphrase→FINAL_INSTRUCTION_END (0.0173)、search_edit::paraphrase→FINAL_INSTRUCTION_END (0.0181) 为有效选择（指令边界功能族 ✓）；test_edit::canonical→ACTION1_END (0.00005)、search_edit::canonical→DETAIL_END (0.0013)、docs_code::canonical→ACTION1_END (0.0016) 退化（≈0）。
- Phase C（28 头逐头 suff）：**pos_set={1,3,5}、neg_set={2,4,6}**（Model A 为 {0,3,Q5}+/{Q2,Q4,Q6}−；负集合完全一致，正集合 2/3 重叠）。

### Preregistered endpoints（held-out confirmation，32 任务全冻结）
| 端点 | mean | CI | pass |
|---|---|---|---|
| selected_V_sufficiency | +0.0254 | [0.0182, 0.0331] | ✅ |
| selected_V_necessity_loss | +0.0266 | [0.0206, 0.0327] | ✅ |
| cross_selected_V_sufficiency | +0.0195 | [0.0139, 0.0254] | ✅ |
| same_state_selected_V_control | **−0.0119** | **[−0.0181, −0.0066]** | ❌ 期望 ≈0 |
| matched_negative_ISSUE_V_suff | +0.0010 | [−0.0017, 0.0039] | ✅ |
| old_absolute_V_sufficiency | −0.1228 | [−0.1688, −0.0813] | 负（与 Model A 一致）|
| reader_target_pos suff/nec | +0.0380 / +0.0375 | [0.0278, 0.0492] / [0.0281, 0.0477] | ✅ |
| reader_negative_neg suff/nec | −0.0119 / −0.0120 | [−0.0153, −0.0086] / [−0.0160, −0.0085] | ✅ |
| reader_non_set suff/nec | −0.0016 / +0.0007 | [−0.0037, +0.0005] / [−0.0013, +0.0029] | ✅ |
| reader_all28 == verified == selected | 精确相等（+0.0254） | — | ✅ |
| reader_leakage_ratio | **0.153** | [0.135, 0.169] | ⚠️ Model A=0.0 |
| target−negative reader suff | +0.0499 | [0.0379, 0.0633] | ✅ |
| selected−old_absolute suff | +0.1482 | [0.1050, 0.1943] | ✅ |
| selected−runner_up suff | +0.0114 | [0.0081, 0.0150] | ✅ |
| selected−runner_up nec | **−0.1078** | **[−0.1540, −0.0645]** | ⚠️ 反常 |
| label0 / label1 selected V suff | +0.0165 / +0.0343 | 均 CI>0 | ✅ 双向 |
| direct_V_sufficiency（specificity） | +0.0035 | [0.0005, 0.0064] | ✅ skill(0.0254)≫direct |

**Verdict：PARTIAL SUCCESS**（`outputs/exp14_cross_model_homolog/summary.json`）。

### Null / negative findings（如实报告）
1. Phase A 在指令边界 anchor 上 pooled 近零：48 组合仅 2 个正分——writer V 接口在 Model B 上同样高度局部化（单点），但整体幅度约减半（+0.025 vs Model A +0.047）。
2. **same-state 控制显著为负**（−0.012）——Model B 的 (20,0) V-code 混入 wording 条件成分（Model A 上为 0）。此为保真度降低的直接证据。
3. **leakage 0.153**——oproj delta 能量约 2.3% 落在注册集外（Model A 精确 0）；行为上 non-set≈0 不受影响。
4. **selected−runner_up necessity 反常**（−0.108）：runner-up anchor 的 V 必要性（+0.134）远大于 selected（+0.027），而 suff 相反——Model B 写点在必要性维度分布更广。
5. 3/8 strata 的 anchor 选择退化（≈0.0001–0.0016）：test_edit::canonical、search_edit::canonical、docs_code::canonical 在 (20,0) 下无可辨别的 writer 位点。

### Bugs / fixes（如实记录）
- discovery Phase C `KeyError ('family','wording')`：Phase B 写字符串 key `"family::wording"`，Phase C 用元组查询 → 改字符串。
- confirmation cross/same donor stratum `KeyError`：查询漏 `["strata"]` 层级 → 补全。
- confirmation direct `compute_schema KeyError 'direct'`：direct entries 无 skill 文本，SKILL_TEXT 查询失败 → compute_schema 容错（缺失段跳过，仅 direct 允许无 SKILL_END）。

### Interpretation
- **结构同构确认**：自由搜索下 Model B 选出 (20,0) —— 与 Model A 的 H20/KV0 完全一致；reader register 负集合 {2,4,6} 与 Model A 完全一致、正集合 {1,3,5} 2/3 重叠；分解恒等式、cross-wording、双向性、specificity、old-absolute 失败模式全部复现。**writer 接口的层/头位点与 reader register 的符号结构不是 Coder 域特异**。
- **保真度降低**：效应减半 + same-state 污染 + leakage 0.15 + runner-up nec 反常 → Model B 的 (20,0) V-code 标签纯度低于 Model A（wording 成分共存），写点必要性在语义邻近位点分布更广。机制是"同构但更松散"的实现。

### Claim boundary
- **可声明**：Qwen2 架构家族内、域分歧 checkpoint（通用 vs Coder）上，功能结构 M 以 (L=20, KV0, +{1,3,5}/−{2,4,6} reader) 复现——跨模型 functional homolog（同架构域内）。
- **不可声明**：跨架构泛化（Llama/Mistral/Gemma 未测，需第三模型）；保真度等同（Model B 效应减半、控制污染）。

### Next decision
1. **首选**：跨架构 homolog 复现（Llama-3-8B / Mistral-7B / Gemma-2-9B 任一即可运行现有 run.py——架构参数全部来自 config.json；仅需该模型加入本地 `models/`）。这是把 claim 从"Qwen2 家族内"升级为"架构无关功能机制"的决定性一步。
2. **并行可选**：exploratory 调查 same-state 污染——(20,0) V-code 的 label/wording 混合是否集中在特定 wording 对，能否经 writer 位点微调恢复纯度。
3. reader register 的 1-head 位移（Q1 vs Q0）是否具功能意义，可在跨架构复现中一并观察。

## 输出（EXP14）

`outputs/exp14_cross_model_homolog/`：gate_report.json/csv、discovery_phaseA.csv、discovery_results.csv、discovery_reader_heads.csv、discovery_summary.csv、selected_homolog.json、confirmation_results.csv、confirmation_summary.csv、paired_contrasts.csv、directional_confirmation.csv、family_wording_confirmation.csv、summary.json、run_manifest.json（+ gate/discovery/confirmation 日志）。

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
        |
        v
exact offset × reader query-head   收敛到边界标记（EXP11：offset -5 im_end 56% + -1 21%；
                                   reader heads Q0 73% / Q3 37% / Q5 14%，Q2/4/6 负；
                                   sanity 全过：Q7-Q27 delta=0、leakage=0、all7 重建=full）
        |
        v
独立复制（新 family/词表）         分层：reader register 复制成功（EXP12：frozen readers
                                   {Q0,Q3,Q5} 4/4 family 显著正、negative {Q2,Q4,Q6} 负、
                                   contrast +0.16/+0.17、cross-wording 同向、措辞稳定）；
                                   positional 编码未复制（frozen offsets V suff −0.013 方向
                                   反转、负对照正、cross 负、措辞翻转）→ confirmatory=FALSE
```

## 当前主张边界

EXP10 之后，可辩护的项目级声明：

> **可恢复的干预位点**：早期窗口内行为效应的因果载体集中在 **H18–H20 三层**（full_H15_H20 = +0.0784；suffix_H18_H20 = +0.0760，即 97% 全效应；H20 单层 +0.0488 为最强单层，但仅 62%，far from sufficient；H15–H17 可移除）。该恢复不需要干预 H21–H28，即可因果转移技能条件化动作偏好。下游两条通路（MLP 中间神经元 EXP06、注意力头输出 EXP07）均已被独立证伪为可移植中介，其 recovery 不对称是伴随表现。

> **消费接口**：H20 残差效应主要通过 block 20 的 **V（KV）投影状态**传递——suff +0.0460（94% 残差）/ nec +0.0443（91%），CI 均排除 0；V 单通道即 89%，K≈0，Q 为负；三层 chain（H18+H19+H20 的 KV/QKV 同时移植）恢复 full 的 82–84%。预注册 KV > Q 假设确认（配对差 +0.051/+0.050 均显著正），但机制细节是 **V 单通道主导**而非 K+V 组合。

> **稀疏 head×position 收敛**：H20 V 效应集中于 **单一 GQA value head（KV0）× 最末端 token 区域（B5，"Choose the single next action now" + assistant turn 边界）**——KV0×B5 一个 cell（suff +0.0448 / nec +0.0445，q 均 <0.001）即完整复现 full-V 效应（~103%）；KV1 近空载，KV2/KV3 轻微抑制；B0–B4 全部无效。procedural state 的路由接口是 **1 head × ~17 token 最末端区域**。

> **精确 path（EXP11）**：B5 内部效应写在 **4 个 causal offset**（-13 `.\n\n`、-5 `<|im_end|>` 56%、-3 `<|im_start|>`、-1 `\n` 21%）——集中在 **chat-template 边界标记与指令尾部**；读取者收敛到 **Q0（73%）、Q3（37%）、Q5（14%）** 三个 query heads（Q2/Q4/Q6 转移 delta 为负，近似线性叠加）；三条 sanity 全部严格通过（Q7–Q27 delta 精确 0、non-KV0 leakage ratio 0.0、all7 重建/移除与 full V effect 逐位相等）→ 该分解是真实因果路径而非事后挑选。

> **EXP12 分层复制**：冻结 **reader register 复制成功**——frozen readers {Q0,Q3,Q5} 在 4 个新 family 全部显著正（suff +0.076 / nec +0.062）、negative readers {Q2,Q4,Q6} 显著负、frozen−negative reader contrast +0.16/+0.17、cross-wording 同向、措辞稳定（canonical-only +0.0998 vs 全措辞 +0.0763）；冻结 **positional 编码未复制**——frozen-offset V suff/nec 显著为负（-0.013/-0.015，方向反转）、内容 token 负对照反而为正（+0.011/+0.012）、frozen−negative offset contrast 显著为负、cross-wording V 亦负、措辞间符号翻转（canonical +0.007 vs paraphrase 约 -0.034）。**confirmatory_replication_pass = FALSE**（10 项 5 过 5 败）。specificity falsification：skill−direct 全部 5 指标 × 4 family 显著为正（frozen-V suff +0.072、reader suff +0.068）→ 冻结路径在 procedural Skill 下显著强于 direct action instruction，direct 下 frozen-V 甚至显著为负 → 支持 **procedural specificity**。

> **不能声称的**：①「H20 单层是 handoff state」——H20 仅 62% 且 full−H20 = +0.030 显著正；②「H15–H20 全宽均匀分布式」——H15–H17 单独为负、可移除；③「H15–H20 residual 本身就是最终因果载体」——Qwen2 block 为 joint computation，前层输出是后层整体计算的输入条件，H18–H20 内部各层各自的 causal contribution 仍未逐层定位；④「Q/K/V 三通道共同负载路由」——H20 上 K≈0、Q 为负，实际是 V 近单通道；⑤「V 投影状态就是最末端载体」——V 状态仍需进入 attention 加权聚合→MLP，其后各步是否可进一步归因仍未实验；⑥「多个 GQA KV head 协同负载」——H20 上 4 个 head 高度不对称，KV0 承载全部正效应；⑦「effect 广泛分布于 prompt 前段内容」——B0–B4 全无效，写入仅发生在 generation boundary 前的最终指令区域；⑧「全部 7 个 reader query heads 协同读取」——EXP11 仅 Q0/Q3/Q5 正贡献（Q0 73%），Q2/Q4/Q6 为负；⑨「B5 内效应均匀铺开」——仅 4 个 offset 显著，其中 -5/-1 即 77%；⑩「frozen offset 编码（{-13,-5,-3,-1} 边界 token）是任务一般的因果 position 载体」——EXP12 在新 family 上方向翻转、负对照为正、措辞不稳定，positional 层未复制；⑪「完整 circuit（1 KV head × 边界 token × 3 readers）是 task-general 的」——EXP12 confirmatory=FALSE，仅 reader register 层复制成功。

尚不可声明：

> 「真正的因果机制完全是一个分布式电路。」——已不成立；EXP10 证伪了分布式假设，路由接口是高度稀疏的。
> 「已定位到单个 GQA KV head / 单个 token 位置的完整路径。」——KV0×B5 定位完成，但 B5 仍有约 17 token 宽，且尚未归因到具体 query heads。
> 「完整 circuit 已证明。」——已由 EXP12 明确否决（confirmatory_replication_pass = FALSE，10 项中 position 层 6 项全败）；当前可辩护的最强声明是 **task-general reader register（Q0/Q3/Q5 正读、Q2/Q4/Q6 抑读，procedural 条件化，跨 family/词表/措辞复制成功）+ family-contingent positional coding（{-13,-5,-3,-1} 方向随 family 与措辞翻转，非任务一般机制）**。

已验证的排除项（EXP01–EXP11）：

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
11. Prompt 位置均匀/前段分布假设（EXP10：B0–B4 全无效，100% 集中于 B5 末端）；
12. 全部 7 个 reader query heads 协同读取（EXP11：Q0 73% 主导、Q3/Q5 次要，Q2/Q4/Q6 负）；
13. B5 内效应均匀铺开（EXP11：仅 offset -13/-5/-3/-1 显著，其余 ≈ 0）；
14. **固定绝对位点 writer 作为任务一般的因果载体**（EXP12：新 4 family 上冻结位点 {-13,-5,-3,-1} 的 V 效应显著为负、内容 token 负对照显著为正、cross-wording 亦负、措辞间符号翻转——同 token 同位置符号随 context 翻转，非 absolute-position artifact 而是 context-conditioned writer，固定位点不可移植；注意：reader register {Q0,Q3,Q5} 的正选择性在全部 4 family 复制成功，不在此排除项内）。

尚未完成的验证：

1. ~~**EXP13：context-conditioned schema write-site remapping**（预注册，已完成 → d706328）~~——writer 高度 schema-stable（USER_END 7/8 strata），其 V 因果效应双向显著正、优于 old absolute 与 runner-up、cross-wording 迁移正；reader register {Q0,Q3,Q5} 全部冻结保持。**结论：不升级为"跨上下文可移植的固定位点 writer"**——writer 位点基本稳定但效应幅度随 family×wording 波动，机制更接近 "schema-stable writer + stable reader"（Pattern B 倾向），已触发下一检验：EXP14 应识别 reader register 消耗的功能性隐变量或测试 USER_END 位点分辨率，而非继续 token-position scan。
2. ~~**reader register 与 writer 接口的跨模型复现**（EXP14，已完成 → PARTIAL SUCCESS）~~——Qwen2-7B-Instruct（通用域，独立权重）上冻结机制定义、不冻结索引：discovery 选出与 Model A **逐索引一致**的 (L=20, KV0)，reader register {1,3,5}+/{2,4,6}−（负集合与 Q2/Q4/Q6 完全一致，正集合 2/3 重叠）；held-out 双向 V suff/nec、cross-wording、分解恒等式、procedural specificity 全部复现；但 same-state 控制 −0.012（Model A=0）、leakage 0.153、效应减半 → 结构同构、保真度降。**跨架构复现（Llama/Mistral/Gemma）仍未验证**（需第三本地模型）。
3. ~~**EXP15：跨架构功能同构复现**（预注册 2026-09-25 → 完成，**PARTIAL / ALGORITHMIC HOMOLOG**）~~——Model C=Mistral-7B-Instruct-v0.3（非 Qwen2 系：32L/32Q/8KV/128、不同宽度深度模板词表）。闸门 PASSED（pooled +1.230，4 family / label0+1.152 / label1+1.308 / canonical+paraphrase 全正）；discovery 自由搜索冻结 **L=30,KV4、GENERATION_BOUNDARY（8/8 strata，提示词尾 `action now . [/INST]`）、reader {0,16,18}+/{1,17,19}−（活跃 head 16 +0.148 主导正读 / head 19 −0.018 抑读；0/1 为 score=0 tie）**；held-out（32 任务，task-level paired bootstrap）：selected V suff/nec **+0.140/+0.151**、cross-wording **+0.141**（措辞完全迁移）、reader pos **+0.165/+0.160** vs neg −0.020/−0.025（contrast **+0.185**）、non-set 精确 0、**leakage 0.0**、all32==verified 精确、matched-negative（ISSUE）干净 null、skill≫direct（6.2×）、4 family 与 label0/1 双向全正；**偏差：same-state +0.002 CI 不含 0（1.4%，效应同号）、runner_up nec +2.37（非边界位点残差通道主导，V/KV 接口边界特异）、old-absolute ≈ selected（偏移与边界窗口重叠）**。**结论：Qwen 系 V/KV 介导的功能组织在独立架构中以完全不同的索引实现复制成功**（层/KV/heads/anchor 全不同 vs EXP14 的逐索引巧合）——支持**功能级同构**，其"V 通道负荷占比"因架构而异（Mistral 边界特异、Qwen USER_END≈89%）。
4. V 投影状态之后 attention 加权聚合→MLP 的剩余归因；
5. H18–H20 状态的维度分解（SAE/SNMF）；
6. 跨真实 agent 泛化。

> 「证据排除了十四个假设——全局线性操控、单 token 精确互换、MLP 神经元级中介、MLP 全量中介、注意力头输出中介、H20 单层 handoff、H15–H17 必要性、H20 Q 投影、H20 K 投影、多 KV head 协同负载、位置均匀分布、7 reader heads 协同读取、B5 均匀铺开、固定绝对位点 writer 的任务一般性——路径收缩为：**H20 残差 → block20 V-projection → KV0 → reader register {Q0/Q3/Q5} 正读 / {Q2/Q4/Q6} 抑读（跨 4 个新 family、新词表、双措辞独立复制成功；procedural 条件化：skill−direct 全指标显著为正）**。写侧机制为 **context-conditioned writer**：同 token 同位置（-13/-5/-3/-1）的 V 因果方向随 family×wording×label 上下文翻转，固定位点不可移植，非 absolute-position artifact。EXP12 confirmatory_replication_pass=FALSE，证据链**不升级**为 task-general frozen sparse circuit；EXP13 预注册检验 9 语义 anchor × 8 strata 的 schema 重映射，若 writer 仍不可移植而 reader register 存续，则永久停止 token-position scan 并转向功能性隐变量识别。」

---

# EXP15 — Cross-Architecture Functional Homolog Replication（跨架构功能同构复现）

## 日期
2026-09-25

## 提交
预注册 `0e0c209`；主实验 `3c19971`（hash 回填提交见索引表）。

## 状态
**预注册（PREREGISTERED）**——本段在读取任何结果之前写入，实验设计、判据与提交顺序全部事先固定。完成本跑后按结果更新本段；不得事后修改判据。

## 为什么是 EXP15（问题）

EXP14 证明 V/KV 介导的功能组织在**同架构**（Qwen2-family，GQA）两个域分歧检查点间复制成功（`79388ea`，PARTIAL SUCCESS：结构同构、保真度降）。但 Model A/B 属于同一模型族：同一 GQA 几何（28L/28H/4KV/128）、同一 chat-template、同一 tokenizer 后代。证据链的最大剩余泛化缺口是**跨架构**：该功能组织是否依赖 Qwen2 特有的架构实现细节（GQA KV-grouping、head_dim、宽度、深度等），还是存在于实现截然不同的解码器模型中？

EXP15 是首个**真正跨架构**检验：在 **Llama-style / Mistral 架构**（MHA、无 KV-grouping、不同宽度/深度/激活）上测试同一**功能机制定义**。

## 冻结的机制（Functional organization — 冻结定义，不冻结索引）

与 EXP13/14 相同：

```
context/schema-conditioned writer（语义位点，chat-template 无关）
→ 单 block 的 attention value/KV 投影接口（self_attn.v_proj → o_proj 之间的 KV 状态）
→ 正读 reader register + 抑读 reader register（分别在 query heads 子集上）
→ 下一动作偏好（Skill 条件化的 label 候选 token 边际对数概率）
```

**明确不冻结**（自由搜索，与 EXP14 同）：层索引 L、KV head 索引 k、reader 正/负 query head 集合 Q。不预设 H20/KV0/Q0,Q3,Q5 的任何等价物。

## 目标模型（Model C）

- **首选：`mistralai/Mistral-7B-Instruct-v0.3`**（Llama-style MHA：q=kv=32 组头、head_dim=128、32 层；架构族与 Qwen2 GQA 不同）。
- 用户显式授权模型准备阶段联网下载（`snapshot_download` 至 `models/Mistral-7B-Instruct-v0.3`，`ignore_patterns=["consolidated.safetensors"]`，先查 `df -h`）；正式实验运行阶段固定本地 snapshot 并 `local_files_only` 离线运行。
- 若 Mistral 下载失败：可改用另一**非 Qwen2 系** decod-only instruct 模型（7B–9B 档），必须记录更换原因；**不得因结果为负而更换模型**。

## 行为闸门（先于一切机制发现）

在 discovery/confirmation 任何干预之前：Model C 必须在 pool 签到基线（batch=1，mean candidate-token log-prob margin）上表现出正的 Skill 条件化下一动作偏好：
- 整体（pooled signed baseline > 0）；
- 每个 task family；
- 每个 wording（canonical/paraphrase）；
- 每个 label（label0/label1 均须正，即双向）。

闸门失败 ⇒ 记录 **BEHAVIORAL GATE FAILURE**，不解释任何电路空结果，实验终止（阴性仍归档）。

## 任务 / 数据

复用 EXP12 的 64 任务受控基准（4 family × 16：test_edit/search_edit/config_command/docs_code；label=0→action0 skill，label=1→action1 skill）与确定性 family-wisely 平衡的 32/32 discovery/confirmation split（任务为单元，不重叠）。EXP15 只改随机种子（seed=5515，与 EXP14 不同以明确独立）；family_split 函数沿用 EXP14（保证 4 个 family 齐整分割）。

## Discovery（仅 discovery split；所有组件搜索只在此阶段）

### Stage A — writer/接口定位（相对深度粗网格 → 局部细化）

- 相对深度网格：10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 96%（映射到层 L，取整、去重、1 ≤ L ≤ L_max−1）；
- 每层 × 每个 KV/value group k：测 {USER_END, FINAL_INSTRUCTION_END} 两个 anchor 的 V 充分性与必要性（donor=opposite-same-wording，两 label 各自聚合）；
- **选择 score = min(suff_label0, suff_label1, nec_label0, nec_label1)**（四个分桶均须为正的保守 score）；
- 在粗网格最优层 L\* 附近局部细化：**L\*−2 … L\*+2**（同 score 同 KV 组），取最优 (L\*, k\*)。
- 只细化一个局部窗口（radius=2），不跨窗口跳跃补搜。

### Stage B — schema writer 位点（在冻结 (L\*, k\*)）

每个 family × wording stratum（8 个）内，在 9 个语义 anchor 上搜索：
SKILL_END / SYSTEM_END / ISSUE_END / DETAIL_END / ACTION0_END / ACTION1_END / FINAL_INSTRUCTION_END / USER_END / GENERATION_BOUNDARY
（direct wording 无 SKILL_END，允许缺省；anchor 一律从渲染后的 message 文本推导，不硬编码 Qwen 特殊 token 偏移。）

每 stratum 用与 Stage A 相同的双向 score（min of 4 buckets）选 top anchor 冻结；记录 runner-up（供 selected−runner_up 对照）。

### Stage C — reader register（在冻结 (L\*, k\*, anchor)）

- 每个 query head q 测 path 充分性与必要性（含需要被充分性/必要性两条证据）；
- **positive score = min(mean_suff, mean_nec)**（取最高 3 head 冻结为正 reader 集合）；
- **negative score = max(mean_suff, mean_nec)**（在排除正集合后取最低 3 head 冻结为抑读集合）。

## 冻结产物 / 确认（held-out confirmation，零组件重选）

`selected_homolog.json` 冻结后进入 confirmation，全部组件不再搜索。报告：

1. selected V 充分性 / 必要性（+ CI）；
2. 双向 label0 / label1；
3. cross-wording（同 family 不同措辞迁移）；
4. 正 reader register suff/nec；
5. 负 reader register suff/nec；
6. positive minus negative reader 对照（paired bootstrap CI）；
7. all-reader 重建（suff）与全移除（nec）；
8. same-state 控制（同 state 反向 donor = 污染指示，期望 CI 含 0）；
9. matched negative anchor 对照（同结构不相关语义位点，期望无效）；
10. reader leakage（register 外 query heads 的转移量，期望 ≈ 0）；
11. selected−runner_up V suff/nec（期望正）；
12. old absolute 对照（负向控制，延续 EXP12/13）。

统计规则不变：task 为推断单元；task-level paired bootstrap 95% CI（seed 派生固定）；batch=1；度量 = mean candidate-token log-prob margin。

## 判据（Verdict）

- **STRONG HOMOLOG**：writer + V/KV 接口 + reader-register 组织全部确认；核心 endpoint CI 下界 >0；fidelity 控制不污染。
- **PARTIAL / ALGORITHMIC HOMOLOG**：部分层级复现，或保真度控制降级（same-state CI 不含 0 / leakage > 0.10 / selected−runner_up 异常），或 endpoint 存活数 ≥ 3。
- **NO V-MEDIATED HOMOLOG**：行为闸门过但主要 endpoint 不存活（含数 < 3）。
- **BEHAVIORAL GATE FAILURE**：闸门失败，机制层不可解释。

**RED ZONE**：判据在此写入后永不修改；不得以事后视角重选组件、重定义 score、补跑搜索来"救"结果。阴性结果有效且必须归档保留。

## 结果（EXP15 — 已完成）

### 日期
2026-09-25（discovery 18:22–20:04，confirmation 20:04–20:08）

### 提交
主实验提交见索引表（本段写入后随结果一起提交，hash 回填）。

### 状态
完成，**PARTIAL / ALGORITHMIC HOMOLOG**（跨架构功能同构复制成功，索引实现不同；核心 endpoint 全过、保真度小降级）。

### 目标模型（Model C）
`mistralai/Mistral-7B-Instruct-v0.3`（`MistralForCausalLM`，32 层 / hidden 4096 / **GQA-8**（32 Q 头 / 8 KV 头 / head_dim 128）/ vocab 32768 / chat-template 与激活族不同；本地 snapshot `models/Mistral-7B-Instruct-v0.3`，config SHA256 见 run_manifest.json）。经 hf-mirror 下载（直连 HF 不可达），权重固定离线运行。

### 行为闸门
**PASSED**：pooled signed +1.230 [1.167, 1.294]（CI_low>0）；4 family 全正（config_command +0.996、docs_code +1.330、search_edit +1.564、test_edit +1.028，CI_low 均 ≥0.88）；label0 +1.152 / label1 +1.308（双向）；canonical +1.274 / paraphrase +1.186（双措辞）。Model B 对照 +1.501 [1.402, 1.601]。

### Discovery（discovery split 32 任务，seed 5515；预注册网格）

- **Stage A**：相对深度粗网格 (3,6,9,12,16,19,22,25,28,30) × 8 KV × {USER_END, FINAL_INSTRUCTION_END}。**全场唯一正值 cell = (L=30, k=4)（score +0.00223，suff0 +0.00282 / suff1 +0.00223 / nec0 +0.00309 / nec1 +0.00230）**，其余全部 ≈±0.001 负值；局部细化窗口 28..31 确认 (30,4) 为最优。→ **USER_END/FINAL_INSTRUCTION_END 的 V 效应在 Mistral 几乎不存在**（与 Qwen 显著不同）。
- **Stage B**（冻结 L=30,k=4，9 anchors × 8 strata）：**8/8 strata 全部选中 `GENERATION_BOUNDARY`**（score 0.066–0.173），runner-up 全 ≈0（0.0002–0.0022，小 60–250 倍）。`GENERATION_BOUNDARY` = 渲染提示词末尾 6 token：`single next action now . [/INST]`（最终指令 + assistant 起始标记）——功能上等价于 Qwen 的 B5/`<|im_end|>` 边界写点，但**不是** USER_END 族。
- **Stage C**（reader register）：正集合 **{0,16,18}**、抑读集合 **{1,17,19}**。实际活跃：**head 16（positive_score +0.1476，suff +0.1503）主导正读、head 18 弱正（+0.012）、head 19（negative_score −0.0184）抑读、head 17 弱负（−0.0014）**；head 0 与 head 1 为 score=0.0 的 tie 填充（positive/negative score 恰为 0），其余 26 头精确 0 贡献。与 Qwen {Q0,Q3,Q5}/{Q2,Q4,Q6} 完全不同。

### Confirmation（held-out 32 任务，零组件重选；task-level paired bootstrap 95% CI）

| 端点 | mean | CI | 32 任务全正？ |
|---|---|---|---|
| selected V suff | **+0.1400** | [0.1279, 0.1515] | ✓（min +0.063） |
| selected V nec | **+0.1509** | [0.1391, 0.1625] | ✓（min +0.076） |
| cross-wording V suff | **+0.1415** | [0.1298, 0.1527] | ✓（= same-wording，措辞完全迁移） |
| reader pos suff | **+0.1651** | [0.1502, 0.1791] | ✓ |
| reader pos nec | **+0.1599** | [0.1457, 0.1732] | ✓ |
| reader neg suff | **−0.0198** | [−0.0225, −0.0172] | 全负 ✓ |
| reader neg nec | **−0.0251** | [−0.0282, −0.0218] | 全负 ✓ |
| pos−neg contrast suff | **+0.1850** | [0.1678, 0.2006] | — |
| pos−neg contrast nec | **+0.1850** | [0.1673, 0.2016] | — |

- 分解恒等式：**all32 suff = verified V = +0.1400；non-set = 0.0 精确；leakage ratio = 0.0 精确**（register 外 26 头零泄漏，三模型中最佳；Model B=0.153）。
- **保真度控制**：
  - same-state 控制 **+0.0020** [0.00095, 0.0033]——CI **不含 0**（75% 任务正），为效应量 1.4% 的轻微污染（Model A=0、Model B=−0.012；方向与效应同号）→ 触发 PARTIAL 降级；
  - matched negative（ISSUE_END）对照 **+0.0005** [−0.00006, 0.00106]——含 0，**阴性对照干净**（Model B 此对照 +0.011 为正，故此处更优）；
  - selected−runner_up suff **+0.137**（runner +0.0031）✓；
  - **runner_up nec +2.374 [2.19, 2.56]（异常）**——非边界 anchor（ISSUE/FINAL_INSTRUCTION/SYSTEM_END）的**残差通道**效应巨大（res_ref≈2.4–2.5），V-clamp 后 retained 很小 → V 在非边界位点"不必要"（残差单独即可携带），而 V/KV 接口仅在 GENERATION_BOUNDARY 必要 → 写点边界特异性证据；
  - old-absolute +0.1365 ≈ selected（Mistral 末尾偏移 {−13,−5,−3,−1} 中 −5/−3/−1 与 GENERATION_BOUNDARY 窗口重叠，非独立性对照；Qwen 上 old-absolute 为负）；
  - **direct specificity**：direct（USER_END anchor）V suff **+0.0226** [0.0207, 0.0246]——skill（+0.140）的 **6.2 倍**，procedural specificity 成立（方向性）；注意 direct 固定锚 USER_END 而 selected 为 GENERATION_BOUNDARY，比值跨 anchor。
- per-family 全正：test_edit +0.175 / search_edit +0.147 / docs_code +0.150 / config_command +0.088；wording：canonical +0.144 / paraphrase +0.136；label：label0 +0.157 / label1 +0.123（**双向正**）。

### 判据裁定

- 5/5 核心 endpoint（selected suff/nec、cross suff、reader pos suff/nec）CI 下界 >0 ✓
- 2/2 对照（pos−neg suff/nec）CI 下界 >0 ✓
- 保真度：same-state CI 不含 0（+0.002，1.4%）→ 按预注册规则 **contaminated** → 判据自动落入 **PARTIAL / ALGORITHMIC HOMOLOG**（而非 STRONG）。泄漏 0.0 为最佳，阴性对照干净，不构成降级因素。

### 三模型功能组织对比

| 维度 | A=Qwen2.5-Coder | B=Qwen2-7B | C=Mistral-7B-v0.3 |
|---|---|---|---|
| writer anchor | USER_END 族（7/8） | USER_END 族 | GENERATION_BOUNDARY（8/8） |
| 层 / KV | H20 / KV0 | L20 / KV0 | L30 / KV4 |
| reader pos / neg | Q0,3,5 / Q2,4,6 | {1,3,5} / {2,4,6} | {0,16,18} / {1,17,19}（16+/19− 活跃） |
| gate pooled | （EXP14 前无同格式） | +1.501 | +1.230 |
| selected V suff | +0.047（EXP13） | +0.025 | **+0.140（最大）** |
| reader contrast | +0.095 | +0.050 | +0.185 |
| same-state | 0 | −0.012（显著负） | +0.002（显著正，1.4%） |
| leakage | 0.0 | 0.153 | **0.0** |

### 解释与 claim boundary

**功能组织跨架构复制成功**：边界写点（context/schema-conditioned writer，Mistral 上为 GENERATION_BOUNDARY=`...action now . [/INST]`，对应 Qwen 的 B5/`<|im_end|>` 角色）→ 单 block 的 V/KV 接口（L30/KV4）→ 稀疏正读/抑读 reader register（16+/19−，register 外精确 0）→ 下一动作偏好（held-out 全 32 任务双向正、措辞完全迁移、skill≫direct）。**索引实现全部不同**（层、KV、heads、anchor 均不同）——与 EXP14 的"索引巧合一致"形成对照，证明该组织是**功能级**而非实现级。

**架构差异**（新的科学信息）：① Mistral 上 USER_END 族 anchor 的 V 效应≈0、残差通道在非边界位点极强（runner nec +2.37），V/KV 接口高度特异性地收敛到边界写点——Qwen 上 V≈89% 残差效应，二者对"V 通道是否承担全部负荷"的分解不同；② reader 更稀疏（单个 16 主导 vs Qwen 三个正头分布）。

**不可声称**：① 非"STRONG HOMOLOG"（same-state CI 不含 0）；② 不推广为"任意解码器模型必然同构"（n=1 个跨架构模型；Gemma/Llama-3 未测）；③ 不声称索引对（L30/KV4/{16,18}/{19,17}）在 Qwen 中会再现或反之；④ 不声称 boundary V 通道是"唯一"因果路径（非边界位点残差通道同样携带 skill 信息，只是不经 V/KV 接口）。阴性对照（matched negative）干净支持 specificity，但 direct 对照为跨 anchor 比较。

**对证据链的意义**：Qwen 系建立的机制（EXP09–13）现被独立架构证实为功能级组织的一部分；下一步可选：① 第二跨架构模型（Gemma-2/Llama-3）增加架构面；② 检验 Mistral 上残差通道 vs V 通道在不同 anchor 的负荷分配（runner-up nec 异常的机制化）；③ 在 Mistral 上验证 reader head 16/19 的职责（如 Qwen 的 Q0/Q3/Q5 分解）。

**RED ZONE 遵守**：判据在预注册提交 `0e0c209` 中冻结，结果全部事后如实记录，未修改任何确认标准；PARTIAL（而非强行 STRONG）是按冻结规则自动得出。阴性/异常项（same-state、runner-up nec、head 0/1 tie）全部保留。

此措辞应保留，直到 EXP09 定位到各层内部投影（Q/K/V/MLP）的 causal handoff 结构。