# EXP16 — Run Notes（工程记录，随提交归档）

状态：执行中（2026-09-29）。

## Model D 决策（门控前，2026-09-29）

- **首选 `google/gemma-2-9b-it` 失败**：HF gated（manual），token 未授权（403）。已实测验证。
- **Llama-3.1-8B / 3.2-3B 失败**：同样 gated/403。
- 探测开放备选：Phi-3-mini / SmolLM2 / OLMo-2 为 **MHA**（trigger 用户指令 #7 暂停主 confirmatory 的配置）；Phi-3-medium（GQA 40/10）可访问且 14B 可装下，但 transformers 5.17 `Phi3Attention` 用融合 `qkv_proj`（无独立 v_proj）→ 需张量切片适配。
- **选定 `ibm-granite/granite-3.0-8b-instruct`**：开放、原生 `GraniteForCausalLM`（transformers 5.17）、**GQA 32Q/8KV / 40L / 4096h / head_dim 128**、独立 `v_proj+o_proj`（零 hook 适配）、instruct + chat-template。第三架构族（IBM Granite）。

## 前置诊断（EXP15 runner-up 修正，已完成）

`diagnostics/recompute_exp15_runnerup.py` 于 2026-09-29 运行：
- 原实现 `runner_up_V_necessity +2.3735` **无效**（复用 selected 的 res_ref）。
- 修正：runner-up 自身残差参考 → nec **+0.00381 [0.0027, 0.0051]**、suff +0.00309、runner 自身残差效应 +0.00420。
- 推导：clamp retained 两次一致 ⇒ res_ref(selected, GENERATION_BOUNDARY) ≈ 2.374 ⇒ 边界位点残差与 V/KV 双通道并存（V≈12%）。
- 写新文件 `outputs/exp15_cross_arch_homolog/diagnostics/`，原 outputs 未动。
- 影响：仅 EXP15 解释更新（见 EXPERIMENTS.md EXP15 段修正），**verdict 不变（PARTIAL）**——冻结判据（endpoints + same-state）不依赖 runner-up 控制。
- 已提交 `41de2a4`。

## 工程要点

- RUN 复用 exp15 的 `load_module`/`EXP14_CODE`/`generic_compute_schema`/`load_cross_arch_model`。
- Granite 40L：`resolve_layers` 命中 `model.model.layers`；`coarse_layers(40)`= (4,8,12,16,20,24,27,31,35,38)（round(frac×39)）；`refined_layers` L\*±2。
- GPU0 独占（CUDA_VISIBLE_DEVICES=0，避免 sglang 争用）；HF 离线。
- delta：原谅 EXP15 的 hoisting 优化未在此 run.py 复用（A2/C ref_succ 对每个 k 重算 residual_effect）；不改变语义，仅时间成本，如实记录不动科学逻辑。

## 运行时观察（2026-09-29 结果回填后）

- **Granite 加载/几何**：40L/32Q/8KV/128，`model.model.layers` 命中，v_proj out=1024（=8×128）✓ o_proj in=4096 ✓；schema 12/12；prompt 尾 `...action now.<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>`（GENERATION_BOUNDARY 含 `<|end_of_role|>`，角色同 Mistral 的 `[/INST]`）。
- 速度：~0.33 s/fwd（8B/40L），discovery 实测 3.3h（含 A1 全 9 anchor × 10 粗层 + A2/C 双层细化 + B + D），confirmation <7min。
- **reader 稀疏性**：register 外头（0–8、12–31）对 suff/nec 的贡献精确 0（与 Mistral/Qwen 相同），但 leakage=0.348 非零（delta 能量泄漏）——说明 register 外头"有激活变化但不产生净 logit 效应"；Mistral 是能量与效应都精确 0。二者机制表述应区分。
- **inhibitory 空集**：最负头 head8 inhibitory_score=−0.0008 > 阈值 −0.002 → 按预注册记录空集；STRONG 判据因此自动不满足（连同 same-state 负、leakage 高），未做任何人工干预。
- **tag 观察**：Granite tokenizer 对 BPE 的 clean_up_tokenization_spaces 警告（transformers 5.x 行为）无害；两条件（recipient/donor）tokenization 均一致，不构成偏差。
- gate label1 偏弱（+0.198 vs label0 +0.927）：贯穿确认期（label1 selected suff +0.100 vs label0 +0.141），非确认期新问题，已在报告如实记录。