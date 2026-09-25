# EXP15 — Run Notes（工程记录，随提交归档）

状态：执行中（2026-09-25）。以下为预注册之后、结果回填之前记录的工程决策，全部为
GREEN/ZONE（工程等价）或 YELLOW（元数据/命名）级别，不影响科学判据。

## 模型准备（Model C）

- 仓库当时仅有两个 Qwen2-family 本地模型（Model A=Qwen2.5-Coder-7B-Instruct、
  Model B=Qwen2-7B-Instruct），无跨架构模型。
- 用户显式授权模型准备阶段联网下载（README 中 “No model download is allowed”
  被用户指令覆盖）。直连 huggingface.co 不可达（网络隔离），经 `hf-mirror.com`
  镜像 `snapshot_download(repo_id="mistralai/Mistral-7B-Instruct-v0.3",
  local_dir="models/Mistral-7B-Instruct-v0.3", ignore_patterns=["consolidated.safetensors"])`
  完成，14.4 GB，3 片 safetensors。磁盘先查（/data 余 ~900G）。
- 权重 snapshot 固定：后续运行一律 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` +
  `local_files_only=True` 离线执行。config.json SHA256 记入 run_manifest.json。

## 架构事实（与 Qwen2 的差异，经验证）

- Mistral-7B-Instruct-v0.3：`MistralForCausalLM`，32 层，hidden 4096，
  **num_attention_heads=32，num_key_value_heads=8（GQA-8，head_dim=128）**，
  vocab 32768，最大位置 32768，**激活是 SiLU**（而 Qwen2 为 SiLU—两者激活其实同族，
  但宽度/深度/GQA 比率/vocab/tokenizer/chat-template 均不同），无 `head_dim` 键
  （由 hidden/q 推导=128）。`resolve_layers` 命中 `model.model.layers` ✓；
  `self_attn.v_proj/o_proj` 存在 ✓。
- smoke test（architecture_sanity.json）：schema 12/12（9 anchors 全过，含 direct）、
  baseline 计时 ~0.23 s/forward、峰值显存 14.6 GB。
- Stage A 粗网格（32 层 → valid_max=31）：(3,6,9,12,16,19,22,25,28,30)，与存档一致。

## GREEN-ZONE 工程优化（结果逐位等价）

1. **Phase A residual_effect hoisting**：`residual_effect` 只修补 layers[L-1] 残差、
   不含任何 v_heads，与 KV 组 k 无关；原实现把它放在 k 循环内导致每个 (L,entry,anchor)
   上重复 8 次。改为按 (L, task, entry, anchor) 计算一次、跨全部 k 复用 → Phase A
   前向数减少 ~29%（65,280 对 92,160）。推理模式确定，同输入必同输出，**数值不变**。
2. **GPU 隔离**：`CUDA_VISIBLE_DEVICES=0`（GPU1 被 sglang 占用 43.5 GB），
   避免显存争用与速度抖动；设备映射 `device_map="auto"` 只在 GPU0。
3. **分阶段 checkpoint**：gate → discovery → confirmation 分进程运行；
   discovery 完成即写 `selected_homolog.json`，confirmation 崩溃可从冻结产物恢复。

## YELLOW-ZONE（元数据/命名，不影响科学语义）

- `reader_all28_sufficiency/necessity` 度量名沿用 EXP14 字面 "all28"，但在 Mistral 上
  语义为 "全部 32 个 query heads"（all-set 重建/移除），非数字 28。已注释。
- `run_discovery` 的 `seed` 形参未使用（split 已含 5515；bootstrap 种子在 confirmation
  的 `make_verdict` 使用）。纯样式，无行为影响。
- 三模型行为对比：Model A（Qwen2.5-Coder）在 EXP14 之前无同一 gate_report 格式
  （门控形式化始于 EXP14），其可比签到基线缺失；Model B=+1.501 [1.402,1.601]、
  Model C（Mistral）=+1.230 [1.167,1.294]，均 gate_pass=True。

## 尚未触及 / 不做的

- 不修改任何确认判据、不重选组件、不因结果方向更换模型。
- 不运行 Model A gate（GPU0 占用；且非本实验任务）。

## 运行时观察（结果回填后补记）

- **Mistral 实为 GQA-8**（32 Q / 8 KV，非预注册参考里的 MHA-32）——粗网格 8 KV 组，仍属"非 Qwen2 系"（不同宽度/深度/vocab/template/激活系），符合跨架构目标；已在预注册段与索引表注明。（pre-reg 文本提到"Llama-style MHA"，实际加载为 GQA-8；归档说明。）
- **GENERATION_BOUNDARY 语义**：= 渲染提示词末尾 6 token `single next action now . [/INST]`（最终指令 + assistant 起始标记），角色等价 Qwen 的 B5/`<|im_end|>` 边界写点。
- **reader head 0/1 tie**：positive/negative score 恰为 0.0（真实贡献精确 0），因 top-3/bottom-3 规则被选中；活跃结构为 16+（正）+ 19−（抑）。非 bug，按规则如实报告。
- **reader_contrasts suff==nec 恒等（0.18498...逐位相同）**：本 patch 方案的 group suff/nec 在该度量下每条目差为同一常数，故 pos−neg 的 suff 与 nec 恒等；结构事实，非错误。
- **runner_up_V_necessity_loss +2.37**：非边界 anchor 的残差 res_ref 很大（2.4–2.5），V-clamp 后必然性计算使其"V 不必要"——解释为 V/KV 接口在边界写点特异、非边界位点残差通道单独承载。
- **old_absolute ≈ selected**：Mistral 末尾偏移 {−13,−5,−3,−1} 中 −5/−3/−1 落入 GENERATION_BOUNDARY 窗口，对照与选择位点重叠，不能作为独立阴性对照（Qwen 上 old-absolute 为负）。
- **direct specificity 跨 anchor**：direct 条目固定锚 USER_END（run_confirmation 硬编码），selected 为 GENERATION_BOUNDARY；skill≫direct 比值 6.2× 是跨 anchor 的，归档说明。
- 全程 GPU0 独占（CUDA_VISIBLE_DEVICES=0），sglang 已自行退出于停止推理期间；实测：discovery ≈ 1h41m，confirmation ≈ 4.5min（远低于预估 5h——hoisting + 实际 coarse 层 10 个 + refine 窗口合并后仅 15 层、GPU 空闲无争用）。