执行我提供的 EXP16，不重新设计主实验，但可以灵活处理下载、环境和跨架构工程兼容。

1. 先核对当前仓库 HEAD、git status，并读取 EXP15 真实 outputs/run.py。
2. 合并 `experiments/exp16_second_cross_arch/`。
3. 在任何 EXP16 结果产生前，把 EXP16 预注册写入 `experiments/EXPERIMENTS.md` 并先 commit。
4. 先运行 `diagnostics/recompute_exp15_runnerup.py`。把 EXP15 原 `runner_up necessity +2.37` 标为原实现无效控制，并记录修正结果；不要覆盖原 outputs，诊断写新文件。若修正后改变任何 EXP15 解释，更新日志。EXP15 PARTIAL verdict 只有在冻结判据确实受影响时才能变化。
5. 运行 EXP16 `--phase inventory`。
6. Model D 首选 `google/gemma-2-9b-it`。如果本地没有，允许联网下载。若官方访问/许可证导致无法下载，在行为 gate 前可换另一个非 Qwen2、非 Mistral 的 GQA instruct decoder（优先 Llama-3.x-8B-Instruct）。必须先记录原因；不能因为机制结果不好换模型。
7. 如果只能使用 MHA 模型（例如 OLMo2），先暂停主 confirmatory run：MHA 的 1:1 V-head/read-head 几何与当前 shared-KV reader-register 定义不同。可以做单独 architecture-adapted exploratory branch，但不要冒充原 EXP16 confirmatory geometry。
8. 正式模型确定后切回本地 snapshot/离线加载，按 gate → discovery → confirmation 跑。`selected_homolog.json` 生成后禁止人工改。
9. 只修工程 bug：module path、Gemma/Llama norm ordering、tokenizer、chat template、tensor shape、head_dim、device/dtype、内存/resume。不得改 split、selection score、threshold、endpoint、verdict 来救结果。
10. 跑完先更新 `EXPERIMENTS.md`，记录 null/negative、bug/fix、claim boundary，再 commit + push GitHub/Gitee。
11. 发回：EXP15 runner-up 修正结果、Model D、gate、selected_homolog.json、confirmation_summary.csv、paired_contrasts.csv、summary.json、commit。
