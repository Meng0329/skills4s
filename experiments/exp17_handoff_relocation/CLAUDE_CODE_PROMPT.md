执行我提供的 EXP17，不重新设计主实验。

1. 合并 `experiments/exp17_handoff_relocation/`。
2. 在产生任何 EXP17 activation 结果前，将预注册追加到 `experiments/EXPERIMENTS.md`，先 commit。
3. 检查已有 Qwen2.5-Coder、Mistral-7B-v0.3、Granite-3.0-8B 本地 checkpoint。缺失可恢复同一 checkpoint，但不得换模型。
4. 先运行：
   `python experiments/exp17_handoff_relocation/run.py --model all --phase audit`
   核对 topology 位移、五个 site 的 token 文本与 overlap。工程映射 bug 可修，但不能改 site 定义来追结果。
5. 再运行：
   `python experiments/exp17_handoff_relocation/run.py --model all --phase confirm`
6. 写完 primary verdict 后才运行：
   `python experiments/exp17_handoff_relocation/run.py --model all --phase reserve`
7. 可灵活修工程兼容：路径、chat template、offset mapping、显存、resume；不得重选 layer/KV/readers、删除 condition/site/family、改判据。
8. direct-choice 是描述性 specificity control，不预设 procedural 一定更强。
9. TEMPLATE_TERMINATOR 与 GENERATION_BOUNDARY 若高度重叠，按预注册报告 HANDOFF_OR_TEMPLATE_BOUNDARY，不要强行解释。
10. 跑完先更新 EXPERIMENTS.md（含 null/negative/bugs），再 commit + push GitHub/Gitee。
11. 返回 summary.json、model_summary.csv、relocation_contrasts.csv、site_overlap_audit.csv、commit。
