# EXP14 — Cross-Model Functional Homolog Replication

Cross-model functional homolog replication of the EXP13 mechanism on a second local
model. The mechanism definition is frozen; the Qwen-Coder indices (H20 / KV0 /
Q0,Q3,Q5 / Q2,Q4,Q6) are NOT frozen — functional homologs are discovered on Model B's
discovery split and frozen for held-out confirmation.

Model B: `/data/mzb/ar2_scratch/models/Qwen2-7B-Instruct` (same Qwen2 architecture
28L/28H/4KV/128 as Model A `models/Qwen2.5-Coder-7B-Instruct`, independent weights,
general instruct domain). Fully offline (HF_HUB_OFFLINE), no download.

## Status
PREREGISTERED (2026-09-25). Design and endpoints in `EXPERIMENT_PLAN.md` and in
`experiments/EXPERIMENTS.md` (EXP14 section, written before the run).

## Run

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/exp14_cross_model_homolog/run.py \
  --model_dir /data/mzb/ar2_scratch/models/Qwen2-7B-Instruct --phase gate
CUDA_VISIBLE_DEVICES=0 python experiments/exp14_cross_model_homolog/run.py \
  --model_dir /data/mzb/ar2_scratch/models/Qwen2-7B-Instruct --phase discovery
CUDA_VISIBLE_DEVICES=0 python experiments/exp14_cross_model_homolog/run.py \
  --model_dir /data/mzb/ar2_scratch/models/Qwen2-7B-Instruct --phase confirmation
```

Phases must run in order; `confirmation` reads the frozen `selected_homolog.json` and
never re-searches. `--phase all` runs gate → discovery → confirmation in one process
(edit `selected_homolog.json` between phases is forbidden by construction — discovery
writes it once).

## Outputs
`outputs/exp14_cross_model_homolog/`:
gate_report.json / gate_report.csv, discovery_phaseA.csv, discovery_results.csv,
discovery_reader_heads.csv, selected_homolog.json, confirmation_results.csv,
confirmation_summary.csv, run_manifest.json (+ summary after confirmation).

## Method invariants
- 64 EXP12 tasks imported from the repository EXP12 implementation (no template
  duplication); split seed 5313 → 32 discovery / 32 confirmation.
- Task is the statistical inference unit; task-level paired bootstrap CIs (5000 draws).
- Sufficiency and necessity both measured; necessity localized to the same write site
  (donor residual restored, then V clamped back to recipient baseline).
- Controls: self, same-state cross-wording, cross-wording functional mapping,
  matched-negative (ISSUE_END content window), frozen runner-up anchor, old-absolute
  offsets, non-set reader heads, direct-instruction specificity.
- Capture/patch sites consistent (residual at layers[L-1] output == block L input; V at
  block L v_proj; reader pre-conv o_proj) — no pre/post-norm mismatch.
- Candidate scoring batch=1 (no padding asymmetry); mean candidate-token log-prob
  margin.
- No post-hoc best-layer/head selection presented as confirmatory: all selection on the
  discovery split, frozen before confirmation.
