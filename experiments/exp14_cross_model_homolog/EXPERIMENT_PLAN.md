# EXP14 — Cross-Model Functional Homolog Replication（跨模型功能同构复现）

## Status
PREREGISTERED (2026-09-25)

## Motivation from verified EXP13 outputs

EXP13 confirmed (confirmatory=TRUE) on Qwen2.5-Coder-7B-Instruct:

- writer is schema-stable: USER_END selected in 7/8 family×wording strata;
- held-out bidirectional selected V sufficiency +0.0473 / necessity +0.0465 (CI>0);
- selected − old_absolute +0.061/+0.066 (old absolute again negative);
- cross-wording selected V > 0; reader target +0.067 vs negative −0.028;
- all sanity exact (all7 == selected == verified; zero leakage);
- bidirectional_writer_pass = true; mechanism_pattern = contextual_writer_plus_stable_reader.

The largest remaining generalization gap is **cross-model**: does the functional
structure M (schema-anchored writer → single-block V(KV) interface → stable
positive/inhibitory reader register → behavior) exist in an independently trained
model in a different domain?

The machine has exactly two local models (offline constraint, no download):

- Model A (origin): `models/Qwen2.5-Coder-7B-Instruct`
- Model B (target): `/data/mzb/ar2_scratch/models/Qwen2-7B-Instruct`
  (general instruction-tuned, non-Coder; same Qwen2 architecture 28L/28H/4KV/128,
  same vocab 152064, independent weights, different pretraining domain)

Cross-architecture replication (Llama/Mistral/Gemma) remains future work and
requires a third model; EXP14's claim boundary is a same-architecture,
different-domain checkpoint homolog, which is the strongest claim available
offline.

## Frozen mechanism definition M (NOT allowed to change)

In procedural skill tasks, action preference is governed by:

- **W1 writer**: a schema-anchored token window (semantic boundary; Model A picks the
  USER_END family) in the recipient context carries a donor-conditional state in the
  residual stream;
- **W2 causal interface**: that state is written through a **single decoder block's**
  V projection via **one efficient KV head** (Model A: H20 residual → block20 V-proj → KV0);
- **W3 reader register**: a small subset of query heads **positively reads** (suff>0),
  a small subset **inhibits** (suff<0), the rest ≈ 0 (Model A: 3 positive + 3 negative);
- **W4 behavior**: the V state moves the mean candidate-token log-prob margin toward the
  donor's preference; procedural skill conditioning is stronger than direct instruction.

## NOT frozen (searched on Model B's discovery split, then FROZEN for confirmation)

- source block **L ∈ {16..27}** (deep band, all 12 layers swept);
- KV head **k ∈ {0,1,2,3}** (all KV heads swept);
- per-stratum **writer anchor** (9 candidates, EXP13 selection rule);
- **reader register** (28 query heads; pos_set = top-3 by discovery suff, neg_set =
  bottom-3; register size 3+3 is preregistered from Model A structure, not tuned to data).

The indices H20 / KV0 / Q0,Q3,Q5 / Q2,Q4,Q6 are NOT frozen. Only M is frozen.

## Split

Exact EXP12 64-task set, EXP13 split (seed 5313, per-family RNG = 5313 + family index):
**32 discovery / 32 confirmation**. Task is the statistical inference unit (bootstrap
over tasks, paired CI).

## Phases and selection rules (discovery split only)

### Phase 0 — engagement gate
All 64 tasks, skill entries, signed baseline margin:
signed = +margin if label==1 else −margin (margin = logprob(action1) − logprob(action0)).
Gate PASS iff pooled signed baseline bootstrap CI excludes 0 in the positive direction
AND all four family means have the correct positive sign.
Gate FAIL → STOP and report null for the engagement axis (Model B does not engage the
task family; mechanism undecidable/false here). No metric substitution.

### Phase A — (L, k) coarse selection
Anchors fixed to {USER_END, FINAL_INSTRUCTION_END} (Model-A instruction-boundary
functional family; both exist in every prompt). Donor relation: opposite_same_wording.
`score(L,k) = max_{a in 2} min(mean_suff_label0, mean_suff_label1)` over the 128
discovery entries. Pick argmax → (L*, k*). Tie-break: (L ascending, k ascending).

### Phase B — per-stratum anchor selection
Frozen (L*, k*). For each family×wording stratum and each of 9 anchors:
`score(a) = min(S0(a), S1(a), N0(a), N1(a))` where S/N are discovery-task means of
suff/nec and 0/1 is recipient label (EXP13 rule). Selected + runner-up frozen.

### Phase C — reader register search
Frozen (L*, k*) plus per-stratum anchors. For each discovery entry capture base and
donor-patched oproj inputs (both candidates); per query head q measure signed
sufficiency by replacing only slot q. pos_set = 3 largest, neg_set = 3 smallest.

## Held-out confirmation (32 tasks, all frozen)

- selected V suff / nec (pooled CI > 0);
- label0 and label1 separately (bidirectionality);
- cross-wording (recipient and donor each use own wording's selected anchor =
  functional-position mapping);
- same-state control (same label, other wording donor; expect ≈ 0);
- matched-negative control (ISSUE_END content window; diagnostic);
- old-absolute {-13,-5,-3,-1} (historical diagnostic; negative on Model A);
- reader register: pos_set suff/nec CI>0, neg_set suff/nec CI<0, pos−neg contrast
  CI>0, all28 suff == verified exactly, non-set ≈ 0 (diagnostic), leakage ≈ 0;
- procedural specificity: direct-condition V suff @ USER_END vs skill-condition
  (diagnostic contrast).

## Preregistered judgment

- **SUCCESS**: gate passes; Phase A/B/C produce non-degenerate choices; confirmation
  hard endpoints all pass (selected V suff/nec CI>0, cross-wording CI>0, pos−neg
  contrast CI>0, same-state ≈ 0, all28 == verified exact, both labels positive); and
  the writer anchor belongs to the instruction-boundary functional family (USER_END /
  FINAL_INSTRUCTION_END / SYSTEM_END / GENERATION_BOUNDARY). → cross-model functional
  homolog replication; claim upgrades to model-general functional mechanism within
  the architecture family.
- **PARTIAL**: reader register replicates (pos>0 / neg<0 / contrast) while the writer V
  is one-directional, or the anchor leaves the functional family, or cross-wording
  fails, or specificity flips. Components that generalize are separated.
- **FAIL**: gate fails, or Phase A best (L,k) is non-positive, or >=half the hard
  confirmation endpoints fail. → mechanism is Model-A (Coder-domain) specific; revise
  the mechanism model to "coder-family-specialized writer protocol"; reader-register
  cross-model generality adjudicated separately.

## Exploratory (explicitly non-preregistered)

- effect-vs-layer profile around (L*,k*) (not an isolated point check);
- anchor-family rank tables for writer-functional-family domain transfer;
- family × wording × label stratified detail;
- if confirmation fails, recheck on (L*,k*) neighbors is explanatory only and cannot
  resurrect a confirmatory claim.

## Outputs

`outputs/exp14_cross_model_homolog/`:
gate_report.json, discovery_results.csv, discovery_summary.csv, selected_homolog.json,
confirmation_results.csv, confirmation_summary.csv, paired_contrasts.csv,
directional_confirmation.csv, family_wording_confirmation.csv, summary.json,
run_manifest.json.

## Run (offline, GPU0 pinned, no padding asymmetry: batch=1; capture/patch sites
consistent: residual at layers[L-1] output == input to block L; V at layer L v_proj;
readers at layer L o_proj pre-conv — no pre/post-norm mismatch)

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/exp14_cross_model_homolog/run.py \
  --model_dir /data/mzb/ar2_scratch/models/Qwen2-7B-Instruct --phase gate
CUDA_VISIBLE_DEVICES=0 python experiments/exp14_cross_model_homolog/run.py \
  --model_dir /data/mzb/ar2_scratch/models/Qwen2-7B-Instruct --phase discovery
CUDA_VISIBLE_DEVICES=0 python experiments/exp14_cross_model_homolog/run.py \
  --model_dir /data/mzb/ar2_scratch/models/Qwen2-7B-Instruct --phase confirmation
```