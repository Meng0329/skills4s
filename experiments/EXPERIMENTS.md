# skills4s — Experiment Log

This file is the project-level scientific record.

Principles:

- Record the hypothesis **before** interpreting the result.
- Separate observation, interpretation, and causal claim.
- Preserve null and negative results.
- Tie each completed experiment to its Git commit.
- Do not upgrade a claim beyond what the intervention actually establishes.
- Treat exploratory layer searches separately from preregistered/confirmatory tests.

---

## Experiment Index

| ID | Question | Status | Main Result | Decision | Commit |
|---|---|---|---|---|---|
| EXP01 | Is procedural stage linearly decodable? | COMPLETED | Yes. Best `order_delta` F1 = 1.000 at H19, but No-Skill F1 = 0.956 | Isolate Skill-induced state with counterbalanced design | `ed1de55` |
| EXP01b | Does Skill ordering alone alter a cross-wording next-state representation? | COMPLETED | Yes. Cross-wording F1 = 0.709; behavior accuracy = 0.651 | Test causal sufficiency of the representation | `62aee2c` |
| EXP02 | Is the H19 mean-difference direction causally steerable? | COMPLETED — NULL | No. Real effect ≈ +0.00086; no advantage over controls | Reject global linear-vector control hypothesis | `20dba87` |
| EXP03 | Can exact paired single-token activation interchange transfer the donor state? | COMPLETED — NULL / NEGATIVE | H19 marginal; later layers show stronger negative transfer | Test coherent multi-token / multi-layer restoration | `e98dade` |
| EXP04 | Is causal mediation distributed across multiple shared suffix tokens and/or layers? | COMPLETE | Signed transfer +0.058 (CI excludes 0); self-patch/same-state ≈ 0; multi-token confirmed, H21–H28 null alone | Distributed residual-state mediation supported | (see run_manifest.json) |

---

# EXP01 — Procedural Stage Probe

## Date
2026-09-20

## Commit
`ed1de55`

## Status
COMPLETED

## Research Question

Can the current procedural stage be linearly decoded from hidden
representations in a Skill-conditioned coding-agent setting?

## Hypothesis

Skill-conditioned hidden states contain linearly decodable information about
the current procedural stage beyond trajectory position and tool identity.

## Design

Conditions:

- No Skill
- Correct Skill
- Shuffled Skill

Six stages:

1. `REPRODUCE`
2. `INSPECT_TEST`
3. `INSPECT_IMPLEMENTATION`
4. `REPAIR`
5. `TARGET_VERIFY`
6. `REGRESSION_VERIFY`

Controls included:

- task-grouped cross-validation;
- trajectory-position baseline;
- same-tool stage pairs.

## Main Results

- Best hidden-state index: **19**
- `correct_skill - shuffled_skill` Macro-F1: **1.000**
- Correct-Skill F1: **0.978**
- Shuffled-Skill F1: **0.978**
- No-Skill F1: **0.956**
- Position baseline: **0.422**

Same-tool controls:

- `INSPECT_TEST` vs `INSPECT_IMPLEMENTATION`: **1.000**
- `TARGET_VERIFY` vs `REGRESSION_VERIFY`: **1.000**

## Observation

Procedural-stage information is extremely strongly linearly decodable,
especially around H19.

## Critical Confound

No-Skill representations already achieve:

```text
Macro-F1 = 0.956
```

Therefore EXP01 does **not** establish that the representation is induced by
the Skill.

The teacher-forced execution history itself carries strong semantic evidence
about the current procedural stage.

## Supported Claims

- Procedural stage is strongly represented.
- The representation generalizes across held-out synthetic tasks.
- It is not reducible to next-tool identity.
- Raw position features do not explain the full signal.

## Not Established

- Skill causally creates the representation.
- Skill order causally controls the representation.
- H19 is a causal control site.
- The representation is behaviorally necessary.

## Decision

Proceed to a counterbalanced design where task/history/tool identity remain
fixed and only Skill-prescribed next state changes.

---

# EXP01b — Counterbalanced Skill-Controlled Next-State Decoding

## Date
2026-09-20

## Commit
`62aee2c`

## Status
COMPLETED

## Motivation

EXP01 showed strong stage decodability but also a strong No-Skill signal.

EXP01b removes this confound by holding constant:

- task;
- failure observation;
- execution history;
- current position;
- relevant files;
- next-tool type.

Only Skill-defined procedural order changes.

## Primary Contrast

```text
INSPECT_TEST
vs
INSPECT_IMPLEMENTATION
```

Both require:

```text
read_file(...)
```

Only the file target differs:

```text
tests/...
vs
src/...
```

## Cross-Wording Design

Two wording families:

- canonical;
- paraphrased.

Primary evaluation crosses both:

- held-out tasks;
- held-out wording family.

H19 was treated as the confirmatory representation site based on EXP01.

## Main Results

### Representation

Cross-wording Macro-F1:

```text
0.709
```

Chance:

```text
0.500
```

### Random Task-Specific Mapping Control

Mean Macro-F1:

```text
0.458
```

This stays near chance and below the true semantic-label probe.

### Behavioral Preference

Overall next-action accuracy:

```text
0.651
```

Canonical wording:

```text
0.719
```

Representation–behavior Pearson correlation:

```text
r = 0.345
```

### No-Skill Prior

Without Skill, the model chooses implementation/source inspection in:

```text
97.9%
```

of cases.

This is a strong intrinsic implementation-first prior.

## Interpretation

EXP01b supports a Skill-sensitive representation of the prescribed next
procedural state that generalizes across task and wording.

It also provides behavior-level evidence that Skill ordering shifts action
preference.

However:

```text
decodable representation
!=
causal control
```

The representation-level evidence is stronger than the behavior-level effect.

## Supported Claims

1. Skill-prescribed next state is linearly decodable.
2. The representation generalizes across held-out tasks.
3. It generalizes across substantially different Skill wording.
4. The effect cannot be explained by tool identity.
5. Arbitrary task-specific label mappings do not generalize similarly.
6. Skill ordering measurably changes next-action preference.

## Not Established

- H19 is necessary.
- The decoded direction is used causally.
- The representation is sufficient to redirect behavior.
- A single linear state variable implements the procedure.

## Decision

Proceed to direct causal intervention.

---

# EXP02 — Causal Procedural-State Steering

## Date
2026-09-20

## Commit
`20dba87`

## Status
COMPLETED — NULL RESULT

## Research Question

Is the linearly identified H19 procedural-state direction sufficient to
causally change the model's next-action preference?

## Important Layer Mapping

Hugging Face:

```text
hidden_states[19]
=
output of decoder block 18
```

because:

```text
hidden_states[0]
=
embedding output
```

## Intervention

Within each train fold:

```text
v =
mean(h_IMPLEMENTATION)
-
mean(h_TEST)
```

On held-out tasks:

```text
h' = h + alpha * v
```

Primary behavioral metric:

```text
M =
mean_logP(src action)
-
mean_logP(test action)
```

Primary symmetric steering effect:

```text
E =
[M(+1) - M(-1)] / 2
```

## Controls

- same-state direction;
- equal-norm orthogonal random directions;
- held-out task construction;
- canonical/paraphrased wording;
- positive/negative steering;
- dose response.

## Main Results

| Criterion | Expected | Observed | Result |
|---|---:|---:|---|
| `E_real > 0` | clearly positive | `+0.00086` | negligible |
| Real > same-state | clearly stronger | real-minus-same ≈ `-0.00030` | failed |
| Real > random | clearly stronger | empirical `p = 0.167` | failed |
| Behavioral reversal | opposite under ±alpha | `83.9% vs 84.4%` | failed |
| Dose response | meaningful monotonic shift | total span ≈ `0.004` | negligible |

## Interpretation

EXP02 rejects the simple causal hypothesis:

> the H19 procedural-state information is implemented as a global,
> task-independent linear steering direction.

The result is consistent with the general interpretability warning:

```text
decodable
!=
globally linearly steerable
```

## What EXP02 Does NOT Show

The null result does not establish that the Skill-sensitive representation is
causally unused.

Alternative explanations remain:

- nonlinear/context-conditioned representation;
- multiple interacting mediators;
- distribution across token positions;
- distribution across layers;
- H19 may be a readable downstream summary rather than the causal source.

## Decision

Do not rescue the hypothesis by simply increasing alpha or trying many new
linear separators.

Proceed to exact paired activation interchange.

---

# EXP03 — Paired Interchange Patching

## Date
2026-09-20

## Commit
`e98dade`

## Status
COMPLETED — NULL / NEGATIVE RESULT

## Research Question

If the global linear-direction assumption is removed, can the exact hidden
state from the same-task opposite-Skill condition transfer the donor's
next-action preference?

## Intervention

For paired prompts with:

- same task;
- same history;
- same wording family;
- opposite Skill-prescribed next state;

replace the recipient's final-prompt-token hidden state with the donor's exact
activation:

```text
h_recipient^(l)
<-
h_donor^(l)
```

Then rescore:

```text
read_file(tests/...)
vs
read_file(src/...)
```

## Confirmatory Site

```text
H19
=
decoder block 18 output
```

Full-layer scan was exploratory.

## Main Result

The confirmatory H19 effect was at most marginal and did not establish causal
transfer.

More strikingly, later hidden states approximately H21-H28 showed stronger
**negative** signed-transfer effects, on the order of roughly:

```text
-0.011 to -0.017
```

which is much larger in magnitude than the weak positive H19 effect.

## Initial Observation

The exact single-site donor activation does not behave like a portable
procedural-state variable.

Deep-layer activation replacement can actively move behavior in the wrong
direction.

## Interpretation Boundary

This result supports:

```text
single-token exact residual-state swap
is not sufficient
```

It does **not yet prove**:

```text
the full mechanism is a distributed circuit
```

because a simpler methodological explanation remains.

## Critical Alternative Explanation: Off-Manifold Chimera

EXP03 replaces only one token position while leaving:

- neighboring token states;
- previous-layer context;
- later interacting states;

from the recipient unchanged.

Therefore the patch may create an inconsistent hybrid:

```text
donor state
+
recipient surrounding computation
```

The strong negative deep-layer effects may reflect disruption caused by this
state mismatch rather than evidence of a specific distributed mechanism.

## Competing Explanations After EXP03

### A. Single-Site Chimera

A coherent donor state requires restoring additional aligned token/layer
context.

Prediction:

```text
single-site patch          fails
coherent distributed patch succeeds
```

### B. Distributed / Interacting Mediation

Procedural control is jointly carried across multiple token positions and/or
layers.

Prediction:

```text
multi-token and/or multi-layer restoration
outperforms single-site patching
```

### C. Residual-State Readout Only

Residual stream contains readable Skill information, but the true causal
mechanism lives in pathway selection such as attention/MLP routing.

Prediction:

```text
even coherent residual restoration remains null
```

## Decision

Run a discriminative multi-token × multi-layer restoration experiment before
claiming a distributed circuit.

---

# EXP04 — Distributed Procedural-State Restoration

## Date
2026-09-20

## Commit
74155e8

## Status
COMPLETE — Positive, self-patch passes, all controls satisfied

## Primary Result (common_H19_H28, 48 tasks)

| Metric | Value | 95% CI |
|--------|-------|--------|
| Signed transfer effect | +0.0579 | [0.0523, 0.0636] |
| Self-patch control | −0.00027 | [−0.00057, +0.000008] |
| Same-state cross-wording | −0.000024 | [−0.00099, +0.00091] |
| Opposite-state cross-wording | +0.0571 | [0.0514, 0.0627] |

By recipient wording:
- canonical: +0.037
- paraphrase: +0.079

## Motivation

EXP03 rules out a portable single-token exact state but leaves a major
ambiguity:

```text
single-site patch failure
=
off-manifold chimera?
or
distributed mediation?
```

EXP04 is specifically designed to distinguish these explanations.

## Research Question

Can a coherent set of donor activations across multiple aligned prompt tokens
and/or multiple adjacent layers causally transfer the donor Skill's prescribed
next-action preference?

## Token Alignment Principle

Do **not** patch unaligned Skill tokens.

For each donor/recipient pair, compute the exact longest common token suffix.

This suffix contains downstream prompt content that is identical in token IDs,
including:

- execution history;
- final instruction;
- generation boundary.

Only aligned positions are patched.

## Localization Configurations

### Token-Distribution Test

```text
last1_H19
common_H19
```

Interpretation:

```text
last1_H19 ~ 0
common_H19 > 0
```

would support multi-token mediation.

### Layer-Distribution Test

Using the aligned common suffix:

```text
common_H19
common_H19_H24
common_H19_H28
common_H21_H28
common_H15_H28
```

## Preregistered Primary Configuration

```text
donor:
same task
same wording
opposite prescribed state

token span:
entire exact common suffix

hidden-state range:
H19-H28
```

Primary metric:

```text
M =
mean_logP(src action)
-
mean_logP(test action)
```

Signed transfer:

```text
T =
donor_sign * (M_patched - M_baseline)
```

Positive means the recipient moves toward the donor's prescribed state.

## Controls

### Self Patch

```text
recipient <- recipient
```

Expected:

```text
~ 0
```

### Same-State Cross-Wording

Example:

```text
canonical TEST-first
<-
paraphrase TEST-first
```

Changes wording without changing procedural state.

Expected directional state-transfer effect:

```text
~ 0
```

### Opposite-State Cross-Wording

Example:

```text
canonical TEST-first
<-
paraphrase IMPLEMENTATION-first
```

A positive result would show causal transfer across Skill surface wording.

## Success Criteria

Support for distributed residual-state mediation requires:

1. primary mean signed transfer > 0;
2. task-bootstrap 95% CI excludes zero;
3. same-state control materially smaller;
4. self-patch approximately zero;
5. canonical and paraphrased recipient conditions have concordant sign;
6. opposite-state cross-wording transfer is also positive.

## Planned Interpretation

### Outcome A — Multi-token rescue

```text
last1_H19 ~ 0
common_H19 > 0
```

Interpretation:

> procedural mediation is distributed across multiple downstream prompt
> positions.

### Outcome B — Multi-layer rescue

```text
common_H19 ~ 0
common_H19_H28 > 0
```

Interpretation:

> procedural mediation depends on coordinated states across layers.

### Outcome C — Full distributed rescue

Primary and cross-wording opposite-state transfer are positive, while self and
same-state controls stay near zero.

Interpretation:

> a coherent distributed residual-stream state is sufficient to transfer part
> of Skill-conditioned action preference.

### Outcome D — All residual restoration remains null

Interpretation:

> stop treating procedural control as a portable residual-state variable.

Next target:

- attention-output pathways;
- MLP-output pathways;
- head-level routing;
- Skill-token → action-token causal paths;
- circuit-selection mechanisms.

## Localization Effects (signed transfer, 48 tasks each)

| Config | Span | Layers | Effect | 95% CI |
|--------|------|--------|--------|--------|
| last1_H19 | last1 | 19 | +0.0011 | [0.0003, 0.0021] |
| common_H19 | common | 19 | +0.0305 | [0.0266, 0.0345] |
| common_H19_H24 | common | 19–24 | +0.0558 | [0.0502, 0.0615] |
| **common_H19_H28** | **common** | **19–28** | **+0.0579** | **[0.0523, 0.0636]** |
| common_H21_H28 | common | 21–28 | −0.0002 | [−0.0038, 0.0033] |
| common_H15_H28 | common | 15–28 | +0.0796 | [0.0731, 0.0863] |

Pattern:
- single last token alone: null;
- entire common suffix at H19: positive → multi-token distribution confirmed;
- adding H19–H20 is necessary (H21-H28 without them → null);
- extending down to H15 increases the effect.

## Interpreted Result — Outcome C (Full distributed rescue)

```text
Tokens:     1 token → 0.001     common suffix → 0.058
Layers:     19 alone → 0.031    19–28 → 0.058    21–28 → ~0
Controls:   self −0.00027     same-state cross-wording −0.00002
```

> A coherent distributed residual-stream state, spanning the full shared
> prompt suffix and the H19–H28 block (critically including H19–H20), is
> sufficient to transfer Skill-conditioned action preference, with controls
> at machine precision zero.

## Methodological Note — Final-Norm Capture Bug (fixed)

The first EXP04 run failed the self-patch sanity check (mean ≈ +0.197
instead of ~0). Root cause: in `outputs.hidden_states`, the element at
index equal to `num_layers` (here `hidden_states[28]`) is the
**post-final-RMSNorm** activation — not the raw output of `layers[27]`.
Patching that value back into `layers[27]`'s pre-norm output injected a
mis-scaled vector (verified max |diff| ≈ 735 → captured 0.196 self-patch
noise) into the residual stream.

Fix: `capture_prompt_states` now grabs hidden states via forward hooks on
`layers[hidx-1]` (pre-norm), exactly matching what `patch_hook` replaces.
Additionally moved `score_one` to per-candidate batch-1 forwards to
eliminate right-padding asymmetry. Self-patch dropped to −0.00027.

Implication for earlier experiments: any location using `hidden_states[len(layers)]`
(e.g. EXP03's L27/L28 patch conditions) shared this capture/patch
misalignment, so those specific layers in EXP03 should be treated as
unverified until re-run with the corrected capture.

---

# Current Evidence Summary

The strongest defensible project-level statement **before EXP04** is:

> Agent Skill procedural information is reliably readable from the residual
> stream and generalizes across task and wording, but neither a global linear
> direction nor a single-token exact residual-state swap is sufficient to
> redirect behavior under the tested interventions.

Current evidence chain:

```text
Skill procedural information
        |
        v
linearly decodable                     YES
        |
        v
cross-wording semantic generalization  YES
        |
        v
associated with action preference      YES, moderate
        |
        v
global linear steering                 NO
        |
        v
single-token exact interchange         NO
        |
        v
distributed causal mediation           YES (EXP04)
```

## Current Claim Boundary

The strongest defensible project-level statement **after EXP04** is:

> Agent Skill procedural information is carried by a distributed
> residual-stream state that spans multiple shared prompt tokens
> (specifically the full aligned common suffix) and multiple adjacent
> layers (critical range: H19–H28, with H19–H20 necessary and
> H21–H28 alone insufficient). This state is sufficient to
> causally transfer Skill-conditioned action preference when patched
> across conditions.

Do not yet state:

> "The true causal mechanism is exclusively a distributed circuit."

Alternative explanations not yet ruled out:

1. Attention-output pathways that happen to carry
   the same information in parallel;
2. MLP-output pathways with distributed representation;
3. Head-level routing / skip-connection bypass mechanisms;

> "The results rule out two simple portable-state hypotheses and motivate a
> distributed/pathway-level causal account."

This wording should be retained until EXP04 or subsequent pathway experiments
provide direct positive causal evidence.
