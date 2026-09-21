# EXP06 — Preregistered Experiment Plan

## Status
PLANNED

## Previous result

EXP05:

```text
early H15–H20 behavior effect = +0.0784
95% CI = [0.0717, 0.0849]

late H21–H28 ≈ 0

full H15–H28 = +0.0797

selector_fraction = 0.984

downstream projection recovery:
MLP      = 0.875
Residual = 0.872
Attention= 0.700
```

## Main hypothesis

A sparse/compositional subset of H21–H28 MLP intermediate neurons mediates the
causal effect of the H15–H20 selector state.

## Discovery

Use 4-fold task splits.

Feature selection uses training tasks only.

For every layer/neuron:

```text
R = recipient baseline
D = opposite-state donor baseline
P = recipient after H15–H20 restoration

recovery_ratio =
mean[(P-R)(D-R)] / mean[(D-R)^2]

impact_score =
max(recovery_ratio,0)
* RMS(D-R)
* ||down_proj column||
```

Select global layer-neuron pairs by impact score.

## Confirmatory group size

```text
K = 256
```

Exploratory:

```text
K = 64
K = 1024
```

## Sufficiency test

Patch selected H21–H28 MLP intermediate-neuron activations from donor into the
baseline recipient.

Primary metric:

```text
signed transfer =
donor_sign * (patched_margin - baseline_margin)
```

Expected:

```text
> 0
```

## Necessity test

Apply H15–H20 selector restoration, then clamp selected downstream neurons back
to baseline recipient values.

Define:

```text
early_effect =
donor_sign * (early_margin - baseline_margin)

retained_effect =
donor_sign * (early+clamp_margin - baseline_margin)

necessity_loss =
early_effect - retained_effect
```

Expected:

```text
necessity_loss > 0
```

## Controls

- all downstream MLP neurons, sufficiency;
- all downstream MLP neurons, necessity;
- five random neuron groups matched to K=256 layer counts;
- held-out task evaluation;
- K=256 cross-wording opposite-state donor.

## Main success criteria

Primary K=256 group is supported if:

1. sufficiency 95% CI > 0;
2. necessity-loss 95% CI > 0;
3. selected sufficiency > mean matched-random sufficiency;
4. selected necessity loss > mean matched-random necessity loss;
5. cross-wording effects have the same sign.

A positive result identifies a causal MLP mediator group, but does not yet
establish semantic interpretability of the group.
