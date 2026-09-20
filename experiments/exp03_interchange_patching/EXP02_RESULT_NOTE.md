# EXP02 — Result Note

## Status
COMPLETED — NULL RESULT

## Commit
20dba87

## Main Result

The preregistered linear causal-steering hypothesis was not supported.

| Criterion | Result |
|---|---:|
| Real primary effect | +0.00086 |
| Real > same-state | No; difference ≈ -0.00030 |
| Real > random controls | No; empirical p = 0.167 |
| Behavioral reversal | No; 83.9% vs 84.4% |
| Dose-response span | ≈ 0.004 |

## Interpretation

EXP01b established that Skill-prescribed next state is linearly decodable from
the residual stream and associated with behavior.

EXP02 shows that the corresponding mean-difference direction at the
preregistered site is not sufficient to causally redirect next-action
preference under the tested intervention.

Therefore:

```text
decodable != globally linearly steerable
```

The result does not establish that the decoded information is causally unused.
Alternative possibilities include:

- context-dependent nonlinear representations;
- multiple interacting mediators;
- distributed representation across tokens/layers;
- a representation that is readable but not on the causal pathway.

## Decision

Do not increase alpha or search for a better linear separator as the immediate
next step.

Proceed to EXP03 exact paired interchange patching to remove the global-linear
direction assumption.
