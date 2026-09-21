# EXP07 — Attention-Head Causal Mediation

## Motivation
EXP06 falsified downstream MLP mediation. EXP07 tests whether H21–H28
attention-head outputs mediate the causal H15–H20 selector effect.

## Intervention point
Qwen2 concatenates per-head attention outputs immediately before
`self_attn.o_proj`. We capture and edit the input to `o_proj`, reshaped as:

`[batch, seq, num_attention_heads, head_dim]`.

## Tests
- Sufficiency: donor head outputs -> baseline recipient.
- Necessity: H15–H20 selector restoration + selected heads clamped back to
  recipient baseline.
- Full-attention upper bound.
- K=16 primary sparse group; K=4/32/64 exploratory.
- Layer-count-matched random groups.
- Cross-wording validation.
- 4-fold held-out-task selection/evaluation.

## Run
```bash
python experiments/exp07_attention_heads/run.py
```

## Main interpretation
Do not claim a sparse attention circuit unless:
1. full attention branch is causally compatible;
2. selected heads are sufficient and necessary;
3. selected groups outperform matched random groups;
4. cross-wording effect preserves sign.
