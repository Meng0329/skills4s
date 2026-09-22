# EXP11 — Exact KV0×Token → Query-Head Value Path

EXP10 localized the H20 value interface to KV0×B5. EXP11 makes two finer cuts.

## Part A — exact B5 offsets
For every exact offset inside B5, patch only block20 V / KV0 / that prompt token.
Measure:
- sufficiency;
- residual-based necessity loss;
- leave-one-token-out loss from full KV0×B5.

Offsets are exploratory; report sign-flip p-values and BH-FDR.

## Part B — reader query-head path
Qwen2.5-Coder-7B-Instruct has 28 query heads and 4 KV heads. `repeat_kv`
duplicates each KV head 7 times, so KV0 feeds Q0..Q6.

This is not another donor post-attention transplant. For each candidate:
1. capture block20 `o_proj` input at baseline;
2. capture it under verified KV0×B5 V patch;
3. compute the actual per-query-head delta caused by the V patch;
4. baseline + only Qh delta = path sufficiency;
5. V patch + Qh clamped to baseline = path necessity.

Sanity:
- all Q0..Q6 deltas must reconstruct the full V effect;
- Q7..Q27 induced delta should be approximately zero.

## Run

```bash
python experiments/exp11_exact_value_path/run.py
```

Optional phases:

```bash
python experiments/exp11_exact_value_path/run.py --phase offsets
python experiments/exp11_exact_value_path/run.py --phase readers
```

## Outputs
`outputs/exp11_exact_value_path/`
