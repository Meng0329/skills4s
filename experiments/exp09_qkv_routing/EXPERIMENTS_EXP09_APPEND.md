# EXP09 — Residual → Q/K/V Routing Mediation

## Status
COMPLETED — H20 层 V（KV）投影状态即消费接口（suff 94% / nec 91%），chain 恢复 full 的 84%。已合并入 EXPERIMENTS.md。

## 日期
2026-09-22

## Motivation
EXP08 identifies a compact H18–H20 causal core, but no single handoff layer is
sufficient. EXP06/07 reject portable downstream MLP and post-attention outputs.

EXP09 tests the consumer interface immediately downstream of each core residual:
Q/K/V projection states.

## Primary hypothesis
Prompt-side KV/QKV routing in the block consuming H20 mediates a substantial
part of the H20 residual effect.

## Design
For H18/H19/H20:
- reproduce residual effect;
- Q/K/V/KV/QKV sufficiency;
- Q/K/V/KV/QKV necessity;
- QKV self and cross-wording controls.

H20 is confirmatory.

## Decision
Positive KV/QKV mediation -> GQA KV-head + token-position path localization.
Necessity-only -> context-conditioned routing.
Null/negative -> move to joint normalization/residual-path mediation.
