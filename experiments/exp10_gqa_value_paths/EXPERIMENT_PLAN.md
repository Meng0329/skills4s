# EXP10 — Preregistered Experiment Plan

## Status
PLANNED

## Previous result

EXP09 H20:
- residual +0.0488
- V sufficiency ~89%
- V necessity ~90%
- KV sufficiency ~94%
- KV necessity ~91%
- K ~0
- Q negative

## Main question

Is H20 value routing concentrated in:
1. one/few of the four GQA KV heads;
2. one/few token regions in the aligned common suffix;
3. specific head × token-region interactions?

## Confirmatory branch

H20 `v_proj`.

## Experiment A: heads

Prespecified heads H0..H3.

For each:
- sufficiency
- necessity

Also:
- all four heads
- all-except-H0 ... all-except-H3

## Experiment B: token bins

Split each common suffix into six contiguous normalized bins B0..B5 using `numpy.array_split`.

Patch all four V heads only within one bin.

Report sufficiency, necessity, and fraction of full-V effect.

## Experiment C: head × bin

Evaluate 24 cells for sufficiency and necessity.

Exploratory multiple-testing policy:
- task-level sign-flip p-value
- Benjamini-Hochberg q-value across 24 cells separately for sufficiency and necessity

## Controls

Full-suffix all-head V:
- H20 residual reference
- full V sufficiency
- full V necessity
- self
- same-state cross-wording
- opposite-state cross-wording

## Claim boundary

The interaction matrix remains exploratory until replicated on new Skills/tasks/models.
