# EXP07 — Attention-Head Causal Mediation

## Status
PLANNED

## Motivation
EXP06 falsified H21–H28 MLP intermediate activations as the causal mediator.
The major remaining downstream branch is self-attention.

## Question
Do H21–H28 attention-head outputs causally mediate the behavioral effect of the
H15–H20 distributed selector state?

## Intervention
Patch at `self_attn.o_proj` input, where per-head outputs are concatenated.

## Primary group
K=16 layer-head components; K=4/32/64 exploratory.

## Standard
A candidate head group must pass both:
- sufficiency: donor head transplant changes behavior toward donor;
- necessity: selector + recipient clamp reduces selector effect.

Full-attention upper bounds and layer-matched random controls are mandatory.

## Decision
- sparse + full attention positive -> move to Q/K/V and path-level circuit analysis;
- only full attention positive -> broad distributed routing;
- full attention null/negative -> reject portable post-attention-output mediation.
