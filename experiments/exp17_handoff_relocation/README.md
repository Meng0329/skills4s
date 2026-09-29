# EXP17 — Dialogue-Handoff Writer Relocation Falsification

EXP17 tests whether the frozen causal writer follows procedural semantics,
absolute position, the end of user content, or the actual generation handoff.

No layer, KV/value group, or reader head is rediscovered.

Frozen circuits:
- Qwen2.5-Coder-7B-Instruct: L20 / KV0 / positive readers Q0,Q3,Q5
- Mistral-7B-Instruct-v0.3: L30 / KV4 / positive readers Q16,Q18
- Granite-3.0-8B-Instruct: L39 / KV2 / positive readers Q10,Q11

Fresh data:
- 48 new tasks total
- 12/family
- first 8/family = 32 confirmatory
- last 4/family = 16 reserve robustness tasks

Topology conditions:
- native
- suffix_short
- suffix_long
- instruction_early

Candidate sites (all 6 tokens):
- FINAL_INSTRUCTION_END
- USER_CONTENT_END
- TEMPLATE_TERMINATOR
- GENERATION_BOUNDARY
- NATIVE_ABSOLUTE

Primary causal endpoints:
- frozen V/KV sufficiency
- local frozen V/KV necessity

Frozen-reader sufficiency/necessity is measured at GENERATION_BOUNDARY,
FINAL_INSTRUCTION_END, and NATIVE_ABSOLUTE for native and suffix_long.

Controls at the actual generation boundary:
- opposite-state cross-wording
- same-state cross-wording
- matched direct-choice opposite-label intervention

Run:
python experiments/exp17_handoff_relocation/run.py --model all --phase audit
python experiments/exp17_handoff_relocation/run.py --model all --phase confirm

Reserve only after primary verdict is written:
python experiments/exp17_handoff_relocation/run.py --model all --phase reserve
