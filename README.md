# skills4s — Minimal Mechanistic-Interpretability MVP

Research question:

> Does an Agent Skill induce a linearly decodable latent procedural state inside an LLM?

This MVP does **not** run a full coding agent yet. It creates paired prompts with the same task/history:

- `NoSkill`: task + teacher-forced trajectory history
- `Skill`: same task/history + `skills/debugging/SKILL.md`

For every procedural stage, it collects the final prompt-token hidden state at every transformer layer and evaluates:

1. Can the current stage be decoded from `h_skill`?
2. Can it be decoded from `h_no_skill`?
3. Can it be decoded from `Δh = h_skill - h_no_skill`?
4. Can same-tool stages still be distinguished?
5. Can simple position/length features explain the result?

## Files

```text
skills4s/
├── README.md
├── requirements.txt
├── run_mvp.py
├── skills/
│   └── debugging/
│       └── SKILL.md
└── outputs/                 # generated
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Main MVP:

```bash
python run_mvp.py --model Qwen/Qwen2.5-Coder-7B-Instruct --num-tasks 12
```

If you only want to smoke-test the pipeline on a smaller GPU:

```bash
python run_mvp.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --num-tasks 8
```

Use the 7B result for the actual first scientific decision.

## Outputs

The script writes:

```text
outputs/mvp/
├── activations.npz
├── metadata.csv
├── layer_probe.csv
├── same_tool_results.csv
├── position_baseline.csv
├── confusion_matrix_best_layer.csv
├── layer_probe.png
└── summary.json
```

Send back these five values first:

```text
best_layer
best_delta_macro_f1
best_skill_macro_f1
best_no_skill_macro_f1
position_baseline_macro_f1
```

Then inspect `same_tool_results.csv`.
