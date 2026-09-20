# EXP02 — Causal Procedural-State Steering

## Research question

Does the procedural-state direction identified in EXP01b **causally control**
the model's next-action preference?

EXP01b showed that Skill-prescribed next state is:

- cross-task / cross-wording decodable at hidden-state index 19;
- associated with next-action preference.

EXP02 moves from association to intervention.

## Important layer mapping

Hugging Face `hidden_states` contains:

```text
hidden_states[0]  = embedding output
hidden_states[1]  = output after decoder block 0
...
hidden_states[19] = output after decoder block 18
```

Therefore the preregistered EXP01/EXP01b hidden-state index 19 maps to:

```text
model.model.layers[18]
```

EXP02 explicitly records both indices to prevent an off-by-one error.

## Causal direction

For every held-out-task fold, the direction is computed **only from training
tasks** in EXP01b:

```text
v = mean(h_impl) - mean(h_test)
```

at hidden-state index 19.

The direction is then injected at the last prompt token of decoder block 18.

## Primary causal endpoint

For each prompt:

```text
margin =
mean_logP(read_file(src/...))
-
mean_logP(read_file(tests/...))
```

Positive values favor implementation inspection.

Primary preregistered effect:

```text
E_real =
mean( margin(alpha=+1) - margin(alpha=-1) ) / 2
```

Expected:

```text
E_real > 0
```

and substantially larger than matched controls.

## Controls

1. **Orthogonal random direction**
   - random vector;
   - explicitly orthogonalized against the real direction;
   - norm-matched to the real direction.

2. **Same-state direction**
   - constructed from two halves of IMPLEMENTATION-state training examples;
   - norm-matched to the real direction;
   - captures arbitrary within-state activation variation.

3. **Held-out tasks**
   - directions are learned from training tasks only;
   - intervention is evaluated only on held-out tasks.

4. **Cross-wording**
   - canonical and paraphrased Skills are both evaluated.

5. **Dose response**
   - real direction: alpha = -2, -1, -0.5, +0.5, +1, +2;
   - alpha = ±1 is the preregistered primary comparison;
   - larger/smaller coefficients are exploratory.

6. **Multiple random controls**
   - five independent random directions by default.

## Why native PyTorch hooks?

This experiment uses a standard forward hook on the Qwen decoder block output
instead of adding an interpretability framework dependency.

It implements exactly the intervention we need:

```text
hidden[last_prompt_token] += alpha * direction
```

If EXP02 succeeds, later experiments can adopt NNsight/pyvene for more complex
multi-component patching.

## Required previous output

EXP02 reads:

```text
outputs/exp01b_counterbalanced_next_state/
├── activations.npz
├── metadata.csv
└── run_manifest.json
```

The original EXP01b activations are used only to estimate directions.
No Hugging Face network access is used.

## Run

From the repository root:

```bash
python experiments/exp02_causal_state_steering/run.py
```

Recommended:

```bash
python experiments/exp02_causal_state_steering/run.py \
  --batch-size 4 \
  --num-random-controls 5
```

If GPU memory is tight:

```bash
python experiments/exp02_causal_state_steering/run.py --batch-size 1
```

## Outputs

```text
outputs/exp02_causal_state_steering/
├── intervention_results.csv
├── fold_directions.csv
├── primary_effects_by_task.csv
├── random_control_effects.csv
├── dose_response.csv
├── dose_response.png
├── primary_effects.png
├── run_manifest.json
└── summary.json
```

## Go / No-Go criterion

Strong support requires all of the following:

1. real-direction primary effect has the predicted sign;
2. task-bootstrap 95% CI excludes zero;
3. real effect exceeds the same-state control;
4. real effect exceeds the distribution of orthogonal random controls;
5. dose response is approximately monotonic around alpha = 0;
6. the effect appears in both canonical and paraphrased Skills.

If these hold, the project has causal evidence that a linearly identified
procedural-state direction participates in controlling next-action preference.
