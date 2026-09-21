# EXP06 — MLP Neuron-Group Causal Mediation

## Motivation

EXP05 supports an MLP-dominant selector mechanism:

- H15–H20 selector restoration captures ~98.4% of the full H15–H28 effect.
- H21–H28 alone is null.
- Unpatched downstream MLP recovery is stronger than attention recovery.

EXP06 asks which downstream MLP intermediate neurons mediate this effect.

## Qwen2 MLP intervention site

Qwen2 computes:

```text
z = SiLU(gate_proj(x)) * up_proj(x)
y = down_proj(z)
```

`z` is the MLP intermediate-neuron activation vector.

EXP06 captures and edits `z` through a `forward_pre_hook` on each layer's
`mlp.down_proj`.

## Discovery statistic

For each training task and downstream neuron:

```text
R = baseline recipient neuron activation
D = baseline donor neuron activation
P = recipient activation after H15–H20 selector restoration

aligned = mean[(P - R) * (D - R)]
energy  = mean[(D - R)^2]
recovery_ratio = aligned / (energy + eps)
```

To rank neurons by potential residual impact:

```text
impact_score =
max(recovery_ratio, 0)
* sqrt(energy)
* ||down_proj[:, neuron]||
```

Neuron groups are selected using training tasks only.

## Causal tests on held-out tasks

### Sufficiency

Without H15–H20 restoration, patch only selected downstream MLP neurons to
their donor values.

If the group is sufficient:

```text
action preference -> donor state
```

### Necessity

First apply the causal H15–H20 selector restoration, then clamp selected
downstream MLP neurons back to their recipient baseline values.

If the group is necessary:

```text
selector behavioral effect decreases
```

### Controls

- full downstream MLP donor patch: sufficiency upper bound;
- full downstream MLP clamp-back: necessity upper bound;
- layer-count-matched random neuron groups;
- held-out tasks;
- primary group size K=256;
- exploratory K=64 and K=1024;
- cross-wording test for the primary K=256 group.

## Run

From repository root:

```bash
python experiments/exp06_mlp_neuron_groups/run.py
```

The script is fully offline and auto-detects the local model under `./models`.

## Outputs

```text
outputs/exp06_mlp_neuron_groups/
├── feature_task_stats.npz
├── selected_neurons.csv
├── intervention_results.csv
├── summary_by_config.csv
├── topk_dose_response.csv
├── topk_dose_response.png
├── selected_layer_counts.png
├── run_manifest.json
└── summary.json
```

## Main success criteria

For primary K=256:

1. selected-group sufficiency effect > 0 with task-bootstrap 95% CI excluding 0;
2. selected-group necessity loss > 0 with CI excluding 0;
3. both exceed matched-random-group controls;
4. full-MLP positive controls are stronger or comparable;
5. cross-wording effects have the same sign.

If successful, EXP07 should interpret the causal neuron group using SNMF /
compositional neuron features and test cross-Skill reuse.
