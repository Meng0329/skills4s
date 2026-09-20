#!/usr/bin/env python3
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import argparse
import hashlib
import importlib.util
import json
import platform
import random
import subprocess
from contextlib import ExitStack, contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = REPO_ROOT / "models"
EXP01B_CODE = REPO_ROOT / "experiments" / "exp01b_counterbalanced_next_state" / "run.py"
EXP01B_MANIFEST = REPO_ROOT / "outputs" / "exp01b_counterbalanced_next_state" / "run_manifest.json"
DEFAULT_OUT = REPO_ROOT / "outputs" / "exp05_selector_pathway"

LABEL_TEST = 0
LABEL_IMPL = 1

# Hidden-state index Hk == decoder block (k-1) output.
EARLY_HIDDEN = list(range(15, 21))   # H15..H20 -> decoder 14..19
LATE_HIDDEN = list(range(21, 29))    # H21..H28 -> decoder 20..27
FULL_HIDDEN = list(range(15, 29))    # H15..H28

BEHAVIOR_CONFIGS = [
    ("early_H15_H20", EARLY_HIDDEN, "primary"),
    ("late_H21_H28", LATE_HIDDEN, "negative_control"),
    ("full_H15_H28", FULL_HIDDEN, "positive_control"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def resolve_model_dir(arg):
    if arg:
        p = Path(arg).expanduser()
        if not p.is_absolute():
            p1 = (REPO_ROOT / p).resolve()
            p2 = (MODELS_ROOT / p).resolve()
            p = p1 if p1.exists() else p2
        p = p.resolve()
        if not (p / "config.json").is_file():
            raise FileNotFoundError(f"Missing config.json: {p}")
        return p

    if (MODELS_ROOT / "config.json").is_file():
        return MODELS_ROOT.resolve()

    candidates = sorted({p.parent.resolve() for p in MODELS_ROOT.rglob("config.json")})
    if not candidates:
        raise FileNotFoundError("No local model found under ./models.")
    if len(candidates) > 1:
        raise RuntimeError("Multiple local models found; specify --model.")
    return candidates[0]


def load_exp01b():
    if not EXP01B_CODE.is_file():
        raise FileNotFoundError(f"Missing EXP01b code: {EXP01B_CODE}")
    spec = importlib.util.spec_from_file_location("skills4s_exp01b", EXP01B_CODE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_manifest():
    if not EXP01B_MANIFEST.is_file():
        raise FileNotFoundError(f"Missing EXP01b manifest: {EXP01B_MANIFEST}")
    return json.loads(EXP01B_MANIFEST.read_text(encoding="utf-8"))


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("Expected Qwen-style model.model.layers")


def render_prompt(tok, messages):
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def recreate_entries(exp01b, manifest):
    tasks = exp01b.make_tasks(int(manifest["num_tasks"]), int(manifest["seed"]))
    out = {}
    for task in tasks:
        ti = int(task["task_idx"])
        history = exp01b.history(task, ti)
        entries = []
        for cond, wording, label in exp01b.CONDS:
            entries.append({
                "task_idx": ti,
                "task_id": task["task_id"],
                "condition": cond,
                "wording": wording,
                "label": int(label),
                "messages": exp01b.messages(
                    task, history, exp01b.SKILLS[cond]
                ),
                "candidates": [
                    f"read_file('{task['test_path']}')",
                    f"read_file('{task['src_path']}')",
                ],
            })
        out[ti] = entries
    return out


def donor_maps(entries):
    lookup = {(e["wording"], e["label"]): i for i, e in enumerate(entries)}
    maps = {}
    for i, e in enumerate(entries):
        opp_label = LABEL_IMPL if e["label"] == LABEL_TEST else LABEL_TEST
        opp_wording = "paraphrase" if e["wording"] == "canonical" else "canonical"
        maps[i] = {
            "opposite_same_wording": lookup[(e["wording"], opp_label)],
            "same_state_cross_wording": lookup[(opp_wording, e["label"])],
            "opposite_cross_wording": lookup[(opp_wording, opp_label)],
            "self": i,
        }
    return maps


def longest_common_suffix(a, b):
    k = 0
    n = min(len(a), len(b))
    while k < n and a[-1-k] == b[-1-k]:
        k += 1
    return k


def first_tensor(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def get_target_modules(layers):
    return {
        "residual": {h: layers[h-1] for h in FULL_HIDDEN},
        "attn": {h: layers[h-1].self_attn for h in LATE_HIDDEN},
        "mlp": {h: layers[h-1].mlp for h in LATE_HIDDEN},
    }


@torch.inference_mode()
def tokenize_entries(tok, entries):
    info = []
    for e in entries:
        text = render_prompt(tok, e["messages"])
        ids = tok(text, add_special_tokens=False)["input_ids"]
        info.append({"ids": ids, "text": text})
    return info


@torch.inference_mode()
def capture_prompt(model, tok, modules, entry, max_suffix):
    """Capture only the last max_suffix prompt positions.

    Residual states: H15..H28 (decoder outputs).
    Downstream pathway outputs: attention / MLP for H21..H28.
    """
    device = model.get_input_embeddings().weight.device
    text = render_prompt(tok, entry["messages"])
    enc = tok(text, add_special_tokens=False, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}

    store = {
        "residual": {},
        "attn": {},
        "mlp": {},
    }
    handles = []

    def make_hook(kind, hidx):
        def fn(module, inputs, output):
            t = first_tensor(output)
            take = min(max_suffix, t.shape[1])
            store[kind][hidx] = (
                t[0, -take:, :]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        return fn

    for hidx, module in modules["residual"].items():
        handles.append(module.register_forward_hook(make_hook("residual", hidx)))
    for hidx, module in modules["attn"].items():
        handles.append(module.register_forward_hook(make_hook("attn", hidx)))
    for hidx, module in modules["mlp"].items():
        handles.append(module.register_forward_hook(make_hook("mlp", hidx)))

    try:
        model(**enc, use_cache=False, return_dict=True)
    finally:
        for h in handles:
            h.remove()

    return {"ids": ids, "states": store}


@contextmanager
def residual_patch_hook(layer, positions, replacements):
    def hook_fn(module, inputs, output):
        if isinstance(output, tuple):
            t, rest = output[0], output[1:]
        else:
            t, rest = output, None

        t = t.clone()
        b = torch.arange(t.shape[0], device=t.device).unsqueeze(1)
        t[b, positions.to(t.device), :] = replacements.to(
            t.device, dtype=t.dtype
        )

        return t if rest is None else (t,) + rest

    handle = layer.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def capture_downstream_hooks(layers, positions, capture_store):
    """Capture unpatched downstream attention/MLP/block outputs H21..H28."""
    handles = []

    def make_capture(kind, hidx):
        def fn(module, inputs, output):
            t = first_tensor(output)
            pos = positions.to(t.device)
            b = torch.arange(t.shape[0], device=t.device).unsqueeze(1)
            picked = t[b, pos, :]
            capture_store[kind][hidx] = (
                picked.detach().float().cpu().numpy().astype(np.float32)
            )
        return fn

    for hidx in LATE_HIDDEN:
        layer = layers[hidx-1]
        handles.append(
            layer.self_attn.register_forward_hook(
                make_capture("attn", hidx)
            )
        )
        handles.append(
            layer.mlp.register_forward_hook(
                make_capture("mlp", hidx)
            )
        )
        handles.append(
            layer.register_forward_hook(
                make_capture("residual", hidx)
            )
        )

    try:
        yield
    finally:
        for h in handles:
            h.remove()


def prepare_patch_positions(prompt_len, common_k, batch_rows=1):
    pos = np.arange(prompt_len-common_k, prompt_len, dtype=np.int64)
    return torch.tensor(
        np.stack([pos] * batch_rows),
        dtype=torch.long,
    )


def get_donor_slice(cache, kind, hidx, k):
    arr = cache["states"][kind][hidx]
    if arr.shape[0] < k:
        raise RuntimeError(
            f"Captured only {arr.shape[0]} tokens but need {k}."
        )
    return arr[-k:, :].astype(np.float32)


@torch.inference_mode()
def prompt_forward_with_residual_patch(
    model, tok, layers, entry, recipient_cache, donor_cache,
    hidden_indices, common_k, capture_downstream=False
):
    """Prompt-only forward with residual restoration."""
    device = model.get_input_embeddings().weight.device
    text = render_prompt(tok, entry["messages"])
    enc = tok(text, add_special_tokens=False, return_tensors="pt")
    plen = enc["input_ids"].shape[1]
    enc = {k: v.to(device) for k, v in enc.items()}

    positions = prepare_patch_positions(plen, common_k, 1).to(device)
    downstream = {"attn": {}, "mlp": {}, "residual": {}}

    with ExitStack() as stack:
        for hidx in hidden_indices:
            donor_vec = get_donor_slice(
                donor_cache, "residual", hidx, common_k
            )
            rep = torch.tensor(
                donor_vec[None, :, :],
                dtype=torch.float32,
                device=device,
            )
            stack.enter_context(
                residual_patch_hook(
                    layers[hidx-1], positions, rep
                )
            )

        if capture_downstream:
            stack.enter_context(
                capture_downstream_hooks(
                    layers, positions, downstream
                )
            )

        model(**enc, use_cache=False, return_dict=True)

    return downstream


@torch.inference_mode()
def score_candidates(
    model, tok, layers, entry, donor_cache=None,
    hidden_indices=None, common_k=None
):
    """Score candidates one at a time to avoid padding asymmetry."""
    device = model.get_input_embeddings().weight.device
    prompt_text = render_prompt(tok, entry["messages"])
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    plen = len(prompt_ids)

    scores = []

    for candidate in entry["candidates"]:
        cids = tok(candidate, add_special_tokens=False)["input_ids"]
        ids = prompt_ids + cids
        input_ids = torch.tensor(
            [ids], dtype=torch.long, device=device
        )
        attention = torch.ones_like(input_ids)

        if donor_cache is None:
            stack = ExitStack()
        else:
            stack = ExitStack()
            positions = prepare_patch_positions(
                plen, common_k, 1
            ).to(device)
            for hidx in hidden_indices:
                donor_vec = get_donor_slice(
                    donor_cache, "residual", hidx, common_k
                )
                rep = torch.tensor(
                    donor_vec[None, :, :],
                    dtype=torch.float32,
                    device=device,
                )
                stack.enter_context(
                    residual_patch_hook(
                        layers[hidx-1], positions, rep
                    )
                )

        with stack:
            out = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            )

        lp = torch.log_softmax(out.logits.float(), dim=-1)
        c = len(cids)
        pred_pos = torch.arange(
            plen-1, plen+c-1, device=device
        )
        targ_pos = torch.arange(
            plen, plen+c, device=device
        )
        targets = input_ids[0, targ_pos]
        scores.append(
            float(lp[0, pred_pos, targets].mean().cpu())
        )

    return {
        "test_mean_logprob": scores[0],
        "impl_mean_logprob": scores[1],
        "margin": scores[1] - scores[0],
    }


def projection_recovery(recipient, donor, patched, eps=1e-12):
    """Flatten token×hidden and measure movement along donor-recipient axis."""
    r = recipient.astype(np.float64).reshape(-1)
    d = donor.astype(np.float64).reshape(-1)
    p = patched.astype(np.float64).reshape(-1)

    axis = d - r
    denom = float(np.dot(axis, axis))
    if denom < eps:
        return np.nan
    return float(np.dot(p-r, axis) / denom)


def distance_recovery(recipient, donor, patched, eps=1e-12):
    r = recipient.astype(np.float64).reshape(-1)
    d = donor.astype(np.float64).reshape(-1)
    p = patched.astype(np.float64).reshape(-1)

    base = float(np.linalg.norm(r-d))
    if base < eps:
        return np.nan
    return float(1.0 - np.linalg.norm(p-d) / base)


def bootstrap_ci(series: pd.Series, seed: int, n_boot=5000):
    x = series.dropna().to_numpy(np.float64)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boots[i] = rng.choice(x, size=len(x), replace=True).mean()
    return (
        float(x.mean()),
        float(np.quantile(boots, 0.025)),
        float(np.quantile(boots, 0.975)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=4505)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    exp01b = load_exp01b()
    manifest01b = load_manifest()
    by_task = recreate_entries(exp01b, manifest01b)

    model_dir = resolve_model_dir(args.model)
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    layers = get_layers(model)
    modules = get_target_modules(layers)

    if max(FULL_HIDDEN) > len(layers):
        raise RuntimeError("Model depth is smaller than requested H28.")

    print("="*80)
    print("EXP05: Selector -> Pathway Reconfiguration")
    print("Model:", model_dir)
    print("Tasks:", len(by_task))
    print("Primary early selector window: H15-H20")
    print("="*80)

    behavior_rows = []
    pathway_rows = []

    for count, ti in enumerate(sorted(by_task), 1):
        entries = by_task[ti]
        maps = donor_maps(entries)
        token_info = tokenize_entries(tok, entries)

        # Determine max suffix needed for each entry across all donor relations.
        max_suffix = []
        for i in range(len(entries)):
            ks = []
            for di in maps[i].values():
                ks.append(
                    longest_common_suffix(
                        token_info[i]["ids"],
                        token_info[di]["ids"],
                    )
                )
            max_suffix.append(max(ks))

        cache = [
            capture_prompt(
                model, tok, modules, entries[i], max_suffix[i]
            )
            for i in range(len(entries))
        ]

        baselines = [
            score_candidates(model, tok, layers, e)
            for e in entries
        ]

        # Main same-wording opposite-state configs.
        for config_name, hidden_indices, role in BEHAVIOR_CONFIGS:
            for i, rec in enumerate(entries):
                di = maps[i]["opposite_same_wording"]
                donor = entries[di]
                k = longest_common_suffix(
                    cache[i]["ids"], cache[di]["ids"]
                )
                patched = score_candidates(
                    model, tok, layers, rec,
                    donor_cache=cache[di],
                    hidden_indices=hidden_indices,
                    common_k=k,
                )
                raw = patched["margin"] - baselines[i]["margin"]
                sign = 1.0 if donor["label"] == LABEL_IMPL else -1.0

                behavior_rows.append({
                    "task_idx": ti,
                    "task_id": rec["task_id"],
                    "recipient_condition": rec["condition"],
                    "recipient_wording": rec["wording"],
                    "recipient_label": rec["label"],
                    "donor_type": "opposite_same_wording",
                    "donor_condition": donor["condition"],
                    "donor_label": donor["label"],
                    "config": config_name,
                    "config_role": role,
                    "common_suffix_tokens": k,
                    "baseline_margin": baselines[i]["margin"],
                    "patched_margin": patched["margin"],
                    "raw_margin_shift": raw,
                    "signed_transfer_effect": sign * raw,
                })

        # Controls for the primary early selector patch.
        early_name, early_hidden, _ = BEHAVIOR_CONFIGS[0]
        for donor_type in [
            "self",
            "same_state_cross_wording",
            "opposite_cross_wording",
        ]:
            for i, rec in enumerate(entries):
                di = maps[i][donor_type]
                donor = entries[di]
                k = longest_common_suffix(
                    cache[i]["ids"], cache[di]["ids"]
                )
                patched = score_candidates(
                    model, tok, layers, rec,
                    donor_cache=cache[di],
                    hidden_indices=early_hidden,
                    common_k=k,
                )
                raw = patched["margin"] - baselines[i]["margin"]

                if donor_type == "self":
                    signed = raw
                else:
                    sign = 1.0 if donor["label"] == LABEL_IMPL else -1.0
                    signed = sign * raw

                behavior_rows.append({
                    "task_idx": ti,
                    "task_id": rec["task_id"],
                    "recipient_condition": rec["condition"],
                    "recipient_wording": rec["wording"],
                    "recipient_label": rec["label"],
                    "donor_type": donor_type,
                    "donor_condition": donor["condition"],
                    "donor_label": donor["label"],
                    "config": early_name,
                    "config_role": "control",
                    "common_suffix_tokens": k,
                    "baseline_margin": baselines[i]["margin"],
                    "patched_margin": patched["margin"],
                    "raw_margin_shift": raw,
                    "signed_transfer_effect": signed,
                })

        # Pathway readout after patching only H15-H20.
        for i, rec in enumerate(entries):
            di = maps[i]["opposite_same_wording"]
            donor = entries[di]
            k = longest_common_suffix(
                cache[i]["ids"], cache[di]["ids"]
            )

            patched_downstream = prompt_forward_with_residual_patch(
                model=model,
                tok=tok,
                layers=layers,
                entry=rec,
                recipient_cache=cache[i],
                donor_cache=cache[di],
                hidden_indices=EARLY_HIDDEN,
                common_k=k,
                capture_downstream=True,
            )

            for kind in ["attn", "mlp", "residual"]:
                for hidx in LATE_HIDDEN:
                    r = get_donor_slice(
                        cache[i], kind, hidx, k
                    )
                    d = get_donor_slice(
                        cache[di], kind, hidx, k
                    )
                    p = patched_downstream[kind][hidx][0].astype(
                        np.float32
                    )

                    pathway_rows.append({
                        "task_idx": ti,
                        "task_id": rec["task_id"],
                        "recipient_condition": rec["condition"],
                        "recipient_wording": rec["wording"],
                        "recipient_label": rec["label"],
                        "donor_condition": donor["condition"],
                        "donor_label": donor["label"],
                        "component": kind,
                        "hidden_state_index": hidx,
                        "decoder_layer_index": hidx-1,
                        "common_suffix_tokens": k,
                        "projection_recovery": projection_recovery(
                            r, d, p
                        ),
                        "distance_recovery": distance_recovery(
                            r, d, p
                        ),
                    })

        print(f"[{count:03d}/{len(by_task):03d}] task {ti}")
        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    behavior = pd.DataFrame(behavior_rows)
    pathway = pd.DataFrame(pathway_rows)

    behavior.to_csv(
        out_dir / "behavior_results.csv", index=False
    )
    pathway.to_csv(
        out_dir / "pathway_recovery.csv", index=False
    )

    # -------------------------------
    # Behavior aggregation by task.
    # -------------------------------
    task_effect = (
        behavior.groupby(
            ["config", "donor_type", "task_idx", "task_id"],
            as_index=False,
        )["signed_transfer_effect"]
        .mean()
    )

    config_rows = []
    for (cfg, donor_type), g in task_effect.groupby(
        ["config", "donor_type"]
    ):
        seed = args.seed + sum(ord(x) for x in (cfg + donor_type))
        mean, lo, hi = bootstrap_ci(
            g["signed_transfer_effect"], seed
        )
        config_rows.append({
            "config": cfg,
            "donor_type": donor_type,
            "mean_signed_transfer": mean,
            "bootstrap_ci_low": lo,
            "bootstrap_ci_high": hi,
            "num_tasks": int(g["task_idx"].nunique()),
        })

    config_effects = pd.DataFrame(config_rows)
    config_effects.to_csv(
        out_dir / "config_effects.csv", index=False
    )

    # -------------------------------
    # Pathway recovery aggregation.
    # -------------------------------
    pathway_task = (
        pathway.groupby(
            ["component", "hidden_state_index", "task_idx"],
            as_index=False,
        )[["projection_recovery", "distance_recovery"]]
        .mean()
    )

    layer_rows = []
    for (component, hidx), g in pathway_task.groupby(
        ["component", "hidden_state_index"]
    ):
        pm, plo, phi = bootstrap_ci(
            g["projection_recovery"],
            args.seed + hidx + sum(ord(c) for c in component),
        )
        dm, dlo, dhi = bootstrap_ci(
            g["distance_recovery"],
            args.seed + 1000 + hidx + sum(ord(c) for c in component),
        )
        layer_rows.append({
            "component": component,
            "hidden_state_index": int(hidx),
            "decoder_layer_index": int(hidx-1),
            "projection_recovery_mean": pm,
            "projection_recovery_ci_low": plo,
            "projection_recovery_ci_high": phi,
            "distance_recovery_mean": dm,
            "distance_recovery_ci_low": dlo,
            "distance_recovery_ci_high": dhi,
        })

    pathway_by_layer = pd.DataFrame(layer_rows)
    pathway_by_layer.to_csv(
        out_dir / "pathway_recovery_by_layer.csv", index=False
    )

    # Overall pathway summaries by component.
    component_task = (
        pathway.groupby(
            ["component", "task_idx"],
            as_index=False,
        )[["projection_recovery", "distance_recovery"]]
        .mean()
    )

    component_summary = []
    for component, g in component_task.groupby("component"):
        pm, plo, phi = bootstrap_ci(
            g["projection_recovery"],
            args.seed + 2000 + sum(ord(c) for c in component),
        )
        dm, dlo, dhi = bootstrap_ci(
            g["distance_recovery"],
            args.seed + 3000 + sum(ord(c) for c in component),
        )
        component_summary.append({
            "component": component,
            "projection_recovery_mean": pm,
            "projection_recovery_95ci": [plo, phi],
            "distance_recovery_mean": dm,
            "distance_recovery_95ci": [dlo, dhi],
        })

    def get_effect(config, donor_type="opposite_same_wording"):
        return config_effects[
            (config_effects["config"] == config) &
            (config_effects["donor_type"] == donor_type)
        ].iloc[0]

    early = get_effect("early_H15_H20")
    late = get_effect("late_H21_H28")
    full = get_effect("full_H15_H28")
    selfrow = get_effect("early_H15_H20", "self")
    same = get_effect(
        "early_H15_H20", "same_state_cross_wording"
    )
    cross = get_effect(
        "early_H15_H20", "opposite_cross_wording"
    )

    selector_fraction = (
        float(early["mean_signed_transfer"])
        / float(full["mean_signed_transfer"])
        if abs(float(full["mean_signed_transfer"])) > 1e-12
        else np.nan
    )

    # Correlate per-task downstream pathway recovery with early behavior effect.
    early_task = task_effect[
        (task_effect["config"] == "early_H15_H20") &
        (task_effect["donor_type"] == "opposite_same_wording")
    ][["task_idx", "signed_transfer_effect"]]

    pathway_component_task = (
        pathway.groupby(
            ["component", "task_idx"], as_index=False
        )["projection_recovery"].mean()
    )

    recovery_behavior_corr = {}
    for component in ["attn", "mlp", "residual"]:
        g = pathway_component_task[
            pathway_component_task["component"] == component
        ].merge(early_task, on="task_idx", how="inner")

        if len(g) > 2 and g["projection_recovery"].std() > 0 and g["signed_transfer_effect"].std() > 0:
            r = float(np.corrcoef(
                g["projection_recovery"],
                g["signed_transfer_effect"],
            )[0, 1])
        else:
            r = np.nan
        recovery_behavior_corr[component] = r

    # -------------------------------
    # Plots.
    # -------------------------------
    fig = plt.figure(figsize=(8, 5.2))
    plot_order = [
        "early_H15_H20",
        "late_H21_H28",
        "full_H15_H28",
    ]
    vals = [
        float(get_effect(x)["mean_signed_transfer"])
        for x in plot_order
    ]
    plt.bar(plot_order, vals)
    plt.axhline(0.0, linestyle="--")
    plt.ylabel("Signed transfer effect")
    plt.title("EXP05 selector-window behavior")
    plt.xticks(rotation=20)
    plt.tight_layout()
    fig.savefig(out_dir / "selector_behavior.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(9, 5.5))
    for component in ["attn", "mlp", "residual"]:
        d = pathway_by_layer[
            pathway_by_layer["component"] == component
        ].sort_values("hidden_state_index")
        plt.plot(
            d["hidden_state_index"],
            d["projection_recovery_mean"],
            marker="o",
            label=component,
        )
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Hidden-state index")
    plt.ylabel("Projection recovery toward donor")
    plt.title("Downstream pathway reconfiguration after H15-H20 restoration")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "pathway_recovery.png", dpi=180)
    plt.close(fig)

    manifest = {
        "experiment": "EXP05_selector_pathway",
        "git_commit": git_commit(),
        "source_exp01b_git_commit": manifest01b.get("git_commit"),
        "seed": args.seed,
        "model_dir": str(model_dir),
        "model_name": model_dir.name,
        "early_hidden_indices": EARLY_HIDDEN,
        "late_hidden_indices": LATE_HIDDEN,
        "full_hidden_indices": FULL_HIDDEN,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "exp01b_script_sha256": sha256_file(EXP01B_CODE),
        "model_config_sha256": sha256_file(model_dir / "config.json"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "offline_only": True,
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "experiment": "EXP05_selector_pathway",
        "hypothesis": (
            "H15-H20 distributed residual state acts as a selector that "
            "reconfigures unpatched H21-H28 pathways and shifts action preference."
        ),
        "early_H15_H20_mean": float(early["mean_signed_transfer"]),
        "early_H15_H20_95ci": [
            float(early["bootstrap_ci_low"]),
            float(early["bootstrap_ci_high"]),
        ],
        "late_H21_H28_mean": float(late["mean_signed_transfer"]),
        "late_H21_H28_95ci": [
            float(late["bootstrap_ci_low"]),
            float(late["bootstrap_ci_high"]),
        ],
        "full_H15_H28_mean": float(full["mean_signed_transfer"]),
        "full_H15_H28_95ci": [
            float(full["bootstrap_ci_low"]),
            float(full["bootstrap_ci_high"]),
        ],
        "selector_fraction_of_full_effect": selector_fraction,
        "self_control_mean": float(selfrow["mean_signed_transfer"]),
        "same_state_cross_wording_control_mean": float(
            same["mean_signed_transfer"]
        ),
        "opposite_state_cross_wording_mean": float(
            cross["mean_signed_transfer"]
        ),
        "pathway_component_summary": component_summary,
        "pathway_recovery_behavior_pearson_r": recovery_behavior_corr,
        "decision_rule": (
            "Positive early_H15_H20 CI, late_H21_H28 near zero, and positive "
            "unpatched downstream pathway recovery support a selector-like "
            "mechanism. Component asymmetry motivates head/MLP-level EXP06."
        ),
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== EXP05 SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
