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
from sklearn.model_selection import GroupKFold
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = REPO_ROOT / "models"
EXP01B_CODE = REPO_ROOT / "experiments" / "exp01b_counterbalanced_next_state" / "run.py"
EXP01B_MANIFEST = REPO_ROOT / "outputs" / "exp01b_counterbalanced_next_state" / "run_manifest.json"
DEFAULT_OUT = REPO_ROOT / "outputs" / "exp06_mlp_neuron_groups"

LABEL_TEST = 0
LABEL_IMPL = 1

EARLY_HIDDEN = list(range(15, 21))   # H15..H20
MLP_HIDDEN = list(range(21, 29))     # H21..H28

PRIMARY_K = 256
EXPLORATORY_K = [64, 1024]
RANDOM_CONTROLS = 5


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


def get_attr_any(obj, names):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"None of these names exist: {names}")


def exp01b_api(mod):
    return {
        "make_tasks": get_attr_any(mod, ["make_tasks"]),
        "history": get_attr_any(mod, ["build_history", "history"]),
        "messages": get_attr_any(mod, ["build_messages", "messages"]),
        "conditions": get_attr_any(mod, ["CONDITIONS", "CONDS"]),
        "skills": get_attr_any(mod, ["SKILLS"]),
    }


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("Expected Qwen-style model.model.layers")


def render_prompt(tok, messages):
    return tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def recreate_entries(mod, manifest):
    api = exp01b_api(mod)
    tasks = api["make_tasks"](int(manifest["num_tasks"]), int(manifest["seed"]))
    out = {}

    for task in tasks:
        ti = int(task["task_idx"])
        hist = api["history"](task, ti)
        entries = []

        for cond, wording, label in api["conditions"]:
            entries.append({
                "task_idx": ti,
                "task_id": task["task_id"],
                "condition": cond,
                "wording": wording,
                "label": int(label),
                "messages": api["messages"](
                    task, hist, api["skills"][cond]
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
            "opposite_cross_wording": lookup[(opp_wording, opp_label)],
        }
    return maps


def longest_common_suffix(a, b):
    k = 0
    n = min(len(a), len(b))
    while k < n and a[-1-k] == b[-1-k]:
        k += 1
    return k


def first_tensor(output):
    return output[0] if isinstance(output, tuple) else output


@contextmanager
def residual_patch_hook(layer, positions, replacements):
    def hook_fn(module, inputs, output):
        if isinstance(output, tuple):
            h, rest = output[0], output[1:]
        else:
            h, rest = output, None

        h = h.clone()
        b = torch.arange(h.shape[0], device=h.device).unsqueeze(1)
        h[b, positions.to(h.device), :] = replacements.to(h.device, dtype=h.dtype)
        return h if rest is None else (h,) + rest

    handle = layer.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def mlp_neuron_patch_hook(down_proj, positions, neuron_idx, replacements):
    """Edit input z to down_proj.

    positions:    [batch, K]
    neuron_idx:   1-D LongTensor, or None for all neurons
    replacements:
      if neuron_idx is None: [batch, K, intermediate]
      else:                  [batch, K, len(neuron_idx)]
    """
    def pre_hook(module, inputs):
        z = inputs[0].clone()
        b = torch.arange(z.shape[0], device=z.device).unsqueeze(1)
        pos = positions.to(z.device)
        rep = replacements.to(z.device, dtype=z.dtype)

        if neuron_idx is None:
            z[b, pos, :] = rep
        else:
            idx = neuron_idx.to(z.device)
            for row in range(z.shape[0]):
                z[row].index_put_(
                    (pos[row][:, None], idx[None, :]),
                    rep[row],
                )

        return (z,) + tuple(inputs[1:])

    handle = down_proj.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


def tokenize_entry(tok, entry):
    text = render_prompt(tok, entry["messages"])
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return {"text": text, "ids": ids}


@torch.inference_mode()
def capture_prompt_states(
    model,
    tok,
    layers,
    entry,
    max_suffix,
    capture_early=True,
    capture_mlp=True,
    early_donor_cache=None,
    early_common_k=None,
):
    """Capture pre-norm decoder outputs H15-H20 and MLP z at H21-H28.

    If early_donor_cache is supplied, restore H15-H20 before capturing the
    downstream MLP neuron activations.
    """
    device = model.get_input_embeddings().weight.device
    text = render_prompt(tok, entry["messages"])
    enc = tok(text, add_special_tokens=False, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    plen = len(ids)
    enc = {k: v.to(device) for k, v in enc.items()}

    store = {"early": {}, "mlp": {}}
    handles = []
    stack = ExitStack()

    if early_donor_cache is not None:
        k = int(early_common_k)
        pos_np = np.arange(plen-k, plen, dtype=np.int64)
        positions = torch.tensor(
            pos_np[None, :],
            dtype=torch.long,
            device=device,
        )

        for hidx in EARLY_HIDDEN:
            donor_arr = early_donor_cache["early"][hidx][-k:, :].astype(np.float32)
            rep = torch.tensor(
                donor_arr[None, :, :],
                dtype=torch.float32,
                device=device,
            )
            stack.enter_context(
                residual_patch_hook(layers[hidx-1], positions, rep)
            )

    def make_early_hook(hidx):
        def fn(module, inputs, output):
            t = first_tensor(output)
            take = min(max_suffix, t.shape[1])
            store["early"][hidx] = (
                t[0, -take:, :]
                .detach().float().cpu().numpy().astype(np.float32)
            )
        return fn

    def make_mlp_pre_hook(hidx):
        def fn(module, inputs):
            z = inputs[0]
            take = min(max_suffix, z.shape[1])
            store["mlp"][hidx] = (
                z[0, -take:, :]
                .detach().float().cpu().numpy().astype(np.float32)
            )
        return fn

    if capture_early:
        for hidx in EARLY_HIDDEN:
            handles.append(
                layers[hidx-1].register_forward_hook(
                    make_early_hook(hidx)
                )
            )

    if capture_mlp:
        for hidx in MLP_HIDDEN:
            handles.append(
                layers[hidx-1].mlp.down_proj.register_forward_pre_hook(
                    make_mlp_pre_hook(hidx)
                )
            )

    try:
        with stack:
            model(**enc, use_cache=False, return_dict=True)
    finally:
        for h in handles:
            h.remove()

    return {"ids": ids, **store}


def build_task_feature_stats(model, tok, layers, entries):
    """Return per-task neuron recovery statistics [8, intermediate]."""
    maps = donor_maps(entries)
    token_info = [tokenize_entry(tok, e) for e in entries]

    max_suffix = []
    for i in range(len(entries)):
        d = maps[i]["opposite_same_wording"]
        max_suffix.append(
            longest_common_suffix(token_info[i]["ids"], token_info[d]["ids"])
        )

    baseline = [
        capture_prompt_states(
            model, tok, layers, entries[i], max_suffix[i],
            capture_early=True, capture_mlp=True
        )
        for i in range(len(entries))
    ]

    num = None
    energy = None
    counts = np.zeros(len(MLP_HIDDEN), dtype=np.int64)

    for i, rec in enumerate(entries):
        di = maps[i]["opposite_same_wording"]
        k = longest_common_suffix(
            baseline[i]["ids"], baseline[di]["ids"]
        )

        patched = capture_prompt_states(
            model, tok, layers, rec, max_suffix=k,
            capture_early=False,
            capture_mlp=True,
            early_donor_cache=baseline[di],
            early_common_k=k,
        )

        if num is None:
            intermediate = baseline[i]["mlp"][MLP_HIDDEN[0]].shape[-1]
            num = np.zeros((len(MLP_HIDDEN), intermediate), dtype=np.float64)
            energy = np.zeros_like(num)

        for li, hidx in enumerate(MLP_HIDDEN):
            r = baseline[i]["mlp"][hidx][-k:, :].astype(np.float64)
            d = baseline[di]["mlp"][hidx][-k:, :].astype(np.float64)
            p = patched["mlp"][hidx][-k:, :].astype(np.float64)

            dr = d - r
            pr = p - r
            num[li] += np.sum(pr * dr, axis=0)
            energy[li] += np.sum(dr * dr, axis=0)
            counts[li] += k

    aligned_mean = num / counts[:, None]
    energy_mean = energy / counts[:, None]
    return aligned_mean.astype(np.float32), energy_mean.astype(np.float32)


def downproj_norms(layers):
    rows = []
    for hidx in MLP_HIDDEN:
        w = (
            layers[hidx-1].mlp.down_proj.weight
            .detach().float().cpu().numpy()
        )
        # weight: [hidden, intermediate]
        rows.append(np.linalg.norm(w, axis=0).astype(np.float32))
    return np.stack(rows)


def select_neurons(train_aligned, train_energy, dp_norms, k):
    aligned = train_aligned.mean(axis=0)
    energy = train_energy.mean(axis=0)
    recovery = aligned / (energy + 1e-12)
    impact = (
        np.maximum(recovery, 0.0)
        * np.sqrt(np.maximum(energy, 0.0))
        * dp_norms
    )

    flat = impact.reshape(-1)
    k_eff = min(k, flat.size)
    idx_flat = np.argpartition(flat, -k_eff)[-k_eff:]
    idx_flat = idx_flat[np.argsort(flat[idx_flat])[::-1]]

    selected = []
    n_neurons = impact.shape[1]
    for rank, flat_idx in enumerate(idx_flat, 1):
        li = int(flat_idx // n_neurons)
        neuron = int(flat_idx % n_neurons)
        selected.append({
            "rank": rank,
            "hidden_state_index": int(MLP_HIDDEN[li]),
            "neuron_index": neuron,
            "impact_score": float(impact[li, neuron]),
            "recovery_ratio": float(recovery[li, neuron]),
            "aligned_mean": float(aligned[li, neuron]),
            "donor_energy_mean": float(energy[li, neuron]),
            "downproj_col_norm": float(dp_norms[li, neuron]),
        })
    return selected


def selected_by_layer(selected):
    out = {}
    for row in selected:
        out.setdefault(row["hidden_state_index"], []).append(row["neuron_index"])
    return {h: np.array(v, dtype=np.int64) for h, v in out.items()}


def random_matched_group(selected, intermediate_size, seed):
    rng = np.random.default_rng(seed)
    by_layer = selected_by_layer(selected)
    out = {}

    for hidx, chosen in by_layer.items():
        mask = np.ones(intermediate_size, dtype=bool)
        mask[chosen] = False
        pool = np.where(mask)[0]
        n = len(chosen)
        out[hidx] = rng.choice(pool, size=n, replace=False).astype(np.int64)

    return out


def selected_dict_to_rows(group):
    rows = []
    for hidx, arr in group.items():
        for neuron in arr:
            rows.append((int(hidx), int(neuron)))
    return rows


def prepare_task_cache(model, tok, layers, entries):
    maps = donor_maps(entries)
    token_info = [tokenize_entry(tok, e) for e in entries]
    max_suffix = []

    for i in range(len(entries)):
        ks = []
        for key in ["opposite_same_wording", "opposite_cross_wording"]:
            di = maps[i][key]
            ks.append(
                longest_common_suffix(
                    token_info[i]["ids"], token_info[di]["ids"]
                )
            )
        max_suffix.append(max(ks))

    cache = [
        capture_prompt_states(
            model, tok, layers, entries[i], max_suffix[i],
            capture_early=True, capture_mlp=True
        )
        for i in range(len(entries))
    ]
    return cache, maps


def get_slice(cache, kind, hidx, k):
    return cache[kind][hidx][-k:, :].astype(np.float32)


@torch.inference_mode()
def score_intervention(
    model,
    tok,
    layers,
    recipient,
    rec_cache,
    donor_cache=None,
    common_k=None,
    early_patch=False,
    mlp_group=None,
    mlp_source=None,  # "donor" or "recipient"
    all_mlp=False,
):
    """Score two candidates one-by-one.

    mlp_source:
      donor     -> sufficiency patch
      recipient -> necessity clamp-back
    """
    device = model.get_input_embeddings().weight.device
    prompt_text = render_prompt(tok, recipient["messages"])
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    plen = len(prompt_ids)

    scores = []

    for candidate in recipient["candidates"]:
        cids = tok(candidate, add_special_tokens=False)["input_ids"]
        ids = prompt_ids + cids
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention = torch.ones_like(input_ids)

        with ExitStack() as stack:
            if donor_cache is not None and common_k is not None:
                k = int(common_k)
                pos_np = np.arange(plen-k, plen, dtype=np.int64)
                positions = torch.tensor(
                    pos_np[None, :],
                    dtype=torch.long,
                    device=device,
                )

                if early_patch:
                    for hidx in EARLY_HIDDEN:
                        d = get_slice(donor_cache, "early", hidx, k)
                        rep = torch.tensor(
                            d[None, :, :],
                            dtype=torch.float32,
                            device=device,
                        )
                        stack.enter_context(
                            residual_patch_hook(
                                layers[hidx-1], positions, rep
                            )
                        )

                if all_mlp or mlp_group:
                    for hidx in MLP_HIDDEN:
                        if all_mlp:
                            idx = None
                        else:
                            if hidx not in mlp_group:
                                continue
                            idx_np = mlp_group[hidx]
                            idx = torch.tensor(
                                idx_np,
                                dtype=torch.long,
                                device=device,
                            )

                        source_cache = (
                            donor_cache if mlp_source == "donor" else rec_cache
                        )
                        full = get_slice(
                            source_cache, "mlp", hidx, k
                        )

                        if idx is None:
                            values = full
                        else:
                            values = full[:, idx_np]

                        rep = torch.tensor(
                            values[None, :, :],
                            dtype=torch.float32,
                            device=device,
                        )

                        stack.enter_context(
                            mlp_neuron_patch_hook(
                                layers[hidx-1].mlp.down_proj,
                                positions,
                                idx,
                                rep,
                            )
                        )

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


def bootstrap_ci(series, seed, n_boot=5000):
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    boots = np.array([
        rng.choice(x, size=len(x), replace=True).mean()
        for _ in range(n_boot)
    ])
    return (
        float(x.mean()),
        float(np.quantile(boots, 0.025)),
        float(np.quantile(boots, 0.975)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=4606)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--random-controls", type=int, default=RANDOM_CONTROLS)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    mod = load_exp01b()
    manifest01b = load_manifest()
    by_task = recreate_entries(mod, manifest01b)

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

    intermediate_size = layers[MLP_HIDDEN[0]-1].mlp.down_proj.in_features
    dp_norm = downproj_norms(layers)

    task_ids = sorted(by_task)
    task_to_pos = {t: i for i, t in enumerate(task_ids)}

    print("="*82)
    print("EXP06: MLP Neuron-Group Causal Mediation")
    print("Model:", model_dir)
    print("Tasks:", len(task_ids))
    print("Intermediate size:", intermediate_size)
    print("Primary K:", PRIMARY_K)
    print("="*82)

    # ------------------------------------------------------------
    # Phase 1: per-task feature sufficient statistics
    # ------------------------------------------------------------
    aligned_tasks = np.zeros(
        (len(task_ids), len(MLP_HIDDEN), intermediate_size),
        dtype=np.float32,
    )
    energy_tasks = np.zeros_like(aligned_tasks)

    for n, ti in enumerate(task_ids, 1):
        aligned, energy = build_task_feature_stats(
            model, tok, layers, by_task[ti]
        )
        aligned_tasks[task_to_pos[ti]] = aligned
        energy_tasks[task_to_pos[ti]] = energy

        print(f"[feature {n:03d}/{len(task_ids):03d}] task {ti}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    np.savez_compressed(
        out_dir / "feature_task_stats.npz",
        task_ids=np.asarray(task_ids, dtype=np.int64),
        hidden_indices=np.asarray(MLP_HIDDEN, dtype=np.int64),
        aligned_mean=aligned_tasks,
        donor_energy_mean=energy_tasks,
        downproj_norm=dp_norm,
    )

    # ------------------------------------------------------------
    # Phase 2: 4-fold held-out causal tests
    # ------------------------------------------------------------
    unique = np.asarray(task_ids, dtype=np.int64)
    gkf = GroupKFold(n_splits=4)
    dummy_x = np.zeros((len(unique), 1))
    dummy_y = np.zeros(len(unique))

    selected_rows = []
    result_rows = []

    ks = [PRIMARY_K] + EXPLORATORY_K

    for fold, (tr_pos, te_pos) in enumerate(
        gkf.split(dummy_x, dummy_y, unique)
    ):
        train_tasks = unique[tr_pos]
        test_tasks = unique[te_pos]

        train_stat_idx = [task_to_pos[int(t)] for t in train_tasks]
        train_aligned = aligned_tasks[train_stat_idx]
        train_energy = energy_tasks[train_stat_idx]

        selected_sets = {}
        for k in ks:
            sel = select_neurons(
                train_aligned, train_energy, dp_norm, k
            )
            selected_sets[k] = sel
            for row in sel:
                selected_rows.append({
                    "fold": fold,
                    "group": f"top{k}",
                    **row,
                })

        primary_group = selected_by_layer(selected_sets[PRIMARY_K])

        random_groups = [
            random_matched_group(
                selected_sets[PRIMARY_K],
                intermediate_size,
                seed=args.seed + fold*1000 + rid,
            )
            for rid in range(args.random_controls)
        ]

        for task_n, ti in enumerate(test_tasks, 1):
            ti = int(ti)
            entries = by_task[ti]
            cache, maps = prepare_task_cache(
                model, tok, layers, entries
            )

            for relation in [
                "opposite_same_wording",
                "opposite_cross_wording",
            ]:
                # Cross-wording is only required for primary K.
                for i, rec in enumerate(entries):
                    di = maps[i][relation]
                    donor = entries[di]
                    k_common = longest_common_suffix(
                        cache[i]["ids"], cache[di]["ids"]
                    )
                    donor_sign = (
                        1.0 if donor["label"] == LABEL_IMPL else -1.0
                    )

                    baseline = score_intervention(
                        model, tok, layers,
                        rec, cache[i]
                    )

                    early = score_intervention(
                        model, tok, layers,
                        rec, cache[i],
                        donor_cache=cache[di],
                        common_k=k_common,
                        early_patch=True,
                    )
                    early_effect = donor_sign * (
                        early["margin"] - baseline["margin"]
                    )

                    # Only same-wording runs the complete intervention suite.
                    if relation == "opposite_same_wording":
                        # Full MLP upper bounds.
                        all_suff = score_intervention(
                            model, tok, layers,
                            rec, cache[i],
                            donor_cache=cache[di],
                            common_k=k_common,
                            mlp_source="donor",
                            all_mlp=True,
                        )
                        all_nec = score_intervention(
                            model, tok, layers,
                            rec, cache[i],
                            donor_cache=cache[di],
                            common_k=k_common,
                            early_patch=True,
                            mlp_source="recipient",
                            all_mlp=True,
                        )

                        all_suff_effect = donor_sign * (
                            all_suff["margin"] - baseline["margin"]
                        )
                        all_retained = donor_sign * (
                            all_nec["margin"] - baseline["margin"]
                        )
                        all_nec_loss = early_effect - all_retained

                        result_rows.extend([
                            {
                                "fold": fold,
                                "task_idx": ti,
                                "task_id": rec["task_id"],
                                "recipient_condition": rec["condition"],
                                "recipient_wording": rec["wording"],
                                "relation": relation,
                                "config": "all_mlp_sufficiency",
                                "group_size": intermediate_size * len(MLP_HIDDEN),
                                "control_id": -1,
                                "value": all_suff_effect,
                                "early_effect": early_effect,
                            },
                            {
                                "fold": fold,
                                "task_idx": ti,
                                "task_id": rec["task_id"],
                                "recipient_condition": rec["condition"],
                                "recipient_wording": rec["wording"],
                                "relation": relation,
                                "config": "all_mlp_necessity_loss",
                                "group_size": intermediate_size * len(MLP_HIDDEN),
                                "control_id": -1,
                                "value": all_nec_loss,
                                "early_effect": early_effect,
                            },
                            {
                                "fold": fold,
                                "task_idx": ti,
                                "task_id": rec["task_id"],
                                "recipient_condition": rec["condition"],
                                "recipient_wording": rec["wording"],
                                "relation": relation,
                                "config": "early_selector_effect",
                                "group_size": 0,
                                "control_id": -1,
                                "value": early_effect,
                                "early_effect": early_effect,
                            },
                        ])

                        # Selected groups across K.
                        for k_group in ks:
                            group = selected_by_layer(
                                selected_sets[k_group]
                            )
                            suff = score_intervention(
                                model, tok, layers,
                                rec, cache[i],
                                donor_cache=cache[di],
                                common_k=k_common,
                                mlp_group=group,
                                mlp_source="donor",
                            )
                            nec = score_intervention(
                                model, tok, layers,
                                rec, cache[i],
                                donor_cache=cache[di],
                                common_k=k_common,
                                early_patch=True,
                                mlp_group=group,
                                mlp_source="recipient",
                            )

                            suff_effect = donor_sign * (
                                suff["margin"] - baseline["margin"]
                            )
                            retained = donor_sign * (
                                nec["margin"] - baseline["margin"]
                            )
                            nec_loss = early_effect - retained

                            result_rows.extend([
                                {
                                    "fold": fold,
                                    "task_idx": ti,
                                    "task_id": rec["task_id"],
                                    "recipient_condition": rec["condition"],
                                    "recipient_wording": rec["wording"],
                                    "relation": relation,
                                    "config": f"top{k_group}_sufficiency",
                                    "group_size": k_group,
                                    "control_id": -1,
                                    "value": suff_effect,
                                    "early_effect": early_effect,
                                },
                                {
                                    "fold": fold,
                                    "task_idx": ti,
                                    "task_id": rec["task_id"],
                                    "recipient_condition": rec["condition"],
                                    "recipient_wording": rec["wording"],
                                    "relation": relation,
                                    "config": f"top{k_group}_necessity_loss",
                                    "group_size": k_group,
                                    "control_id": -1,
                                    "value": nec_loss,
                                    "early_effect": early_effect,
                                },
                            ])

                        # Random controls only for primary K.
                        for rid, group in enumerate(random_groups):
                            suff = score_intervention(
                                model, tok, layers,
                                rec, cache[i],
                                donor_cache=cache[di],
                                common_k=k_common,
                                mlp_group=group,
                                mlp_source="donor",
                            )
                            nec = score_intervention(
                                model, tok, layers,
                                rec, cache[i],
                                donor_cache=cache[di],
                                common_k=k_common,
                                early_patch=True,
                                mlp_group=group,
                                mlp_source="recipient",
                            )

                            suff_effect = donor_sign * (
                                suff["margin"] - baseline["margin"]
                            )
                            retained = donor_sign * (
                                nec["margin"] - baseline["margin"]
                            )
                            nec_loss = early_effect - retained

                            result_rows.extend([
                                {
                                    "fold": fold,
                                    "task_idx": ti,
                                    "task_id": rec["task_id"],
                                    "recipient_condition": rec["condition"],
                                    "recipient_wording": rec["wording"],
                                    "relation": relation,
                                    "config": "random256_sufficiency",
                                    "group_size": PRIMARY_K,
                                    "control_id": rid,
                                    "value": suff_effect,
                                    "early_effect": early_effect,
                                },
                                {
                                    "fold": fold,
                                    "task_idx": ti,
                                    "task_id": rec["task_id"],
                                    "recipient_condition": rec["condition"],
                                    "recipient_wording": rec["wording"],
                                    "relation": relation,
                                    "config": "random256_necessity_loss",
                                    "group_size": PRIMARY_K,
                                    "control_id": rid,
                                    "value": nec_loss,
                                    "early_effect": early_effect,
                                },
                            ])

                    else:
                        # Cross-wording validation for primary selected group.
                        suff = score_intervention(
                            model, tok, layers,
                            rec, cache[i],
                            donor_cache=cache[di],
                            common_k=k_common,
                            mlp_group=primary_group,
                            mlp_source="donor",
                        )
                        nec = score_intervention(
                            model, tok, layers,
                            rec, cache[i],
                            donor_cache=cache[di],
                            common_k=k_common,
                            early_patch=True,
                            mlp_group=primary_group,
                            mlp_source="recipient",
                        )

                        suff_effect = donor_sign * (
                            suff["margin"] - baseline["margin"]
                        )
                        retained = donor_sign * (
                            nec["margin"] - baseline["margin"]
                        )
                        nec_loss = early_effect - retained

                        result_rows.extend([
                            {
                                "fold": fold,
                                "task_idx": ti,
                                "task_id": rec["task_id"],
                                "recipient_condition": rec["condition"],
                                "recipient_wording": rec["wording"],
                                "relation": relation,
                                "config": "top256_sufficiency_cross_wording",
                                "group_size": PRIMARY_K,
                                "control_id": -1,
                                "value": suff_effect,
                                "early_effect": early_effect,
                            },
                            {
                                "fold": fold,
                                "task_idx": ti,
                                "task_id": rec["task_id"],
                                "recipient_condition": rec["condition"],
                                "recipient_wording": rec["wording"],
                                "relation": relation,
                                "config": "top256_necessity_loss_cross_wording",
                                "group_size": PRIMARY_K,
                                "control_id": -1,
                                "value": nec_loss,
                                "early_effect": early_effect,
                            },
                        ])

            print(
                f"[fold {fold+1}/4 task {task_n:02d}/{len(test_tasks):02d}] {ti}"
            )
            del cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(out_dir / "selected_neurons.csv", index=False)

    results = pd.DataFrame(result_rows)
    results.to_csv(out_dir / "intervention_results.csv", index=False)

    # ------------------------------------------------------------
    # Task-level inference
    # ------------------------------------------------------------
    task_level = (
        results.groupby(
            ["config", "relation", "control_id", "task_idx", "task_id"],
            as_index=False,
        )["value"].mean()
    )

    summary_rows = []
    for (config, relation, control_id), g in task_level.groupby(
        ["config", "relation", "control_id"]
    ):
        seed = args.seed + sum(ord(c) for c in config+relation) + int(control_id+1)*17
        mean, lo, hi = bootstrap_ci(g["value"], seed)
        summary_rows.append({
            "config": config,
            "relation": relation,
            "control_id": int(control_id),
            "mean": mean,
            "ci_low": lo,
            "ci_high": hi,
            "num_tasks": int(g["task_idx"].nunique()),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary_by_config.csv", index=False)

    def cfg_mean(name, relation="opposite_same_wording", control_id=-1):
        row = summary_df[
            (summary_df["config"] == name)
            & (summary_df["relation"] == relation)
            & (summary_df["control_id"] == control_id)
        ].iloc[0]
        return row

    top256_suff = cfg_mean("top256_sufficiency")
    top256_nec = cfg_mean("top256_necessity_loss")
    all_suff = cfg_mean("all_mlp_sufficiency")
    all_nec = cfg_mean("all_mlp_necessity_loss")
    early_row = cfg_mean("early_selector_effect")

    # Random controls: average random effect per task, then paired bootstrap.
    random_task = (
        task_level[
            (task_level["config"].isin([
                "random256_sufficiency",
                "random256_necessity_loss",
            ]))
            & (task_level["relation"] == "opposite_same_wording")
        ]
        .groupby(["config", "task_idx"], as_index=False)["value"]
        .mean()
    )

    random_suff_task = random_task[
        random_task["config"] == "random256_sufficiency"
    ][["task_idx", "value"]].rename(columns={"value": "random_value"})

    random_nec_task = random_task[
        random_task["config"] == "random256_necessity_loss"
    ][["task_idx", "value"]].rename(columns={"value": "random_value"})

    selected_suff_task = task_level[
        (task_level["config"] == "top256_sufficiency")
        & (task_level["relation"] == "opposite_same_wording")
        & (task_level["control_id"] == -1)
    ][["task_idx", "value"]].rename(columns={"value": "selected_value"})

    selected_nec_task = task_level[
        (task_level["config"] == "top256_necessity_loss")
        & (task_level["relation"] == "opposite_same_wording")
        & (task_level["control_id"] == -1)
    ][["task_idx", "value"]].rename(columns={"value": "selected_value"})

    suff_diff = selected_suff_task.merge(
        random_suff_task, on="task_idx"
    )
    suff_diff["diff"] = (
        suff_diff["selected_value"] - suff_diff["random_value"]
    )
    suff_diff_mean, suff_diff_lo, suff_diff_hi = bootstrap_ci(
        suff_diff["diff"], args.seed + 9001
    )

    nec_diff = selected_nec_task.merge(
        random_nec_task, on="task_idx"
    )
    nec_diff["diff"] = (
        nec_diff["selected_value"] - nec_diff["random_value"]
    )
    nec_diff_mean, nec_diff_lo, nec_diff_hi = bootstrap_ci(
        nec_diff["diff"], args.seed + 9002
    )

    # K dose table.
    dose_rows = []
    for k in ks:
        suff = cfg_mean(f"top{k}_sufficiency")
        nec = cfg_mean(f"top{k}_necessity_loss")
        dose_rows.append({
            "k": k,
            "sufficiency_mean": float(suff["mean"]),
            "sufficiency_ci_low": float(suff["ci_low"]),
            "sufficiency_ci_high": float(suff["ci_high"]),
            "necessity_loss_mean": float(nec["mean"]),
            "necessity_ci_low": float(nec["ci_low"]),
            "necessity_ci_high": float(nec["ci_high"]),
        })

    dose_df = pd.DataFrame(dose_rows)
    dose_df.to_csv(out_dir / "topk_dose_response.csv", index=False)

    # Layer counts for primary selected set.
    layer_counts = (
        selected_df[selected_df["group"] == f"top{PRIMARY_K}"]
        .groupby(["fold", "hidden_state_index"])
        .size()
        .reset_index(name="count")
    )

    # Cross-wording primary.
    cross_suff = cfg_mean(
        "top256_sufficiency_cross_wording",
        relation="opposite_cross_wording",
    )
    cross_nec = cfg_mean(
        "top256_necessity_loss_cross_wording",
        relation="opposite_cross_wording",
    )

    # Plots.
    fig = plt.figure(figsize=(8.5, 5.5))
    plt.plot(
        dose_df["k"],
        dose_df["sufficiency_mean"],
        marker="o",
        label="sufficiency",
    )
    plt.plot(
        dose_df["k"],
        dose_df["necessity_loss_mean"],
        marker="o",
        label="necessity loss",
    )
    plt.axhline(0.0, linestyle="--")
    plt.xscale("log")
    plt.xlabel("Selected neuron-group size K")
    plt.ylabel("Causal effect")
    plt.title("EXP06 MLP neuron-group dose response")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "topk_dose_response.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(9, 5.5))
    pivot = layer_counts.pivot(
        index="hidden_state_index",
        columns="fold",
        values="count",
    ).fillna(0)
    pivot.mean(axis=1).plot(kind="bar")
    plt.ylabel("Mean selected neurons per fold")
    plt.xlabel("Hidden-state index")
    plt.title("Primary K=256 group layer distribution")
    plt.tight_layout()
    fig.savefig(out_dir / "selected_layer_counts.png", dpi=180)
    plt.close(fig)

    manifest = {
        "experiment": "EXP06_mlp_neuron_groups",
        "git_commit": git_commit(),
        "source_exp01b_git_commit": manifest01b.get("git_commit"),
        "seed": args.seed,
        "model_dir": str(model_dir),
        "model_name": model_dir.name,
        "early_hidden_indices": EARLY_HIDDEN,
        "mlp_hidden_indices": MLP_HIDDEN,
        "primary_k": PRIMARY_K,
        "exploratory_k": EXPLORATORY_K,
        "random_controls": args.random_controls,
        "intermediate_size": int(intermediate_size),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "exp01b_script_sha256": sha256_file(EXP01B_CODE),
        "model_config_sha256": sha256_file(model_dir / "config.json"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "offline_only": True,
    }

    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "experiment": "EXP06_mlp_neuron_groups",
        "primary_k": PRIMARY_K,
        "early_selector_effect_mean": float(early_row["mean"]),
        "top256_sufficiency_mean": float(top256_suff["mean"]),
        "top256_sufficiency_95ci": [
            float(top256_suff["ci_low"]),
            float(top256_suff["ci_high"]),
        ],
        "top256_necessity_loss_mean": float(top256_nec["mean"]),
        "top256_necessity_loss_95ci": [
            float(top256_nec["ci_low"]),
            float(top256_nec["ci_high"]),
        ],
        "top256_sufficiency_minus_matched_random_mean": suff_diff_mean,
        "top256_sufficiency_minus_matched_random_95ci": [
            suff_diff_lo, suff_diff_hi
        ],
        "top256_necessity_minus_matched_random_mean": nec_diff_mean,
        "top256_necessity_minus_matched_random_95ci": [
            nec_diff_lo, nec_diff_hi
        ],
        "all_mlp_sufficiency_mean": float(all_suff["mean"]),
        "all_mlp_necessity_loss_mean": float(all_nec["mean"]),
        "cross_wording_top256_sufficiency_mean": float(cross_suff["mean"]),
        "cross_wording_top256_sufficiency_95ci": [
            float(cross_suff["ci_low"]),
            float(cross_suff["ci_high"]),
        ],
        "cross_wording_top256_necessity_loss_mean": float(cross_nec["mean"]),
        "cross_wording_top256_necessity_loss_95ci": [
            float(cross_nec["ci_low"]),
            float(cross_nec["ci_high"]),
        ],
        "dose_response": dose_df.to_dict(orient="records"),
        "decision_rule": (
            "Primary neuron-group mediation is supported if K=256 is both "
            "sufficient and necessary with CI>0, exceeds matched-random "
            "groups, and generalizes in sign across wording."
        ),
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== EXP06 SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
