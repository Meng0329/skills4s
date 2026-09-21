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
EXP01B_CODE = (
    REPO_ROOT / "experiments" / "exp01b_counterbalanced_next_state" / "run.py"
)
EXP01B_MANIFEST = (
    REPO_ROOT / "outputs" / "exp01b_counterbalanced_next_state" / "run_manifest.json"
)
DEFAULT_OUT = REPO_ROOT / "outputs" / "exp08_selector_boundary"

LABEL_TEST = 0
LABEL_IMPL = 1
SELECTOR_HIDDEN = list(range(15, 21))  # H15..H20


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

    candidates = sorted(
        {p.parent.resolve() for p in MODELS_ROOT.rglob("config.json")}
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one local model under ./models, found {len(candidates)}"
        )
    return candidates[0]


def load_exp01b():
    if not EXP01B_CODE.is_file():
        raise FileNotFoundError(EXP01B_CODE)
    spec = importlib.util.spec_from_file_location("exp01b", EXP01B_CODE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_manifest():
    if not EXP01B_MANIFEST.is_file():
        raise FileNotFoundError(EXP01B_MANIFEST)
    return json.loads(EXP01B_MANIFEST.read_text(encoding="utf-8"))


def get_attr_any(obj, names):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"Missing API aliases: {names}")


def exp01b_api(mod):
    return {
        "make_tasks": get_attr_any(mod, ["make_tasks"]),
        "history": get_attr_any(mod, ["build_history", "history"]),
        "messages": get_attr_any(mod, ["build_messages", "messages"]),
        "conditions": get_attr_any(mod, ["CONDITIONS", "CONDS"]),
        "skills": get_attr_any(mod, ["SKILLS"]),
    }


def recreate_entries(mod, manifest):
    api = exp01b_api(mod)
    tasks = api["make_tasks"](
        int(manifest["num_tasks"]),
        int(manifest["seed"]),
    )
    by_task = {}

    for task in tasks:
        ti = int(task["task_idx"])
        hist = api["history"](task, ti)
        entries = []

        for cond, wording, label in api["conditions"]:
            entries.append(
                {
                    "task_idx": ti,
                    "task_id": task["task_id"],
                    "condition": cond,
                    "wording": wording,
                    "label": int(label),
                    "messages": api["messages"](
                        task,
                        hist,
                        api["skills"][cond],
                    ),
                    "candidates": [
                        f"read_file('{task['test_path']}')",
                        f"read_file('{task['src_path']}')",
                    ],
                }
            )

        by_task[ti] = entries

    return by_task


def donor_maps(entries):
    lookup = {
        (e["wording"], e["label"]): i
        for i, e in enumerate(entries)
    }
    out = {}

    for i, e in enumerate(entries):
        opposite_label = (
            LABEL_IMPL if e["label"] == LABEL_TEST else LABEL_TEST
        )
        opposite_wording = (
            "paraphrase"
            if e["wording"] == "canonical"
            else "canonical"
        )

        out[i] = {
            "opposite_same_wording": lookup[
                (e["wording"], opposite_label)
            ],
            "same_state_cross_wording": lookup[
                (opposite_wording, e["label"])
            ],
            "opposite_cross_wording": lookup[
                (opposite_wording, opposite_label)
            ],
            "self": i,
        }

    return out


def longest_common_suffix(a, b):
    k = 0
    n = min(len(a), len(b))
    while k < n and a[-1-k] == b[-1-k]:
        k += 1
    return k


def render_prompt(tok, messages):
    return tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("Expected Qwen-style model.model.layers")


def configs():
    out = []

    # Single-layer sufficiency.
    for h in SELECTOR_HIDDEN:
        out.append(
            {
                "name": f"single_H{h}",
                "hidden_indices": [h],
                "family": "single",
                "boundary": h,
            }
        )

    # Prefixes. H15 alone already exists as single_H15.
    for end in range(16, 21):
        out.append(
            {
                "name": (
                    "full_H15_H20"
                    if end == 20
                    else f"prefix_H15_H{end}"
                ),
                "hidden_indices": list(range(15, end + 1)),
                "family": "prefix",
                "boundary": end,
            }
        )

    # Suffixes. H20 alone and full H15-H20 already exist.
    for start in range(16, 20):
        out.append(
            {
                "name": f"suffix_H{start}_H20",
                "hidden_indices": list(range(start, 21)),
                "family": "suffix",
                "boundary": start,
            }
        )

    return out


CONFIGS = configs()


@contextmanager
def residual_patch_hook(layer, positions, replacements):
    def hook_fn(module, inputs, output):
        if isinstance(output, tuple):
            h, rest = output[0], output[1:]
        else:
            h, rest = output, None

        h = h.clone()
        batch = torch.arange(
            h.shape[0],
            device=h.device,
        ).unsqueeze(1)

        h[
            batch,
            positions.to(h.device),
            :,
        ] = replacements.to(
            h.device,
            dtype=h.dtype,
        )

        return h if rest is None else (h,) + rest

    handle = layer.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


@torch.inference_mode()
def capture_prompt(model, tok, layers, entry, max_suffix):
    """Capture exact decoder-output patch sites H15..H20 in float32."""
    device = model.get_input_embeddings().weight.device
    text = render_prompt(tok, entry["messages"])
    enc = tok(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    ids = enc["input_ids"][0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}

    states = {}
    handles = []

    def make_hook(hidx):
        def fn(module, inputs, output):
            t = output[0] if isinstance(output, tuple) else output
            take = min(max_suffix, t.shape[1])

            states[hidx] = (
                t[0, -take:, :]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        return fn

    for hidx in SELECTOR_HIDDEN:
        handles.append(
            layers[hidx - 1].register_forward_hook(
                make_hook(hidx)
            )
        )

    try:
        model(
            **enc,
            use_cache=False,
            return_dict=True,
        )
    finally:
        for h in handles:
            h.remove()

    return {
        "ids": ids,
        "states": states,
    }


def tokenize_entry(tok, entry):
    text = render_prompt(tok, entry["messages"])
    return tok(
        text,
        add_special_tokens=False,
    )["input_ids"]


def prepare_task_cache(model, tok, layers, entries):
    maps = donor_maps(entries)
    ids = [
        tokenize_entry(tok, e)
        for e in entries
    ]

    max_suffix = []
    for i in range(len(entries)):
        ks = []
        for relation in maps[i].values():
            ks.append(
                longest_common_suffix(
                    ids[i],
                    ids[relation],
                )
            )
        max_suffix.append(max(ks))

    cache = [
        capture_prompt(
            model,
            tok,
            layers,
            entries[i],
            max_suffix[i],
        )
        for i in range(len(entries))
    ]

    return cache, maps


def get_state_slice(cache, hidx, k):
    arr = cache["states"][hidx]
    if arr.shape[0] < k:
        raise RuntimeError(
            f"Captured {arr.shape[0]} tokens but need {k}"
        )
    return arr[-k:, :].astype(np.float32)


@torch.inference_mode()
def score(
    model,
    tok,
    layers,
    recipient,
    recipient_cache,
    donor_cache=None,
    hidden_indices=None,
    common_k=None,
):
    """Score candidates one at a time; patch only prompt common-suffix positions."""
    device = model.get_input_embeddings().weight.device

    prompt_text = render_prompt(
        tok,
        recipient["messages"],
    )
    prompt_ids = tok(
        prompt_text,
        add_special_tokens=False,
    )["input_ids"]
    prompt_len = len(prompt_ids)

    scores = []

    for candidate in recipient["candidates"]:
        cids = tok(
            candidate,
            add_special_tokens=False,
        )["input_ids"]

        ids = prompt_ids + cids
        input_ids = torch.tensor(
            [ids],
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.ones_like(input_ids)

        with ExitStack() as stack:
            if donor_cache is not None:
                k = int(common_k)
                pos_np = np.arange(
                    prompt_len - k,
                    prompt_len,
                    dtype=np.int64,
                )
                positions = torch.tensor(
                    pos_np[None, :],
                    dtype=torch.long,
                    device=device,
                )

                for hidx in hidden_indices:
                    donor_values = get_state_slice(
                        donor_cache,
                        hidx,
                        k,
                    )

                    replacement = torch.tensor(
                        donor_values[None, :, :],
                        dtype=torch.float32,
                        device=device,
                    )

                    stack.enter_context(
                        residual_patch_hook(
                            layers[hidx - 1],
                            positions,
                            replacement,
                        )
                    )

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )

        lp = torch.log_softmax(
            out.logits.float(),
            dim=-1,
        )

        c = len(cids)
        pred_pos = torch.arange(
            prompt_len - 1,
            prompt_len + c - 1,
            device=device,
        )
        targ_pos = torch.arange(
            prompt_len,
            prompt_len + c,
            device=device,
        )

        targets = input_ids[0, targ_pos]

        scores.append(
            float(
                lp[
                    0,
                    pred_pos,
                    targets,
                ]
                .mean()
                .cpu()
            )
        )

    return {
        "test_mean_logprob": scores[0],
        "impl_mean_logprob": scores[1],
        "margin": scores[1] - scores[0],
    }


def bootstrap_ci(values, seed, n_boot=5000):
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)

    boots = np.asarray(
        [
            rng.choice(
                x,
                size=len(x),
                replace=True,
            ).mean()
            for _ in range(n_boot)
        ]
    )

    return (
        float(x.mean()),
        float(np.quantile(boots, 0.025)),
        float(np.quantile(boots, 0.975)),
    )


def paired_bootstrap(a, b, seed):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if len(a) != len(b):
        raise RuntimeError("Paired arrays differ in length")

    return bootstrap_ci(
        a - b,
        seed,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=4808)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    exp01b = load_exp01b()
    manifest01b = load_manifest()
    by_task = recreate_entries(
        exp01b,
        manifest01b,
    )

    model_dir = resolve_model_dir(args.model)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    print("=" * 80)
    print("EXP08: Selector Boundary / Handoff Localization")
    print("Model:", model_dir)
    print("Tasks:", len(by_task))
    print("Selector candidate depth: H15-H20")
    print("=" * 80)

    result_rows = []

    for task_no, ti in enumerate(
        sorted(by_task),
        start=1,
    ):
        entries = by_task[ti]
        cache, maps = prepare_task_cache(
            model,
            tok,
            layers,
            entries,
        )

        baselines = [
            score(
                model,
                tok,
                layers,
                e,
                cache[i],
            )
            for i, e in enumerate(entries)
        ]

        # Main same-wording opposite-state scan.
        for cfg in CONFIGS:
            for i, rec in enumerate(entries):
                di = maps[i]["opposite_same_wording"]
                donor = entries[di]

                k = longest_common_suffix(
                    cache[i]["ids"],
                    cache[di]["ids"],
                )

                patched = score(
                    model,
                    tok,
                    layers,
                    rec,
                    cache[i],
                    donor_cache=cache[di],
                    hidden_indices=cfg["hidden_indices"],
                    common_k=k,
                )

                sign = (
                    1.0
                    if donor["label"] == LABEL_IMPL
                    else -1.0
                )

                raw = (
                    patched["margin"]
                    - baselines[i]["margin"]
                )

                result_rows.append(
                    {
                        "task_idx": ti,
                        "task_id": rec["task_id"],
                        "recipient_condition": rec["condition"],
                        "recipient_wording": rec["wording"],
                        "recipient_label": rec["label"],
                        "donor_type": "opposite_same_wording",
                        "donor_condition": donor["condition"],
                        "donor_label": donor["label"],
                        "config": cfg["name"],
                        "family": cfg["family"],
                        "boundary": cfg["boundary"],
                        "hidden_indices": ",".join(
                            map(
                                str,
                                cfg["hidden_indices"],
                            )
                        ),
                        "common_suffix_tokens": k,
                        "baseline_margin": baselines[i]["margin"],
                        "patched_margin": patched["margin"],
                        "raw_margin_shift": raw,
                        "signed_transfer_effect": sign * raw,
                    }
                )

        # Controls only for H20-only and full H15-H20.
        key_configs = [
            {
                "name": "single_H20",
                "hidden_indices": [20],
            },
            {
                "name": "full_H15_H20",
                "hidden_indices": list(range(15, 21)),
            },
        ]

        for key_cfg in key_configs:
            for donor_type in [
                "self",
                "same_state_cross_wording",
                "opposite_cross_wording",
            ]:
                for i, rec in enumerate(entries):
                    di = maps[i][donor_type]
                    donor = entries[di]

                    k = longest_common_suffix(
                        cache[i]["ids"],
                        cache[di]["ids"],
                    )

                    patched = score(
                        model,
                        tok,
                        layers,
                        rec,
                        cache[i],
                        donor_cache=cache[di],
                        hidden_indices=key_cfg["hidden_indices"],
                        common_k=k,
                    )

                    raw = (
                        patched["margin"]
                        - baselines[i]["margin"]
                    )

                    if donor_type == "self":
                        signed = raw
                    else:
                        sign = (
                            1.0
                            if donor["label"] == LABEL_IMPL
                            else -1.0
                        )
                        signed = sign * raw

                    result_rows.append(
                        {
                            "task_idx": ti,
                            "task_id": rec["task_id"],
                            "recipient_condition": rec["condition"],
                            "recipient_wording": rec["wording"],
                            "recipient_label": rec["label"],
                            "donor_type": donor_type,
                            "donor_condition": donor["condition"],
                            "donor_label": donor["label"],
                            "config": key_cfg["name"],
                            "family": "control",
                            "boundary": 20,
                            "hidden_indices": ",".join(
                                map(
                                    str,
                                    key_cfg["hidden_indices"],
                                )
                            ),
                            "common_suffix_tokens": k,
                            "baseline_margin": baselines[i]["margin"],
                            "patched_margin": patched["margin"],
                            "raw_margin_shift": raw,
                            "signed_transfer_effect": signed,
                        }
                    )

        print(
            f"[{task_no:03d}/{len(by_task):03d}] task {ti}"
        )

        del cache

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = pd.DataFrame(result_rows)
    results.to_csv(
        out_dir / "intervention_results.csv",
        index=False,
    )

    # Inferential unit = task.
    task_level = (
        results.groupby(
            [
                "config",
                "donor_type",
                "family",
                "boundary",
                "task_idx",
                "task_id",
            ],
            as_index=False,
        )["signed_transfer_effect"]
        .mean()
    )

    summary_rows = []

    for keys, g in task_level.groupby(
        [
            "config",
            "donor_type",
            "family",
            "boundary",
        ]
    ):
        config, donor_type, family, boundary = keys

        seed = (
            args.seed
            + sum(
                ord(c)
                for c in config + donor_type
            )
        )

        mean, lo, hi = bootstrap_ci(
            g["signed_transfer_effect"],
            seed,
        )

        summary_rows.append(
            {
                "config": config,
                "donor_type": donor_type,
                "family": family,
                "boundary": int(boundary),
                "mean_signed_transfer": mean,
                "ci_low": lo,
                "ci_high": hi,
                "num_tasks": int(
                    g["task_idx"].nunique()
                ),
            }
        )

    effects = pd.DataFrame(summary_rows)
    effects.to_csv(
        out_dir / "config_effects.csv",
        index=False,
    )

    main_effects = effects[
        effects["donor_type"]
        == "opposite_same_wording"
    ].copy()

    def task_values(config):
        x = task_level[
            (task_level["config"] == config)
            & (
                task_level["donor_type"]
                == "opposite_same_wording"
            )
        ].sort_values("task_idx")
        return x["signed_transfer_effect"].to_numpy(
            np.float64
        )

    e20 = task_values("single_H20")
    epre19 = task_values("prefix_H15_H19")
    efull = task_values("full_H15_H20")

    contrast_rows = []

    for name, a, b in [
        ("full_minus_H20", efull, e20),
        ("full_minus_H15_H19", efull, epre19),
        ("H20_minus_H15_H19", e20, epre19),
    ]:
        mean, lo, hi = paired_bootstrap(
            a,
            b,
            args.seed + sum(ord(c) for c in name),
        )

        contrast_rows.append(
            {
                "contrast": name,
                "mean_difference": mean,
                "ci_low": lo,
                "ci_high": hi,
            }
        )

    contrast_df = pd.DataFrame(contrast_rows)
    contrast_df.to_csv(
        out_dir / "paired_contrasts.csv",
        index=False,
    )

    # Descriptive boundary profile.
    profile_rows = []

    for cfg in CONFIGS:
        row = main_effects[
            main_effects["config"] == cfg["name"]
        ].iloc[0]

        profile_rows.append(
            {
                "config": cfg["name"],
                "family": cfg["family"],
                "boundary": cfg["boundary"],
                "mean": float(
                    row["mean_signed_transfer"]
                ),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
            }
        )

    profile = pd.DataFrame(profile_rows)
    profile.to_csv(
        out_dir / "boundary_profile.csv",
        index=False,
    )

    # Cross-wording and controls.
    def get_effect(config, donor_type):
        return effects[
            (effects["config"] == config)
            & (effects["donor_type"] == donor_type)
        ].iloc[0]

    h20_main = get_effect(
        "single_H20",
        "opposite_same_wording",
    )
    full_main = get_effect(
        "full_H15_H20",
        "opposite_same_wording",
    )
    pre19_main = get_effect(
        "prefix_H15_H19",
        "opposite_same_wording",
    )

    h20_cross = get_effect(
        "single_H20",
        "opposite_cross_wording",
    )
    full_cross = get_effect(
        "full_H15_H20",
        "opposite_cross_wording",
    )

    h20_self = get_effect(
        "single_H20",
        "self",
    )
    full_self = get_effect(
        "full_H15_H20",
        "self",
    )

    h20_same = get_effect(
        "single_H20",
        "same_state_cross_wording",
    )
    full_same = get_effect(
        "full_H15_H20",
        "same_state_cross_wording",
    )

    h20_fraction = (
        float(h20_main["mean_signed_transfer"])
        / float(full_main["mean_signed_transfer"])
        if abs(
            float(full_main["mean_signed_transfer"])
        )
        > 1e-12
        else np.nan
    )

    pre19_fraction = (
        float(pre19_main["mean_signed_transfer"])
        / float(full_main["mean_signed_transfer"])
        if abs(
            float(full_main["mean_signed_transfer"])
        )
        > 1e-12
        else np.nan
    )

    # Plot singles + prefix sequence.
    fig = plt.figure(
        figsize=(10, 5.8)
    )

    singles = profile[
        profile["family"] == "single"
    ].sort_values("boundary")

    prefixes = profile[
        profile["family"] == "prefix"
    ].sort_values("boundary")

    plt.plot(
        singles["boundary"],
        singles["mean"],
        marker="o",
        label="single layer",
    )

    # Add H15 single as prefix starting point.
    h15 = profile[
        profile["config"] == "single_H15"
    ].copy()
    prefix_plot = pd.concat(
        [h15, prefixes],
        ignore_index=True,
    ).sort_values("boundary")

    plt.plot(
        prefix_plot["boundary"],
        prefix_plot["mean"],
        marker="o",
        label="cumulative H15→Hk",
    )

    plt.axhline(
        0.0,
        linestyle="--",
    )

    plt.xlabel(
        "Boundary hidden-state index"
    )
    plt.ylabel(
        "Signed transfer effect"
    )
    plt.title(
        "EXP08 selector boundary localization"
    )
    plt.legend()
    plt.tight_layout()

    fig.savefig(
        out_dir / "selector_boundary.png",
        dpi=180,
    )
    plt.close(fig)

    manifest = {
        "experiment": "EXP08_selector_boundary",
        "git_commit": git_commit(),
        "source_exp01b_git_commit": (
            manifest01b.get("git_commit")
        ),
        "seed": args.seed,
        "model_dir": str(model_dir),
        "model_name": model_dir.name,
        "selector_hidden_indices": (
            SELECTOR_HIDDEN
        ),
        "configs": CONFIGS,
        "script_sha256": sha256_file(
            Path(__file__).resolve()
        ),
        "exp01b_script_sha256": sha256_file(
            EXP01B_CODE
        ),
        "model_config_sha256": sha256_file(
            model_dir / "config.json"
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": bool(
            torch.cuda.is_available()
        ),
        "cuda_version": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "offline_only": True,
    }

    (
        out_dir / "run_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    def ci(row):
        return [
            float(row["ci_low"]),
            float(row["ci_high"]),
        ]

    summary = {
        "experiment": (
            "EXP08_selector_boundary"
        ),
        "single_H20_mean": float(
            h20_main["mean_signed_transfer"]
        ),
        "single_H20_95ci": ci(h20_main),
        "prefix_H15_H19_mean": float(
            pre19_main[
                "mean_signed_transfer"
            ]
        ),
        "prefix_H15_H19_95ci": ci(
            pre19_main
        ),
        "full_H15_H20_mean": float(
            full_main["mean_signed_transfer"]
        ),
        "full_H15_H20_95ci": ci(
            full_main
        ),
        "H20_fraction_of_full": (
            h20_fraction
        ),
        "H15_H19_fraction_of_full": (
            pre19_fraction
        ),
        "paired_contrasts": (
            contrast_df.to_dict(
                orient="records"
            )
        ),
        "single_H20_cross_wording_mean": float(
            h20_cross[
                "mean_signed_transfer"
            ]
        ),
        "full_H15_H20_cross_wording_mean": float(
            full_cross[
                "mean_signed_transfer"
            ]
        ),
        "single_H20_self_mean": float(
            h20_self[
                "mean_signed_transfer"
            ]
        ),
        "full_H15_H20_self_mean": float(
            full_self[
                "mean_signed_transfer"
            ]
        ),
        "single_H20_same_state_cross_wording_mean": float(
            h20_same[
                "mean_signed_transfer"
            ]
        ),
        "full_H15_H20_same_state_cross_wording_mean": float(
            full_same[
                "mean_signed_transfer"
            ]
        ),
        "all_boundary_effects": (
            profile.to_dict(
                orient="records"
            )
        ),
        "decision_rule": (
            "H20-handoff interpretation is favored if H20-only is strongly "
            "positive and close to full H15-H20 while H15-H19 without H20 "
            "is much weaker. A depth-distributed interpretation is favored "
            "if cumulative prefixes build substantial effect before H20 and "
            "no single layer approaches the full effect."
        ),
    }

    (
        out_dir / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
