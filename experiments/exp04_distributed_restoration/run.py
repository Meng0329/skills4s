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
DEFAULT_OUT = REPO_ROOT / "outputs" / "exp04_distributed_restoration"

LABEL_TEST = 0
LABEL_IMPL = 1

CONFIGS = [
    ("last1_H19", "last1", [19], "token_scan"),
    ("common_H19", "common", [19], "token_scan"),
    ("common_H19_H24", "common", list(range(19, 25)), "layer_scan"),
    ("common_H19_H28", "common", list(range(19, 29)), "primary"),
    ("common_H21_H28", "common", list(range(21, 29)), "layer_scan"),
    ("common_H15_H28", "common", list(range(15, 29)), "layer_scan"),
]
PRIMARY_CONFIG = "common_H19_H28"


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
            cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
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
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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
                "messages": exp01b.messages(task, history, exp01b.SKILLS[cond]),
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


@torch.inference_mode()
def capture_prompt_states(model, tok, entries, hidden_indices):
    """Capture PRE-norm hidden states via forward hooks on layers[hidx-1].

    IMPORTANT: `outputs.hidden_states[hidx]` for hidx == num_layers is the
    POST-final-RMSNorm value, which is NOT the output of `layers[hidx-1]`.
    Patching the final layer with a normalized activation would inject a
    mis-scaled vector (verified |diff| ~ 735) into the pre-norm residual
    stream, breaking the self-patch sanity control. Hooks grab exactly the
    tensor that `patch_hook` replaces.
    """
    device = model.get_input_embeddings().weight.device
    layers = get_layers(model)
    cache = []
    for e in entries:
        text = render_prompt(tok, e["messages"])
        enc = tok(text, add_special_tokens=False, return_tensors="pt")
        ids = enc["input_ids"][0].tolist()
        enc = {k: v.to(device) for k, v in enc.items()}
        captured = {}
        handles = []
        for hidx in hidden_indices:
            layer = layers[hidx - 1]
            def make_fn(hidx):
                def fn(mod, inputs, output):
                    h = output[0] if isinstance(output, tuple) else output
                    captured[hidx] = h.detach().float().cpu()
                return fn
            handles.append(layer.register_forward_hook(make_fn(hidx)))
        model(**enc, output_hidden_states=False, use_cache=False, return_dict=True)
        for h in handles:
            h.remove()
        states = {
            hidx: captured[hidx][0].numpy().astype(np.float32)
            for hidx in hidden_indices
        }
        cache.append({"ids": ids, "states": states})
        del captured
    return cache


@contextmanager
def patch_hook(layer, positions, replacements):
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


@torch.inference_mode()
def score_one(model, tok, layers, recipient, rec_cache, donor_cache=None,
              hidden_indices=None, span=None):
    prompt_text = render_prompt(tok, recipient["messages"])
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    plen = len(prompt_ids)

    cand_ids = [tok(c, add_special_tokens=False)["input_ids"]
                for c in recipient["candidates"]]

    common = None
    k = 0
    rec_pos_np = donor_pos_np = per_layer = None
    if donor_cache is not None:
        common = longest_common_suffix(rec_cache["ids"], donor_cache["ids"])
        if common <= 0:
            raise RuntimeError("No aligned common suffix.")
        k = 1 if span == "last1" else common
        rec_pos_np = np.arange(plen - k, plen, dtype=np.int64)
        dlen = len(donor_cache["ids"])
        donor_pos_np = np.arange(dlen - k, dlen, dtype=np.int64)
        per_layer = {
            hidx: donor_cache["states"][hidx][donor_pos_np]
            for hidx in hidden_indices
        }

    device = model.get_input_embeddings().weight.device
    scores = []
    for ci in cand_ids:
        row = prompt_ids + ci
        c = len(ci)
        input_ids = torch.tensor([row], dtype=torch.long, device=device)
        attention = torch.ones_like(input_ids)  # no padding → no mask asymmetry

        with ExitStack() as stack:
            if donor_cache is not None:
                positions = torch.tensor(
                    rec_pos_np.reshape(1, -1), dtype=torch.long, device=device)
                for hidx in hidden_indices:
                    replacement = torch.tensor(
                        per_layer[hidx].reshape(1, k, -1), dtype=torch.float32, device=device)
                    stack.enter_context(
                        patch_hook(layers[hidx - 1], positions, replacement))

            out = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            )

        lp = torch.log_softmax(out.logits.float(), dim=-1)[0]
        pred_pos = torch.arange(plen - 1, plen + c - 1, device=device)
        targ_pos = torch.arange(plen, plen + c, device=device)
        targets = input_ids[0, targ_pos]
        scores.append(float(lp[pred_pos, targets].mean().cpu()))

    return {
        "test_mean_logprob": scores[0],
        "impl_mean_logprob": scores[1],
        "margin": scores[1] - scores[0],
        "common_suffix_len": common,
        "patched_token_count": k,
    }


def bootstrap_ci(series, seed, n_boot=5000):
    x = series.to_numpy(np.float64)
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)])
    return float(x.mean()), float(np.quantile(boots, .025)), float(np.quantile(boots, .975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=4404)
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
        str(model_dir), local_files_only=True, trust_remote_code=False
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

    needed = sorted({h for _, _, hs, _ in CONFIGS for h in hs})
    if max(needed) > len(layers):
        raise RuntimeError("Requested hidden-state index exceeds model depth.")

    print("="*78)
    print("EXP04: Distributed Procedural-State Restoration")
    print("Model:", model_dir)
    print("Tasks:", len(by_task))
    print("Primary:", PRIMARY_CONFIG)
    print("="*78)

    rows, suffix_rows = [], []

    for count, ti in enumerate(sorted(by_task), 1):
        entries = by_task[ti]
        maps = donor_maps(entries)
        cache = capture_prompt_states(model, tok, entries, needed)

        baselines = [
            score_one(model, tok, layers, e, cache[i])
            for i, e in enumerate(entries)
        ]

        for i, e in enumerate(entries):
            for dtype, di in maps[i].items():
                suffix_rows.append({
                    "task_idx": ti,
                    "task_id": e["task_id"],
                    "recipient_condition": e["condition"],
                    "donor_type": dtype,
                    "donor_condition": entries[di]["condition"],
                    "common_suffix_tokens": longest_common_suffix(
                        cache[i]["ids"], cache[di]["ids"]
                    ),
                })

        # Opposite state / same wording across all localization configs.
        for name, span, hidden_indices, role in CONFIGS:
            for i, rec in enumerate(entries):
                di = maps[i]["opposite_same_wording"]
                donor = entries[di]
                p = score_one(
                    model, tok, layers, rec, cache[i],
                    cache[di], hidden_indices, span
                )
                raw = p["margin"] - baselines[i]["margin"]
                sign = 1.0 if donor["label"] == LABEL_IMPL else -1.0
                rows.append({
                    "task_idx": ti,
                    "task_id": rec["task_id"],
                    "recipient_condition": rec["condition"],
                    "recipient_wording": rec["wording"],
                    "recipient_label": rec["label"],
                    "donor_type": "opposite_same_wording",
                    "donor_condition": donor["condition"],
                    "donor_label": donor["label"],
                    "config": name,
                    "config_role": role,
                    "span": span,
                    "hidden_indices": ",".join(map(str, hidden_indices)),
                    "common_suffix_len": p["common_suffix_len"],
                    "patched_token_count": p["patched_token_count"],
                    "baseline_margin": baselines[i]["margin"],
                    "patched_margin": p["margin"],
                    "raw_margin_shift": raw,
                    "signed_transfer_effect": sign * raw,
                })

        # Primary controls.
        primary = next(c for c in CONFIGS if c[0] == PRIMARY_CONFIG)
        _, span, hidden_indices, _ = primary
        for dtype in ["same_state_cross_wording", "opposite_cross_wording", "self"]:
            for i, rec in enumerate(entries):
                di = maps[i][dtype]
                donor = entries[di]
                p = score_one(
                    model, tok, layers, rec, cache[i],
                    cache[di], hidden_indices, span
                )
                raw = p["margin"] - baselines[i]["margin"]
                if dtype == "self":
                    signed = raw
                else:
                    sign = 1.0 if donor["label"] == LABEL_IMPL else -1.0
                    signed = sign * raw
                rows.append({
                    "task_idx": ti,
                    "task_id": rec["task_id"],
                    "recipient_condition": rec["condition"],
                    "recipient_wording": rec["wording"],
                    "recipient_label": rec["label"],
                    "donor_type": dtype,
                    "donor_condition": donor["condition"],
                    "donor_label": donor["label"],
                    "config": PRIMARY_CONFIG,
                    "config_role": "control",
                    "span": span,
                    "hidden_indices": ",".join(map(str, hidden_indices)),
                    "common_suffix_len": p["common_suffix_len"],
                    "patched_token_count": p["patched_token_count"],
                    "baseline_margin": baselines[i]["margin"],
                    "patched_margin": p["margin"],
                    "raw_margin_shift": raw,
                    "signed_transfer_effect": signed,
                })

        print(f"[{count:03d}/{len(by_task):03d}] task {ti}")
        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    suffix = pd.DataFrame(suffix_rows)
    df.to_csv(out_dir / "patch_results.csv", index=False)
    suffix.to_csv(out_dir / "common_suffix_lengths.csv", index=False)

    task_level = (
        df.groupby(["config", "donor_type", "task_idx", "task_id"], as_index=False)
          ["signed_transfer_effect"].mean()
    )

    summaries = []
    for (config, donor_type), g in task_level.groupby(["config", "donor_type"]):
        seed = args.seed + sum(ord(ch) for ch in (config + donor_type))
        mean, lo, hi = bootstrap_ci(
            g.set_index("task_idx")["signed_transfer_effect"], seed
        )
        summaries.append({
            "config": config,
            "donor_type": donor_type,
            "mean_signed_transfer": mean,
            "bootstrap_ci_low": lo,
            "bootstrap_ci_high": hi,
            "num_tasks": int(g["task_idx"].nunique()),
        })
    effects = pd.DataFrame(summaries)
    effects.to_csv(out_dir / "config_effects.csv", index=False)

    primary_task = task_level[
        (task_level["config"] == PRIMARY_CONFIG) &
        (task_level["donor_type"] == "opposite_same_wording")
    ].copy()
    primary_task.to_csv(out_dir / "task_effects_primary.csv", index=False)

    def get_effect(dtype):
        return effects[
            (effects["config"] == PRIMARY_CONFIG) &
            (effects["donor_type"] == dtype)
        ].iloc[0]

    primary = get_effect("opposite_same_wording")
    same = get_effect("same_state_cross_wording")
    cross = get_effect("opposite_cross_wording")
    selfrow = get_effect("self")

    primary_raw = df[
        (df["config"] == PRIMARY_CONFIG) &
        (df["donor_type"] == "opposite_same_wording")
    ]
    wording = (
        primary_raw.groupby(["recipient_wording", "task_idx"], as_index=False)
        ["signed_transfer_effect"].mean()
        .groupby("recipient_wording")["signed_transfer_effect"]
        .agg(["mean", "count"]).reset_index()
    )

    loc = effects[effects["donor_type"] == "opposite_same_wording"].copy()
    order = {name: i for i, (name, _, _, _) in enumerate(CONFIGS)}
    loc["order"] = loc["config"].map(order)
    loc = loc.sort_values("order")

    fig = plt.figure(figsize=(10, 5.8))
    x = np.arange(len(loc))
    means = loc["mean_signed_transfer"].to_numpy()
    low = means - loc["bootstrap_ci_low"].to_numpy()
    high = loc["bootstrap_ci_high"].to_numpy() - means
    plt.errorbar(x, means, yerr=np.vstack([low, high]), marker="o", capsize=4)
    plt.axhline(0.0, linestyle="--")
    plt.xticks(x, loc["config"], rotation=30, ha="right")
    plt.ylabel("Signed donor-state transfer effect")
    plt.title("EXP04 distributed restoration")
    plt.tight_layout()
    fig.savefig(out_dir / "distributed_restoration.png", dpi=180)
    plt.close(fig)

    primary_suffix = suffix[suffix["donor_type"] == "opposite_same_wording"]

    manifest = {
        "experiment": "EXP04_distributed_restoration",
        "git_commit": git_commit(),
        "source_exp01b_git_commit": manifest01b.get("git_commit"),
        "seed": args.seed,
        "model_dir": str(model_dir),
        "model_name": model_dir.name,
        "primary_config": PRIMARY_CONFIG,
        "configs": [
            {"name": n, "span": s, "hidden_indices": hs, "role": r}
            for n, s, hs, r in CONFIGS
        ],
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
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "experiment": "EXP04_distributed_restoration",
        "primary_config": PRIMARY_CONFIG,
        "primary_mean_signed_transfer": float(primary["mean_signed_transfer"]),
        "primary_bootstrap_95ci": [
            float(primary["bootstrap_ci_low"]), float(primary["bootstrap_ci_high"])
        ],
        "primary_effect_by_recipient_wording": wording.to_dict(orient="records"),
        "same_state_cross_wording_control_mean": float(same["mean_signed_transfer"]),
        "same_state_cross_wording_control_95ci": [
            float(same["bootstrap_ci_low"]), float(same["bootstrap_ci_high"])
        ],
        "opposite_state_cross_wording_mean": float(cross["mean_signed_transfer"]),
        "opposite_state_cross_wording_95ci": [
            float(cross["bootstrap_ci_low"]), float(cross["bootstrap_ci_high"])
        ],
        "self_patch_mean": float(selfrow["mean_signed_transfer"]),
        "self_patch_95ci": [
            float(selfrow["bootstrap_ci_low"]), float(selfrow["bootstrap_ci_high"])
        ],
        "minimum_common_suffix_tokens": int(primary_suffix["common_suffix_tokens"].min()),
        "median_common_suffix_tokens": float(primary_suffix["common_suffix_tokens"].median()),
        "localization_effects": loc[
            ["config", "mean_signed_transfer", "bootstrap_ci_low", "bootstrap_ci_high"]
        ].to_dict(orient="records"),
        "decision_rule": (
            "Positive primary CI excluding zero, small self/same-state controls, "
            "and concordant opposite-state cross-wording transfer supports "
            "distributed residual-state mediation."
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== EXP04 SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
