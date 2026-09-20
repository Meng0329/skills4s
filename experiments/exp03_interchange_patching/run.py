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
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = REPO_ROOT / "models"
EXP01B_DIR = REPO_ROOT / "outputs" / "exp01b_counterbalanced_next_state"
EXP01B_CODE = REPO_ROOT / "experiments" / "exp01b_counterbalanced_next_state" / "run.py"
DEFAULT_OUT = REPO_ROOT / "outputs" / "exp03_interchange_patching"

CONFIRMATORY_HIDDEN_INDEX = 19
CONFIRMATORY_DECODER_INDEX = CONFIRMATORY_HIDDEN_INDEX - 1

LABEL_TEST = 0
LABEL_IMPL = 1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def resolve_model_dir(model_arg: str | None) -> Path:
    if model_arg:
        p = Path(model_arg).expanduser()
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

    candidates = sorted(
        {p.parent.resolve() for p in MODELS_ROOT.rglob("config.json")}
    )
    if not candidates:
        raise FileNotFoundError("No local model found under ./models.")
    if len(candidates) > 1:
        raise RuntimeError(
            "Multiple local models found; specify --model:\n"
            + "\n".join(f"  - {p}" for p in candidates)
        )
    return candidates[0]


def load_exp01b_module():
    if not EXP01B_CODE.is_file():
        raise FileNotFoundError(f"Missing EXP01b code: {EXP01B_CODE}")
    spec = importlib.util.spec_from_file_location("skills4s_exp01b", EXP01B_CODE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_previous_artifacts():
    activation_path = EXP01B_DIR / "activations.npz"
    metadata_path = EXP01B_DIR / "metadata.csv"
    behavior_path = EXP01B_DIR / "behavior.csv"
    manifest_path = EXP01B_DIR / "run_manifest.json"

    for p in [activation_path, metadata_path, behavior_path, manifest_path]:
        if not p.is_file():
            raise FileNotFoundError(f"Missing required EXP01b artifact: {p}")

    arr = np.load(activation_path)
    H = arr["hidden"]
    md = pd.read_csv(metadata_path)
    behavior = pd.read_csv(behavior_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if H.shape[0] != len(md) or len(md) != len(behavior):
        raise RuntimeError("EXP01b artifacts have inconsistent row counts.")

    return H, md, behavior, manifest


def recreate_entries(exp01b, manifest, metadata):
    num_tasks = int(manifest["num_tasks"])
    seed = int(manifest["seed"])
    tasks = exp01b.make_tasks(num_tasks, seed)

    entries = []
    for task in tasks:
        ti = int(task["task_idx"])
        history = exp01b.history(task, ti)
        for cond_name, wording, label in exp01b.CONDS:
            messages = exp01b.messages(
                task, history, exp01b.SKILLS[cond_name]
            )
            entries.append(
                {
                    "task_idx": ti,
                    "task_id": task["task_id"],
                    "condition": cond_name,
                    "wording": wording,
                    "label": int(label),
                    "messages": messages,
                    "candidates": [
                        f"read_file('{task['test_path']}')",
                        f"read_file('{task['src_path']}')",
                    ],
                }
            )

    if len(entries) != len(metadata):
        raise RuntimeError("Recreated EXP01b stimuli count mismatch.")

    for i, entry in enumerate(entries):
        row = metadata.iloc[i]
        if not (
            int(row["task_idx"]) == entry["task_idx"]
            and str(row["condition"]) == entry["condition"]
            and str(row["wording"]) == entry["wording"]
            and int(row["label"]) == entry["label"]
        ):
            raise RuntimeError(
                f"Stimulus recreation mismatch at row {i}; refusing to run."
            )
    return entries


def make_pairs(entries):
    """Return recipient->donor mapping within same task and wording."""
    lookup = {
        (e["task_idx"], e["wording"], e["label"]): i
        for i, e in enumerate(entries)
    }
    pairs = []
    for i, e in enumerate(entries):
        donor_label = LABEL_IMPL if e["label"] == LABEL_TEST else LABEL_TEST
        donor_idx = lookup[(e["task_idx"], e["wording"], donor_label)]
        pairs.append((i, donor_idx))
    return pairs


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("Expected Qwen-style model.model.layers.")


def render_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_items(tokenizer, entries):
    items = []
    for prompt_idx, e in enumerate(entries):
        prompt_text = render_prompt(tokenizer, e["messages"])
        prompt_ids = tokenizer(
            prompt_text, add_special_tokens=False
        )["input_ids"]
        for cand_idx, candidate in enumerate(e["candidates"]):
            cand_ids = tokenizer(
                candidate, add_special_tokens=False
            )["input_ids"]
            items.append(
                {
                    "prompt_idx": prompt_idx,
                    "cand_idx": cand_idx,
                    "prompt_len": len(prompt_ids),
                    "cand_len": len(cand_ids),
                    "ids": prompt_ids + cand_ids,
                }
            )
    return items


@contextmanager
def exact_patch_hook(layer_module, prompt_last_positions, replacement_vectors):
    """Replace each row's final-prompt-token layer output by an exact donor state."""
    def hook_fn(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = None

        h = h.clone()
        batch_idx = torch.arange(h.shape[0], device=h.device)
        pos = prompt_last_positions.to(h.device)
        rep = replacement_vectors.to(h.device, dtype=h.dtype)

        h[batch_idx, pos, :] = rep

        if rest is None:
            return h
        return (h,) + rest

    handle = layer_module.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def null_context():
    yield


@torch.inference_mode()
def score_entries(
    model,
    tokenizer,
    layer_module,
    entries,
    replacements_by_prompt: np.ndarray | None,
    batch_size: int,
):
    """Score TEST and IMPL candidate actions for each prompt.

    If replacements_by_prompt is supplied, it has shape [num_prompts, hidden]
    and is inserted at the final prompt token of the selected decoder block.
    """
    items = build_items(tokenizer, entries)
    by_prompt = {}
    for item in items:
        by_prompt.setdefault(item["prompt_idx"], []).append(item)

    prompt_indices = sorted(by_prompt)
    results = [[None, None] for _ in entries]
    pad_id = tokenizer.pad_token_id
    device = model.get_input_embeddings().weight.device

    for start in range(0, len(prompt_indices), batch_size):
        batch_prompts = prompt_indices[start:start + batch_size]
        batch_items = []
        for pidx in batch_prompts:
            batch_items.extend(
                sorted(by_prompt[pidx], key=lambda x: x["cand_idx"])
            )

        max_len = max(len(x["ids"]) for x in batch_items)
        ids_rows, attn_rows, last_positions = [], [], []
        rep_rows = []

        for item in batch_items:
            ids = item["ids"]
            pad_n = max_len - len(ids)
            ids_rows.append(ids + [pad_id] * pad_n)
            attn_rows.append([1] * len(ids) + [0] * pad_n)
            last_positions.append(item["prompt_len"] - 1)

            if replacements_by_prompt is not None:
                rep_rows.append(
                    replacements_by_prompt[item["prompt_idx"]]
                )

        input_ids = torch.tensor(ids_rows, dtype=torch.long, device=device)
        attention = torch.tensor(attn_rows, dtype=torch.long, device=device)
        last_t = torch.tensor(last_positions, dtype=torch.long, device=device)

        if replacements_by_prompt is None:
            ctx = null_context()
        else:
            rep_t = torch.tensor(
                np.stack(rep_rows),
                dtype=torch.float32,
                device=device,
            )
            ctx = exact_patch_hook(layer_module, last_t, rep_t)

        with ctx:
            out = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            )

        lp = torch.log_softmax(out.logits.float(), dim=-1)

        for row_idx, item in enumerate(batch_items):
            p = item["prompt_len"]
            c = item["cand_len"]
            pred_pos = torch.arange(p - 1, p + c - 1, device=device)
            target_pos = torch.arange(p, p + c, device=device)
            targets = input_ids[row_idx, target_pos]
            token_lp = lp[row_idx, pred_pos, targets]
            results[item["prompt_idx"]][item["cand_idx"]] = {
                "mean_logprob": float(token_lp.mean().cpu()),
                "sum_logprob": float(token_lp.sum().cpu()),
            }

    return results


def margins(scores):
    return np.array(
        [
            pair[1]["mean_logprob"] - pair[0]["mean_logprob"]
            for pair in scores
        ],
        dtype=np.float64,
    )


def bootstrap_task_ci(task_values: pd.Series, seed: int, n_boot: int = 5000):
    values = task_values.to_numpy(np.float64)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boots[i] = rng.choice(values, size=len(values), replace=True).mean()
    return (
        float(values.mean()),
        float(np.quantile(boots, 0.025)),
        float(np.quantile(boots, 0.975)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=4303)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    H, metadata, old_behavior, exp01b_manifest = load_previous_artifacts()
    exp01b = load_exp01b_module()
    entries = recreate_entries(exp01b, exp01b_manifest, metadata)
    pairs = make_pairs(entries)

    model_dir = resolve_model_dir(args.model)
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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

    n_hidden_states = H.shape[1]
    if n_hidden_states != len(layers) + 1:
        raise RuntimeError(
            f"Expected hidden_states = decoder_layers + 1, got "
            f"{n_hidden_states} vs {len(layers)}."
        )

    print("=" * 78)
    print("EXP03: Paired Interchange Patching")
    print(f"Model                 : {model_dir}")
    print(f"Prompts               : {len(entries)}")
    print(f"Decoder layers        : {len(layers)}")
    print(f"Confirmatory hidden   : {CONFIRMATORY_HIDDEN_INDEX}")
    print(f"Confirmatory decoder  : {CONFIRMATORY_DECODER_INDEX}")
    print(f"Output                : {out_dir}")
    print("=" * 78)

    # Recompute baseline once.
    baseline_scores = score_entries(
        model, tokenizer, layers[0], entries,
        replacements_by_prompt=None,
        batch_size=args.batch_size,
    )
    baseline_margin = margins(baseline_scores)

    # Donor map is the same for every layer.
    donor_idx = np.array([d for _, d in pairs], dtype=np.int64)
    donor_labels = metadata.iloc[donor_idx]["label"].to_numpy(np.int64)
    donor_sign = np.where(donor_labels == LABEL_IMPL, 1.0, -1.0)

    rows = []

    # Exploratory full-layer exact paired patch scan.
    for hidden_idx in range(1, n_hidden_states):
        decoder_idx = hidden_idx - 1
        replacements = H[donor_idx, hidden_idx, :].astype(np.float32)

        patched_scores = score_entries(
            model=model,
            tokenizer=tokenizer,
            layer_module=layers[decoder_idx],
            entries=entries,
            replacements_by_prompt=replacements,
            batch_size=args.batch_size,
        )
        patched_margin = margins(patched_scores)

        raw_delta = patched_margin - baseline_margin
        signed_transfer = donor_sign * raw_delta

        for i, e in enumerate(entries):
            rows.append(
                {
                    "hidden_state_index": hidden_idx,
                    "decoder_layer_index": decoder_idx,
                    "task_idx": e["task_idx"],
                    "task_id": e["task_id"],
                    "condition": e["condition"],
                    "wording": e["wording"],
                    "recipient_label": e["label"],
                    "donor_label": int(donor_labels[i]),
                    "baseline_margin": baseline_margin[i],
                    "patched_margin": patched_margin[i],
                    "raw_margin_shift": raw_delta[i],
                    "signed_transfer_effect": signed_transfer[i],
                }
            )

        print(
            f"[hidden {hidden_idx:02d} / decoder {decoder_idx:02d}] "
            f"mean signed transfer = {signed_transfer.mean():+.6f}"
        )

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "patch_results.csv", index=False)

    # Confirmatory self-patch sanity at hidden index 19.
    self_replacements = H[
        np.arange(len(entries)),
        CONFIRMATORY_HIDDEN_INDEX,
        :
    ].astype(np.float32)

    self_scores = score_entries(
        model=model,
        tokenizer=tokenizer,
        layer_module=layers[CONFIRMATORY_DECODER_INDEX],
        entries=entries,
        replacements_by_prompt=self_replacements,
        batch_size=args.batch_size,
    )
    self_margin = margins(self_scores)
    self_delta = self_margin - baseline_margin

    # Aggregate layer effects by task to avoid treating wording/condition rows
    # as independent inferential units.
    layer_task = (
        result.groupby(
            ["hidden_state_index", "decoder_layer_index", "task_idx"],
            as_index=False,
        )["signed_transfer_effect"]
        .mean()
    )

    layer_summary_rows = []
    for hidden_idx, g in layer_task.groupby("hidden_state_index"):
        mean, lo, hi = bootstrap_task_ci(
            g.set_index("task_idx")["signed_transfer_effect"],
            seed=args.seed + int(hidden_idx),
        )
        layer_summary_rows.append(
            {
                "hidden_state_index": int(hidden_idx),
                "decoder_layer_index": int(hidden_idx - 1),
                "mean_signed_transfer": mean,
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
            }
        )

    layer_effects = pd.DataFrame(layer_summary_rows)
    layer_effects.to_csv(out_dir / "layer_effects.csv", index=False)

    confirm = result[
        result["hidden_state_index"] == CONFIRMATORY_HIDDEN_INDEX
    ].copy()

    confirm_task = (
        confirm.groupby(["task_idx", "task_id"], as_index=False)[
            "signed_transfer_effect"
        ].mean()
    )
    confirm_task.to_csv(
        out_dir / "task_effects_confirmatory.csv",
        index=False,
    )

    confirm_mean, confirm_lo, confirm_hi = bootstrap_task_ci(
        confirm_task.set_index("task_idx")["signed_transfer_effect"],
        seed=args.seed + 1900,
    )

    confirm_wording = (
        confirm.groupby("wording")["signed_transfer_effect"]
        .agg(["mean", "count"])
        .reset_index()
    )

    best_row = layer_effects.loc[
        layer_effects["mean_signed_transfer"].idxmax()
    ]

    # Fraction of individual patches that move toward donor Skill state.
    confirm_positive_rate = float(
        (confirm["signed_transfer_effect"] > 0).mean()
    )

    # Self-patch numerical sanity.
    self_mean_abs = float(np.mean(np.abs(self_delta)))
    self_max_abs = float(np.max(np.abs(self_delta)))

    # Plot.
    fig = plt.figure(figsize=(9, 5.5))
    plt.plot(
        layer_effects["hidden_state_index"],
        layer_effects["mean_signed_transfer"],
        marker="o",
        label="Exact paired interchange",
    )
    plt.fill_between(
        layer_effects["hidden_state_index"],
        layer_effects["bootstrap_ci_low"],
        layer_effects["bootstrap_ci_high"],
        alpha=0.2,
    )
    plt.axhline(0.0, linestyle="--")
    plt.axvline(
        CONFIRMATORY_HIDDEN_INDEX,
        linestyle=":",
        label="Confirmatory index 19",
    )
    plt.xlabel("Hidden-state index (0 = embedding)")
    plt.ylabel("Signed donor-state transfer effect")
    plt.title("EXP03 exact activation interchange across layers")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "layer_effects.png", dpi=180)
    plt.close(fig)

    manifest = {
        "experiment": "EXP03_interchange_patching",
        "git_commit": git_commit(),
        "source_exp01b_git_commit": exp01b_manifest.get("git_commit"),
        "seed": args.seed,
        "model_dir": str(model_dir),
        "model_name": model_dir.name,
        "confirmatory_hidden_state_index": CONFIRMATORY_HIDDEN_INDEX,
        "confirmatory_decoder_layer_index": CONFIRMATORY_DECODER_INDEX,
        "num_prompts": len(entries),
        "num_tasks": int(metadata["task_idx"].nunique()),
        "batch_size": args.batch_size,
        "script_sha256": sha256_file(Path(__file__).resolve()),
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
        "experiment": "EXP03_interchange_patching",
        "hypothesis": (
            "Exact same-task opposite-Skill residual-state interchange "
            "causally transfers next-action preference."
        ),
        "confirmatory_hidden_state_index": CONFIRMATORY_HIDDEN_INDEX,
        "confirmatory_decoder_layer_index": CONFIRMATORY_DECODER_INDEX,
        "confirmatory_mean_signed_transfer": confirm_mean,
        "confirmatory_bootstrap_95ci": [confirm_lo, confirm_hi],
        "confirmatory_positive_patch_rate": confirm_positive_rate,
        "confirmatory_effect_by_wording": confirm_wording.to_dict(
            orient="records"
        ),
        "self_patch_mean_abs_margin_change": self_mean_abs,
        "self_patch_max_abs_margin_change": self_max_abs,
        "exploratory_best_hidden_state_index": int(
            best_row["hidden_state_index"]
        ),
        "exploratory_best_decoder_layer_index": int(
            best_row["decoder_layer_index"]
        ),
        "exploratory_best_mean_signed_transfer": float(
            best_row["mean_signed_transfer"]
        ),
        "exploratory_best_bootstrap_95ci": [
            float(best_row["bootstrap_ci_low"]),
            float(best_row["bootstrap_ci_high"]),
        ],
        "decision_rule": (
            "Positive index-19 transfer with task-bootstrap CI excluding zero, "
            "same sign across wordings, and near-zero self-patch supports a "
            "context-specific causal state. Otherwise inspect exploratory "
            "layer localization before moving to multi-token/multi-layer patching."
        ),
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== EXP03 SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nReturn:")
    print("  summary.json")
    print("  layer_effects.csv")
    print("  task_effects_confirmatory.csv")
    print("  layer_effects.png")
    print("  run_manifest.json")


if __name__ == "__main__":
    main()
