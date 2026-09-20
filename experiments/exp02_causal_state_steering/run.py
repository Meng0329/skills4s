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
import math
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
from sklearn.model_selection import GroupKFold
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = REPO_ROOT / "models"
EXP01B_DIR = REPO_ROOT / "outputs" / "exp01b_counterbalanced_next_state"
EXP01B_CODE = REPO_ROOT / "experiments" / "exp01b_counterbalanced_next_state" / "run.py"
DEFAULT_OUT = REPO_ROOT / "outputs" / "exp02_causal_state_steering"

# EXP01/EXP01b preregistered representation index.
HIDDEN_STATE_INDEX = 19

# Hugging Face hidden_states[0] is embedding output, so:
# hidden_states[19] == output after decoder block 18.
DECODER_LAYER_INDEX = HIDDEN_STATE_INDEX - 1

LABEL_TEST = 0
LABEL_IMPL = 1

REAL_ALPHAS = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
PRIMARY_ALPHA = 1.0


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
            repo_candidate = (REPO_ROOT / p).resolve()
            model_candidate = (MODELS_ROOT / p).resolve()
            if repo_candidate.exists():
                p = repo_candidate
            elif model_candidate.exists():
                p = model_candidate
            else:
                p = repo_candidate
        p = p.resolve()
        if not (p / "config.json").is_file():
            raise FileNotFoundError(f"Missing config.json in model directory: {p}")
        return p

    if (MODELS_ROOT / "config.json").is_file():
        return MODELS_ROOT.resolve()

    candidates = sorted(
        {
            cfg.parent.resolve()
            for cfg in MODELS_ROOT.rglob("config.json")
            if cfg.is_file()
        }
    )
    if not candidates:
        raise FileNotFoundError(
            "No local Hugging Face-format model found under ./models."
        )
    if len(candidates) > 1:
        listing = "\n".join(f"  - {p}" for p in candidates)
        raise RuntimeError(
            "Multiple local models found. Choose one with --model:\n" + listing
        )
    return candidates[0]


def load_exp01b_module():
    if not EXP01B_CODE.is_file():
        raise FileNotFoundError(
            f"EXP01b code not found: {EXP01B_CODE}\n"
            "EXP02 reuses the exact EXP01b stimulus-generation functions."
        )
    spec = importlib.util.spec_from_file_location("skills4s_exp01b", EXP01B_CODE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_exp01b_outputs():
    activation_path = EXP01B_DIR / "activations.npz"
    metadata_path = EXP01B_DIR / "metadata.csv"
    manifest_path = EXP01B_DIR / "run_manifest.json"

    for p in [activation_path, metadata_path, manifest_path]:
        if not p.is_file():
            raise FileNotFoundError(
                f"Required EXP01b output not found: {p}\n"
                "Run EXP01b first and keep its outputs."
            )

    arr = np.load(activation_path)
    if "hidden" not in arr:
        raise KeyError(
            f"{activation_path} does not contain an array named 'hidden'."
        )
    H = arr["hidden"]
    metadata = pd.read_csv(metadata_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if H.shape[0] != len(metadata):
        raise RuntimeError(
            f"Activation/metadata mismatch: {H.shape[0]} vs {len(metadata)}"
        )
    if HIDDEN_STATE_INDEX >= H.shape[1]:
        raise RuntimeError(
            f"Hidden-state index {HIDDEN_STATE_INDEX} unavailable; "
            f"activation shape is {H.shape}."
        )

    return H, metadata, manifest


def normalize_to(vec: np.ndarray, target_norm: float) -> np.ndarray:
    vec = vec.astype(np.float64, copy=False)
    norm = np.linalg.norm(vec)
    if not np.isfinite(norm) or norm < 1e-12:
        raise RuntimeError("Control direction has near-zero or invalid norm.")
    return (vec / norm * target_norm).astype(np.float32)


def make_real_direction(
    H: np.ndarray,
    metadata: pd.DataFrame,
    train_tasks: set[int],
) -> np.ndarray:
    train_mask = metadata["task_idx"].isin(train_tasks).to_numpy()
    y = metadata["label"].to_numpy()

    h = H[:, HIDDEN_STATE_INDEX, :].astype(np.float32)
    h_train = h[train_mask]
    y_train = y[train_mask]

    impl = h_train[y_train == LABEL_IMPL]
    test = h_train[y_train == LABEL_TEST]

    if len(impl) == 0 or len(test) == 0:
        raise RuntimeError("Training fold lacks one of the semantic labels.")

    return (impl.mean(axis=0) - test.mean(axis=0)).astype(np.float32)


def make_same_state_direction(
    H: np.ndarray,
    metadata: pd.DataFrame,
    train_tasks: set[int],
    target_norm: float,
) -> np.ndarray:
    """Within-IMPLEMENTATION arbitrary variation control.

    Split training tasks deterministically into two groups, then compare only
    IMPLEMENTATION-state activations. This has the same state on both sides.
    """
    train_task_list = sorted(train_tasks)
    group_a = set(train_task_list[::2])
    group_b = set(train_task_list[1::2])

    md = metadata
    mask_impl = md["label"].to_numpy() == LABEL_IMPL
    mask_a = mask_impl & md["task_idx"].isin(group_a).to_numpy()
    mask_b = mask_impl & md["task_idx"].isin(group_b).to_numpy()

    h = H[:, HIDDEN_STATE_INDEX, :].astype(np.float32)
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        raise RuntimeError("Unable to build same-state control direction.")

    raw = h[mask_a].mean(axis=0) - h[mask_b].mean(axis=0)
    return normalize_to(raw, target_norm)


def make_random_direction(
    hidden_size: int,
    real_direction: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Equal-norm random vector explicitly orthogonal to the real direction."""
    rng = np.random.default_rng(seed)
    real = real_direction.astype(np.float64)
    real_norm = np.linalg.norm(real)
    u = real / real_norm

    r = rng.standard_normal(hidden_size)
    r = r - np.dot(r, u) * u
    return normalize_to(r, real_norm)


def get_decoder_layers(model):
    """Locate the causal-LM decoder block list.

    Qwen2ForCausalLM exposes model.model.layers.
    Keep a guarded fallback for closely related HF architectures.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers

    candidates = [
        ("model", "model", "layers"),
        ("transformer", "h"),
    ]
    raise RuntimeError(
        "Unable to locate decoder layers automatically. "
        "Expected Qwen2ForCausalLM with model.model.layers."
    )


@contextmanager
def residual_steering_hook(
    layer_module,
    prompt_last_positions: torch.Tensor,
    direction: torch.Tensor,
    alpha: float,
):
    """Add alpha*direction at each row's last prompt token.

    Robust to decoder layers that return either a Tensor or a tuple whose first
    item is the residual-stream Tensor.
    """
    def hook_fn(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        hidden = hidden.clone()
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)

        pos = prompt_last_positions.to(hidden.device)
        delta = (direction.to(hidden.device, dtype=hidden.dtype) * alpha)

        hidden[batch_idx, pos, :] = hidden[batch_idx, pos, :] + delta

        if rest is None:
            return hidden
        return (hidden,) + rest

    handle = layer_module.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


def render_prompt(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_scoring_items(
    tokenizer,
    prompt_messages: list[list[dict]],
    candidate_pairs: list[list[str]],
):
    items = []
    for prompt_idx, (messages, candidates) in enumerate(
        zip(prompt_messages, candidate_pairs)
    ):
        prompt_text = render_prompt(tokenizer, messages)
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]

        for cand_idx, candidate in enumerate(candidates):
            cand_ids = tokenizer(
                candidate,
                add_special_tokens=False,
            )["input_ids"]
            if not cand_ids:
                raise RuntimeError(f"Empty candidate tokenization: {candidate!r}")

            items.append(
                {
                    "prompt_idx": prompt_idx,
                    "cand_idx": cand_idx,
                    "candidate": candidate,
                    "prompt_len": len(prompt_ids),
                    "cand_len": len(cand_ids),
                    "ids": prompt_ids + cand_ids,
                }
            )
    return items


@torch.inference_mode()
def score_messages_with_steering(
    model,
    tokenizer,
    layer_module,
    prompt_messages: list[list[dict]],
    candidate_pairs: list[list[str]],
    direction: np.ndarray | None,
    alpha: float,
    batch_size: int,
):
    """Score two candidate actions for every prompt under one intervention.

    `batch_size` counts prompts. Internally there are two sequences per prompt.
    """
    items = build_scoring_items(tokenizer, prompt_messages, candidate_pairs)
    pad_id = tokenizer.pad_token_id

    # Group the two candidates for each prompt and batch by prompt.
    prompt_to_items = {}
    for item in items:
        prompt_to_items.setdefault(item["prompt_idx"], []).append(item)

    prompt_indices = sorted(prompt_to_items.keys())
    results = [[None, None] for _ in prompt_messages]

    if direction is not None:
        direction_t = torch.from_numpy(direction.astype(np.float32))
    else:
        direction_t = None

    for start in range(0, len(prompt_indices), batch_size):
        batch_prompts = prompt_indices[start:start + batch_size]
        batch_items = []
        for pidx in batch_prompts:
            pair = sorted(prompt_to_items[pidx], key=lambda x: x["cand_idx"])
            batch_items.extend(pair)

        max_len = max(len(x["ids"]) for x in batch_items)
        input_ids = []
        attention = []
        prompt_last_positions = []

        for item in batch_items:
            ids = item["ids"]
            pad_n = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_n)
            attention.append([1] * len(ids) + [0] * pad_n)
            prompt_last_positions.append(item["prompt_len"] - 1)

        device = model.get_input_embeddings().weight.device
        input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)
        attention_t = torch.tensor(attention, dtype=torch.long, device=device)
        prompt_last_t = torch.tensor(
            prompt_last_positions,
            dtype=torch.long,
            device=device,
        )

        if direction_t is None or alpha == 0.0:
            ctx = _null_context()
        else:
            ctx = residual_steering_hook(
                layer_module,
                prompt_last_t,
                direction_t,
                alpha,
            )

        with ctx:
            out = model(
                input_ids=input_ids_t,
                attention_mask=attention_t,
                use_cache=False,
                return_dict=True,
            )

        log_probs = torch.log_softmax(out.logits.float(), dim=-1)

        for row_idx, item in enumerate(batch_items):
            p_len = item["prompt_len"]
            c_len = item["cand_len"]

            pred_positions = torch.arange(
                p_len - 1,
                p_len + c_len - 1,
                device=device,
            )
            target_positions = torch.arange(
                p_len,
                p_len + c_len,
                device=device,
            )
            targets = input_ids_t[row_idx, target_positions]
            token_lp = log_probs[row_idx, pred_positions, targets]

            results[item["prompt_idx"]][item["cand_idx"]] = {
                "candidate": item["candidate"],
                "token_count": c_len,
                "sum_logprob": float(token_lp.sum().cpu()),
                "mean_logprob": float(token_lp.mean().cpu()),
            }

    return results


@contextmanager
def _null_context():
    yield


def bootstrap_task_ci(
    effects_by_task: pd.Series,
    seed: int,
    n_boot: int = 5000,
):
    values = effects_by_task.to_numpy(dtype=np.float64)
    if len(values) < 2:
        return float(values.mean()), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    boots = []
    n = len(values)
    for _ in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boots.append(sample.mean())

    boots = np.asarray(boots)
    return (
        float(values.mean()),
        float(np.quantile(boots, 0.025)),
        float(np.quantile(boots, 0.975)),
    )


def load_or_recreate_prompts(exp01b, metadata, num_tasks: int, seed: int):
    """Recreate exact EXP01b prompts from its committed stimulus code."""
    tasks = exp01b.make_tasks(num_tasks, seed)

    by_task = {}
    for task in tasks:
        ti = int(task["task_idx"])
        history = exp01b.history(task, ti)

        entries = []
        for cond_name, wording, label in exp01b.CONDS:
            messages = exp01b.messages(
                task,
                history,
                exp01b.SKILLS[cond_name],
            )
            cand_test = f"read_file('{task['test_path']}')"
            cand_impl = f"read_file('{task['src_path']}')"

            entries.append(
                {
                    "task_idx": ti,
                    "task_id": task["task_id"],
                    "condition": cond_name,
                    "wording": wording,
                    "label": int(label),
                    "messages": messages,
                    "candidates": [cand_test, cand_impl],
                }
            )
        by_task[ti] = entries

    # Validate row order against saved metadata.
    recreated = []
    for ti in sorted(by_task):
        recreated.extend(by_task[ti])

    if len(recreated) != len(metadata):
        raise RuntimeError(
            f"Recreated prompt count {len(recreated)} != metadata {len(metadata)}"
        )

    for i, entry in enumerate(recreated):
        row = metadata.iloc[i]
        checks = [
            int(row["task_idx"]) == entry["task_idx"],
            str(row["condition"]) == entry["condition"],
            str(row["wording"]) == entry["wording"],
            int(row["label"]) == entry["label"],
        ]
        if not all(checks):
            raise RuntimeError(
                "EXP01b prompt recreation does not match saved metadata at "
                f"row {i}. Refusing to run a non-identical causal experiment."
            )

    return by_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-random-controls", type=int, default=5)
    parser.add_argument("--seed", type=int, default=4202)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.num_random_controls < 1:
        raise ValueError("--num-random-controls must be >= 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    H, metadata, exp01b_manifest = load_exp01b_outputs()
    exp01b = load_exp01b_module()

    num_tasks = int(exp01b_manifest["num_tasks"])
    exp01b_seed = int(exp01b_manifest["seed"])

    if num_tasks < 16:
        raise RuntimeError(
            f"EXP01b used only {num_tasks} tasks; EXP02 expects >=16."
        )

    prompts_by_task = load_or_recreate_prompts(
        exp01b,
        metadata,
        num_tasks=num_tasks,
        seed=exp01b_seed,
    )

    model_dir = resolve_model_dir(args.model)
    out_dir = Path(args.out).expanduser()
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("EXP02: Causal Procedural-State Steering")
    print(f"Local model             : {model_dir}")
    print(f"EXP01b tasks            : {num_tasks}")
    print(f"Hidden-state index      : {HIDDEN_STATE_INDEX}")
    print(f"Decoder module index    : {DECODER_LAYER_INDEX}")
    print(f"Batch size (prompts)    : {args.batch_size}")
    print(f"Random controls / fold  : {args.num_random_controls}")
    print(f"Output dir              : {out_dir}")
    print("Network mode            : OFFLINE / local_files_only=True")
    print("=" * 78)

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

    layers = get_decoder_layers(model)
    if DECODER_LAYER_INDEX >= len(layers):
        raise RuntimeError(
            f"Decoder layer index {DECODER_LAYER_INDEX} invalid for "
            f"{len(layers)} layers."
        )
    intervention_layer = layers[DECODER_LAYER_INDEX]

    unique_tasks = np.array(sorted(metadata["task_idx"].unique()), dtype=np.int64)
    n_splits = 4
    cv = GroupKFold(n_splits=n_splits)

    # Split at task level.
    dummy_X = np.zeros((len(unique_tasks), 1))
    dummy_y = np.zeros(len(unique_tasks))

    result_rows = []
    direction_rows = []

    for fold, (train_pos, test_pos) in enumerate(
        cv.split(dummy_X, dummy_y, unique_tasks)
    ):
        train_tasks = set(unique_tasks[train_pos].tolist())
        test_tasks = set(unique_tasks[test_pos].tolist())

        real = make_real_direction(H, metadata, train_tasks)
        real_norm = float(np.linalg.norm(real))
        same = make_same_state_direction(
            H,
            metadata,
            train_tasks,
            target_norm=real_norm,
        )

        random_dirs = [
            make_random_direction(
                hidden_size=real.shape[0],
                real_direction=real,
                seed=args.seed + fold * 1000 + r,
            )
            for r in range(args.num_random_controls)
        ]

        cos_same = float(
            np.dot(real, same)
            / (np.linalg.norm(real) * np.linalg.norm(same))
        )

        direction_rows.append(
            {
                "fold": fold,
                "train_tasks": ",".join(map(str, sorted(train_tasks))),
                "test_tasks": ",".join(map(str, sorted(test_tasks))),
                "real_direction_norm": real_norm,
                "same_state_direction_norm": float(np.linalg.norm(same)),
                "real_vs_same_cosine": cos_same,
            }
        )

        # Assemble held-out prompts once.
        held_entries = []
        for ti in sorted(test_tasks):
            held_entries.extend(prompts_by_task[int(ti)])

        messages = [e["messages"] for e in held_entries]
        candidate_pairs = [e["candidates"] for e in held_entries]

        # Baseline (no hook).
        baseline_scores = score_messages_with_steering(
            model=model,
            tokenizer=tokenizer,
            layer_module=intervention_layer,
            prompt_messages=messages,
            candidate_pairs=candidate_pairs,
            direction=None,
            alpha=0.0,
            batch_size=args.batch_size,
        )

        baseline_margin = []
        for s in baseline_scores:
            margin = s[1]["mean_logprob"] - s[0]["mean_logprob"]
            baseline_margin.append(margin)

        for idx, entry in enumerate(held_entries):
            result_rows.append(
                {
                    "fold": fold,
                    "task_idx": entry["task_idx"],
                    "task_id": entry["task_id"],
                    "condition": entry["condition"],
                    "wording": entry["wording"],
                    "label": entry["label"],
                    "direction_type": "baseline",
                    "control_id": -1,
                    "alpha": 0.0,
                    "test_mean_logprob": baseline_scores[idx][0]["mean_logprob"],
                    "impl_mean_logprob": baseline_scores[idx][1]["mean_logprob"],
                    "margin_impl_minus_test": baseline_margin[idx],
                    "delta_margin_from_baseline": 0.0,
                }
            )

        interventions = []

        # Real direction full dose-response.
        for alpha in REAL_ALPHAS:
            interventions.append(("real", -1, real, alpha))

        # Same-state control at primary symmetric alpha only.
        for alpha in [-PRIMARY_ALPHA, PRIMARY_ALPHA]:
            interventions.append(("same_state", -1, same, alpha))

        # Multiple orthogonal random controls at primary symmetric alpha.
        for rid, rdir in enumerate(random_dirs):
            for alpha in [-PRIMARY_ALPHA, PRIMARY_ALPHA]:
                interventions.append(("random", rid, rdir, alpha))

        for direction_type, control_id, direction, alpha in interventions:
            scores = score_messages_with_steering(
                model=model,
                tokenizer=tokenizer,
                layer_module=intervention_layer,
                prompt_messages=messages,
                candidate_pairs=candidate_pairs,
                direction=direction,
                alpha=float(alpha),
                batch_size=args.batch_size,
            )

            for idx, entry in enumerate(held_entries):
                margin = (
                    scores[idx][1]["mean_logprob"]
                    - scores[idx][0]["mean_logprob"]
                )
                result_rows.append(
                    {
                        "fold": fold,
                        "task_idx": entry["task_idx"],
                        "task_id": entry["task_id"],
                        "condition": entry["condition"],
                        "wording": entry["wording"],
                        "label": entry["label"],
                        "direction_type": direction_type,
                        "control_id": control_id,
                        "alpha": float(alpha),
                        "test_mean_logprob": scores[idx][0]["mean_logprob"],
                        "impl_mean_logprob": scores[idx][1]["mean_logprob"],
                        "margin_impl_minus_test": margin,
                        "delta_margin_from_baseline": margin - baseline_margin[idx],
                    }
                )

        print(
            f"[fold {fold + 1}/{n_splits}] "
            f"train={len(train_tasks)} tasks, test={len(test_tasks)} tasks, "
            f"||v||={real_norm:.4f}"
        )

    results = pd.DataFrame(result_rows)
    results.to_csv(out_dir / "intervention_results.csv", index=False)
    pd.DataFrame(direction_rows).to_csv(
        out_dir / "fold_directions.csv",
        index=False,
    )

    # ---------------------------------------------------------------------
    # Primary symmetric alpha=1 effects, computed per prompt then per task.
    # E = [M(+1) - M(-1)] / 2
    # ---------------------------------------------------------------------
    def symmetric_effects(direction_type: str, control_id: int = -1):
        subset = results[
            (results["direction_type"] == direction_type)
            & (results["control_id"] == control_id)
            & (results["alpha"].isin([-PRIMARY_ALPHA, PRIMARY_ALPHA]))
        ].copy()

        pivot = subset.pivot_table(
            index=[
                "fold",
                "task_idx",
                "task_id",
                "condition",
                "wording",
                "label",
            ],
            columns="alpha",
            values="margin_impl_minus_test",
            aggfunc="first",
        ).reset_index()

        if -PRIMARY_ALPHA not in pivot.columns or PRIMARY_ALPHA not in pivot.columns:
            raise RuntimeError(
                f"Missing +/-{PRIMARY_ALPHA} results for {direction_type}."
            )

        pivot["symmetric_effect"] = (
            pivot[PRIMARY_ALPHA] - pivot[-PRIMARY_ALPHA]
        ) / 2.0
        pivot["direction_type"] = direction_type
        pivot["control_id"] = control_id
        return pivot

    real_eff = symmetric_effects("real", -1)
    same_eff = symmetric_effects("same_state", -1)

    random_effect_frames = []
    for rid in range(args.num_random_controls):
        random_effect_frames.append(symmetric_effects("random", rid))
    random_eff = pd.concat(random_effect_frames, ignore_index=True)

    primary_all = pd.concat(
        [real_eff, same_eff, random_eff],
        ignore_index=True,
    )

    # Task-level aggregation is the inferential unit.
    task_effects = (
        primary_all.groupby(
            ["direction_type", "control_id", "task_idx", "task_id"],
            as_index=False,
        )["symmetric_effect"]
        .mean()
    )
    task_effects.to_csv(
        out_dir / "primary_effects_by_task.csv",
        index=False,
    )

    real_task = task_effects[
        task_effects["direction_type"] == "real"
    ].set_index("task_idx")["symmetric_effect"]
    same_task = task_effects[
        task_effects["direction_type"] == "same_state"
    ].set_index("task_idx")["symmetric_effect"]

    real_mean, real_lo, real_hi = bootstrap_task_ci(
        real_task,
        seed=args.seed + 10,
    )
    same_mean, same_lo, same_hi = bootstrap_task_ci(
        same_task,
        seed=args.seed + 11,
    )

    # Random controls: one global mean per random direction.
    random_control_summary = (
        task_effects[task_effects["direction_type"] == "random"]
        .groupby("control_id", as_index=False)["symmetric_effect"]
        .mean()
        .rename(columns={"symmetric_effect": "mean_symmetric_effect"})
    )
    random_control_summary.to_csv(
        out_dir / "random_control_effects.csv",
        index=False,
    )

    random_means = random_control_summary["mean_symmetric_effect"].to_numpy()
    random_mean = float(random_means.mean())
    random_std = float(random_means.std())

    # Empirical one-sided random-control p with +1 correction.
    random_empirical_p = float(
        (1 + np.sum(random_means >= real_mean))
        / (1 + len(random_means))
    )

    # Paired task-level difference real - same-state.
    paired = pd.concat(
        [real_task.rename("real"), same_task.rename("same")],
        axis=1,
        join="inner",
    )
    paired["difference"] = paired["real"] - paired["same"]
    diff_mean, diff_lo, diff_hi = bootstrap_task_ci(
        paired["difference"],
        seed=args.seed + 12,
    )

    # ---------------------------------------------------------------------
    # Word-family effects.
    # ---------------------------------------------------------------------
    real_wording = (
        real_eff.groupby(["wording", "task_idx"], as_index=False)[
            "symmetric_effect"
        ].mean()
        .groupby("wording")["symmetric_effect"]
        .agg(["mean", "count"])
        .reset_index()
    )

    # ---------------------------------------------------------------------
    # Dose response for the real direction.
    # Use delta from each prompt's baseline, then average.
    # ---------------------------------------------------------------------
    dose = (
        results[results["direction_type"] == "real"]
        .groupby(["alpha", "wording"], as_index=False)[
            "delta_margin_from_baseline"
        ].mean()
    )
    dose_all = (
        results[results["direction_type"] == "real"]
        .groupby("alpha", as_index=False)["delta_margin_from_baseline"]
        .mean()
    )
    dose["scope"] = dose["wording"]
    dose_all["wording"] = "all"
    dose_all["scope"] = "all"
    dose_out = pd.concat([dose, dose_all], ignore_index=True)
    dose_out.to_csv(out_dir / "dose_response.csv", index=False)

    # Simple monotonic association alpha -> mean delta margin.
    alpha_vals = dose_all["alpha"].to_numpy(np.float64)
    delta_vals = dose_all["delta_margin_from_baseline"].to_numpy(np.float64)
    dose_pearson_r = float(np.corrcoef(alpha_vals, delta_vals)[0, 1])

    # Behavioral flip diagnostics at primary alphas.
    real_primary_rows = results[
        (results["direction_type"] == "real")
        & (results["alpha"].isin([-PRIMARY_ALPHA, PRIMARY_ALPHA]))
    ].copy()
    real_primary_rows["prefers_impl"] = (
        real_primary_rows["margin_impl_minus_test"] > 0
    ).astype(int)

    plus = real_primary_rows[
        real_primary_rows["alpha"] == PRIMARY_ALPHA
    ]
    minus = real_primary_rows[
        real_primary_rows["alpha"] == -PRIMARY_ALPHA
    ]

    plus_impl_rate = float(plus["prefers_impl"].mean())
    minus_impl_rate = float(minus["prefers_impl"].mean())

    # For TEST-first prompts: +v should push toward impl.
    # For IMPL-first prompts: -v should push toward test.
    test_plus_impl_rate = float(
        plus.loc[plus["label"] == LABEL_TEST, "prefers_impl"].mean()
    )
    impl_minus_test_rate = float(
        1.0
        - minus.loc[minus["label"] == LABEL_IMPL, "prefers_impl"].mean()
    )

    # ---------------------------------------------------------------------
    # Plots.
    # ---------------------------------------------------------------------
    fig = plt.figure(figsize=(8.5, 5.5))
    for wording_name in ["canonical", "paraphrase"]:
        d = dose[dose["wording"] == wording_name].sort_values("alpha")
        plt.plot(
            d["alpha"],
            d["delta_margin_from_baseline"],
            marker="o",
            label=wording_name,
        )
    d = dose_all.sort_values("alpha")
    plt.plot(
        d["alpha"],
        d["delta_margin_from_baseline"],
        marker="o",
        linewidth=2.5,
        label="all",
    )
    plt.axhline(0.0, linestyle="--")
    plt.axvline(0.0, linestyle=":")
    plt.xlabel("Steering coefficient α")
    plt.ylabel("Δ [logP(src) - logP(test)] vs baseline")
    plt.title("EXP02 real-direction dose response")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "dose_response.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(7.5, 5.2))
    labels = ["real", "same-state", "random mean"]
    means = [real_mean, same_mean, random_mean]
    plt.bar(labels, means)
    plt.axhline(0.0, linestyle="--")
    plt.ylabel("Primary symmetric steering effect")
    plt.title("EXP02 causal steering: real vs controls")
    plt.tight_layout()
    fig.savefig(out_dir / "primary_effects.png", dpi=180)
    plt.close(fig)

    # ---------------------------------------------------------------------
    # Manifest and summary.
    # ---------------------------------------------------------------------
    script_path = Path(__file__).resolve()
    config_path = model_dir / "config.json"

    manifest = {
        "experiment": "EXP02_causal_state_steering",
        "git_commit": git_commit(),
        "seed": args.seed,
        "source_exp01b_git_commit": exp01b_manifest.get("git_commit"),
        "source_exp01b_num_tasks": num_tasks,
        "source_exp01b_seed": exp01b_seed,
        "hidden_state_index": HIDDEN_STATE_INDEX,
        "decoder_layer_index": DECODER_LAYER_INDEX,
        "primary_alpha": PRIMARY_ALPHA,
        "real_alphas": REAL_ALPHAS,
        "num_random_controls": args.num_random_controls,
        "batch_size": args.batch_size,
        "model_dir": str(model_dir),
        "model_name": model_dir.name,
        "model_config_sha256": (
            sha256_file(config_path) if config_path.is_file() else None
        ),
        "script_sha256": sha256_file(script_path),
        "exp01b_script_sha256": (
            sha256_file(EXP01B_CODE) if EXP01B_CODE.is_file() else None
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
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
        "experiment": "EXP02_causal_state_steering",
        "primary_claim_test": (
            "Does the held-out impl-minus-test representation direction "
            "causally shift impl-vs-test action preference?"
        ),
        "hidden_state_index": HIDDEN_STATE_INDEX,
        "decoder_layer_index": DECODER_LAYER_INDEX,
        "primary_alpha": PRIMARY_ALPHA,
        "num_tasks": num_tasks,
        "real_primary_effect_mean": real_mean,
        "real_primary_effect_bootstrap_95ci": [real_lo, real_hi],
        "same_state_primary_effect_mean": same_mean,
        "same_state_primary_effect_bootstrap_95ci": [same_lo, same_hi],
        "real_minus_same_state_mean": diff_mean,
        "real_minus_same_state_bootstrap_95ci": [diff_lo, diff_hi],
        "random_control_effect_mean": random_mean,
        "random_control_effect_std": random_std,
        "random_control_empirical_one_sided_p": random_empirical_p,
        "real_effect_by_wording": real_wording.to_dict(orient="records"),
        "dose_response_alpha_vs_delta_margin_pearson_r": dose_pearson_r,
        "plus_alpha_impl_choice_rate_all": plus_impl_rate,
        "minus_alpha_impl_choice_rate_all": minus_impl_rate,
        "test_skill_plus_alpha_impl_flip_pressure": test_plus_impl_rate,
        "impl_skill_minus_alpha_test_flip_pressure": impl_minus_test_rate,
        "interpretation_rule": (
            "Positive real effect with CI excluding zero, stronger than "
            "same-state and orthogonal-random controls, plus monotonic dose "
            "response, supports causal steering."
        ),
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== EXP02 SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nResults saved to: {out_dir}")
    print("\nReturn these files:")
    print("  - summary.json")
    print("  - dose_response.csv")
    print("  - primary_effects_by_task.csv")
    print("  - random_control_effects.csv")
    print("  - dose_response.png")
    print("  - primary_effects.png")
    print("  - run_manifest.json")


if __name__ == "__main__":
    main()
