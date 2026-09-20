#!/usr/bin/env python3
from __future__ import annotations

# Hard offline mode: this experiment must never fetch model/tokenizer files.
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import argparse
import json
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = REPO_ROOT / "models"
SKILL_PATH = REPO_ROOT / "skills" / "debugging" / "SKILL.md"
DEFAULT_OUT = REPO_ROOT / "outputs" / "exp01_stage_probe"

STAGES = [
    "REPRODUCE",
    "INSPECT_TEST",
    "INSPECT_IMPLEMENTATION",
    "REPAIR",
    "TARGET_VERIFY",
    "REGRESSION_VERIFY",
]

# Deliberate same-tool controls.
NEXT_TOOLS = [
    "run_tests",
    "read_file",
    "read_file",
    "edit_file",
    "run_tests",
    "run_tests",
]


def resolve_model_dir(model_arg: str | None) -> Path:
    """Resolve a Hugging Face-format model directory using local files only."""
    if model_arg:
        p = Path(model_arg).expanduser()
        if not p.is_absolute():
            # Prefer repository-relative path; then models/<name>.
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
            raise FileNotFoundError(
                f"Model directory does not contain config.json: {p}"
            )
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
            "No local model found.\n"
            f"Put a Hugging Face-format model under: {MODELS_ROOT}\n"
            "Example: models/Qwen2.5-Coder-7B-Instruct/config.json"
        )
    if len(candidates) > 1:
        listing = "\n".join(f"  - {p}" for p in candidates)
        raise RuntimeError(
            "Multiple local models found. Select one with --model:\n" + listing
        )
    return candidates[0]


def make_shuffled_skill(skill_text: str) -> str:
    """Reorder Skill sections while preserving the exact section text.

    This creates a content/length-matched procedural-order corruption.
    """
    match = re.search(r"(?m)^##\s+1\.", skill_text)
    if not match:
        raise ValueError("SKILL.md must contain headings like '## 1. REPRODUCE'.")

    prefix = skill_text[: match.start()]
    blocks = re.findall(
        r"(?ms)^##\s+\d+\.\s+[A-Z_]+\s*\n.*?(?=^##\s+\d+\.|\Z)",
        skill_text[match.start():],
    )
    if len(blocks) != 6:
        raise ValueError(f"Expected 6 Skill sections, found {len(blocks)}.")

    # Fixed, intentionally wrong order. No randomization across runs.
    permutation = [3, 0, 5, 2, 4, 1]
    return prefix + "\n".join(blocks[i].rstrip() for i in permutation) + "\n"


def make_tasks(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)

    templates = [
        (
            "boundary comparison",
            "A boundary case is rejected although the test expects it to pass.",
            "the boundary value should be accepted",
        ),
        (
            "wrong default",
            "A function returns the wrong default for an empty input.",
            "the documented default should be returned",
        ),
        (
            "sign error",
            "A numeric transformation returns a value with the wrong sign.",
            "the sign should match the test expectation",
        ),
        (
            "off-by-one",
            "A sequence helper returns one element too few.",
            "the full requested range should be returned",
        ),
        (
            "wrong branch",
            "A valid input incorrectly takes the fallback branch.",
            "the primary branch should be selected",
        ),
        (
            "normalization bug",
            "A text helper normalizes a valid input inconsistently.",
            "the normalized value should match the assertion",
        ),
    ]

    prefixes = [
        "CI reports one deterministic regression.",
        "A previously passing unit test is now failing.",
        "A narrow edge case is broken after a refactor.",
        "One reproducible test failure remains in the repository.",
    ]

    tasks = []
    for i in range(n):
        kind, failure, expected = templates[i % len(templates)]
        prefix = prefixes[(i * 7 + rng.randrange(len(prefixes))) % len(prefixes)]
        symbol = f"feature_{i:02d}"
        package = f"pkg_{i:02d}"
        tasks.append(
            {
                "task_id": f"task_{i:02d}",
                "kind": kind,
                "issue": f"{prefix} {failure}",
                "expected": expected,
                "test_path": f"tests/test_{symbol}.py",
                "src_path": f"src/{package}/{symbol}.py",
                "symbol": symbol,
            }
        )
    return tasks


def gold_step(task: dict, stage_idx: int) -> tuple[str, str]:
    t = task["test_path"]
    s = task["src_path"]
    fn = task["symbol"]

    steps = [
        (
            f"run_tests('{t}')",
            f"FAIL: {t}::test_{fn}; observed behavior contradicts the expected behavior.",
        ),
        (
            f"read_file('{t}')",
            f"The test asserts that {task['expected']}.",
        ),
        (
            f"read_file('{s}')",
            f"The implementation of `{fn}` contains a small local discrepancy consistent with the failure.",
        ),
        (
            f"edit_file('{s}', '<buggy expression>', '<minimal corrected expression>')",
            "The minimal local patch was applied successfully.",
        ),
        (
            f"run_tests('{t}')",
            "PASS: the directly affected test now passes.",
        ),
        (
            "run_tests('ALL')",
            "PASS: the complete test suite passes with no new failures.",
        ),
    ]
    return steps[stage_idx]


def detour(task_idx: int, stage_idx: int, detour_idx: int) -> tuple[str, str]:
    choices = [
        (
            "read_file('README.md')",
            "The README contains general usage notes but nothing specific to this failure.",
        ),
        (
            "search('TODO')",
            "The search returns unrelated TODO comments in other modules.",
        ),
        (
            "read_file('pyproject.toml')",
            "Package metadata and test configuration look normal.",
        ),
        (
            "search('deprecated')",
            "No result is relevant to the failing symbol.",
        ),
    ]
    return choices[(task_idx * 11 + stage_idx * 5 + detour_idx * 3) % len(choices)]


def build_history(
    task: dict,
    task_idx: int,
    target_stage: int,
    seed: int,
) -> tuple[str, int]:
    """History before target_stage, with detours that break stage==turn index.

    Stage names/numbers are intentionally NOT written into the history.
    """
    rng = random.Random(seed + task_idx * 1009)
    records: list[str] = []
    tool_calls = 0

    prefix_n = rng.randint(0, 2)
    for j in range(prefix_n):
        action, obs = detour(task_idx, 0, j)
        records.append(f"Action: {action}\nObservation: {obs}")
        tool_calls += 1

    for completed_stage in range(target_stage):
        action, obs = gold_step(task, completed_stage)
        records.append(f"Action: {action}\nObservation: {obs}")
        tool_calls += 1

        extra_n = rng.randint(0, 2)
        for j in range(extra_n):
            action, obs = detour(task_idx, completed_stage + 1, j)
            records.append(f"Action: {action}\nObservation: {obs}")
            tool_calls += 1

    history = "\n\n".join(records) if records else "(No actions have been taken yet.)"
    return history, tool_calls


def build_messages(
    task: dict,
    history: str,
    skill_text: str | None,
) -> list[dict]:
    system = (
        "You are a coding agent. Decide the single next action from the issue "
        "and execution history. Do not solve the whole task at once."
    )

    if skill_text is None:
        skill_block = ""
    else:
        skill_block = (
            "\n\nUse the following workflow guidance while deciding the next action:\n"
            "----- SKILL START -----\n"
            f"{skill_text.strip()}\n"
            "----- SKILL END -----\n"
        )

    user = f"""Repository issue:
{task['issue']}

Expected behavior:
{task['expected']}

Potentially relevant paths:
- {task['test_path']}
- {task['src_path']}
{skill_block}
Execution history:
{history}

Decide the next action.
"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


@torch.inference_mode()
def extract_batch(
    model,
    tokenizer,
    batch_messages: list[list[dict]],
) -> tuple[np.ndarray, list[int]]:
    """Extract final-prompt-token hidden states.

    Returns:
        hidden: [batch, num_hidden_states, hidden_size], float16 CPU numpy
        lengths: unpadded token counts
    """
    texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for messages in batch_messages
    ]

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )

    attention_mask = encoded["attention_mask"]
    lengths = attention_mask.sum(dim=1).tolist()
    last_idx = attention_mask.sum(dim=1) - 1

    device = model.get_input_embeddings().weight.device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    last_idx = last_idx.to(device)

    out = model(
        **encoded,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    batch_idx = torch.arange(len(batch_messages), device=device)
    per_layer = []
    for layer_h in out.hidden_states:
        # [batch, seq, hidden] -> [batch, hidden]
        selected = layer_h[batch_idx, last_idx]
        per_layer.append(selected.detach().float().cpu())

    hidden = torch.stack(per_layer, dim=1)
    return hidden.numpy().astype(np.float16), [int(x) for x in lengths]


def make_probe():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            C=0.5,
        ),
    )


def grouped_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> np.ndarray:
    cv = GroupKFold(n_splits=n_splits)
    pred = np.empty_like(y)
    for train_idx, test_idx in cv.split(X, y, groups):
        clf = make_probe()
        clf.fit(X[train_idx], y[train_idx])
        pred[test_idx] = clf.predict(X[test_idx])
    return pred


def grouped_layer_scores(
    X_by_layer: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> list[tuple[float, float]]:
    cv = GroupKFold(n_splits=n_splits)
    rows = []

    for layer in range(X_by_layer.shape[1]):
        X = X_by_layer[:, layer, :].astype(np.float32)
        fold_scores = []
        for train_idx, test_idx in cv.split(X, y, groups):
            clf = make_probe()
            clf.fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[test_idx])
            fold_scores.append(
                f1_score(y[test_idx], pred, average="macro")
            )
        rows.append((float(np.mean(fold_scores)), float(np.std(fold_scores))))
    return rows


def binary_pair_score(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    stage_a: int,
    stage_b: int,
    n_splits: int,
) -> float:
    mask = np.isin(y, [stage_a, stage_b])
    X2 = X[mask]
    y2 = (y[mask] == stage_b).astype(np.int64)
    g2 = groups[mask]
    pred = grouped_oof_predictions(X2, y2, g2, n_splits)
    return float(f1_score(y2, pred, average="macro"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=None,
        help="Local model dir. If omitted, auto-detect one model under ./models.",
    )
    parser.add_argument("--num-tasks", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    if args.num_tasks < 8:
        raise ValueError("--num-tasks must be >= 8 for grouped cross-validation.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_dir = resolve_model_dir(args.model)
    out_dir = Path(args.out).expanduser()
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    correct_skill = SKILL_PATH.read_text(encoding="utf-8")
    shuffled_skill = make_shuffled_skill(correct_skill)

    print("=" * 72)
    print("EXP01: Procedural Stage Probe")
    print(f"Repository root : {REPO_ROOT}")
    print(f"Local model     : {model_dir}")
    print(f"Output dir      : {out_dir}")
    print("Network mode    : OFFLINE / local_files_only=True")
    print("=" * 72)

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

    tasks = make_tasks(args.num_tasks, args.seed)

    no_skill_all = []
    correct_all = []
    shuffled_all = []
    rows = []

    total = len(tasks) * len(STAGES)
    count = 0

    for task_idx, task in enumerate(tasks):
        for stage_idx, stage_name in enumerate(STAGES):
            history, history_tool_calls = build_history(
                task=task,
                task_idx=task_idx,
                target_stage=stage_idx,
                seed=args.seed,
            )

            conditions = [
                build_messages(task, history, None),
                build_messages(task, history, correct_skill),
                build_messages(task, history, shuffled_skill),
            ]

            hidden, lengths = extract_batch(model, tokenizer, conditions)
            no_skill_all.append(hidden[0])
            correct_all.append(hidden[1])
            shuffled_all.append(hidden[2])

            rows.append(
                {
                    "task_id": task["task_id"],
                    "task_idx": task_idx,
                    "bug_kind": task["kind"],
                    "stage_idx": stage_idx,
                    "stage": stage_name,
                    "next_tool": NEXT_TOOLS[stage_idx],
                    "history_tool_calls": history_tool_calls,
                    "history_chars": len(history),
                    "no_skill_tokens": lengths[0],
                    "correct_skill_tokens": lengths[1],
                    "shuffled_skill_tokens": lengths[2],
                }
            )

            count += 1
            print(f"[{count:03d}/{total:03d}] {task['task_id']} / {stage_name}")

    h_no = np.stack(no_skill_all)
    h_correct = np.stack(correct_all)
    h_shuffled = np.stack(shuffled_all)

    # Main contrast: same Skill content, wrong vs correct procedure order.
    h_order_delta = (
        h_correct.astype(np.float32) - h_shuffled.astype(np.float32)
    ).astype(np.float16)

    # Secondary contrast: Skill presence.
    h_presence_delta = (
        h_correct.astype(np.float32) - h_no.astype(np.float32)
    ).astype(np.float16)

    metadata = pd.DataFrame(rows)
    metadata.to_csv(out_dir / "metadata.csv", index=False)

    np.savez_compressed(
        out_dir / "activations.npz",
        h_no_skill=h_no,
        h_correct_skill=h_correct,
        h_shuffled_skill=h_shuffled,
        h_order_delta=h_order_delta,
        h_presence_delta=h_presence_delta,
    )

    y = metadata["stage_idx"].to_numpy(np.int64)
    groups = metadata["task_idx"].to_numpy(np.int64)
    n_splits = min(4, args.num_tasks)

    print("Running task-grouped layer-wise linear probes...")
    no_scores = grouped_layer_scores(h_no, y, groups, n_splits)
    correct_scores = grouped_layer_scores(h_correct, y, groups, n_splits)
    shuffled_scores = grouped_layer_scores(h_shuffled, y, groups, n_splits)
    order_delta_scores = grouped_layer_scores(h_order_delta, y, groups, n_splits)

    probe_rows = []
    for layer in range(h_no.shape[1]):
        probe_rows.append(
            {
                "layer": layer,
                "no_skill_f1": no_scores[layer][0],
                "no_skill_std": no_scores[layer][1],
                "correct_skill_f1": correct_scores[layer][0],
                "correct_skill_std": correct_scores[layer][1],
                "shuffled_skill_f1": shuffled_scores[layer][0],
                "shuffled_skill_std": shuffled_scores[layer][1],
                "order_delta_f1": order_delta_scores[layer][0],
                "order_delta_std": order_delta_scores[layer][1],
            }
        )

    probe_df = pd.DataFrame(probe_rows)
    probe_df.to_csv(out_dir / "layer_probe.csv", index=False)

    # Position/history-length baseline.
    X_pos = metadata[
        [
            "history_tool_calls",
            "history_chars",
            "no_skill_tokens",
        ]
    ].to_numpy(np.float32)
    pos_pred = grouped_oof_predictions(X_pos, y, groups, n_splits)
    pos_f1 = float(f1_score(y, pos_pred, average="macro"))
    pd.DataFrame(
        [{"position_baseline_macro_f1": pos_f1}]
    ).to_csv(out_dir / "position_baseline.csv", index=False)

    # Best layer from the main content-matched order contrast.
    best_idx = int(probe_df["order_delta_f1"].idxmax())
    best_row = probe_df.loc[best_idx]
    best_layer = int(best_row["layer"])
    X_best = h_order_delta[:, best_layer, :].astype(np.float32)

    best_pred = grouped_oof_predictions(X_best, y, groups, n_splits)
    cm = confusion_matrix(y, best_pred, labels=np.arange(len(STAGES)))
    pd.DataFrame(
        cm,
        index=STAGES,
        columns=STAGES,
    ).to_csv(out_dir / "confusion_matrix_best_layer.csv")

    same_tool = [
        {
            "pair": "INSPECT_TEST_vs_INSPECT_IMPLEMENTATION",
            "shared_tool": "read_file",
            "macro_f1": binary_pair_score(
                X_best, y, groups, 1, 2, n_splits
            ),
        },
        {
            "pair": "TARGET_VERIFY_vs_REGRESSION_VERIFY",
            "shared_tool": "run_tests",
            "macro_f1": binary_pair_score(
                X_best, y, groups, 4, 5, n_splits
            ),
        },
    ]
    pd.DataFrame(same_tool).to_csv(
        out_dir / "same_tool_results.csv",
        index=False,
    )

    chance = 1.0 / len(STAGES)

    fig = plt.figure(figsize=(9, 5.5))
    plt.plot(
        probe_df["layer"],
        probe_df["no_skill_f1"],
        label="No skill",
    )
    plt.plot(
        probe_df["layer"],
        probe_df["correct_skill_f1"],
        label="Correct skill",
    )
    plt.plot(
        probe_df["layer"],
        probe_df["shuffled_skill_f1"],
        label="Shuffled skill",
    )
    plt.plot(
        probe_df["layer"],
        probe_df["order_delta_f1"],
        label="Correct - Shuffled",
        linewidth=2.2,
    )
    plt.axhline(chance, linestyle="--", label="Chance")
    plt.axhline(pos_f1, linestyle=":", label="Position baseline")
    plt.xlabel("Hidden-state index (0 = embedding output)")
    plt.ylabel("Task-grouped CV Macro-F1")
    plt.title("Procedural-stage decodability by layer")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "layer_probe.png", dpi=180)
    plt.close(fig)

    summary = {
        "experiment": "exp01_stage_probe",
        "model_dir": str(model_dir),
        "model_name": model_dir.name,
        "num_tasks": args.num_tasks,
        "num_stage_examples": int(len(metadata)),
        "num_hidden_states": int(h_no.shape[1]),
        "hidden_size": int(h_no.shape[2]),
        "chance_macro_f1": chance,
        "best_layer": best_layer,
        "best_order_delta_macro_f1": float(best_row["order_delta_f1"]),
        "best_correct_skill_macro_f1": float(
            probe_df["correct_skill_f1"].max()
        ),
        "best_shuffled_skill_macro_f1": float(
            probe_df["shuffled_skill_f1"].max()
        ),
        "best_no_skill_macro_f1": float(
            probe_df["no_skill_f1"].max()
        ),
        "position_baseline_macro_f1": pos_f1,
        "same_tool_results": same_tool,
        "primary_contrast": "correct_skill - shuffled_skill",
        "offline_only": True,
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== EXP01 SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
