#!/usr/bin/env python3
"""
Minimal MVP for testing whether an Agent Skill induces a decodable
procedural-state representation inside an open-weight LLM.

No real tool execution yet:
- synthetic coding tasks
- teacher-forced gold histories
- paired Skill / NoSkill prompts
- hidden-state extraction at the final prompt token
- task-grouped layer-wise linear probes
"""

from __future__ import annotations

import argparse
import json
import random
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


STAGES = [
    "REPRODUCE",
    "INSPECT_TEST",
    "INSPECT_IMPLEMENTATION",
    "REPAIR",
    "TARGET_VERIFY",
    "REGRESSION_VERIFY",
]

# Important controls:
# - stage 1 and 2 both use read_file
# - stage 0, 4 and 5 all use run_tests
TOOLS = [
    "run_tests",
    "read_file",
    "read_file",
    "edit_file",
    "run_tests",
    "run_tests",
]


def make_tasks(n: int, seed: int) -> list[dict]:
    """Create varied synthetic coding tasks.

    These are deliberately simple. We are testing representation first,
    not coding competence.
    """
    rng = random.Random(seed)

    bug_templates = [
        (
            "boundary comparison",
            "A boundary case is rejected although the test expects it to pass.",
            "the boundary value should be accepted",
        ),
        (
            "wrong default",
            "A function returns the wrong default when the input is empty.",
            "the documented default should be returned",
        ),
        (
            "sign error",
            "A numeric transformation flips the expected sign.",
            "the sign should match the test expectation",
        ),
        (
            "off-by-one",
            "A sequence helper returns one element too few.",
            "the full requested range should be returned",
        ),
        (
            "wrong branch",
            "A conditional selects the fallback branch for a valid input.",
            "the primary branch should be selected",
        ),
        (
            "normalization bug",
            "A text helper fails to normalize a valid input consistently.",
            "the normalized value should match the assertion",
        ),
    ]

    tasks = []
    for i in range(n):
        kind, failure, expected = bug_templates[i % len(bug_templates)]
        pkg = f"pkg_{i:02d}"
        fn = f"feature_{i:02d}"
        test_path = f"tests/test_{fn}.py"
        src_path = f"src/{pkg}/{fn}.py"

        # Vary wording enough to reduce trivial lexical memorization.
        prefixes = [
            "A regression was reported after a small refactor.",
            "CI now reports one deterministic failure.",
            "A previously passing unit test is failing.",
            "A narrow edge case is broken in the current implementation.",
        ]
        rng.shuffle(prefixes)

        tasks.append(
            {
                "task_id": f"task_{i:02d}",
                "kind": kind,
                "issue": f"{prefixes[0]} {failure}",
                "expected": expected,
                "test_path": test_path,
                "src_path": src_path,
                "symbol": fn,
            }
        )
    return tasks


def gold_step(task: dict, stage_idx: int) -> tuple[str, str]:
    """Return a teacher-forced action and observation for one stage."""
    test_path = task["test_path"]
    src_path = task["src_path"]
    symbol = task["symbol"]

    if stage_idx == 0:
        return (
            f"run_tests('{test_path}')",
            f"FAIL: {test_path}::test_{symbol}; observed behavior contradicts: {task['expected']}.",
        )
    if stage_idx == 1:
        return (
            f"read_file('{test_path}')",
            f"The test explicitly asserts that {task['expected']}.",
        )
    if stage_idx == 2:
        return (
            f"read_file('{src_path}')",
            f"The implementation around `{symbol}` contains a small local discrepancy consistent with the failure.",
        )
    if stage_idx == 3:
        return (
            f"edit_file('{src_path}', '<buggy expression>', '<minimal corrected expression>')",
            "Patch applied successfully; only the relevant local expression changed.",
        )
    if stage_idx == 4:
        return (
            f"run_tests('{test_path}')",
            f"PASS: the directly affected test in {test_path} now passes.",
        )
    if stage_idx == 5:
        return (
            "run_tests('ALL')",
            "PASS: full test suite passes with no new failures.",
        )
    raise ValueError(stage_idx)


def detour(task: dict, task_idx: int, stage_idx: int, detour_idx: int) -> tuple[str, str]:
    """Irrelevant but plausible extra tool use to break stage==turn-index."""
    choices = [
        (
            "read_file('README.md')",
            "Project README contains general usage notes but nothing specific to this failure.",
        ),
        (
            "search('TODO')",
            "Found unrelated TODO comments in other modules.",
        ),
        (
            "read_file('pyproject.toml')",
            "Package metadata and test configuration look normal.",
        ),
        (
            "search('deprecated')",
            "No result relevant to the failing symbol.",
        ),
    ]
    idx = (task_idx * 11 + stage_idx * 5 + detour_idx * 3) % len(choices)
    return choices[idx]


def build_history(task: dict, task_idx: int, target_stage: int, seed: int) -> tuple[str, int]:
    """Build history *before* target_stage.

    Random detours are inserted before and between gold stages so the number
    of previous tool calls is not identical to the procedural stage.
    """
    rng = random.Random(seed + task_idx * 1009)

    records: list[str] = []
    tool_calls = 0

    # Random prefix detours, including before stage 0.
    prefix_n = rng.randint(0, 3)
    for j in range(prefix_n):
        action, obs = detour(task, task_idx, 0, j)
        records.append(f"Extra action: {action}\nObservation: {obs}")
        tool_calls += 1

    for completed_stage in range(target_stage):
        action, obs = gold_step(task, completed_stage)
        records.append(
            f"Completed procedural step {completed_stage + 1}:\n"
            f"Action: {action}\nObservation: {obs}"
        )
        tool_calls += 1

        # 0–3 extra calls after each completed stage.
        extra_n = rng.randint(0, 3)
        for j in range(extra_n):
            action, obs = detour(task, task_idx, completed_stage + 1, j)
            records.append(f"Extra action: {action}\nObservation: {obs}")
            tool_calls += 1

    history = "\n\n".join(records) if records else "(No actions have been taken yet.)"
    return history, tool_calls


def build_messages(task: dict, history: str, skill_text: str | None) -> list[dict]:
    system = (
        "You are a coding agent. Given the issue and the execution history, "
        "decide what procedural action should come next. "
        "Do not solve the whole task at once."
    )

    skill_block = ""
    if skill_text is not None:
        skill_block = (
            "\n\nYou have access to the following procedural skill. "
            "Use it as the governing workflow:\n\n"
            "----- SKILL START -----\n"
            f"{skill_text.strip()}\n"
            "----- SKILL END -----\n"
        )

    user = f"""Repository issue:
{task['issue']}

Expected behavior:
{task['expected']}

Relevant paths:
- test: {task['test_path']}
- implementation: {task['src_path']}
{skill_block}
Execution history:
{history}

Choose the next procedural action now.
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


@torch.inference_mode()
def get_hidden_states(model, tokenizer, messages: list[dict]) -> tuple[np.ndarray, int]:
    """Return [num_hidden_states, hidden_size] at final prompt token."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt")

    # Works for ordinary single-device loading and device_map='auto'.
    input_device = model.get_input_embeddings().weight.device
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    out = model(
        **inputs,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    # hidden_states = embedding output + every transformer layer.
    h = torch.stack(
        [layer[0, -1].detach().float().cpu() for layer in out.hidden_states],
        dim=0,
    )
    return h.numpy().astype(np.float16), int(inputs["input_ids"].shape[-1])


def grouped_probe_scores(
    X_by_layer: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> list[tuple[float, float]]:
    """Task-grouped CV, one linear probe per layer."""
    cv = GroupKFold(n_splits=n_splits)
    results = []

    for layer in range(X_by_layer.shape[1]):
        fold_scores = []
        X = X_by_layer[:, layer, :].astype(np.float32)

        for train_idx, test_idx in cv.split(X, y, groups):
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    C=1.0,
                ),
            )
            clf.fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[test_idx])
            fold_scores.append(f1_score(y[test_idx], pred, average="macro"))

        results.append((float(np.mean(fold_scores)), float(np.std(fold_scores))))
    return results


def grouped_predictions(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> np.ndarray:
    """Out-of-fold predictions for a fixed feature matrix."""
    cv = GroupKFold(n_splits=n_splits)
    pred = np.empty_like(y)

    for train_idx, test_idx in cv.split(X, y, groups):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                C=1.0,
            ),
        )
        clf.fit(X[train_idx], y[train_idx])
        pred[test_idx] = clf.predict(X[test_idx])
    return pred


def pair_probe(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    a: int,
    b: int,
    n_splits: int,
) -> float:
    mask = np.isin(y, [a, b])
    X2 = X[mask]
    y2 = (y[mask] == b).astype(int)
    g2 = groups[mask]
    pred = grouped_predictions(X2, y2, g2, n_splits)
    return float(f1_score(y2, pred, average="macro"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
    )
    parser.add_argument("--num-tasks", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="outputs/mvp")
    parser.add_argument(
        "--skill",
        default="skills/debugging/SKILL.md",
    )
    args = parser.parse_args()

    if args.num_tasks < 8:
        raise ValueError("--num-tasks should be at least 8 for grouped CV.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    skill_text = Path(args.skill).read_text(encoding="utf-8")
    tasks = make_tasks(args.num_tasks, args.seed)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    h_skill_list = []
    h_base_list = []
    rows = []

    total = len(tasks) * len(STAGES)
    done = 0

    for task_idx, task in enumerate(tasks):
        for stage_idx, stage_name in enumerate(STAGES):
            history, tool_calls = build_history(
                task=task,
                task_idx=task_idx,
                target_stage=stage_idx,
                seed=args.seed,
            )

            base_messages = build_messages(task, history, skill_text=None)
            skill_messages = build_messages(task, history, skill_text=skill_text)

            h_base, base_tokens = get_hidden_states(model, tokenizer, base_messages)
            h_skill, skill_tokens = get_hidden_states(model, tokenizer, skill_messages)

            if h_base.shape != h_skill.shape:
                raise RuntimeError(
                    f"Hidden-state shape mismatch: {h_base.shape} vs {h_skill.shape}"
                )

            h_base_list.append(h_base)
            h_skill_list.append(h_skill)

            rows.append(
                {
                    "task_id": task["task_id"],
                    "task_idx": task_idx,
                    "stage_idx": stage_idx,
                    "stage": stage_name,
                    "next_tool": TOOLS[stage_idx],
                    "history_tool_calls": tool_calls,
                    "base_tokens": base_tokens,
                    "skill_tokens": skill_tokens,
                    "bug_kind": task["kind"],
                }
            )

            done += 1
            print(f"[{done:03d}/{total:03d}] {task['task_id']} / {stage_name}")

    h_base = np.stack(h_base_list, axis=0)
    h_skill = np.stack(h_skill_list, axis=0)
    h_delta = (h_skill.astype(np.float32) - h_base.astype(np.float32)).astype(np.float16)

    metadata = pd.DataFrame(rows)
    metadata.to_csv(out_dir / "metadata.csv", index=False)

    np.savez_compressed(
        out_dir / "activations.npz",
        h_base=h_base,
        h_skill=h_skill,
        h_delta=h_delta,
    )

    y = metadata["stage_idx"].to_numpy(dtype=np.int64)
    groups = metadata["task_idx"].to_numpy(dtype=np.int64)
    n_splits = min(4, args.num_tasks)

    print("Running task-grouped layer-wise probes...")
    base_scores = grouped_probe_scores(h_base, y, groups, n_splits)
    skill_scores = grouped_probe_scores(h_skill, y, groups, n_splits)
    delta_scores = grouped_probe_scores(h_delta, y, groups, n_splits)

    probe_rows = []
    for layer in range(h_base.shape[1]):
        probe_rows.append(
            {
                "layer": layer,
                "no_skill_f1": base_scores[layer][0],
                "no_skill_f1_std": base_scores[layer][1],
                "skill_f1": skill_scores[layer][0],
                "skill_f1_std": skill_scores[layer][1],
                "delta_f1": delta_scores[layer][0],
                "delta_f1_std": delta_scores[layer][1],
            }
        )
    probe_df = pd.DataFrame(probe_rows)
    probe_df.to_csv(out_dir / "layer_probe.csv", index=False)

    # Position/length baseline.
    X_pos = metadata[
        ["history_tool_calls", "base_tokens"]
    ].to_numpy(dtype=np.float32)
    pos_pred = grouped_predictions(X_pos, y, groups, n_splits)
    pos_f1 = float(f1_score(y, pos_pred, average="macro"))
    pd.DataFrame(
        [{"position_baseline_macro_f1": pos_f1}]
    ).to_csv(out_dir / "position_baseline.csv", index=False)

    # Best delta layer.
    best_row = probe_df.loc[probe_df["delta_f1"].idxmax()]
    best_layer = int(best_row["layer"])
    X_best_delta = h_delta[:, best_layer, :].astype(np.float32)

    best_pred = grouped_predictions(X_best_delta, y, groups, n_splits)
    cm = confusion_matrix(y, best_pred, labels=np.arange(len(STAGES)))
    cm_df = pd.DataFrame(cm, index=STAGES, columns=STAGES)
    cm_df.to_csv(out_dir / "confusion_matrix_best_layer.csv")

    # Same-tool controls:
    # INSPECT_TEST vs INSPECT_IMPLEMENTATION: read_file vs read_file
    # TARGET_VERIFY vs REGRESSION_VERIFY: run_tests vs run_tests
    same_tool_rows = [
        {
            "pair": "INSPECT_TEST_vs_INSPECT_IMPLEMENTATION",
            "tool": "read_file",
            "macro_f1": pair_probe(
                X_best_delta, y, groups, 1, 2, n_splits
            ),
        },
        {
            "pair": "TARGET_VERIFY_vs_REGRESSION_VERIFY",
            "tool": "run_tests",
            "macro_f1": pair_probe(
                X_best_delta, y, groups, 4, 5, n_splits
            ),
        },
    ]
    same_tool_df = pd.DataFrame(same_tool_rows)
    same_tool_df.to_csv(out_dir / "same_tool_results.csv", index=False)

    # Plot.
    fig = plt.figure(figsize=(8, 5))
    plt.plot(probe_df["layer"], probe_df["no_skill_f1"], label="NoSkill")
    plt.plot(probe_df["layer"], probe_df["skill_f1"], label="Skill")
    plt.plot(probe_df["layer"], probe_df["delta_f1"], label="Skill - NoSkill")
    plt.axhline(1.0 / len(STAGES), linestyle="--", label="Chance")
    plt.xlabel("Hidden-state index (0 = embedding output)")
    plt.ylabel("Task-grouped CV Macro-F1")
    plt.title("Procedural-stage decodability by layer")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "layer_probe.png", dpi=180)
    plt.close(fig)

    summary = {
        "model": args.model,
        "num_tasks": args.num_tasks,
        "num_examples": int(len(metadata)),
        "num_hidden_states": int(h_base.shape[1]),
        "hidden_size": int(h_base.shape[2]),
        "best_layer": best_layer,
        "best_delta_macro_f1": float(best_row["delta_f1"]),
        "best_skill_macro_f1": float(
            probe_df.loc[probe_df["skill_f1"].idxmax(), "skill_f1"]
        ),
        "best_no_skill_macro_f1": float(
            probe_df.loc[probe_df["no_skill_f1"].idxmax(), "no_skill_f1"]
        ),
        "position_baseline_macro_f1": pos_f1,
        "chance_macro_f1": 1.0 / len(STAGES),
        "same_tool": same_tool_rows,
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== MVP SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved results to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
