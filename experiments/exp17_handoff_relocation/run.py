#!/usr/bin/env python3
"""EXP17 — Dialogue-Handoff Writer Relocation Falsification.

All model components are frozen from EXP13/15/16.  EXP17 manipulates only
prompt topology and asks where the causal V/KV writer follows.
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import argparse
import gc
import hashlib
import importlib.util
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP14_CODE = REPO_ROOT / "experiments" / "exp14_cross_model_homolog" / "run.py"
EXP15_CODE = REPO_ROOT / "experiments" / "exp15_cross_arch_homolog" / "run.py"
EXP13_SELECTED = REPO_ROOT / "outputs" / "exp13_contextual_write_remap" / "selected_schema.json"
DEFAULT_OUT = REPO_ROOT / "outputs" / "exp17_handoff_relocation"

SEED = 5717
ANCHOR_WIDTH = 6
N_BOOT = 5000

CONDITIONS = ("native", "suffix_short", "suffix_long", "instruction_early")
SITES = (
    "FINAL_INSTRUCTION_END",
    "USER_CONTENT_END",
    "TEMPLATE_TERMINATOR",
    "GENERATION_BOUNDARY",
    "NATIVE_ABSOLUTE",
)
PRIMARY_READER_SITES = (
    "GENERATION_BOUNDARY",
    "FINAL_INSTRUCTION_END",
    "NATIVE_ABSOLUTE",
)
CONTROL_CONDITIONS = ("native", "suffix_long")

SHORT_SUFFIX = "Reference note: protocol marker R17."
LONG_SUFFIX = (
    "Reference note: protocol marker R17. "
    "Session metadata: channel standard; priority normal; checksum stable."
)

FAMILIES = ("test_edit", "search_edit", "config_command", "docs_code")

MODULES = (
    "indexer", "normalizer", "rewriter", "backoff",
    "snapshot", "merger", "compiler", "gateway",
    "tracker", "allocator", "sanitizer", "matcher",
)
SYMBOLS = (
    "index_record", "normalize_event", "rewrite_path", "apply_backoff",
    "capture_snapshot", "merge_state", "compile_rule", "route_gateway",
    "track_update", "allocate_slot", "sanitize_entry", "match_pattern",
)
PACKAGES = (
    "anyio", "orjson", "msgspec", "tenacity",
    "uvicorn", "networkx", "pyyaml", "aiohttp",
    "structlog", "cachetools", "jsonschema", "cattrs",
)
APIS = (
    "create_task_group", "loads", "Struct", "retry",
    "Server.run", "DiGraph.add_edge", "safe_load", "ClientSession.get",
    "get_logger", "TTLCache", "validate", "structure",
)
COMMANDS = (
    "python -m app.inspect_indexer",
    "python -m app.check_normalizer",
    "python -m app.trace_rewriter",
    "python -m app.verify_backoff",
    "python -m app.inspect_snapshot",
    "python -m app.check_merger",
    "python -m app.trace_compiler",
    "python -m app.verify_gateway",
    "python -m app.inspect_tracker",
    "python -m app.check_allocator",
    "python -m app.trace_sanitizer",
    "python -m app.verify_matcher",
)

MODEL_SPECS = {
    "qwen": {
        "name": "Qwen2.5-Coder-7B-Instruct",
        "path_hints": ("Qwen2.5-Coder-7B-Instruct", "qwen2.5-coder-7b-instruct"),
        "model_type_contains": "qwen2",
        "L": 20,
        "k": 0,
        "positive_readers": (0, 3, 5),
        "negative_readers": (2, 4, 6),
        "native_anchor_source": "exp13",
    },
    "mistral": {
        "name": "Mistral-7B-Instruct-v0.3",
        "path_hints": ("Mistral-7B-Instruct-v0.3", "mistral-7b-instruct-v0.3"),
        "model_type_contains": "mistral",
        "L": 30,
        "k": 4,
        # EXP15 Q0 was an exact-zero quota tie; Q16/Q18 were actually positive.
        "positive_readers": (16, 18),
        "negative_readers": (17, 19),
        "native_anchor_source": "generation_boundary",
    },
    "granite": {
        "name": "Granite-3.0-8B-Instruct",
        "path_hints": ("granite-3.0-8b-instruct", "Granite-3.0-8B-Instruct"),
        "model_type_contains": "granite",
        "L": 39,
        "k": 2,
        "positive_readers": (10, 11),
        "negative_readers": (),
        "native_anchor_source": "generation_boundary",
    },
}

SEARCH_ROOTS = (
    REPO_ROOT / "models",
    Path("/data/mzb/ar2_scratch/models"),
    Path("/data/mzb/skills4s/models"),
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def find_model_dir(model_id: str, override: str | None):
    if override:
        p = Path(override).expanduser().resolve()
        if not (p / "config.json").is_file():
            raise FileNotFoundError(f"Missing config.json: {p}")
        return p

    spec = MODEL_SPECS[model_id]
    candidates = []
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for cfg in root.rglob("config.json"):
            p = cfg.parent.resolve()
            low = str(p).lower()
            if any(h.lower() in low for h in spec["path_hints"]):
                candidates.append(p)

    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise RuntimeError(
            f"{model_id}: expected one local checkpoint matching "
            f"{spec['path_hints']}, found {candidates}"
        )
    return candidates[0]


def make_task(family: str, idx: int):
    module = MODULES[idx]
    symbol = SYMBOLS[idx]
    package = PACKAGES[idx]
    api = APIS[idx]
    command = COMMANDS[idx]
    case = f"{module}_handoff_{idx:02d}"

    if family == "test_edit":
        action0 = f'run_tests("tests/test_{module}.py::test_{case}")'
        action1 = f'open_file("src/{module}.py")'
        issue = (
            f"A newly reported regression in `{module}` appears in the focused "
            f"`{case}` scenario. Both the target test and implementation are available."
        )
        detail = f"The suspected implementation entry point is `{symbol}`."
    elif family == "search_edit":
        action0 = f'search_code("{symbol}")'
        action1 = f'open_file("src/{module}.py")'
        issue = (
            f"A maintenance change around `{symbol}` may affect several callers. "
            "Repository-wide references and the target source are both available."
        )
        detail = f"The primary implementation file is `src/{module}.py`."
    elif family == "config_command":
        action0 = f'inspect_config("config/{module}.toml")'
        action1 = f'run_command("{command}")'
        issue = (
            f"The `{module}` subsystem has an environment-dependent failure. "
            "A local config and a narrow diagnostic command are both available."
        )
        detail = f"The diagnostic concerns `{symbol}`."
    elif family == "docs_code":
        action0 = f'search_docs("{package}.{api}")'
        action1 = f'open_file("src/{module}.py")'
        issue = (
            f"An integration with `{package}` may depend on a version-sensitive API. "
            "The external reference and the local implementation are both available."
        )
        detail = f"The project integration point is `{symbol}`."
    else:
        raise ValueError(family)

    return {
        "task_id": f"exp17_{family}_{idx:02d}",
        "family": family,
        "index": idx,
        "module": module,
        "symbol": symbol,
        "package": package,
        "api": api,
        "command": command,
        "action0": action0,
        "action1": action1,
        "issue": issue,
        "detail": detail,
    }


def make_tasks():
    return [make_task(f, i) for f in FAMILIES for i in range(12)]


def task_partition(task):
    return "confirm" if int(task["index"]) < 8 else "reserve"


def build_user_content(task, condition):
    prefix = (
        "Repository task:\n"
        f"{task['issue']}\n\n"
        f"Project detail:\n{task['detail']}\n\n"
    )
    actions = (
        "Available next actions:\n"
        f"- {task['action0']}\n"
        f"- {task['action1']}"
    )
    final_instruction = "Choose the single next action now."

    if condition == "native":
        return f"{prefix}{actions}\n\n{final_instruction}"
    if condition == "suffix_short":
        return f"{prefix}{actions}\n\n{final_instruction}\n\n{SHORT_SUFFIX}"
    if condition == "suffix_long":
        return f"{prefix}{actions}\n\n{final_instruction}\n\n{LONG_SUFFIX}"
    if condition == "instruction_early":
        return f"{prefix}{final_instruction}\n\n{actions}\n\n{LONG_SUFFIX}"
    raise ValueError(condition)


def skill_messages(exp12, task, label, wording, condition):
    system = (
        "You are an autonomous software-engineering agent.\n\n"
        "Workflow Skill:\n"
        f"{exp12.SKILL_TEXT[task['family']][label][wording]}\n\n"
        "Follow the Workflow Skill exactly when deciding the first action."
    )
    user = build_user_content(task, condition)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], user


def direct_messages(task, label, condition):
    required = task["action1"] if label == 1 else task["action0"]
    system = (
        "You are an autonomous software-engineering agent.\n\n"
        "For this task, do not infer or apply a workflow. "
        "The required next action is explicitly fixed as:\n"
        f"{required}\n\n"
        "Choose that required action."
    )
    user = build_user_content(task, condition)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], user


def make_skill_entries(exp12, task, condition):
    out = []
    for label in (0, 1):
        for wording in ("canonical", "paraphrase"):
            messages, user = skill_messages(exp12, task, label, wording, condition)
            out.append({
                "task_id": task["task_id"],
                "family": task["family"],
                "label": label,
                "wording": wording,
                "condition": condition,
                "messages": messages,
                "user_content": user,
                "candidates": [task["action0"], task["action1"]],
            })
    return out


def make_direct_entries(task, condition):
    out = []
    for label in (0, 1):
        messages, user = direct_messages(task, label, condition)
        out.append({
            "task_id": task["task_id"],
            "family": task["family"],
            "label": label,
            "wording": "direct",
            "condition": condition,
            "messages": messages,
            "user_content": user,
            "candidates": [task["action0"], task["action1"]],
        })
    return out


def donor_maps_skill(entries):
    lookup = {(e["wording"], int(e["label"])): i for i, e in enumerate(entries)}
    out = {}
    for i, e in enumerate(entries):
        opp = 1 - int(e["label"])
        other = "paraphrase" if e["wording"] == "canonical" else "canonical"
        out[i] = {
            "opposite_same_wording": lookup[(e["wording"], opp)],
            "opposite_cross_wording": lookup[(other, opp)],
            "same_state_cross_wording": lookup[(other, int(e["label"]))],
        }
    return out


def fixed_end_window(end_idx, n, width=ANCHOR_WIDTH):
    start = int(end_idx) - width + 1
    if start < 0:
        raise RuntimeError("Anchor too close to prompt start")
    p = list(range(start, int(end_idx) + 1))
    if max(p) >= n or len(p) != width:
        raise RuntimeError("Invalid fixed-end window")
    return p


def token_span_for_text(text, offsets, segment):
    start = text.rfind(segment)
    if start < 0:
        raise RuntimeError(f"Segment missing from rendered prompt: {segment!r}")
    end = start + len(segment)
    ids = []
    for i, pair in enumerate(offsets):
        s, e = int(pair[0]), int(pair[1])
        if e <= s:
            continue
        if e > start and s < end:
            ids.append(i)
    if not ids:
        raise RuntimeError(f"No tokens overlap segment: {segment!r}")
    return ids, start, end


def basic_schema(exp12, tok, task, entry):
    text = exp12.render_prompt(tok, entry["messages"])
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(x) for x in enc["input_ids"]]
    offsets = list(enc["offset_mapping"])
    n = len(ids)

    final_toks, _, _ = token_span_for_text(
        text, offsets, "Choose the single next action now."
    )
    user_toks, _, user_end_char = token_span_for_text(
        text, offsets, entry["user_content"]
    )

    after_user = []
    for i, pair in enumerate(offsets):
        s, e = int(pair[0]), int(pair[1])
        if e <= s:
            continue
        if s >= user_end_char:
            after_user.append(i)

    if len(after_user) >= ANCHOR_WIDTH:
        template_terminator = after_user[:ANCHOR_WIDTH]
    else:
        # Some fast tokenizers report template special-token offsets poorly.
        # Use the earliest available suffix positions and pad backward, while
        # preserving an audited distinction from the semantic content end.
        first = after_user[0] if after_user else max(user_toks) + 1
        end_idx = min(n - 1, first + ANCHOR_WIDTH - 1)
        template_terminator = list(range(max(0, end_idx - ANCHOR_WIDTH + 1), end_idx + 1))

    anchors = {
        "FINAL_INSTRUCTION_END": fixed_end_window(final_toks[-1], n),
        "USER_CONTENT_END": fixed_end_window(user_toks[-1], n),
        "TEMPLATE_TERMINATOR": template_terminator,
        "GENERATION_BOUNDARY": list(range(n - ANCHOR_WIDTH, n)),
    }

    if "native_abs_positions" in entry:
        p = [int(x) for x in entry["native_abs_positions"]]
        if min(p) < 0 or max(p) >= n:
            raise RuntimeError(
                f"NATIVE_ABSOLUTE outside shifted prompt: {p} / len={n}"
            )
        anchors["NATIVE_ABSOLUTE"] = p

    return {"text": text, "ids": ids, "offsets": offsets, "anchors": anchors}


def compute_schema(exp12, tok, task, entry):
    schema = basic_schema(exp12, tok, task, entry)
    missing = [x for x in SITES if x not in schema["anchors"]]
    if missing:
        raise RuntimeError(f"Missing EXP17 sites: {missing}")
    for k in SITES:
        if len(schema["anchors"][k]) != ANCHOR_WIDTH:
            raise RuntimeError(f"Bad site width: {k}")
    return schema


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def qwen_historical_user_end(tok, schema):
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    positions = [i for i, x in enumerate(schema["ids"]) if int(x) == int(im_end)]
    if not positions:
        raise RuntimeError("Qwen historical USER_END requires <|im_end|>")
    return fixed_end_window(positions[-1], len(schema["ids"]))


def load_qwen_anchor_map():
    obj = json.loads(EXP13_SELECTED.read_text(encoding="utf-8"))
    strata = obj.get("strata", obj.get("families", {}))
    return {k: v["selected_anchor"] for k, v in strata.items()}


def attach_native_absolute(exp12, tok, task, entries, model_id, qwen_map):
    # Native references are computed separately for each label/wording entry.
    native_entries = make_skill_entries(exp12, task, "native")
    lookup = {(e["wording"], int(e["label"])): e for e in native_entries}

    for e in entries:
        native = lookup[(e["wording"], int(e["label"]))]
        schema = basic_schema(exp12, tok, task, native)

        if model_id == "qwen":
            stratum = f"{task['family']}::{e['wording']}"
            historical = qwen_map[stratum]
            if historical == "USER_END":
                p = qwen_historical_user_end(tok, schema)
            elif historical == "FINAL_INSTRUCTION_END":
                p = schema["anchors"]["FINAL_INSTRUCTION_END"]
            else:
                raise RuntimeError(f"Unexpected EXP13 Qwen anchor: {historical}")
        else:
            p = schema["anchors"]["GENERATION_BOUNDARY"]

        e["native_abs_positions"] = list(p)


def attach_direct_native_absolute(exp12, tok, task, entries):
    # Not a historical mechanism claim; only needed to satisfy schema invariants.
    native_entries = make_direct_entries(task, "native")
    lookup = {int(e["label"]): e for e in native_entries}
    for e in entries:
        native = lookup[int(e["label"])]
        schema = basic_schema(exp12, tok, task, native)
        e["native_abs_positions"] = schema["anchors"]["GENERATION_BOUNDARY"]


def bootstrap_ci(values, seed, n_boot=N_BOOT):
    x = np.asarray(values, dtype=np.float64)
    if len(x) == 0:
        raise RuntimeError("Empty bootstrap input")
    rng = np.random.default_rng(seed)
    b = np.asarray([
        rng.choice(x, size=len(x), replace=True).mean()
        for _ in range(n_boot)
    ])
    return (
        float(x.mean()),
        float(np.quantile(b, 0.025)),
        float(np.quantile(b, 0.975)),
    )


def paired_bootstrap(a, b, seed):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) != len(b):
        raise RuntimeError("Paired length mismatch")
    return bootstrap_ci(a - b, seed)


def summarize_rows(df, seed):
    rows = []
    keys = ["model", "cohort", "condition", "site", "metric"]
    for key, g in df.groupby(keys, dropna=False):
        by_task = g.groupby("task_id")["value"].mean().to_numpy(np.float64)
        m, lo, hi = bootstrap_ci(by_task, seed + sum(map(ord, str(key))))
        rows.append(dict(zip(keys, key), mean=m, ci_low=lo, ci_high=hi, num_tasks=len(by_task)))
    return pd.DataFrame(rows)


def task_metric(df, *, model, condition, site, metric, cohort="skill"):
    d = df[
        (df["model"] == model)
        & (df["cohort"] == cohort)
        & (df["condition"] == condition)
        & (df["site"] == site)
        & (df["metric"] == metric)
    ]
    return d.groupby("task_id")["value"].mean().sort_index()


def contrast(df, model, condition, site_a, site_b, metric, seed, name):
    a = task_metric(df, model=model, condition=condition, site=site_a, metric=metric)
    b = task_metric(df, model=model, condition=condition, site=site_b, metric=metric)
    ix = a.index.intersection(b.index)
    m, lo, hi = paired_bootstrap(a.loc[ix].values, b.loc[ix].values, seed)
    return {
        "model": model,
        "contrast": name,
        "condition": condition,
        "metric": metric,
        "mean": m,
        "ci_low": lo,
        "ci_high": hi,
        "num_tasks": len(ix),
    }


def native_to_long_contrast(df, model, site, metric, seed, name):
    native = task_metric(df, model=model, condition="native", site=site, metric=metric)
    long = task_metric(df, model=model, condition="suffix_long", site=site, metric=metric)
    ix = native.index.intersection(long.index)
    m, lo, hi = paired_bootstrap(native.loc[ix].values, long.loc[ix].values, seed)
    return {
        "model": model,
        "contrast": name,
        "condition": "native_minus_suffix_long",
        "metric": metric,
        "mean": m,
        "ci_low": lo,
        "ci_high": hi,
        "num_tasks": len(ix),
    }


def classify_model(summary, contrasts, audit, model_id):
    def row(condition, site, metric):
        d = summary[
            (summary["model"] == model_id)
            & (summary["cohort"] == "skill")
            & (summary["condition"] == condition)
            & (summary["site"] == site)
            & (summary["metric"] == metric)
        ]
        return None if d.empty else d.iloc[0]

    def c(name, metric):
        d = contrasts[
            (contrasts["model"] == model_id)
            & (contrasts["contrast"] == name)
            & (contrasts["metric"] == metric)
        ]
        return None if d.empty else d.iloc[0]

    hs, hn = row("suffix_long", "GENERATION_BOUNDARY", "V_sufficiency"), row("suffix_long", "GENERATION_BOUNDARY", "V_necessity")
    fs, fn = row("suffix_long", "FINAL_INSTRUCTION_END", "V_sufficiency"), row("suffix_long", "FINAL_INSTRUCTION_END", "V_necessity")
    us, un = row("suffix_long", "USER_CONTENT_END", "V_sufficiency"), row("suffix_long", "USER_CONTENT_END", "V_necessity")
    os_, on = row("suffix_long", "NATIVE_ABSOLUTE", "V_sufficiency"), row("suffix_long", "NATIVE_ABSOLUTE", "V_necessity")

    h_minus_o = c("handoff_minus_native_absolute", "V_sufficiency")
    h_minus_f = c("handoff_minus_final_instruction", "V_sufficiency")
    h_minus_u = c("handoff_minus_user_content_end", "V_sufficiency")
    old_decay = c("native_absolute_native_minus_long", "V_sufficiency")

    positive = lambda r: r is not None and float(r["ci_low"]) > 0
    not_weaker = lambda r: r is not None and float(r["ci_low"]) <= 0

    handoff = (
        positive(hs) and positive(hn)
        and h_minus_o is not None and float(h_minus_o["ci_low"]) > 0
        and old_decay is not None and float(old_decay["ci_low"]) > 0
    )
    semantic = positive(fs) and positive(fn) and not_weaker(h_minus_f)
    user_end = positive(us) and positive(un) and not_weaker(h_minus_u)
    absolute = positive(os_) and positive(on) and not_weaker(h_minus_o)

    overlap = audit[
        (audit["model"] == model_id)
        & (audit["condition"] == "suffix_long")
    ]["template_generation_jaccard"].mean()
    overlap = float(overlap) if np.isfinite(overlap) else np.nan

    active = []
    if handoff:
        active.append("HANDOFF_TRACKING")
    if semantic:
        active.append("SEMANTIC_INSTRUCTION")
    if user_end:
        active.append("USER_END_TRACKING")
    if absolute:
        active.append("ABSOLUTE_POSITION")

    if handoff and overlap >= 0.8:
        active = [x for x in active if x != "HANDOFF_TRACKING"]
        active.insert(0, "HANDOFF_OR_TEMPLATE_BOUNDARY")

    if len(active) == 0:
        verdict = "NO_CLEAR_RELOCATION_PATTERN"
    elif len(active) == 1:
        verdict = active[0]
    else:
        verdict = "MIXED:" + "+".join(active)

    return {
        "model": model_id,
        "verdict": verdict,
        "template_generation_jaccard_suffix_long": overlap,
        "handoff_criteria": bool(handoff),
        "semantic_criteria": bool(semantic),
        "user_end_criteria": bool(user_end),
        "absolute_criteria": bool(absolute),
    }


def run_audit_for_model(exp14, exp12, tok, tasks, model_id, qwen_map, partition):
    rows = []
    subset = [t for t in tasks if task_partition(t) == partition]
    for task in subset:
        for condition in CONDITIONS:
            entries = make_skill_entries(exp12, task, condition)
            attach_native_absolute(exp12, tok, task, entries, model_id, qwen_map)
            for e in entries:
                s = compute_schema(exp12, tok, task, e)
                a = s["anchors"]
                rows.append({
                    "model": model_id,
                    "task_id": task["task_id"],
                    "family": task["family"],
                    "condition": condition,
                    "wording": e["wording"],
                    "label": int(e["label"]),
                    "prompt_len": len(s["ids"]),
                    "final_to_generation_distance": min(a["GENERATION_BOUNDARY"]) - max(a["FINAL_INSTRUCTION_END"]),
                    "native_abs_to_generation_distance": min(a["GENERATION_BOUNDARY"]) - max(a["NATIVE_ABSOLUTE"]),
                    "user_to_generation_distance": min(a["GENERATION_BOUNDARY"]) - max(a["USER_CONTENT_END"]),
                    "template_generation_jaccard": jaccard(a["TEMPLATE_TERMINATOR"], a["GENERATION_BOUNDARY"]),
                    "final_generation_jaccard": jaccard(a["FINAL_INSTRUCTION_END"], a["GENERATION_BOUNDARY"]),
                    "user_generation_jaccard": jaccard(a["USER_CONTENT_END"], a["GENERATION_BOUNDARY"]),
                    "final_tokens": tok.decode([s["ids"][p] for p in a["FINAL_INSTRUCTION_END"]], skip_special_tokens=False),
                    "user_end_tokens": tok.decode([s["ids"][p] for p in a["USER_CONTENT_END"]], skip_special_tokens=False),
                    "template_tokens": tok.decode([s["ids"][p] for p in a["TEMPLATE_TERMINATOR"]], skip_special_tokens=False),
                    "generation_tokens": tok.decode([s["ids"][p] for p in a["GENERATION_BOUNDARY"]], skip_special_tokens=False),
                    "native_abs_tokens": tok.decode([s["ids"][p] for p in a["NATIVE_ABSOLUTE"]], skip_special_tokens=False),
                })
    return rows


def add_result(rows, model_id, task, entry, condition, site, metric, value, cohort="skill", relation="opposite_same_wording"):
    rows.append({
        "model": model_id,
        "cohort": cohort,
        "task_id": task["task_id"],
        "family": task["family"],
        "condition": condition,
        "site": site,
        "wording": entry["wording"],
        "label": int(entry["label"]),
        "relation": relation,
        "metric": metric,
        "value": float(value),
    })


def run_model(exp14, exp15, exp12, model_id, model_dir, tasks, partition, out_dir):
    spec = MODEL_SPECS[model_id]
    model, tok, layers, cfg, arch = exp15.load_cross_arch_model(model_dir)

    if spec["model_type_contains"] not in str(getattr(cfg, "model_type", "")).lower():
        raise RuntimeError(
            f"{model_id}: wrong checkpoint type {getattr(cfg, 'model_type', None)}"
        )
    if spec["L"] >= len(layers):
        raise RuntimeError(f"{model_id}: frozen layer out of range")
    if spec["k"] >= arch["num_key_value_heads"]:
        raise RuntimeError(f"{model_id}: frozen KV out of range")
    if max(spec["positive_readers"]) >= arch["num_attention_heads"]:
        raise RuntimeError(f"{model_id}: frozen reader out of range")

    qwen_map = load_qwen_anchor_map() if model_id == "qwen" else {}
    subset = [t for t in tasks if task_partition(t) == partition]
    rows = []

    print("=" * 90)
    print(f"EXP17 model={model_id} checkpoint={model_dir}")
    print("Frozen:", spec)
    print("Architecture:", arch)
    print("Tasks:", len(subset), "partition:", partition)
    print("=" * 90)

    # Make EXP14 use EXP17's topology-aware schema.
    exp14.compute_schema = compute_schema

    for ti, task in enumerate(subset, 1):
        for condition in CONDITIONS:
            entries = make_skill_entries(exp12, task, condition)
            attach_native_absolute(exp12, tok, task, entries, model_id, qwen_map)
            maps = donor_maps_skill(entries)

            cache = exp14.build_task_cache(
                model, tok, layers, exp12, task, entries,
                spec["L"], arch["num_key_value_heads"], arch["head_dim"]
            )
            baselines = exp14.baseline_margins(
                model, tok, layers, exp12, entries, cache,
                spec["L"], arch["num_key_value_heads"], arch["head_dim"]
            )

            for i, entry in enumerate(entries):
                donor_i = maps[i]["opposite_same_wording"]
                donor = entries[donor_i]
                rec_cache, don_cache = cache[i], cache[donor_i]
                base = baselines[i]

                for site in SITES:
                    rp = exp14.anchor_positions(rec_cache, site)
                    dp = exp14.anchor_positions(don_cache, site)
                    res_ref = exp14.residual_effect(
                        model, tok, layers, exp12,
                        entry, rec_cache, donor, don_cache, base,
                        spec["L"], rp, dp,
                        arch["num_key_value_heads"], arch["head_dim"]
                    )
                    suff, nec = exp14.v_suff_nec(
                        model, tok, layers, exp12,
                        entry, rec_cache, donor, don_cache, base, res_ref,
                        spec["L"], rp, dp,
                        arch["num_key_value_heads"], arch["head_dim"],
                        v_heads=(spec["k"],)
                    )
                    add_result(rows, model_id, task, entry, condition, site, "residual_reference", res_ref)
                    add_result(rows, model_id, task, entry, condition, site, "V_sufficiency", suff)
                    add_result(rows, model_id, task, entry, condition, site, "V_necessity", nec)

                    if condition in CONTROL_CONDITIONS and site in PRIMARY_READER_SITES:
                        path = exp14.reader_path_effects(
                            model, tok, layers, exp12,
                            entry, rec_cache, donor, don_cache,
                            rp, dp, base,
                            spec["L"], (spec["k"],),
                            arch["num_key_value_heads"],
                            arch["num_attention_heads"],
                            arch["head_dim"],
                            leak_register_heads=spec["positive_readers"],
                            include_nec=True,
                        )
                        ps, pn = path["groups"](spec["positive_readers"])
                        add_result(rows, model_id, task, entry, condition, site, "reader_positive_sufficiency", ps)
                        add_result(rows, model_id, task, entry, condition, site, "reader_positive_necessity", pn)
                        add_result(rows, model_id, task, entry, condition, site, "reader_verified_V", path["verified_v_effect"])
                        add_result(rows, model_id, task, entry, condition, site, "reader_leakage_ratio", path["leakage_ratio"])

                if condition in CONTROL_CONDITIONS:
                    site = "GENERATION_BOUNDARY"
                    rp = exp14.anchor_positions(rec_cache, site)

                    cross_i = maps[i]["opposite_cross_wording"]
                    cross = entries[cross_i]
                    cross_cache = cache[cross_i]
                    cross_dp = exp14.anchor_positions(cross_cache, site)
                    cross_res = exp14.residual_effect(
                        model, tok, layers, exp12,
                        entry, rec_cache, cross, cross_cache, base,
                        spec["L"], rp, cross_dp,
                        arch["num_key_value_heads"], arch["head_dim"]
                    )
                    cross_s, cross_n = exp14.v_suff_nec(
                        model, tok, layers, exp12,
                        entry, rec_cache, cross, cross_cache, base, cross_res,
                        spec["L"], rp, cross_dp,
                        arch["num_key_value_heads"], arch["head_dim"],
                        v_heads=(spec["k"],)
                    )
                    add_result(rows, model_id, task, entry, condition, site, "cross_wording_V_sufficiency", cross_s, relation="opposite_cross_wording")
                    add_result(rows, model_id, task, entry, condition, site, "cross_wording_V_necessity", cross_n, relation="opposite_cross_wording")

                    same_i = maps[i]["same_state_cross_wording"]
                    same = entries[same_i]
                    same_cache = cache[same_i]
                    same_dp = exp14.anchor_positions(same_cache, site)
                    same_s, _ = exp14.v_suff_nec(
                        model, tok, layers, exp12,
                        entry, rec_cache, same, same_cache, base, None,
                        spec["L"], rp, same_dp,
                        arch["num_key_value_heads"], arch["head_dim"],
                        v_heads=(spec["k"],)
                    )
                    add_result(rows, model_id, task, entry, condition, site, "same_state_V_control", same_s, relation="same_state_cross_wording")

            # Direct-choice control at the same actual generation boundary.
            if condition in CONTROL_CONDITIONS:
                direct = make_direct_entries(task, condition)
                attach_direct_native_absolute(exp12, tok, task, direct)
                dcache = exp14.build_task_cache(
                    model, tok, layers, exp12, task, direct,
                    spec["L"], arch["num_key_value_heads"], arch["head_dim"]
                )
                dbases = exp14.baseline_margins(
                    model, tok, layers, exp12, direct, dcache,
                    spec["L"], arch["num_key_value_heads"], arch["head_dim"]
                )
                for di, de in enumerate(direct):
                    dj = 1 - di
                    donor = direct[dj]
                    rp = exp14.anchor_positions(dcache[di], "GENERATION_BOUNDARY")
                    dp = exp14.anchor_positions(dcache[dj], "GENERATION_BOUNDARY")
                    res_ref = exp14.residual_effect(
                        model, tok, layers, exp12,
                        de, dcache[di], donor, dcache[dj], dbases[di],
                        spec["L"], rp, dp,
                        arch["num_key_value_heads"], arch["head_dim"]
                    )
                    ds, dn = exp14.v_suff_nec(
                        model, tok, layers, exp12,
                        de, dcache[di], donor, dcache[dj], dbases[di], res_ref,
                        spec["L"], rp, dp,
                        arch["num_key_value_heads"], arch["head_dim"],
                        v_heads=(spec["k"],)
                    )
                    add_result(rows, model_id, task, de, condition, "GENERATION_BOUNDARY", "V_sufficiency", ds, cohort="direct", relation="opposite_label")
                    add_result(rows, model_id, task, de, condition, "GENERATION_BOUNDARY", "V_necessity", dn, cohort="direct", relation="opposite_label")

        print(f"[{model_id} {ti:02d}/{len(subset):02d}] {task['task_id']}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest = {
        "model_id": model_id,
        "model_dir": str(model_dir),
        "model_config_sha256": sha256_file(model_dir / "config.json"),
        "architecture": arch,
        "frozen": spec,
    }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, manifest


def make_contrasts(df, seed):
    rows = []
    for model_id in sorted(df["model"].unique()):
        for metric in ("V_sufficiency", "V_necessity"):
            rows.extend([
                contrast(df, model_id, "suffix_long", "GENERATION_BOUNDARY", "NATIVE_ABSOLUTE", metric, seed+1, "handoff_minus_native_absolute"),
                contrast(df, model_id, "suffix_long", "GENERATION_BOUNDARY", "FINAL_INSTRUCTION_END", metric, seed+2, "handoff_minus_final_instruction"),
                contrast(df, model_id, "suffix_long", "GENERATION_BOUNDARY", "USER_CONTENT_END", metric, seed+3, "handoff_minus_user_content_end"),
                contrast(df, model_id, "suffix_long", "GENERATION_BOUNDARY", "TEMPLATE_TERMINATOR", metric, seed+4, "handoff_minus_template_terminator"),
                native_to_long_contrast(df, model_id, "NATIVE_ABSOLUTE", metric, seed+5, "native_absolute_native_minus_long"),
                native_to_long_contrast(df, model_id, "GENERATION_BOUNDARY", metric, seed+6, "handoff_native_minus_long"),
            ])
    return pd.DataFrame(rows)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("all", "qwen", "mistral", "granite"), default="all")
    ap.add_argument("--phase", choices=("audit", "confirm", "reserve"), default="audit")
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--qwen_dir", default=None)
    ap.add_argument("--mistral_dir", default=None)
    ap.add_argument("--granite_dir", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exp14 = load_module(EXP14_CODE, "skills4s_exp14_for_exp17")
    exp15 = load_module(EXP15_CODE, "skills4s_exp15_for_exp17")
    exp12 = exp14.load_exp12()
    exp14.compute_schema = compute_schema

    tasks = make_tasks()
    (out_dir / "tasks.jsonl").write_text(
        "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks),
        encoding="utf-8",
    )
    split = {
        "confirm": [t["task_id"] for t in tasks if task_partition(t) == "confirm"],
        "reserve": [t["task_id"] for t in tasks if task_partition(t) == "reserve"],
    }
    (out_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    selected_models = list(MODEL_SPECS) if args.model == "all" else [args.model]
    overrides = {
        "qwen": args.qwen_dir,
        "mistral": args.mistral_dir,
        "granite": args.granite_dir,
    }
    model_dirs = {m: find_model_dir(m, overrides[m]) for m in selected_models}

    partition = "confirm" if args.phase in ("audit", "confirm") else "reserve"

    # Token/site audit can run without loading model weights.
    audit_rows = []
    qwen_map = load_qwen_anchor_map()
    for model_id in selected_models:
        tok = AutoTokenizer.from_pretrained(
            str(model_dirs[model_id]),
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        audit_rows.extend(
            run_audit_for_model(exp14, exp12, tok, tasks, model_id, qwen_map, partition)
        )

    audit_df = pd.DataFrame(audit_rows)
    audit_name = "site_overlap_audit.csv" if partition == "confirm" else "reserve_site_overlap_audit.csv"
    audit_df.to_csv(out_dir / audit_name, index=False)

    if args.phase == "audit":
        print(
            audit_df.groupby(["model", "condition"])[
                ["final_to_generation_distance", "native_abs_to_generation_distance",
                 "user_to_generation_distance", "template_generation_jaccard"]
            ].mean().to_string()
        )
        return

    all_rows = []
    manifests = {}
    for model_id in selected_models:
        rows, manifest = run_model(
            exp14, exp15, exp12, model_id, model_dirs[model_id],
            tasks, partition, out_dir
        )
        all_rows.extend(rows)
        manifests[model_id] = manifest

    df = pd.DataFrame(all_rows)
    prefix = "" if partition == "confirm" else "reserve_"
    df.to_csv(out_dir / f"{prefix}results.csv", index=False)

    summary = summarize_rows(df, args.seed)
    summary.to_csv(out_dir / f"{prefix}model_summary.csv", index=False)

    contrasts = make_contrasts(df, args.seed + 1000)
    contrasts.to_csv(out_dir / f"{prefix}relocation_contrasts.csv", index=False)

    verdicts = [
        classify_model(summary, contrasts, audit_df, m)
        for m in selected_models
    ]

    direct_specificity = []
    for m in selected_models:
        for cond in CONTROL_CONDITIONS:
            skill = task_metric(
                df, model=m, condition=cond, site="GENERATION_BOUNDARY",
                metric="V_sufficiency", cohort="skill"
            )
            direct = task_metric(
                df, model=m, condition=cond, site="GENERATION_BOUNDARY",
                metric="V_sufficiency", cohort="direct"
            )
            ix = skill.index.intersection(direct.index)
            if len(ix):
                mean, lo, hi = paired_bootstrap(
                    skill.loc[ix].values, direct.loc[ix].values, args.seed + 2000
                )
                direct_specificity.append({
                    "model": m,
                    "condition": cond,
                    "contrast": "procedural_minus_direct_V_sufficiency",
                    "mean": mean, "ci_low": lo, "ci_high": hi, "num_tasks": len(ix),
                })

    direct_df = pd.DataFrame(direct_specificity)
    direct_df.to_csv(out_dir / f"{prefix}specificity_contrasts.csv", index=False)

    result = {
        "experiment": "EXP17_handoff_relocation",
        "partition": partition,
        "models": selected_models,
        "verdicts": verdicts,
        "specificity": direct_specificity,
        "claim_boundary": (
            "EXP17 distinguishes semantic/position/handoff relocation hypotheses "
            "for already frozen V/KV subpaths; it does not establish that these "
            "subpaths are the only procedural-control mechanism."
        ),
    }
    (out_dir / f"{prefix}summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest = {
        "experiment": "EXP17_handoff_relocation",
        "git_commit": git_commit(),
        "seed": args.seed,
        "phase": args.phase,
        "partition": partition,
        "models": manifests,
        "conditions": list(CONDITIONS),
        "sites": list(SITES),
        "anchor_width": ANCHOR_WIDTH,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "exp14_sha256": sha256_file(EXP14_CODE),
        "exp15_sha256": sha256_file(EXP15_CODE),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "offline_only": True,
    }
    (out_dir / f"{prefix}run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
