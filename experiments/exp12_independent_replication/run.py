#!/usr/bin/env python3
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import argparse
import hashlib
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
DEFAULT_OUT = REPO_ROOT / "outputs" / "exp12_independent_replication"

SOURCE_H = 20
CONSUMER_BLOCK = 20
KV_HEAD = 0

FROZEN_OFFSETS = (-13, -5, -3, -1)
NEGATIVE_OFFSETS = (-12, -10, -8, -6)
FROZEN_READERS = (0, 3, 5)
NEGATIVE_READERS = (2, 4, 6)
ALL_KV0_READERS = tuple(range(7))
NON_KV0_READERS = tuple(range(7, 28))

N_TASKS_PER_FAMILY = 16
FAMILIES = ("test_edit", "search_edit", "config_command", "docs_code")


MODULES = [
    "parser", "cache", "router", "serializer",
    "scheduler", "registry", "validator", "transport",
    "loader", "formatter", "resolver", "tokenizer",
    "planner", "monitor", "dispatcher", "adapter",
]

SYMBOLS = [
    "parse_record", "cache_lookup", "route_request", "encode_payload",
    "schedule_job", "register_plugin", "validate_schema", "send_packet",
    "load_manifest", "format_result", "resolve_alias", "tokenize_input",
    "build_plan", "check_health", "dispatch_event", "adapt_response",
]

PACKAGES = [
    "pydantic", "httpx", "click", "pytest",
    "sqlalchemy", "fastapi", "numpy", "pandas",
    "jinja2", "typer", "attrs", "rich",
    "starlette", "requests", "packaging", "tomllib",
]

APIS = [
    "BaseModel.model_validate", "Client.send", "Command.invoke", "raises",
    "Session.execute", "APIRouter.add_api_route", "array", "DataFrame.merge",
    "Environment.from_string", "Argument.make_metavar", "define", "Console.print",
    "Request.url_for", "Session.request", "Version", "loads",
]

COMMANDS = [
    "python -m app.check_cache",
    "python -m app.trace_router",
    "python -m app.validate_payload",
    "python -m app.inspect_scheduler",
    "python -m app.check_registry",
    "python -m app.probe_transport",
    "python -m app.verify_loader",
    "python -m app.inspect_formatter",
    "python -m app.trace_resolver",
    "python -m app.check_tokenizer",
    "python -m app.verify_planner",
    "python -m app.inspect_monitor",
    "python -m app.trace_dispatcher",
    "python -m app.check_adapter",
    "python -m app.verify_parser",
    "python -m app.inspect_validator",
]


SKILL_TEXT = {
    "test_edit": {
        0: {
            "canonical": (
                "Observe before editing. Run the narrowest relevant test first to "
                "capture current behavior. Inspect implementation only after the test result."
            ),
            "paraphrase": (
                "Start with evidence, not source inspection: execute the targeted test "
                "as the first step, then look at implementation."
            ),
        },
        1: {
            "canonical": (
                "Inspect before testing. Open the relevant implementation first to "
                "understand the code path. Run the targeted test only afterward."
            ),
            "paraphrase": (
                "Begin from the implementation: inspect the source as the first step, "
                "and defer the focused test until after that inspection."
            ),
        },
    },
    "search_edit": {
        0: {
            "canonical": (
                "Map references before opening the target. Search the codebase for the "
                "relevant symbol first; inspect the implementation only after locating its uses."
            ),
            "paraphrase": (
                "Start by finding callers and references. Code search is the required "
                "first action, with source inspection following later."
            ),
        },
        1: {
            "canonical": (
                "Inspect the target before mapping references. Open the implementation "
                "first; search callers only after understanding the target code."
            ),
            "paraphrase": (
                "Begin at the source file itself. The first action is implementation "
                "inspection, and reference search comes afterward."
            ),
        },
    },
    "config_command": {
        0: {
            "canonical": (
                "Check configuration before execution. Inspect the relevant config file "
                "first, then run any diagnostic command."
            ),
            "paraphrase": (
                "Configuration is the first checkpoint: read it before executing the "
                "diagnostic command."
            ),
        },
        1: {
            "canonical": (
                "Capture runtime evidence before configuration inspection. Run the "
                "diagnostic command first, then inspect configuration."
            ),
            "paraphrase": (
                "Start from runtime behavior: execute the diagnostic as the first step "
                "and review configuration afterward."
            ),
        },
    },
    "docs_code": {
        0: {
            "canonical": (
                "Confirm the external API contract before source inspection. Consult the "
                "relevant documentation first, then inspect local implementation."
            ),
            "paraphrase": (
                "Documentation comes first: verify the library API before opening the "
                "project source."
            ),
        },
        1: {
            "canonical": (
                "Inspect local implementation before consulting external documentation. "
                "Open the source first and check the API reference afterward."
            ),
            "paraphrase": (
                "Begin with the project's code path. Source inspection is the first "
                "action; documentation lookup is deferred."
            ),
        },
    },
}


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

    candidates = sorted({p.parent.resolve() for p in MODELS_ROOT.rglob("config.json")})
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one local model under ./models, found {len(candidates)}"
        )
    return candidates[0]


def make_task(family: str, idx: int):
    module = MODULES[idx]
    symbol = SYMBOLS[idx]
    package = PACKAGES[idx]
    api = APIS[idx]
    command = COMMANDS[idx]
    case = f"{module}_edge_{idx:02d}"

    if family == "test_edit":
        action0 = f'run_tests("tests/test_{module}.py::test_{case}")'
        action1 = f'open_file("src/{module}.py")'
        issue = (
            f"A regression involving `{module}` appears only in the focused `{case}` "
            f"scenario. The implementation and its targeted test are both available."
        )
        detail = f"The suspected implementation entry point is `{symbol}`."

    elif family == "search_edit":
        action0 = f'search_code("{symbol}")'
        action1 = f'open_file("src/{module}.py")'
        issue = (
            f"A refactor around `{symbol}` may affect multiple callers. The target file "
            f"is known, but the repository-wide reference pattern has not been checked."
        )
        detail = f"The primary implementation file is `src/{module}.py`."

    elif family == "config_command":
        action0 = f'inspect_config("config/{module}.toml")'
        action1 = f'run_command("{command}")'
        issue = (
            f"The `{module}` subsystem behaves differently across environments. A local "
            f"configuration file and a narrow diagnostic command are both available."
        )
        detail = f"The diagnostic focuses on `{symbol}`."

    elif family == "docs_code":
        action0 = f'search_docs("{package}.{api}")'
        action1 = f'open_file("src/{module}.py")'
        issue = (
            f"Code using `{package}` may rely on an API contract that changed across "
            f"versions. Both the external API reference and local implementation are available."
        )
        detail = f"The local integration point is `{symbol}`."

    else:
        raise ValueError(family)

    return {
        "task_id": f"exp12_{family}_{idx:02d}",
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
    return [
        make_task(family, idx)
        for family in FAMILIES
        for idx in range(N_TASKS_PER_FAMILY)
    ]


def user_content(task):
    return (
        "Repository task:\n"
        f"{task['issue']}\n\n"
        f"Project detail:\n{task['detail']}\n\n"
        "Available next actions:\n"
        f"- {task['action0']}\n"
        f"- {task['action1']}\n\n"
        "Choose the single next action now."
    )


def skill_messages(task, label, wording):
    system = (
        "You are an autonomous software-engineering agent.\n\n"
        "Workflow Skill:\n"
        f"{SKILL_TEXT[task['family']][label][wording]}\n\n"
        "Follow the Workflow Skill exactly when deciding the first action."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content(task)},
    ]


def direct_messages(task, label):
    required = task["action1"] if label == 1 else task["action0"]
    system = (
        "You are an autonomous software-engineering agent.\n\n"
        "For this task, do not infer or apply a workflow. "
        "The required next action is explicitly fixed as:\n"
        f"{required}\n\n"
        "Choose that required action."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content(task)},
    ]


def make_skill_entries(task):
    entries = []
    for label in (0, 1):
        for wording in ("canonical", "paraphrase"):
            entries.append({
                "task_id": task["task_id"],
                "family": task["family"],
                "label": label,
                "wording": wording,
                "messages": skill_messages(task, label, wording),
                "candidates": [task["action0"], task["action1"]],
            })
    return entries


def make_direct_entries(task):
    return [
        {
            "task_id": task["task_id"],
            "family": task["family"],
            "label": label,
            "wording": "direct",
            "messages": direct_messages(task, label),
            "candidates": [task["action0"], task["action1"]],
        }
        for label in (0, 1)
    ]


def donor_maps_skill(entries):
    lookup = {(e["wording"], e["label"]): i for i, e in enumerate(entries)}
    out = {}
    for i, e in enumerate(entries):
        opp = 1 - e["label"]
        other_wording = "paraphrase" if e["wording"] == "canonical" else "canonical"
        out[i] = {
            "opposite_same_wording": lookup[(e["wording"], opp)],
            "opposite_cross_wording": lookup[(other_wording, opp)],
            "same_state_cross_wording": lookup[(other_wording, e["label"])],
        }
    return out


def donor_maps_direct(entries):
    return {0: {"opposite": 1}, 1: {"opposite": 0}}


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
def value_patch_hook(v_proj, positions, head_indices, replacements, num_kv_heads, head_dim):
    def hook_fn(module, inputs, output):
        y = output.clone()
        shape = y.shape
        yh = y.view(*shape[:-1], num_kv_heads, head_dim)

        pos = positions.to(y.device)
        heads = head_indices.to(y.device)
        rep = replacements.to(y.device, dtype=y.dtype)

        for row in range(yh.shape[0]):
            for pi, token_pos in enumerate(pos[row]):
                yh[row, token_pos, heads, :] = rep[row, pi]

        return yh.reshape(*shape)

    handle = v_proj.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def oproj_capture_hook(o_proj, store, key, num_q_heads, head_dim):
    def pre_hook(module, inputs):
        x = inputs[0]
        b, s, _ = x.shape
        store[key] = (
            x.view(b, s, num_q_heads, head_dim)
            .detach().float().cpu().numpy().astype(np.float32)
        )

    handle = o_proj.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def oproj_replace_heads_hook(o_proj, head_indices, replacement_heads, num_q_heads, head_dim):
    def pre_hook(module, inputs):
        x = inputs[0].clone()
        shape = x.shape
        xh = x.view(*shape[:-1], num_q_heads, head_dim)
        heads = head_indices.to(x.device)
        rep = replacement_heads.to(x.device, dtype=x.dtype)
        xh[:, :, heads, :] = rep
        return (xh.reshape(*shape),) + tuple(inputs[1:])

    handle = o_proj.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


@torch.inference_mode()
def capture_prompt(model, tok, layers, entry, num_kv_heads, head_dim):
    """Capture full-prompt H20 residual and block20 V states."""
    device = model.get_input_embeddings().weight.device
    text = render_prompt(tok, entry["messages"])
    enc = tok(text, add_special_tokens=False, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}

    store = {"residual": None, "v": None}
    handles = []

    def residual_hook(module, inputs, output):
        t = output[0] if isinstance(output, tuple) else output
        store["residual"] = (
            t[0].detach().float().cpu().numpy().astype(np.float32)
        )

    def v_hook(module, inputs, output):
        x = output[0].view(output.shape[1], num_kv_heads, head_dim)
        store["v"] = x.detach().float().cpu().numpy().astype(np.float32)

    handles.append(layers[SOURCE_H - 1].register_forward_hook(residual_hook))
    handles.append(
        layers[CONSUMER_BLOCK].self_attn.v_proj.register_forward_hook(v_hook)
    )

    try:
        model(**enc, use_cache=False, return_dict=True)
    finally:
        for h in handles:
            h.remove()

    return {"ids": ids, **store}


def prompt_cache(model, tok, layers, entries, num_kv_heads, head_dim):
    return [
        capture_prompt(model, tok, layers, e, num_kv_heads, head_dim)
        for e in entries
    ]


def offset_positions(prompt_len, offsets, device):
    arr = np.asarray([prompt_len + int(o) for o in offsets], dtype=np.int64)
    if np.any(arr < 0) or np.any(arr >= prompt_len):
        raise RuntimeError(
            f"Frozen offset outside prompt: len={prompt_len}, offsets={offsets}"
        )
    return torch.tensor(arr[None, :], dtype=torch.long, device=device)


def donor_v_values(cache, offsets, heads):
    idx = np.asarray([len(cache["ids"]) + int(o) for o in offsets], dtype=np.int64)
    return cache["v"][idx][:, list(heads), :].astype(np.float32)


def common_residual_values(cache, k):
    return cache["residual"][-k:, :].astype(np.float32)


def anchor_audit(tok, entry, cohort, task_id, family):
    text = render_prompt(tok, entry["messages"])
    ids = tok(text, add_special_tokens=False)["input_ids"]

    rows = []
    all_offsets = sorted(set(FROZEN_OFFSETS + NEGATIVE_OFFSETS))

    for off in all_offsets:
        pos = len(ids) + off
        token_id = int(ids[pos])
        rows.append({
            "cohort": cohort,
            "task_id": task_id,
            "family": family,
            "wording": entry["wording"],
            "label": entry["label"],
            "offset": int(off),
            "token_id": token_id,
            "token": tok.convert_ids_to_tokens(token_id),
            "decoded": tok.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
        })

    return rows


def validate_frozen_anchors(tok, audit_df):
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    im_start = tok.convert_tokens_to_ids("<|im_start|>")

    at_m5 = audit_df[audit_df["offset"] == -5]["token_id"].unique().tolist()
    at_m3 = audit_df[audit_df["offset"] == -3]["token_id"].unique().tolist()

    if at_m5 != [im_end]:
        raise RuntimeError(
            f"Frozen offset -5 no longer maps uniquely to <|im_end|>: {at_m5}"
        )
    if at_m3 != [im_start]:
        raise RuntimeError(
            f"Frozen offset -3 no longer maps uniquely to <|im_start|>: {at_m3}"
        )


@torch.inference_mode()
def score_margin(
    model,
    tok,
    layers,
    entry,
    num_kv_heads,
    head_dim,
    donor_cache=None,
    recipient_cache=None,
    residual_common_k=None,
    v_source_cache=None,
    v_offsets=None,
    v_heads=(KV_HEAD,),
):
    """Score candidate1 - candidate0; interventions apply only to prompt positions."""
    device = model.get_input_embeddings().weight.device
    prompt_text = render_prompt(tok, entry["messages"])
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    plen = len(prompt_ids)

    scores = []

    for candidate in entry["candidates"]:
        cids = tok(candidate, add_special_tokens=False)["input_ids"]
        ids = prompt_ids + cids
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)

        with ExitStack() as stack:
            if residual_common_k is not None:
                if donor_cache is None:
                    raise RuntimeError("Residual intervention requires donor_cache")
                k = int(residual_common_k)
                pos = torch.tensor(
                    np.arange(plen-k, plen, dtype=np.int64)[None, :],
                    dtype=torch.long,
                    device=device,
                )
                rv = common_residual_values(donor_cache, k)
                rep = torch.tensor(rv[None], dtype=torch.float32, device=device)
                stack.enter_context(
                    residual_patch_hook(
                        layers[SOURCE_H - 1], pos, rep
                    )
                )

            if v_source_cache is not None:
                if v_offsets is None:
                    raise RuntimeError("V intervention requires v_offsets")
                pos = offset_positions(plen, v_offsets, device)
                vv = donor_v_values(v_source_cache, v_offsets, v_heads)
                rep = torch.tensor(vv[None], dtype=torch.float32, device=device)
                heads = torch.tensor(list(v_heads), dtype=torch.long, device=device)
                stack.enter_context(
                    value_patch_hook(
                        layers[CONSUMER_BLOCK].self_attn.v_proj,
                        pos, heads, rep, num_kv_heads, head_dim,
                    )
                )

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )

        lp = torch.log_softmax(out.logits.float(), dim=-1)
        c = len(cids)
        pred_pos = torch.arange(plen-1, plen+c-1, device=device)
        targ_pos = torch.arange(plen, plen+c, device=device)
        targets = input_ids[0, targ_pos]
        scores.append(float(lp[0, pred_pos, targets].mean().cpu()))

    return scores[1] - scores[0]


@torch.inference_mode()
def candidate_capture_oproj(
    model,
    tok,
    layers,
    entry,
    candidate,
    num_kv_heads,
    num_q_heads,
    head_dim,
    v_source_cache=None,
    v_offsets=None,
):
    """Return candidate mean logP and block20 o_proj input."""
    device = model.get_input_embeddings().weight.device
    prompt_text = render_prompt(tok, entry["messages"])
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    cids = tok(candidate, add_special_tokens=False)["input_ids"]
    plen = len(prompt_ids)

    input_ids = torch.tensor(
        [prompt_ids + cids],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    store = {}

    with ExitStack() as stack:
        if v_source_cache is not None:
            pos = offset_positions(plen, v_offsets, device)
            vv = donor_v_values(v_source_cache, v_offsets, (KV_HEAD,))
            rep = torch.tensor(vv[None], dtype=torch.float32, device=device)
            heads = torch.tensor([KV_HEAD], dtype=torch.long, device=device)
            stack.enter_context(
                value_patch_hook(
                    layers[CONSUMER_BLOCK].self_attn.v_proj,
                    pos, heads, rep, num_kv_heads, head_dim,
                )
            )

        stack.enter_context(
            oproj_capture_hook(
                layers[CONSUMER_BLOCK].self_attn.o_proj,
                store, "oproj", num_q_heads, head_dim,
            )
        )

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

    lp = torch.log_softmax(out.logits.float(), dim=-1)
    c = len(cids)
    pred_pos = torch.arange(plen-1, plen+c-1, device=device)
    targ_pos = torch.arange(plen, plen+c, device=device)
    targets = input_ids[0, targ_pos]
    score = float(lp[0, pred_pos, targets].mean().cpu())

    return score, store["oproj"]


@torch.inference_mode()
def candidate_replace_readers(
    model,
    tok,
    layers,
    entry,
    candidate,
    num_kv_heads,
    num_q_heads,
    head_dim,
    replacement_np,
    reader_heads,
    upstream_v_cache=None,
    v_offsets=None,
):
    """Replace selected block20 reader-head outputs; optionally keep V intervention upstream."""
    device = model.get_input_embeddings().weight.device
    prompt_text = render_prompt(tok, entry["messages"])
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    cids = tok(candidate, add_special_tokens=False)["input_ids"]
    plen = len(prompt_ids)

    input_ids = torch.tensor(
        [prompt_ids + cids],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    heads = torch.tensor(list(reader_heads), dtype=torch.long, device=device)
    rep = torch.tensor(replacement_np, dtype=torch.float32, device=device)

    with ExitStack() as stack:
        if upstream_v_cache is not None:
            pos = offset_positions(plen, v_offsets, device)
            vv = donor_v_values(upstream_v_cache, v_offsets, (KV_HEAD,))
            vrep = torch.tensor(vv[None], dtype=torch.float32, device=device)
            kvhead = torch.tensor([KV_HEAD], dtype=torch.long, device=device)
            stack.enter_context(
                value_patch_hook(
                    layers[CONSUMER_BLOCK].self_attn.v_proj,
                    pos, kvhead, vrep, num_kv_heads, head_dim,
                )
            )

        stack.enter_context(
            oproj_replace_heads_hook(
                layers[CONSUMER_BLOCK].self_attn.o_proj,
                heads, rep, num_q_heads, head_dim,
            )
        )

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

    lp = torch.log_softmax(out.logits.float(), dim=-1)
    c = len(cids)
    pred_pos = torch.arange(plen-1, plen+c-1, device=device)
    targ_pos = torch.arange(plen, plen+c, device=device)
    targets = input_ids[0, targ_pos]
    return float(lp[0, pred_pos, targets].mean().cpu())


def pair_margin(scores):
    return scores[1] - scores[0]


def donor_sign(donor):
    return 1.0 if donor["label"] == 1 else -1.0


def signed_effect(patched_margin, baseline_margin, donor):
    return donor_sign(donor) * (patched_margin - baseline_margin)


def bootstrap_ci(values, seed, n_boot=5000):
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = np.asarray([
        rng.choice(x, size=len(x), replace=True).mean()
        for _ in range(n_boot)
    ])
    return (
        float(x.mean()),
        float(np.quantile(boots, 0.025)),
        float(np.quantile(boots, 0.975)),
    )


def paired_ci(a, b, seed):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) != len(b):
        raise RuntimeError("Paired arrays differ in length")
    return bootstrap_ci(a-b, seed)


def reader_path_effects(
    model,
    tok,
    layers,
    entry,
    donor,
    recipient_cache,
    donor_cache,
    num_kv_heads,
    num_q_heads,
    head_dim,
    offsets=FROZEN_OFFSETS,
):
    """Decompose the actual block20 response caused by frozen KV0 V patch."""
    sign = donor_sign(donor)

    base_scores = []
    patch_scores = []
    base_caps = []
    patch_caps = []

    for candidate in entry["candidates"]:
        bs, bc = candidate_capture_oproj(
            model, tok, layers, entry, candidate,
            num_kv_heads, num_q_heads, head_dim,
        )
        ps, pc = candidate_capture_oproj(
            model, tok, layers, entry, candidate,
            num_kv_heads, num_q_heads, head_dim,
            v_source_cache=donor_cache,
            v_offsets=offsets,
        )

        base_scores.append(bs)
        patch_scores.append(ps)
        base_caps.append(bc)
        patch_caps.append(pc)

    base_margin = pair_margin(base_scores)
    patch_margin = pair_margin(patch_scores)
    full_effect = sign * (patch_margin - base_margin)

    num = 0.0
    den = 0.0
    for bc, pc in zip(base_caps, patch_caps):
        delta = pc - bc
        num += float(np.sum(delta[:, :, list(NON_KV0_READERS), :] ** 2))
        den += float(np.sum(delta[:, :, list(ALL_KV0_READERS), :] ** 2))
    leakage_ratio = float(np.sqrt(num / (den + 1e-30)))

    def group_effect(heads):
        suff_scores = []
        nec_scores = []

        for ci, candidate in enumerate(entry["candidates"]):
            bc = base_caps[ci]
            pc = patch_caps[ci]

            suff_scores.append(
                candidate_replace_readers(
                    model, tok, layers, entry, candidate,
                    num_kv_heads, num_q_heads, head_dim,
                    replacement_np=pc[:, :, list(heads), :],
                    reader_heads=heads,
                )
            )

            nec_scores.append(
                candidate_replace_readers(
                    model, tok, layers, entry, candidate,
                    num_kv_heads, num_q_heads, head_dim,
                    replacement_np=bc[:, :, list(heads), :],
                    reader_heads=heads,
                    upstream_v_cache=donor_cache,
                    v_offsets=offsets,
                )
            )

        suff_margin = pair_margin(suff_scores)
        nec_margin = pair_margin(nec_scores)
        suff = sign * (suff_margin - base_margin)
        retained = sign * (nec_margin - base_margin)
        necessity_loss = full_effect - retained
        return suff, necessity_loss

    groups = {
        "target": FROZEN_READERS,
        "negative": NEGATIVE_READERS,
        "all7": ALL_KV0_READERS,
        "non_kv0": NON_KV0_READERS,
    }

    out = {
        "base_margin": base_margin,
        "patch_margin": patch_margin,
        "verified_v_effect": full_effect,
        "leakage_ratio": leakage_ratio,
    }

    for name, heads in groups.items():
        suff, nec = group_effect(heads)
        out[f"{name}_reader_sufficiency"] = suff
        out[f"{name}_reader_necessity"] = nec

    return out


def add_result(
    rows,
    *,
    cohort,
    task,
    entry,
    donor,
    relation,
    metric,
    value,
    extra=None,
):
    row = {
        "cohort": cohort,
        "task_id": task["task_id"],
        "family": task["family"],
        "recipient_label": entry["label"],
        "recipient_wording": entry["wording"],
        "donor_label": donor["label"] if donor is not None else -1,
        "donor_wording": donor["wording"] if donor is not None else "",
        "relation": relation,
        "metric": metric,
        "value": float(value),
    }
    if extra:
        row.update(extra)
    rows.append(row)


def skill_task_experiment(
    model,
    tok,
    layers,
    task,
    num_kv_heads,
    num_q_heads,
    head_dim,
):
    entries = make_skill_entries(task)
    maps = donor_maps_skill(entries)
    cache = prompt_cache(model, tok, layers, entries, num_kv_heads, head_dim)

    rows = []
    audits = []
    baselines = []

    for i, entry in enumerate(entries):
        baseline = score_margin(
            model, tok, layers, entry,
            num_kv_heads, head_dim,
        )
        baselines.append(baseline)

        pred = 1 if baseline > 0 else 0
        add_result(
            rows,
            cohort="skill",
            task=task,
            entry=entry,
            donor=None,
            relation="baseline",
            metric="behavior_correct",
            value=1.0 if pred == entry["label"] else 0.0,
        )
        add_result(
            rows,
            cohort="skill",
            task=task,
            entry=entry,
            donor=None,
            relation="baseline",
            metric="baseline_signed_preference",
            value=(1.0 if entry["label"] == 1 else -1.0) * baseline,
        )

        audits.extend(
            anchor_audit(
                tok, entry, "skill", task["task_id"], task["family"]
            )
        )

    for i, entry in enumerate(entries):
        base = baselines[i]

        # Primary same-wording opposite-state donor.
        di = maps[i]["opposite_same_wording"]
        donor = entries[di]
        k = longest_common_suffix(cache[i]["ids"], cache[di]["ids"])
        if k < abs(min(FROZEN_OFFSETS + NEGATIVE_OFFSETS)):
            raise RuntimeError(
                f"{task['task_id']} common suffix too short: {k}"
            )

        residual_margin = score_margin(
            model, tok, layers, entry,
            num_kv_heads, head_dim,
            donor_cache=cache[di],
            recipient_cache=cache[i],
            residual_common_k=k,
        )
        e_res = signed_effect(residual_margin, base, donor)

        target_v_margin = score_margin(
            model, tok, layers, entry,
            num_kv_heads, head_dim,
            v_source_cache=cache[di],
            v_offsets=FROZEN_OFFSETS,
        )
        e_target_v_suff = signed_effect(target_v_margin, base, donor)

        residual_target_clamp_margin = score_margin(
            model, tok, layers, entry,
            num_kv_heads, head_dim,
            donor_cache=cache[di],
            residual_common_k=k,
            v_source_cache=cache[i],
            v_offsets=FROZEN_OFFSETS,
        )
        target_retained = signed_effect(
            residual_target_clamp_margin, base, donor
        )
        e_target_v_nec = e_res - target_retained

        neg_v_margin = score_margin(
            model, tok, layers, entry,
            num_kv_heads, head_dim,
            v_source_cache=cache[di],
            v_offsets=NEGATIVE_OFFSETS,
        )
        e_neg_v_suff = signed_effect(neg_v_margin, base, donor)

        residual_neg_clamp_margin = score_margin(
            model, tok, layers, entry,
            num_kv_heads, head_dim,
            donor_cache=cache[di],
            residual_common_k=k,
            v_source_cache=cache[i],
            v_offsets=NEGATIVE_OFFSETS,
        )
        neg_retained = signed_effect(
            residual_neg_clamp_margin, base, donor
        )
        e_neg_v_nec = e_res - neg_retained

        for metric, value in [
            ("H20_residual_reference", e_res),
            ("frozen_V_sufficiency", e_target_v_suff),
            ("frozen_V_necessity_loss", e_target_v_nec),
            ("negative_offset_V_sufficiency", e_neg_v_suff),
            ("negative_offset_V_necessity_loss", e_neg_v_nec),
        ]:
            add_result(
                rows,
                cohort="skill",
                task=task,
                entry=entry,
                donor=donor,
                relation="opposite_same_wording",
                metric=metric,
                value=value,
            )

        reader = reader_path_effects(
            model, tok, layers, entry, donor,
            cache[i], cache[di],
            num_kv_heads, num_q_heads, head_dim,
            offsets=FROZEN_OFFSETS,
        )

        for metric in [
            "verified_v_effect",
            "target_reader_sufficiency",
            "target_reader_necessity",
            "negative_reader_sufficiency",
            "negative_reader_necessity",
            "all7_reader_sufficiency",
            "all7_reader_necessity",
            "non_kv0_reader_sufficiency",
            "non_kv0_reader_necessity",
        ]:
            add_result(
                rows,
                cohort="skill",
                task=task,
                entry=entry,
                donor=donor,
                relation="opposite_same_wording",
                metric=metric,
                value=reader[metric],
                extra={
                    "reader_leakage_ratio": reader["leakage_ratio"],
                    "reader_baseline_margin_diff": reader["base_margin"] - base,
                    "reader_v_effect_diff": (
                        reader["verified_v_effect"] - e_target_v_suff
                    ),
                },
            )

        # Cross-wording frozen V replication.
        cdi = maps[i]["opposite_cross_wording"]
        cross_donor = entries[cdi]
        ck = longest_common_suffix(cache[i]["ids"], cache[cdi]["ids"])

        cross_res_margin = score_margin(
            model, tok, layers, entry,
            num_kv_heads, head_dim,
            donor_cache=cache[cdi],
            residual_common_k=ck,
        )
        cross_res = signed_effect(cross_res_margin, base, cross_donor)

        cross_v_margin = score_margin(
            model, tok, layers, entry,
            num_kv_heads, head_dim,
            v_source_cache=cache[cdi],
            v_offsets=FROZEN_OFFSETS,
        )
        cross_suff = signed_effect(cross_v_margin, base, cross_donor)

        cross_clamp_margin = score_margin(
            model, tok, layers, entry,
            num_kv_heads, head_dim,
            donor_cache=cache[cdi],
            residual_common_k=ck,
            v_source_cache=cache[i],
            v_offsets=FROZEN_OFFSETS,
        )
        cross_retained = signed_effect(
            cross_clamp_margin, base, cross_donor
        )
        cross_nec = cross_res - cross_retained

        for metric, value in [
            ("cross_H20_residual_reference", cross_res),
            ("cross_frozen_V_sufficiency", cross_suff),
            ("cross_frozen_V_necessity_loss", cross_nec),
        ]:
            add_result(
                rows,
                cohort="skill",
                task=task,
                entry=entry,
                donor=cross_donor,
                relation="opposite_cross_wording",
                metric=metric,
                value=value,
            )

        # Same-state cross-wording semantic control.
        sdi = maps[i]["same_state_cross_wording"]
        same_donor = entries[sdi]
        same_v_margin = score_margin(
            model, tok, layers, entry,
            num_kv_heads, head_dim,
            v_source_cache=cache[sdi],
            v_offsets=FROZEN_OFFSETS,
        )
        same_effect = (
            (1.0 if same_donor["label"] == 1 else -1.0)
            * (same_v_margin - base)
        )
        add_result(
            rows,
            cohort="skill",
            task=task,
            entry=entry,
            donor=same_donor,
            relation="same_state_cross_wording",
            metric="same_state_frozen_V_control",
            value=same_effect,
        )

    return rows, audits


def direct_task_experiment(
    model,
    tok,
    layers,
    task,
    num_kv_heads,
    num_q_heads,
    head_dim,
):
    entries = make_direct_entries(task)
    maps = donor_maps_direct(entries)
    cache = prompt_cache(model, tok, layers, entries, num_kv_heads, head_dim)

    rows = []
    audits = []
    baselines = []

    for i, entry in enumerate(entries):
        baseline = score_margin(
            model, tok, layers, entry,
            num_kv_heads, head_dim,
        )
        baselines.append(baseline)
        pred = 1 if baseline > 0 else 0

        add_result(
            rows,
            cohort="direct",
            task=task,
            entry=entry,
            donor=None,
            relation="baseline",
            metric="behavior_correct",
            value=1.0 if pred == entry["label"] else 0.0,
        )
        add_result(
            rows,
            cohort="direct",
            task=task,
            entry=entry,
            donor=None,
            relation="baseline",
            metric="baseline_signed_preference",
            value=(1.0 if entry["label"] == 1 else -1.0) * baseline,
        )

        audits.extend(
            anchor_audit(
                tok, entry, "direct", task["task_id"], task["family"]
            )
        )

    for i, entry in enumerate(entries):
        di = maps[i]["opposite"]
        donor = entries[di]
        base = baselines[i]
        k = longest_common_suffix(cache[i]["ids"], cache[di]["ids"])

        residual_margin = score_margin(
            model, tok, layers, entry,
            num_kv_heads, head_dim,
            donor_cache=cache[di],
            residual_common_k=k,
        )
        e_res = signed_effect(residual_margin, base, donor)

        v_margin = score_margin(
            model, tok, layers, entry,
            num_kv_heads, head_dim,
            v_source_cache=cache[di],
            v_offsets=FROZEN_OFFSETS,
        )
        e_v_suff = signed_effect(v_margin, base, donor)

        clamp_margin = score_margin(
            model, tok, layers, entry,
            num_kv_heads, head_dim,
            donor_cache=cache[di],
            residual_common_k=k,
            v_source_cache=cache[i],
            v_offsets=FROZEN_OFFSETS,
        )
        retained = signed_effect(clamp_margin, base, donor)
        e_v_nec = e_res - retained

        for metric, value in [
            ("H20_residual_reference", e_res),
            ("frozen_V_sufficiency", e_v_suff),
            ("frozen_V_necessity_loss", e_v_nec),
        ]:
            add_result(
                rows,
                cohort="direct",
                task=task,
                entry=entry,
                donor=donor,
                relation="opposite",
                metric=metric,
                value=value,
            )

        reader = reader_path_effects(
            model, tok, layers, entry, donor,
            cache[i], cache[di],
            num_kv_heads, num_q_heads, head_dim,
            offsets=FROZEN_OFFSETS,
        )

        for metric in [
            "verified_v_effect",
            "target_reader_sufficiency",
            "target_reader_necessity",
            "all7_reader_sufficiency",
            "all7_reader_necessity",
            "non_kv0_reader_sufficiency",
            "non_kv0_reader_necessity",
        ]:
            add_result(
                rows,
                cohort="direct",
                task=task,
                entry=entry,
                donor=donor,
                relation="opposite",
                metric=metric,
                value=reader[metric],
                extra={
                    "reader_leakage_ratio": reader["leakage_ratio"],
                    "reader_baseline_margin_diff": reader["base_margin"] - base,
                    "reader_v_effect_diff": (
                        reader["verified_v_effect"] - e_v_suff
                    ),
                },
            )

    return rows, audits


def summarize_metric(task_df, metric, seed):
    x = task_df[task_df["metric"] == metric]["value"].to_numpy(np.float64)
    if len(x) == 0:
        return None
    mean, lo, hi = bootstrap_ci(x, seed)
    return {
        "metric": metric,
        "mean": mean,
        "ci_low": lo,
        "ci_high": hi,
        "num_tasks": int(len(x)),
    }


def paired_metric_contrast(task_df, metric_a, metric_b, seed, name):
    a = task_df[task_df["metric"] == metric_a][
        ["task_id", "value"]
    ].rename(columns={"value": "a"})
    b = task_df[task_df["metric"] == metric_b][
        ["task_id", "value"]
    ].rename(columns={"value": "b"})
    m = a.merge(b, on="task_id", how="inner")
    mean, lo, hi = paired_ci(m["a"], m["b"], seed)
    return {
        "contrast": name,
        "metric_a": metric_a,
        "metric_b": metric_b,
        "mean_difference": mean,
        "ci_low": lo,
        "ci_high": hi,
        "num_tasks": int(len(m)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=5212)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--phase",
        choices=["all", "replication", "specificity"],
        default="all",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tasks = make_tasks()

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Freeze/audit the generated task set itself.
    with (out_dir / "replication_tasks.jsonl").open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    model_dir = resolve_model_dir(args.model)

    tok = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Validate frozen boundary structure before expensive intervention runs.
    pre_audit = []
    for task in tasks:
        for entry in make_skill_entries(task):
            pre_audit.extend(
                anchor_audit(tok, entry, "skill", task["task_id"], task["family"])
            )
        if args.phase in ("all", "specificity"):
            for entry in make_direct_entries(task):
                pre_audit.extend(
                    anchor_audit(tok, entry, "direct", task["task_id"], task["family"])
                )

    audit_df = pd.DataFrame(pre_audit)
    validate_frozen_anchors(tok, audit_df)
    audit_df.to_csv(out_dir / "token_anchor_audit.csv", index=False)

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

    num_q_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    hidden_size = int(model.config.hidden_size)
    head_dim = int(getattr(model.config, "head_dim", hidden_size // num_q_heads))

    if num_q_heads != 28 or num_kv_heads != 4:
        raise RuntimeError(
            f"Frozen EXP11 architecture expects 28 Q / 4 KV heads; got "
            f"{num_q_heads} / {num_kv_heads}"
        )
    if num_q_heads // num_kv_heads != 7:
        raise RuntimeError("Frozen KV0 -> Q0..Q6 mapping is not valid")
    if layers[CONSUMER_BLOCK].self_attn.v_proj.out_features != num_kv_heads * head_dim:
        raise RuntimeError("Unexpected block20 v_proj geometry")

    print("=" * 88)
    print("EXP12: Frozen-Circuit Independent Replication")
    print("Model:", model_dir)
    print("Tasks:", len(tasks), "| families:", FAMILIES)
    print("Frozen offsets:", FROZEN_OFFSETS)
    print("Frozen readers:", FROZEN_READERS)
    print("Phase:", args.phase)
    print("=" * 88)

    skill_rows = []
    direct_rows = []

    for n, task in enumerate(tasks, 1):
        # Specificity requires the procedural comparator too, so both `all`
        # and `specificity` run the frozen Skill cohort.
        if args.phase in ("all", "replication", "specificity"):
            rows, _ = skill_task_experiment(
                model, tok, layers, task,
                num_kv_heads, num_q_heads, head_dim,
            )
            skill_rows.extend(rows)

        if args.phase in ("all", "specificity"):
            rows, _ = direct_task_experiment(
                model, tok, layers, task,
                num_kv_heads, num_q_heads, head_dim,
            )
            direct_rows.extend(rows)

        print(
            f"[{n:03d}/{len(tasks):03d}] "
            f"{task['task_id']} ({task['family']})"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # Replication inference
    # ------------------------------------------------------------
    replication_summary_rows = []
    family_summary_rows = []
    contrasts = []
    summary = {
        "experiment": "EXP12_independent_replication",
        "phase": args.phase,
        "frozen_offsets": list(FROZEN_OFFSETS),
        "negative_offsets": list(NEGATIVE_OFFSETS),
        "frozen_readers": list(FROZEN_READERS),
        "negative_readers": list(NEGATIVE_READERS),
        "num_tasks": len(tasks),
        "families": list(FAMILIES),
    }

    if skill_rows:
        rep_df = pd.DataFrame(skill_rows)
        rep_df.to_csv(out_dir / "replication_results.csv", index=False)

        # Task-level inference: average labels and wording variants within task.
        task_rep = (
            rep_df.groupby(
                ["task_id", "family", "relation", "metric"],
                as_index=False,
            )["value"].mean()
        )

        primary_relation = task_rep[
            task_rep["relation"] == "opposite_same_wording"
        ].copy()

        metrics = [
            "H20_residual_reference",
            "frozen_V_sufficiency",
            "frozen_V_necessity_loss",
            "negative_offset_V_sufficiency",
            "negative_offset_V_necessity_loss",
            "verified_v_effect",
            "target_reader_sufficiency",
            "target_reader_necessity",
            "negative_reader_sufficiency",
            "negative_reader_necessity",
            "all7_reader_sufficiency",
            "all7_reader_necessity",
            "non_kv0_reader_sufficiency",
            "non_kv0_reader_necessity",
        ]

        for metric in metrics:
            item = summarize_metric(
                primary_relation, metric,
                args.seed + sum(map(ord, metric)),
            )
            if item:
                item["scope"] = "overall"
                replication_summary_rows.append(item)

        # Cross-wording and baseline controls.
        for relation, metric in [
            ("opposite_cross_wording", "cross_H20_residual_reference"),
            ("opposite_cross_wording", "cross_frozen_V_sufficiency"),
            ("opposite_cross_wording", "cross_frozen_V_necessity_loss"),
            ("same_state_cross_wording", "same_state_frozen_V_control"),
            ("baseline", "behavior_correct"),
            ("baseline", "baseline_signed_preference"),
        ]:
            d = task_rep[task_rep["relation"] == relation]
            item = summarize_metric(
                d, metric,
                args.seed + 10000 + sum(map(ord, relation + metric)),
            )
            if item:
                item["scope"] = "overall"
                replication_summary_rows.append(item)

        # Prespecified paired controls.
        contrasts = [
            paired_metric_contrast(
                primary_relation,
                "frozen_V_sufficiency",
                "negative_offset_V_sufficiency",
                args.seed + 20001,
                "frozen_offsets_minus_negative_offsets_sufficiency",
            ),
            paired_metric_contrast(
                primary_relation,
                "frozen_V_necessity_loss",
                "negative_offset_V_necessity_loss",
                args.seed + 20002,
                "frozen_offsets_minus_negative_offsets_necessity",
            ),
            paired_metric_contrast(
                primary_relation,
                "target_reader_sufficiency",
                "negative_reader_sufficiency",
                args.seed + 20003,
                "frozen_readers_minus_negative_readers_sufficiency",
            ),
            paired_metric_contrast(
                primary_relation,
                "target_reader_necessity",
                "negative_reader_necessity",
                args.seed + 20004,
                "frozen_readers_minus_negative_readers_necessity",
            ),
        ]

        # Family-level secondary replication.
        family_metrics = [
            "H20_residual_reference",
            "frozen_V_sufficiency",
            "frozen_V_necessity_loss",
            "target_reader_sufficiency",
            "target_reader_necessity",
        ]

        for family in FAMILIES:
            fd = primary_relation[primary_relation["family"] == family]
            for metric in family_metrics:
                vals = fd[fd["metric"] == metric]["value"].to_numpy(np.float64)
                mean, lo, hi = bootstrap_ci(
                    vals,
                    args.seed + 30000 + sum(map(ord, family + metric)),
                )
                family_summary_rows.append({
                    "family": family,
                    "metric": metric,
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "num_tasks": int(len(vals)),
                })

        rep_summary_df = pd.DataFrame(replication_summary_rows)
        rep_summary_df.to_csv(out_dir / "replication_summary.csv", index=False)
        pd.DataFrame(family_summary_rows).to_csv(
            out_dir / "family_summary.csv", index=False
        )
        pd.DataFrame(contrasts).to_csv(
            out_dir / "replication_contrasts.csv", index=False
        )

        def rep_row(metric):
            x = rep_summary_df[rep_summary_df["metric"] == metric]
            return None if x.empty else x.iloc[0]

        primary_required = [
            "frozen_V_sufficiency",
            "frozen_V_necessity_loss",
            "target_reader_sufficiency",
            "target_reader_necessity",
        ]

        endpoint_flags = {}
        for metric in primary_required:
            r = rep_row(metric)
            endpoint_flags[metric] = bool(
                r is not None and float(r["ci_low"]) > 0
            )

        contrast_flags = {
            c["contrast"]: bool(c["ci_low"] > 0)
            for c in contrasts
        }

        cross_s = rep_row("cross_frozen_V_sufficiency")
        cross_n = rep_row("cross_frozen_V_necessity_loss")
        cross_flags = {
            "cross_frozen_V_sufficiency_positive": bool(
                cross_s is not None and float(cross_s["ci_low"]) > 0
            ),
            "cross_frozen_V_necessity_positive": bool(
                cross_n is not None and float(cross_n["ci_low"]) > 0
            ),
        }

        # Reader decomposition sanity from raw rows.
        same = rep_df[
            rep_df["relation"] == "opposite_same_wording"
        ]
        verified = (
            same[same["metric"] == "verified_v_effect"]
            .groupby("task_id")["value"].mean()
        )
        all7s = (
            same[same["metric"] == "all7_reader_sufficiency"]
            .groupby("task_id")["value"].mean()
        )
        all7n = (
            same[same["metric"] == "all7_reader_necessity"]
            .groupby("task_id")["value"].mean()
        )
        non_s = (
            same[same["metric"] == "non_kv0_reader_sufficiency"]
            .groupby("task_id")["value"].mean()
        )
        non_n = (
            same[same["metric"] == "non_kv0_reader_necessity"]
            .groupby("task_id")["value"].mean()
        )

        sanity = {
            "all7_minus_verified_suff_mean": float(
                (all7s - verified).mean()
            ),
            "all7_minus_verified_nec_mean": float(
                (all7n - verified).mean()
            ),
            "non_kv0_reader_sufficiency_mean": float(non_s.mean()),
            "non_kv0_reader_necessity_mean": float(non_n.mean()),
            "mean_reader_leakage_ratio": float(
                same["reader_leakage_ratio"].dropna().mean()
            ),
            "max_abs_reader_baseline_margin_diff": float(
                same["reader_baseline_margin_diff"].dropna().abs().max()
            ),
            "max_abs_reader_v_effect_diff": float(
                same["reader_v_effect_diff"].dropna().abs().max()
            ),
        }

        behavior = rep_row("behavior_correct")
        summary["replication"] = {
            "baseline_behavior_accuracy": (
                float(behavior["mean"]) if behavior is not None else None
            ),
            "primary_endpoint_flags": endpoint_flags,
            "paired_control_flags": contrast_flags,
            "cross_wording_flags": cross_flags,
            "sanity": sanity,
            "confirmatory_replication_pass": bool(
                all(endpoint_flags.values())
                and all(contrast_flags.values())
                and all(cross_flags.values())
            ),
            "overall_metrics": (
                rep_summary_df.to_dict(orient="records")
            ),
            "paired_contrasts": contrasts,
        }

        # Plot family replication.
        fdf = pd.DataFrame(family_summary_rows)
        plot_metrics = [
            "frozen_V_sufficiency",
            "frozen_V_necessity_loss",
            "target_reader_sufficiency",
            "target_reader_necessity",
        ]

        fig = plt.figure(figsize=(11, 6))
        x = np.arange(len(FAMILIES))
        width = 0.19

        for j, metric in enumerate(plot_metrics):
            vals = []
            for family in FAMILIES:
                vals.append(
                    float(
                        fdf[
                            (fdf["family"] == family)
                            & (fdf["metric"] == metric)
                        ].iloc[0]["mean"]
                    )
                )
            plt.bar(
                x + (j - 1.5) * width,
                vals,
                width,
                label=metric,
            )

        plt.axhline(0.0, linestyle="--")
        plt.xticks(x, FAMILIES, rotation=15)
        plt.ylabel("Causal effect")
        plt.title("EXP12 frozen-circuit replication by new Skill family")
        plt.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(out_dir / "family_replication.png", dpi=180)
        plt.close(fig)

    # ------------------------------------------------------------
    # Specificity falsification
    # ------------------------------------------------------------
    if direct_rows:
        direct_df = pd.DataFrame(direct_rows)
        direct_df.to_csv(out_dir / "specificity_results.csv", index=False)

        skill_df = pd.DataFrame(skill_rows)

        # Canonical Skill only, to pair one procedural wording against direct.
        skill_can = skill_df[
            (
                (skill_df["relation"] == "opposite_same_wording")
                & (skill_df["recipient_wording"] == "canonical")
            )
        ]

        direct_main = direct_df[
            direct_df["relation"] == "opposite"
        ]

        compare_metrics = [
            "H20_residual_reference",
            "frozen_V_sufficiency",
            "frozen_V_necessity_loss",
            "target_reader_sufficiency",
            "target_reader_necessity",
        ]

        comp_rows = []

        for metric in compare_metrics:
            s = (
                skill_can[skill_can["metric"] == metric]
                .groupby(["task_id", "family"], as_index=False)["value"]
                .mean()
                .rename(columns={"value": "skill"})
            )
            d = (
                direct_main[direct_main["metric"] == metric]
                .groupby(["task_id", "family"], as_index=False)["value"]
                .mean()
                .rename(columns={"value": "direct"})
            )
            m = s.merge(d, on=["task_id", "family"], how="inner")
            diff = m["skill"] - m["direct"]

            sm, slo, shi = bootstrap_ci(
                m["skill"],
                args.seed + 40000 + sum(map(ord, metric)),
            )
            dm, dlo, dhi = bootstrap_ci(
                m["direct"],
                args.seed + 41000 + sum(map(ord, metric)),
            )
            qm, qlo, qhi = bootstrap_ci(
                diff,
                args.seed + 42000 + sum(map(ord, metric)),
            )

            comp_rows.append({
                "scope": "overall",
                "family": "ALL",
                "metric": metric,
                "skill_mean": sm,
                "skill_ci_low": slo,
                "skill_ci_high": shi,
                "direct_mean": dm,
                "direct_ci_low": dlo,
                "direct_ci_high": dhi,
                "skill_minus_direct_mean": qm,
                "skill_minus_direct_ci_low": qlo,
                "skill_minus_direct_ci_high": qhi,
                "num_tasks": int(len(m)),
            })

            for family in FAMILIES:
                fm = m[m["family"] == family]
                fqm, fqlo, fqhi = bootstrap_ci(
                    fm["skill"] - fm["direct"],
                    args.seed + 43000 + sum(map(ord, family + metric)),
                )
                comp_rows.append({
                    "scope": "family",
                    "family": family,
                    "metric": metric,
                    "skill_mean": float(fm["skill"].mean()),
                    "skill_ci_low": np.nan,
                    "skill_ci_high": np.nan,
                    "direct_mean": float(fm["direct"].mean()),
                    "direct_ci_low": np.nan,
                    "direct_ci_high": np.nan,
                    "skill_minus_direct_mean": fqm,
                    "skill_minus_direct_ci_low": fqlo,
                    "skill_minus_direct_ci_high": fqhi,
                    "num_tasks": int(len(fm)),
                })

        comp_df = pd.DataFrame(comp_rows)
        comp_df.to_csv(out_dir / "specificity_comparison.csv", index=False)

        direct_behavior = (
            direct_df[
                (direct_df["relation"] == "baseline")
                & (direct_df["metric"] == "behavior_correct")
            ]["value"].mean()
        )

        summary["specificity"] = {
            "direct_choice_behavior_accuracy": float(direct_behavior),
            "comparison": comp_df[
                comp_df["scope"] == "overall"
            ].to_dict(orient="records"),
            "interpretation_rule": (
                "If frozen-path effects are similar under procedural Skills and "
                "direct-choice instructions, broaden the claim to an "
                "instruction-conditioned action-selection boundary circuit. "
                "A positive skill-minus-direct difference supports procedural "
                "specificity; a negative difference indicates broader/general "
                "instruction routing."
            ),
        }

        overall = comp_df[comp_df["scope"] == "overall"].copy()
        x = np.arange(len(compare_metrics))
        width = 0.36

        fig = plt.figure(figsize=(11, 6))
        plt.bar(
            x - width/2,
            overall["skill_mean"],
            width,
            label="procedural Skill",
        )
        plt.bar(
            x + width/2,
            overall["direct_mean"],
            width,
            label="direct choice",
        )
        plt.axhline(0.0, linestyle="--")
        plt.xticks(x, compare_metrics, rotation=25, ha="right")
        plt.ylabel("Causal effect")
        plt.title("EXP12 specificity falsification")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "specificity_comparison.png", dpi=180)
        plt.close(fig)

    # ------------------------------------------------------------
    # Manifest + final summary
    # ------------------------------------------------------------
    anchor_signature = (
        audit_df.groupby("offset")["decoded"]
        .agg(lambda x: sorted(set(x)))
        .to_dict()
    )

    manifest = {
        "experiment": "EXP12_independent_replication",
        "git_commit": git_commit(),
        "seed": args.seed,
        "phase": args.phase,
        "model_dir": str(model_dir),
        "model_name": model_dir.name,
        "num_tasks": len(tasks),
        "tasks_per_family": N_TASKS_PER_FAMILY,
        "families": list(FAMILIES),
        "source_hidden_index": SOURCE_H,
        "consumer_decoder_block": CONSUMER_BLOCK,
        "kv_head": KV_HEAD,
        "frozen_offsets": list(FROZEN_OFFSETS),
        "negative_offsets": list(NEGATIVE_OFFSETS),
        "frozen_readers": list(FROZEN_READERS),
        "negative_readers": list(NEGATIVE_READERS),
        "num_attention_heads": num_q_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "anchor_signature": {str(k): v for k, v in anchor_signature.items()},
        "script_sha256": sha256_file(Path(__file__).resolve()),
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
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
