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
DEFAULT_OUT = REPO_ROOT / "outputs" / "exp10_gqa_value_paths"

LABEL_TEST = 0
LABEL_IMPL = 1

SOURCE_H = 20
CONSUMER_BLOCK = 20
N_BINS = 6


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
    tasks = api["make_tasks"](int(manifest["num_tasks"]), int(manifest["seed"]))
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
                    "messages": api["messages"](task, hist, api["skills"][cond]),
                    "candidates": [
                        f"read_file('{task['test_path']}')",
                        f"read_file('{task['src_path']}')",
                    ],
                }
            )

        by_task[ti] = entries

    return by_task


def donor_maps(entries):
    lookup = {(e["wording"], e["label"]): i for i, e in enumerate(entries)}
    out = {}

    for i, e in enumerate(entries):
        opp_label = LABEL_IMPL if e["label"] == LABEL_TEST else LABEL_TEST
        opp_wording = "paraphrase" if e["wording"] == "canonical" else "canonical"

        out[i] = {
            "opposite_same_wording": lookup[(e["wording"], opp_label)],
            "same_state_cross_wording": lookup[(opp_wording, e["label"])],
            "opposite_cross_wording": lookup[(opp_wording, opp_label)],
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


def suffix_bins(k: int, n_bins: int = N_BINS):
    if k < n_bins:
        raise RuntimeError(f"Common suffix length {k} < number of bins {n_bins}")
    return [x.astype(np.int64) for x in np.array_split(np.arange(k), n_bins)]


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
    """Patch selected V-head outputs at selected token positions.

    positions: [batch, T]
    head_indices: 1-D tensor or None for all KV heads
    replacements:
      selected heads: [batch, T, Hsel, head_dim]
      all heads:      [batch, T, num_kv_heads, head_dim]
    """
    def hook_fn(module, inputs, output):
        y = output.clone()
        shape = y.shape
        yh = y.view(*shape[:-1], num_kv_heads, head_dim)

        pos = positions.to(y.device)
        rep = replacements.to(y.device, dtype=y.dtype)

        for row in range(yh.shape[0]):
            if head_indices is None:
                yh[row, pos[row], :, :] = rep[row]
            else:
                idx = head_indices.to(y.device)
                for pi, token_pos in enumerate(pos[row]):
                    yh[row, token_pos, idx, :] = rep[row, pi]

        return yh.reshape(*shape)

    handle = v_proj.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


def tokenize_entry(tok, entry):
    text = render_prompt(tok, entry["messages"])
    return tok(text, add_special_tokens=False)["input_ids"]


@torch.inference_mode()
def capture_prompt(model, tok, layers, entry, max_suffix, num_kv_heads, head_dim):
    """Capture H20 residual and block20 pre-attention V states in float32."""
    device = model.get_input_embeddings().weight.device

    text = render_prompt(tok, entry["messages"])
    enc = tok(text, add_special_tokens=False, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}

    store = {"residual": None, "v": None}
    handles = []

    def residual_hook(module, inputs, output):
        t = output[0] if isinstance(output, tuple) else output
        take = min(max_suffix, t.shape[1])
        store["residual"] = (
            t[0, -take:, :].detach().float().cpu().numpy().astype(np.float32)
        )

    def v_hook(module, inputs, output):
        take = min(max_suffix, output.shape[1])
        x = output[0, -take:, :].view(take, num_kv_heads, head_dim)
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


def prepare_task_cache(model, tok, layers, entries, num_kv_heads, head_dim):
    maps = donor_maps(entries)
    ids = [tokenize_entry(tok, e) for e in entries]

    max_suffix = []
    for i in range(len(entries)):
        ks = [longest_common_suffix(ids[i], ids[j]) for j in maps[i].values()]
        max_suffix.append(max(ks))

    cache = [
        capture_prompt(
            model, tok, layers, entries[i], max_suffix[i], num_kv_heads, head_dim
        )
        for i in range(len(entries))
    ]
    return cache, maps


def get_residual_slice(cache, k):
    if cache["residual"].shape[0] < k:
        raise RuntimeError("Residual capture shorter than requested suffix")
    return cache["residual"][-k:, :].astype(np.float32)


def get_v_slice(cache, k):
    if cache["v"].shape[0] < k:
        raise RuntimeError("V capture shorter than requested suffix")
    return cache["v"][-k:, :, :].astype(np.float32)


@torch.inference_mode()
def score(
    model,
    tok,
    layers,
    recipient,
    recipient_cache,
    num_kv_heads,
    head_dim,
    donor_cache=None,
    common_k=None,
    residual_patch=False,
    v_source_cache=None,
    head_indices=None,
    suffix_relative_indices=None,
):
    """Score the two candidates one at a time.

    suffix_relative_indices are 0..common_k-1 positions within the aligned suffix.
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
        attention_mask = torch.ones_like(input_ids)

        with ExitStack() as stack:
            if common_k is not None:
                k = int(common_k)
                full_pos_np = np.arange(plen-k, plen, dtype=np.int64)
                full_positions = torch.tensor(
                    full_pos_np[None, :], dtype=torch.long, device=device
                )

                if residual_patch:
                    if donor_cache is None:
                        raise RuntimeError("Residual patch requires donor_cache")
                    values = get_residual_slice(donor_cache, k)
                    rep = torch.tensor(
                        values[None, :, :], dtype=torch.float32, device=device
                    )
                    stack.enter_context(
                        residual_patch_hook(
                            layers[SOURCE_H - 1], full_positions, rep
                        )
                    )

                if v_source_cache is not None:
                    rel = (
                        np.arange(k, dtype=np.int64)
                        if suffix_relative_indices is None
                        else np.asarray(suffix_relative_indices, dtype=np.int64)
                    )
                    if len(rel) == 0:
                        raise RuntimeError("Empty token-position intervention")

                    pos_np = full_pos_np[rel]
                    positions = torch.tensor(
                        pos_np[None, :], dtype=torch.long, device=device
                    )

                    full_v = get_v_slice(v_source_cache, k)
                    selected_v = full_v[rel]

                    if head_indices is None:
                        values = selected_v
                        idx_tensor = None
                    else:
                        heads = np.asarray(head_indices, dtype=np.int64)
                        values = selected_v[:, heads, :]
                        idx_tensor = torch.tensor(
                            heads, dtype=torch.long, device=device
                        )

                    rep = torch.tensor(
                        values[None, :, :, :],
                        dtype=torch.float32,
                        device=device,
                    )

                    stack.enter_context(
                        value_patch_hook(
                            layers[CONSUMER_BLOCK].self_attn.v_proj,
                            positions,
                            idx_tensor,
                            rep,
                            num_kv_heads,
                            head_dim,
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

    return {
        "test_mean_logprob": scores[0],
        "impl_mean_logprob": scores[1],
        "margin": scores[1] - scores[0],
    }


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


def sign_flip_p(values, seed, n_perm=20000):
    x = np.asarray(values, dtype=np.float64)
    obs = abs(float(x.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0

    # Chunked to avoid a large temporary matrix.
    remaining = n_perm
    while remaining > 0:
        m = min(2000, remaining)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(m, len(x)))
        means = np.abs((signs * x[None, :]).mean(axis=1))
        exceed += int(np.sum(means >= obs))
        remaining -= m

    return float((exceed + 1) / (n_perm + 1))


def bh_qvalues(pvals):
    p = np.asarray(pvals, dtype=np.float64)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=np.float64)

    running = 1.0
    for i in range(n-1, -1, -1):
        rank = i + 1
        running = min(running, ranked[i] * n / rank)
        q[order[i]] = min(running, 1.0)

    return q


def append_result(
    rows, *, ti, rec, donor, relation, config, family,
    effect_type, value, heads="all", bin_index=-1, full_reference=np.nan
):
    rows.append({
        "task_idx": ti,
        "task_id": rec["task_id"],
        "recipient_condition": rec["condition"],
        "recipient_wording": rec["wording"],
        "recipient_label": rec["label"],
        "donor_condition": donor["condition"],
        "donor_label": donor["label"],
        "relation": relation,
        "config": config,
        "family": family,
        "effect_type": effect_type,
        "heads": str(heads),
        "bin_index": int(bin_index),
        "value": float(value),
        "full_reference": float(full_reference)
        if np.isfinite(full_reference) else np.nan,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=5010)
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

    num_q_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    hidden_size = int(model.config.hidden_size)
    head_dim = int(getattr(model.config, "head_dim", hidden_size // num_q_heads))

    if num_kv_heads != 4:
        raise RuntimeError(
            f"EXP10 preregistered for 4 KV heads; model has {num_kv_heads}"
        )
    if layers[CONSUMER_BLOCK].self_attn.v_proj.out_features != num_kv_heads * head_dim:
        raise RuntimeError("Unexpected v_proj geometry")

    print("=" * 84)
    print("EXP10: GQA Value-Head × Token-Position Causal Localization")
    print("Model:", model_dir)
    print("Tasks:", len(by_task))
    print("Query heads:", num_q_heads)
    print("KV heads:", num_kv_heads)
    print("Queries per KV group:", num_q_heads // num_kv_heads)
    print("Head dim:", head_dim)
    print("=" * 84)

    rows = []
    token_rows = []

    for task_no, ti in enumerate(sorted(by_task), start=1):
        entries = by_task[ti]
        cache, maps = prepare_task_cache(
            model, tok, layers, entries, num_kv_heads, head_dim
        )

        baselines = [
            score(
                model, tok, layers, rec, cache[i], num_kv_heads, head_dim
            )
            for i, rec in enumerate(entries)
        ]

        # Token audit: one exact suffix per task × wording family.
        for wording in sorted({e["wording"] for e in entries}):
            i = next(
                idx for idx, e in enumerate(entries)
                if e["wording"] == wording and e["label"] == LABEL_TEST
            )
            di = maps[i]["opposite_same_wording"]
            k = longest_common_suffix(cache[i]["ids"], cache[di]["ids"])
            bins = suffix_bins(k)
            bin_lookup = {}
            for b, rels in enumerate(bins):
                for rel in rels:
                    bin_lookup[int(rel)] = b

            suffix_ids = cache[i]["ids"][-k:]
            for rel, token_id in enumerate(suffix_ids):
                token_rows.append({
                    "task_idx": ti,
                    "task_id": entries[i]["task_id"],
                    "wording": wording,
                    "common_suffix_length": k,
                    "suffix_index": rel,
                    "offset_from_generation_boundary": rel - k,
                    "bin_index": bin_lookup[rel],
                    "token_id": int(token_id),
                    "decoded_token": tok.decode(
                        [int(token_id)],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                })

        # Main same-wording opposite-state experiments.
        for i, rec in enumerate(entries):
            di = maps[i]["opposite_same_wording"]
            donor = entries[di]
            k = longest_common_suffix(cache[i]["ids"], cache[di]["ids"])
            bins = suffix_bins(k)
            sign = 1.0 if donor["label"] == LABEL_IMPL else -1.0
            base = baselines[i]["margin"]

            residual = score(
                model, tok, layers, rec, cache[i], num_kv_heads, head_dim,
                donor_cache=cache[di],
                common_k=k,
                residual_patch=True,
            )
            e_res = sign * (residual["margin"] - base)

            append_result(
                rows, ti=ti, rec=rec, donor=donor,
                relation="opposite_same_wording",
                config="H20_residual_reference",
                family="reference",
                effect_type="residual_reference",
                value=e_res,
            )

            # Full V positive reference.
            full_suff = score(
                model, tok, layers, rec, cache[i], num_kv_heads, head_dim,
                common_k=k,
                v_source_cache=cache[di],
                head_indices=None,
            )
            e_full_suff = sign * (full_suff["margin"] - base)

            full_nec = score(
                model, tok, layers, rec, cache[i], num_kv_heads, head_dim,
                donor_cache=cache[di],
                common_k=k,
                residual_patch=True,
                v_source_cache=cache[i],
                head_indices=None,
            )
            full_retained = sign * (full_nec["margin"] - base)
            e_full_nec = e_res - full_retained

            append_result(
                rows, ti=ti, rec=rec, donor=donor,
                relation="opposite_same_wording",
                config="full_V_sufficiency",
                family="reference",
                effect_type="sufficiency",
                value=e_full_suff,
                full_reference=e_full_suff,
            )
            append_result(
                rows, ti=ti, rec=rec, donor=donor,
                relation="opposite_same_wording",
                config="full_V_necessity_loss",
                family="reference",
                effect_type="necessity_loss",
                value=e_full_nec,
                full_reference=e_full_nec,
            )

            # A: individual heads and leave-one-out.
            all_heads = np.arange(num_kv_heads, dtype=np.int64)

            for h in range(num_kv_heads):
                single = np.asarray([h], dtype=np.int64)
                without = all_heads[all_heads != h]

                for group_name, group in [
                    (f"head{h}", single),
                    (f"all_except_head{h}", without),
                ]:
                    suff = score(
                        model, tok, layers, rec, cache[i], num_kv_heads, head_dim,
                        common_k=k,
                        v_source_cache=cache[di],
                        head_indices=group,
                    )
                    e_suff = sign * (suff["margin"] - base)

                    nec = score(
                        model, tok, layers, rec, cache[i], num_kv_heads, head_dim,
                        donor_cache=cache[di],
                        common_k=k,
                        residual_patch=True,
                        v_source_cache=cache[i],
                        head_indices=group,
                    )
                    retained = sign * (nec["margin"] - base)
                    e_nec = e_res - retained

                    append_result(
                        rows, ti=ti, rec=rec, donor=donor,
                        relation="opposite_same_wording",
                        config=f"{group_name}_sufficiency",
                        family="head",
                        effect_type="sufficiency",
                        value=e_suff,
                        heads=",".join(map(str, group.tolist())),
                        full_reference=e_full_suff,
                    )
                    append_result(
                        rows, ti=ti, rec=rec, donor=donor,
                        relation="opposite_same_wording",
                        config=f"{group_name}_necessity_loss",
                        family="head",
                        effect_type="necessity_loss",
                        value=e_nec,
                        heads=",".join(map(str, group.tolist())),
                        full_reference=e_full_nec,
                    )

            # B: all-head position bins.
            for b, rels in enumerate(bins):
                suff = score(
                    model, tok, layers, rec, cache[i], num_kv_heads, head_dim,
                    common_k=k,
                    v_source_cache=cache[di],
                    head_indices=None,
                    suffix_relative_indices=rels,
                )
                e_suff = sign * (suff["margin"] - base)

                nec = score(
                    model, tok, layers, rec, cache[i], num_kv_heads, head_dim,
                    donor_cache=cache[di],
                    common_k=k,
                    residual_patch=True,
                    v_source_cache=cache[i],
                    head_indices=None,
                    suffix_relative_indices=rels,
                )
                retained = sign * (nec["margin"] - base)
                e_nec = e_res - retained

                append_result(
                    rows, ti=ti, rec=rec, donor=donor,
                    relation="opposite_same_wording",
                    config=f"bin{b}_allheads_sufficiency",
                    family="position",
                    effect_type="sufficiency",
                    value=e_suff,
                    heads="all",
                    bin_index=b,
                    full_reference=e_full_suff,
                )
                append_result(
                    rows, ti=ti, rec=rec, donor=donor,
                    relation="opposite_same_wording",
                    config=f"bin{b}_allheads_necessity_loss",
                    family="position",
                    effect_type="necessity_loss",
                    value=e_nec,
                    heads="all",
                    bin_index=b,
                    full_reference=e_full_nec,
                )

                # C: head × position matrix.
                for h in range(num_kv_heads):
                    single = np.asarray([h], dtype=np.int64)

                    suff = score(
                        model, tok, layers, rec, cache[i], num_kv_heads, head_dim,
                        common_k=k,
                        v_source_cache=cache[di],
                        head_indices=single,
                        suffix_relative_indices=rels,
                    )
                    e_suff = sign * (suff["margin"] - base)

                    nec = score(
                        model, tok, layers, rec, cache[i], num_kv_heads, head_dim,
                        donor_cache=cache[di],
                        common_k=k,
                        residual_patch=True,
                        v_source_cache=cache[i],
                        head_indices=single,
                        suffix_relative_indices=rels,
                    )
                    retained = sign * (nec["margin"] - base)
                    e_nec = e_res - retained

                    append_result(
                        rows, ti=ti, rec=rec, donor=donor,
                        relation="opposite_same_wording",
                        config=f"head{h}_bin{b}_sufficiency",
                        family="head_position",
                        effect_type="sufficiency",
                        value=e_suff,
                        heads=str(h),
                        bin_index=b,
                        full_reference=e_full_suff,
                    )
                    append_result(
                        rows, ti=ti, rec=rec, donor=donor,
                        relation="opposite_same_wording",
                        config=f"head{h}_bin{b}_necessity_loss",
                        family="head_position",
                        effect_type="necessity_loss",
                        value=e_nec,
                        heads=str(h),
                        bin_index=b,
                        full_reference=e_full_nec,
                    )

        # Full-V controls.
        for relation in [
            "self",
            "same_state_cross_wording",
            "opposite_cross_wording",
        ]:
            for i, rec in enumerate(entries):
                di = maps[i][relation]
                donor = entries[di]
                k = longest_common_suffix(cache[i]["ids"], cache[di]["ids"])
                base = baselines[i]["margin"]
                sign = 1.0 if relation == "self" else (
                    1.0 if donor["label"] == LABEL_IMPL else -1.0
                )

                suff = score(
                    model, tok, layers, rec, cache[i], num_kv_heads, head_dim,
                    common_k=k,
                    v_source_cache=cache[di],
                    head_indices=None,
                )
                e_suff = sign * (suff["margin"] - base)

                append_result(
                    rows, ti=ti, rec=rec, donor=donor,
                    relation=relation,
                    config="full_V_control_sufficiency",
                    family="control",
                    effect_type="control_sufficiency",
                    value=e_suff,
                )

                if relation == "opposite_cross_wording":
                    residual = score(
                        model, tok, layers, rec, cache[i], num_kv_heads, head_dim,
                        donor_cache=cache[di],
                        common_k=k,
                        residual_patch=True,
                    )
                    e_res = sign * (residual["margin"] - base)

                    nec = score(
                        model, tok, layers, rec, cache[i], num_kv_heads, head_dim,
                        donor_cache=cache[di],
                        common_k=k,
                        residual_patch=True,
                        v_source_cache=cache[i],
                        head_indices=None,
                    )
                    retained = sign * (nec["margin"] - base)
                    e_nec = e_res - retained

                    append_result(
                        rows, ti=ti, rec=rec, donor=donor,
                        relation=relation,
                        config="full_V_control_necessity_loss",
                        family="control",
                        effect_type="control_necessity_loss",
                        value=e_nec,
                        full_reference=e_res,
                    )

        print(f"[{task_no:03d}/{len(by_task):03d}] task {ti}")

        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = pd.DataFrame(rows)
    token_map = pd.DataFrame(token_rows)

    results.to_csv(out_dir / "intervention_results.csv", index=False)
    token_map.to_csv(out_dir / "token_position_map.csv", index=False)

    # Task is the inferential unit.
    task_level = (
        results.groupby(
            [
                "config", "relation", "family", "effect_type",
                "heads", "bin_index", "task_idx", "task_id",
            ],
            as_index=False,
        )["value"].mean()
    )

    summary_rows = []

    for keys, g in task_level.groupby(
        ["config", "relation", "family", "effect_type", "heads", "bin_index"]
    ):
        config, relation, family, effect_type, heads, bin_index = keys
        seed = args.seed + sum(
            ord(c) for c in (config + relation + family + effect_type + heads)
        )
        mean, lo, hi = bootstrap_ci(g["value"], seed)
        p = sign_flip_p(g["value"], seed + 100000)

        summary_rows.append({
            "config": config,
            "relation": relation,
            "family": family,
            "effect_type": effect_type,
            "heads": heads,
            "bin_index": int(bin_index),
            "mean": mean,
            "ci_low": lo,
            "ci_high": hi,
            "sign_flip_p": p,
            "num_tasks": int(g["task_idx"].nunique()),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df["bh_q"] = np.nan

    # BH only within exploratory head×position family, separately by effect type.
    for effect_type in ["sufficiency", "necessity_loss"]:
        mask = (
            (summary_df["family"] == "head_position")
            & (summary_df["effect_type"] == effect_type)
            & (summary_df["relation"] == "opposite_same_wording")
        )
        if mask.any():
            summary_df.loc[mask, "bh_q"] = bh_qvalues(
                summary_df.loc[mask, "sign_flip_p"].to_numpy()
            )

    summary_df.to_csv(out_dir / "summary_by_config.csv", index=False)

    def task_values(config, relation="opposite_same_wording"):
        return (
            task_level[
                (task_level["config"] == config)
                & (task_level["relation"] == relation)
            ]
            .sort_values("task_idx")["value"]
            .to_numpy(np.float64)
        )

    def get_summary(config, relation="opposite_same_wording"):
        return summary_df[
            (summary_df["config"] == config)
            & (summary_df["relation"] == relation)
        ].iloc[0]

    full_s = task_values("full_V_sufficiency")
    full_n = task_values("full_V_necessity_loss")

    # Head profile including direct and leave-one-out marginal effects.
    head_rows = []
    for h in range(num_kv_heads):
        direct_s = task_values(f"head{h}_sufficiency")
        direct_n = task_values(f"head{h}_necessity_loss")
        without_s = task_values(f"all_except_head{h}_sufficiency")
        without_n = task_values(f"all_except_head{h}_necessity_loss")

        marginal_s = full_s - without_s
        marginal_n = full_n - without_n

        ds = bootstrap_ci(direct_s, args.seed + 2000 + h)
        dn = bootstrap_ci(direct_n, args.seed + 2100 + h)
        ms = bootstrap_ci(marginal_s, args.seed + 2200 + h)
        mn = bootstrap_ci(marginal_n, args.seed + 2300 + h)

        head_rows.append({
            "head": h,
            "direct_sufficiency_mean": ds[0],
            "direct_sufficiency_ci_low": ds[1],
            "direct_sufficiency_ci_high": ds[2],
            "direct_necessity_mean": dn[0],
            "direct_necessity_ci_low": dn[1],
            "direct_necessity_ci_high": dn[2],
            "leave_one_out_marginal_sufficiency_mean": ms[0],
            "leave_one_out_marginal_sufficiency_ci_low": ms[1],
            "leave_one_out_marginal_sufficiency_ci_high": ms[2],
            "leave_one_out_marginal_necessity_mean": mn[0],
            "leave_one_out_marginal_necessity_ci_low": mn[1],
            "leave_one_out_marginal_necessity_ci_high": mn[2],
        })

    head_profile = pd.DataFrame(head_rows)
    head_profile.to_csv(out_dir / "head_profile.csv", index=False)

    # Position profile.
    pos_rows = []
    for b in range(N_BINS):
        s = get_summary(f"bin{b}_allheads_sufficiency")
        n = get_summary(f"bin{b}_allheads_necessity_loss")
        pos_rows.append({
            "bin_index": b,
            "sufficiency_mean": float(s["mean"]),
            "sufficiency_ci_low": float(s["ci_low"]),
            "sufficiency_ci_high": float(s["ci_high"]),
            "necessity_mean": float(n["mean"]),
            "necessity_ci_low": float(n["ci_low"]),
            "necessity_ci_high": float(n["ci_high"]),
            "sufficiency_fraction_of_full": float(s["mean"]) / float(full_s.mean()),
            "necessity_fraction_of_full": float(n["mean"]) / float(full_n.mean()),
        })

    position_profile = pd.DataFrame(pos_rows)
    position_profile.to_csv(out_dir / "position_profile.csv", index=False)

    # Interaction matrix.
    matrix = summary_df[
        (summary_df["family"] == "head_position")
        & (summary_df["relation"] == "opposite_same_wording")
    ].copy()

    matrix["head"] = matrix["heads"].astype(int)
    matrix = matrix[
        [
            "head", "bin_index", "effect_type", "mean",
            "ci_low", "ci_high", "sign_flip_p", "bh_q"
        ]
    ].sort_values(["effect_type", "head", "bin_index"])

    matrix.to_csv(out_dir / "head_position_matrix.csv", index=False)

    # Plots.
    fig = plt.figure(figsize=(9, 5.5))
    x = np.arange(num_kv_heads)
    width = 0.36
    plt.bar(
        x-width/2,
        head_profile["direct_sufficiency_mean"],
        width,
        label="direct sufficiency",
    )
    plt.bar(
        x+width/2,
        head_profile["direct_necessity_mean"],
        width,
        label="direct necessity",
    )
    plt.axhline(0.0, linestyle="--")
    plt.xticks(x, [f"KV{h}" for h in range(num_kv_heads)])
    plt.ylabel("Causal effect")
    plt.title("EXP10 H20 GQA value-head profile")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "head_profile.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(9, 5.5))
    plt.plot(
        position_profile["bin_index"],
        position_profile["sufficiency_mean"],
        marker="o",
        label="sufficiency",
    )
    plt.plot(
        position_profile["bin_index"],
        position_profile["necessity_mean"],
        marker="o",
        label="necessity",
    )
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Normalized common-suffix bin (B0 earliest → B5 latest)")
    plt.ylabel("Causal effect")
    plt.title("EXP10 token-position profile")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "position_profile.png", dpi=180)
    plt.close(fig)

    for effect_type, filename, title in [
        ("sufficiency", "head_position_sufficiency.png", "Head × position sufficiency"),
        ("necessity_loss", "head_position_necessity.png", "Head × position necessity"),
    ]:
        pivot = (
            matrix[matrix["effect_type"] == effect_type]
            .pivot(index="head", columns="bin_index", values="mean")
            .reindex(index=range(num_kv_heads), columns=range(N_BINS))
        )

        fig = plt.figure(figsize=(9, 4.5))
        im = plt.imshow(pivot.to_numpy(), aspect="auto")
        plt.colorbar(im, label="Causal effect")
        plt.xticks(range(N_BINS), [f"B{i}" for i in range(N_BINS)])
        plt.yticks(range(num_kv_heads), [f"KV{i}" for i in range(num_kv_heads)])
        plt.xlabel("Common-suffix bin")
        plt.ylabel("KV head")
        plt.title(title)
        plt.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)

    # Controls + compact summary.
    residual_row = get_summary("H20_residual_reference")
    full_s_row = get_summary("full_V_sufficiency")
    full_n_row = get_summary("full_V_necessity_loss")
    self_row = get_summary("full_V_control_sufficiency", "self")
    same_row = get_summary(
        "full_V_control_sufficiency", "same_state_cross_wording"
    )
    cross_s_row = get_summary(
        "full_V_control_sufficiency", "opposite_cross_wording"
    )
    cross_n_row = get_summary(
        "full_V_control_necessity_loss", "opposite_cross_wording"
    )

    best_direct_s = head_profile.loc[
        head_profile["direct_sufficiency_mean"].idxmax()
    ]
    best_direct_n = head_profile.loc[
        head_profile["direct_necessity_mean"].idxmax()
    ]
    best_bin_s = position_profile.loc[
        position_profile["sufficiency_mean"].idxmax()
    ]
    best_bin_n = position_profile.loc[
        position_profile["necessity_mean"].idxmax()
    ]

    sig_both = []
    suff_m = matrix[matrix["effect_type"] == "sufficiency"].set_index(
        ["head", "bin_index"]
    )
    nec_m = matrix[matrix["effect_type"] == "necessity_loss"].set_index(
        ["head", "bin_index"]
    )

    for key in sorted(set(suff_m.index).intersection(set(nec_m.index))):
        srow = suff_m.loc[key]
        nrow = nec_m.loc[key]
        if float(srow["bh_q"]) < 0.05 and float(nrow["bh_q"]) < 0.05:
            sig_both.append({
                "head": int(key[0]),
                "bin_index": int(key[1]),
                "sufficiency_mean": float(srow["mean"]),
                "sufficiency_q": float(srow["bh_q"]),
                "necessity_mean": float(nrow["mean"]),
                "necessity_q": float(nrow["bh_q"]),
            })

    manifest = {
        "experiment": "EXP10_gqa_value_paths",
        "git_commit": git_commit(),
        "source_exp01b_git_commit": manifest01b.get("git_commit"),
        "seed": args.seed,
        "model_dir": str(model_dir),
        "model_name": model_dir.name,
        "source_hidden_index": SOURCE_H,
        "consumer_decoder_block": CONSUMER_BLOCK,
        "num_attention_heads": num_q_heads,
        "num_key_value_heads": num_kv_heads,
        "queries_per_kv_group": num_q_heads // num_kv_heads,
        "head_dim": head_dim,
        "num_position_bins": N_BINS,
        "matrix_multiple_testing": "BH-FDR separately for sufficiency and necessity",
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

    def ci(row):
        return [float(row["ci_low"]), float(row["ci_high"])]

    summary = {
        "experiment": "EXP10_gqa_value_paths",
        "H20_residual_reference_mean": float(residual_row["mean"]),
        "H20_residual_reference_95ci": ci(residual_row),
        "full_V_sufficiency_mean": float(full_s_row["mean"]),
        "full_V_sufficiency_95ci": ci(full_s_row),
        "full_V_necessity_mean": float(full_n_row["mean"]),
        "full_V_necessity_95ci": ci(full_n_row),
        "full_V_self_control_mean": float(self_row["mean"]),
        "full_V_same_state_cross_wording_mean": float(same_row["mean"]),
        "full_V_opposite_cross_wording_sufficiency_mean": float(cross_s_row["mean"]),
        "full_V_opposite_cross_wording_necessity_mean": float(cross_n_row["mean"]),
        "best_direct_sufficiency_head_exploratory": int(best_direct_s["head"]),
        "best_direct_sufficiency_mean": float(best_direct_s["direct_sufficiency_mean"]),
        "best_direct_necessity_head_exploratory": int(best_direct_n["head"]),
        "best_direct_necessity_mean": float(best_direct_n["direct_necessity_mean"]),
        "best_position_bin_sufficiency_exploratory": int(best_bin_s["bin_index"]),
        "best_position_bin_sufficiency_mean": float(best_bin_s["sufficiency_mean"]),
        "best_position_bin_necessity_exploratory": int(best_bin_n["bin_index"]),
        "best_position_bin_necessity_mean": float(best_bin_n["necessity_mean"]),
        "head_position_cells_fdr_lt_0_05_in_both": sig_both,
        "head_profile": head_profile.to_dict(orient="records"),
        "position_profile": position_profile.to_dict(orient="records"),
        "decision_rule": (
            "Head or position concentration requires causal sufficiency and necessity. "
            "The 24 head×position cells are exploratory and must satisfy BH-FDR in both "
            "effect families before being prioritized for independent replication."
        ),
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
