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
DEFAULT_OUT = REPO_ROOT / "outputs" / "exp11_exact_value_path"

LABEL_TEST = 0
LABEL_IMPL = 1

SOURCE_H = 20
CONSUMER_BLOCK = 20
KV_HEAD = 0
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
            entries.append({
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
            })
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


def suffix_bins(k, n_bins=N_BINS):
    if k < n_bins:
        raise RuntimeError(f"Common suffix length {k} < {n_bins} bins")
    return [x.astype(np.int64) for x in np.array_split(np.arange(k), n_bins)]


def render_prompt(tok, messages):
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("Expected Qwen-style model.model.layers")


def tokenize_entry(tok, entry):
    text = render_prompt(tok, entry["messages"])
    return tok(text, add_special_tokens=False)["input_ids"]


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
    """Replace selected query-head slices of o_proj input for all sequence positions."""
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
def capture_prompt(model, tok, layers, entry, max_suffix, num_kv_heads, head_dim):
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
    handles.append(layers[CONSUMER_BLOCK].self_attn.v_proj.register_forward_hook(v_hook))

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
            model, tok, layers, entries[i], max_suffix[i],
            num_kv_heads, head_dim
        )
        for i in range(len(entries))
    ]
    return cache, maps


def get_residual_slice(cache, k):
    return cache["residual"][-k:, :].astype(np.float32)


def get_v_slice(cache, k):
    return cache["v"][-k:, :, :].astype(np.float32)


def prompt_positions(plen, k, rel_indices, device):
    full = np.arange(plen-k, plen, dtype=np.int64)
    rel = np.asarray(rel_indices, dtype=np.int64)
    return torch.tensor(full[rel][None, :], dtype=torch.long, device=device)


@torch.inference_mode()
def score_value_intervention(
    model, tok, layers, recipient, recipient_cache,
    num_kv_heads, head_dim,
    donor_cache=None, common_k=None,
    residual_patch=False,
    v_source_cache=None,
    v_rel_indices=None,
    v_heads=(KV_HEAD,),
):
    """Standard causal score used for exact-offset and leave-one-out experiments."""
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
                full_pos = torch.tensor(
                    np.arange(plen-k, plen, dtype=np.int64)[None, :],
                    dtype=torch.long, device=device,
                )

                if residual_patch:
                    if donor_cache is None:
                        raise RuntimeError("Residual patch requires donor_cache")
                    rv = get_residual_slice(donor_cache, k)
                    rep = torch.tensor(rv[None], dtype=torch.float32, device=device)
                    stack.enter_context(
                        residual_patch_hook(layers[SOURCE_H - 1], full_pos, rep)
                    )

                if v_source_cache is not None:
                    rel = (
                        np.arange(k, dtype=np.int64)
                        if v_rel_indices is None
                        else np.asarray(v_rel_indices, dtype=np.int64)
                    )
                    pos = prompt_positions(plen, k, rel, device)
                    vv = get_v_slice(v_source_cache, k)[rel][:, list(v_heads), :]
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
def forward_candidate_capture_oproj(
    model, tok, layers, recipient, candidate,
    recipient_cache, num_kv_heads, num_q_heads, head_dim,
    donor_cache=None, common_k=None, v_rel_indices=None,
):
    """Return candidate mean logP and block20 o_proj input.

    If donor_cache is supplied, apply verified KV0 V patch at v_rel_indices.
    """
    device = model.get_input_embeddings().weight.device
    prompt_text = render_prompt(tok, recipient["messages"])
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    cids = tok(candidate, add_special_tokens=False)["input_ids"]
    plen = len(prompt_ids)
    ids = prompt_ids + cids

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    store = {}

    with ExitStack() as stack:
        if donor_cache is not None:
            k = int(common_k)
            rel = np.asarray(v_rel_indices, dtype=np.int64)
            pos = prompt_positions(plen, k, rel, device)
            vv = get_v_slice(donor_cache, k)[rel][:, [KV_HEAD], :]
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
                store, "oproj_in", num_q_heads, head_dim,
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

    return score, store["oproj_in"]


@torch.inference_mode()
def forward_candidate_replace_oproj(
    model, tok, layers, recipient, candidate,
    num_q_heads, head_dim,
    replacement_heads_np, head_indices,
    recipient_cache=None,
    num_kv_heads=None,
    donor_cache=None,
    common_k=None,
    v_rel_indices=None,
):
    """Run a candidate with selected block20 o_proj-input heads replaced.

    Optional donor_cache + v_rel_indices applies the verified upstream V patch
    before replacing/clamping query-head outputs.
    """
    device = model.get_input_embeddings().weight.device
    prompt_text = render_prompt(tok, recipient["messages"])
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    cids = tok(candidate, add_special_tokens=False)["input_ids"]
    plen = len(prompt_ids)
    ids = prompt_ids + cids

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    heads = torch.tensor(head_indices, dtype=torch.long, device=device)
    rep = torch.tensor(replacement_heads_np, dtype=torch.float32, device=device)

    with ExitStack() as stack:
        if donor_cache is not None:
            k = int(common_k)
            rel = np.asarray(v_rel_indices, dtype=np.int64)
            pos = prompt_positions(plen, k, rel, device)
            vv = get_v_slice(donor_cache, k)[rel][:, [KV_HEAD], :]
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


def bootstrap_ci(values, seed, n_boot=5000):
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = np.asarray([
        rng.choice(x, size=len(x), replace=True).mean()
        for _ in range(n_boot)
    ])
    return (
        float(x.mean()),
        float(np.quantile(boots, .025)),
        float(np.quantile(boots, .975)),
    )


def sign_flip_p(values, seed, n_perm=20000):
    x = np.asarray(values, dtype=np.float64)
    obs = abs(float(x.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    left = n_perm
    while left > 0:
        m = min(2000, left)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(m, len(x)))
        means = np.abs((signs * x[None, :]).mean(axis=1))
        exceed += int(np.sum(means >= obs))
        left -= m
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


def signed_effect(margin, baseline_margin, donor):
    sign = 1.0 if donor["label"] == LABEL_IMPL else -1.0
    return sign * (margin - baseline_margin)


def score_pair_from_candidate_scores(scores):
    # [test_mean_logp, impl_mean_logp] -> margin
    return scores[1] - scores[0]


def append_offset(rows, ti, rec, donor, offset, endpoint, value, wording):
    rows.append({
        "task_idx": ti,
        "task_id": rec["task_id"],
        "recipient_condition": rec["condition"],
        "recipient_wording": wording,
        "donor_condition": donor["condition"],
        "donor_label": donor["label"],
        "offset": int(offset),
        "endpoint": endpoint,
        "value": float(value),
    })


def append_reader(rows, ti, rec, donor, reader, endpoint, value, leakage_ratio, wording):
    rows.append({
        "task_idx": ti,
        "task_id": rec["task_id"],
        "recipient_condition": rec["condition"],
        "recipient_wording": wording,
        "donor_condition": donor["condition"],
        "donor_label": donor["label"],
        "reader": str(reader),
        "endpoint": endpoint,
        "value": float(value),
        "non_kv0_delta_ratio": float(leakage_ratio),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=5111)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--phase",
        choices=["all", "offsets", "readers"],
        default="all",
    )
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

    if num_q_heads != 28 or num_kv_heads != 4:
        raise RuntimeError(
            f"EXP11 preregistered for 28 Q / 4 KV heads; got "
            f"{num_q_heads} / {num_kv_heads}"
        )
    if num_q_heads % num_kv_heads != 0:
        raise RuntimeError("Query heads are not evenly divisible by KV heads")

    q_per_kv = num_q_heads // num_kv_heads
    reader_heads = list(range(KV_HEAD*q_per_kv, (KV_HEAD+1)*q_per_kv))
    non_reader_heads = [h for h in range(num_q_heads) if h not in reader_heads]

    print("=" * 86)
    print("EXP11: Exact KV0×Token -> Query-Head Value Path")
    print("Model:", model_dir)
    print("Tasks:", len(by_task))
    print("Q heads:", num_q_heads, "KV heads:", num_kv_heads)
    print("KV0 readers:", reader_heads)
    print("Phase:", args.phase)
    print("=" * 86)

    offset_rows = []
    reader_rows = []
    token_rows = []

    for task_no, ti in enumerate(sorted(by_task), start=1):
        entries = by_task[ti]
        cache, maps = prepare_task_cache(
            model, tok, layers, entries, num_kv_heads, head_dim
        )

        # Baseline margins once per entry.
        base_margins = [
            score_value_intervention(
                model, tok, layers, rec, cache[i],
                num_kv_heads, head_dim,
            )
            for i, rec in enumerate(entries)
        ]

        for i, rec in enumerate(entries):
            di = maps[i]["opposite_same_wording"]
            donor = entries[di]
            k = longest_common_suffix(cache[i]["ids"], cache[di]["ids"])
            bins = suffix_bins(k)
            b5 = bins[-1]
            b5_offsets = [int(rel-k) for rel in b5]
            sign = 1.0 if donor["label"] == LABEL_IMPL else -1.0
            base_margin = base_margins[i]

            # Token audit for B5, once for every recipient condition.
            suffix_ids = cache[i]["ids"][-k:]
            for rel in b5:
                token_id = int(suffix_ids[int(rel)])
                token_rows.append({
                    "task_idx": ti,
                    "task_id": rec["task_id"],
                    "recipient_condition": rec["condition"],
                    "recipient_wording": rec["wording"],
                    "recipient_label": rec["label"],
                    "common_suffix_length": k,
                    "suffix_index": int(rel),
                    "offset": int(rel-k),
                    "token_id": token_id,
                    "decoded_token": tok.decode(
                        [token_id],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                })

            # References needed by offsets and sanity checks.
            residual_margin = score_value_intervention(
                model, tok, layers, rec, cache[i],
                num_kv_heads, head_dim,
                donor_cache=cache[di],
                common_k=k,
                residual_patch=True,
            )
            residual_effect = sign * (residual_margin - base_margin)

            full_b5_margin = score_value_intervention(
                model, tok, layers, rec, cache[i],
                num_kv_heads, head_dim,
                common_k=k,
                v_source_cache=cache[di],
                v_rel_indices=b5,
                v_heads=(KV_HEAD,),
            )
            full_b5_effect = sign * (full_b5_margin - base_margin)

            residual_plus_clamp_margin = score_value_intervention(
                model, tok, layers, rec, cache[i],
                num_kv_heads, head_dim,
                donor_cache=cache[di],
                common_k=k,
                residual_patch=True,
                v_source_cache=cache[i],
                v_rel_indices=b5,
                v_heads=(KV_HEAD,),
            )
            full_b5_nec = residual_effect - sign * (
                residual_plus_clamp_margin - base_margin
            )

            if args.phase in ("all", "offsets"):
                # Exact offset interventions.
                for rel, offset in zip(b5, b5_offsets):
                    rel_arr = np.asarray([int(rel)], dtype=np.int64)

                    single_margin = score_value_intervention(
                        model, tok, layers, rec, cache[i],
                        num_kv_heads, head_dim,
                        common_k=k,
                        v_source_cache=cache[di],
                        v_rel_indices=rel_arr,
                        v_heads=(KV_HEAD,),
                    )
                    suff = sign * (single_margin - base_margin)

                    residual_clamp_margin = score_value_intervention(
                        model, tok, layers, rec, cache[i],
                        num_kv_heads, head_dim,
                        donor_cache=cache[di],
                        common_k=k,
                        residual_patch=True,
                        v_source_cache=cache[i],
                        v_rel_indices=rel_arr,
                        v_heads=(KV_HEAD,),
                    )
                    nec = residual_effect - sign * (
                        residual_clamp_margin - base_margin
                    )

                    loo_rel = np.asarray(
                        [int(x) for x in b5 if int(x) != int(rel)],
                        dtype=np.int64,
                    )
                    if len(loo_rel) == 0:
                        loo_loss = full_b5_effect
                    else:
                        loo_margin = score_value_intervention(
                            model, tok, layers, rec, cache[i],
                            num_kv_heads, head_dim,
                            common_k=k,
                            v_source_cache=cache[di],
                            v_rel_indices=loo_rel,
                            v_heads=(KV_HEAD,),
                        )
                        loo_effect = sign * (loo_margin - base_margin)
                        loo_loss = full_b5_effect - loo_effect

                    append_offset(
                        offset_rows, ti, rec, donor, offset,
                        "single_token_sufficiency", suff, rec["wording"]
                    )
                    append_offset(
                        offset_rows, ti, rec, donor, offset,
                        "single_token_necessity_loss", nec, rec["wording"]
                    )
                    append_offset(
                        offset_rows, ti, rec, donor, offset,
                        "leave_one_out_loss", loo_loss, rec["wording"]
                    )

                # Store references as offset=0 pseudo rows for audit only.
                append_offset(
                    offset_rows, ti, rec, donor, 0,
                    "full_KV0_B5_sufficiency", full_b5_effect, rec["wording"]
                )
                append_offset(
                    offset_rows, ti, rec, donor, 0,
                    "full_KV0_B5_necessity_loss", full_b5_nec, rec["wording"]
                )
                append_offset(
                    offset_rows, ti, rec, donor, 0,
                    "H20_residual_reference", residual_effect, rec["wording"]
                )

            if args.phase in ("all", "readers"):
                # Candidate-specific baseline and verified V-patched captures.
                base_candidate_scores = []
                patch_candidate_scores = []
                base_caps = []
                patch_caps = []

                for candidate in rec["candidates"]:
                    bs, bc = forward_candidate_capture_oproj(
                        model, tok, layers, rec, candidate,
                        cache[i], num_kv_heads, num_q_heads, head_dim,
                    )
                    ps, pc = forward_candidate_capture_oproj(
                        model, tok, layers, rec, candidate,
                        cache[i], num_kv_heads, num_q_heads, head_dim,
                        donor_cache=cache[di],
                        common_k=k,
                        v_rel_indices=b5,
                    )
                    base_candidate_scores.append(bs)
                    patch_candidate_scores.append(ps)
                    base_caps.append(bc)
                    patch_caps.append(pc)

                reader_base_margin = score_pair_from_candidate_scores(
                    base_candidate_scores
                )
                reader_patch_margin = score_pair_from_candidate_scores(
                    patch_candidate_scores
                )
                reader_full_effect = sign * (
                    reader_patch_margin - reader_base_margin
                )

                # Exact architecture sanity: V0 should only induce Q0..Q6 delta.
                num = 0.0
                den = 0.0
                for bc, pc in zip(base_caps, patch_caps):
                    delta = pc - bc
                    num += float(np.sum(delta[:, :, non_reader_heads, :] ** 2))
                    den += float(np.sum(delta[:, :, reader_heads, :] ** 2))
                leakage_ratio = float(np.sqrt(num / (den + 1e-30)))

                append_reader(
                    reader_rows, ti, rec, donor, "full_KV0_B5",
                    "verified_V_effect", reader_full_effect,
                    leakage_ratio, rec["wording"]
                )

                # Single reader heads.
                for qh in reader_heads:
                    suff_scores = []
                    nec_scores = []

                    for candidate_idx, candidate in enumerate(rec["candidates"]):
                        bc = base_caps[candidate_idx]
                        pc = patch_caps[candidate_idx]

                        # Path sufficiency: baseline upstream, patched Qh output.
                        suff_score = forward_candidate_replace_oproj(
                            model, tok, layers, rec, candidate,
                            num_q_heads, head_dim,
                            replacement_heads_np=pc[:, :, [qh], :],
                            head_indices=[qh],
                        )
                        suff_scores.append(suff_score)

                        # Path necessity: verified V patch upstream, clamp Qh baseline.
                        nec_score = forward_candidate_replace_oproj(
                            model, tok, layers, rec, candidate,
                            num_q_heads, head_dim,
                            replacement_heads_np=bc[:, :, [qh], :],
                            head_indices=[qh],
                            recipient_cache=cache[i],
                            num_kv_heads=num_kv_heads,
                            donor_cache=cache[di],
                            common_k=k,
                            v_rel_indices=b5,
                        )
                        nec_scores.append(nec_score)

                    suff_margin = score_pair_from_candidate_scores(suff_scores)
                    nec_margin = score_pair_from_candidate_scores(nec_scores)

                    suff_effect = sign * (suff_margin - reader_base_margin)
                    retained = sign * (nec_margin - reader_base_margin)
                    necessity_loss = reader_full_effect - retained

                    append_reader(
                        reader_rows, ti, rec, donor, qh,
                        "path_sufficiency", suff_effect,
                        leakage_ratio, rec["wording"]
                    )
                    append_reader(
                        reader_rows, ti, rec, donor, qh,
                        "path_necessity_loss", necessity_loss,
                        leakage_ratio, rec["wording"]
                    )

                # all7 reconstruction / removal sanity.
                all7_suff_scores = []
                all7_nec_scores = []
                nonreader_suff_scores = []
                nonreader_nec_scores = []

                for candidate_idx, candidate in enumerate(rec["candidates"]):
                    bc = base_caps[candidate_idx]
                    pc = patch_caps[candidate_idx]

                    all7_suff_scores.append(
                        forward_candidate_replace_oproj(
                            model, tok, layers, rec, candidate,
                            num_q_heads, head_dim,
                            replacement_heads_np=pc[:, :, reader_heads, :],
                            head_indices=reader_heads,
                        )
                    )
                    all7_nec_scores.append(
                        forward_candidate_replace_oproj(
                            model, tok, layers, rec, candidate,
                            num_q_heads, head_dim,
                            replacement_heads_np=bc[:, :, reader_heads, :],
                            head_indices=reader_heads,
                            recipient_cache=cache[i],
                            num_kv_heads=num_kv_heads,
                            donor_cache=cache[di],
                            common_k=k,
                            v_rel_indices=b5,
                        )
                    )

                    # Negative architecture control: transmit/clamp only Q7..Q27 delta.
                    nonreader_suff_scores.append(
                        forward_candidate_replace_oproj(
                            model, tok, layers, rec, candidate,
                            num_q_heads, head_dim,
                            replacement_heads_np=pc[:, :, non_reader_heads, :],
                            head_indices=non_reader_heads,
                        )
                    )
                    nonreader_nec_scores.append(
                        forward_candidate_replace_oproj(
                            model, tok, layers, rec, candidate,
                            num_q_heads, head_dim,
                            replacement_heads_np=bc[:, :, non_reader_heads, :],
                            head_indices=non_reader_heads,
                            recipient_cache=cache[i],
                            num_kv_heads=num_kv_heads,
                            donor_cache=cache[di],
                            common_k=k,
                            v_rel_indices=b5,
                        )
                    )

                all7_suff_margin = score_pair_from_candidate_scores(all7_suff_scores)
                all7_nec_margin = score_pair_from_candidate_scores(all7_nec_scores)
                non_suff_margin = score_pair_from_candidate_scores(nonreader_suff_scores)
                non_nec_margin = score_pair_from_candidate_scores(nonreader_nec_scores)

                all7_suff = sign * (all7_suff_margin - reader_base_margin)
                all7_retained = sign * (all7_nec_margin - reader_base_margin)
                all7_nec = reader_full_effect - all7_retained

                non_suff = sign * (non_suff_margin - reader_base_margin)
                non_retained = sign * (non_nec_margin - reader_base_margin)
                non_nec = reader_full_effect - non_retained

                append_reader(
                    reader_rows, ti, rec, donor, "all7",
                    "path_sufficiency", all7_suff,
                    leakage_ratio, rec["wording"]
                )
                append_reader(
                    reader_rows, ti, rec, donor, "all7",
                    "path_necessity_loss", all7_nec,
                    leakage_ratio, rec["wording"]
                )
                append_reader(
                    reader_rows, ti, rec, donor, "Q7_Q27",
                    "path_sufficiency", non_suff,
                    leakage_ratio, rec["wording"]
                )
                append_reader(
                    reader_rows, ti, rec, donor, "Q7_Q27",
                    "path_necessity_loss", non_nec,
                    leakage_ratio, rec["wording"]
                )

        print(f"[{task_no:03d}/{len(by_task):03d}] task {ti}")

        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    token_df = pd.DataFrame(token_rows)
    token_df.to_csv(out_dir / "token_audit.csv", index=False)

    if offset_rows:
        pd.DataFrame(offset_rows).to_csv(out_dir / "offset_results.csv", index=False)
    if reader_rows:
        pd.DataFrame(reader_rows).to_csv(out_dir / "reader_results.csv", index=False)

    # ---------------------------
    # Offset inference
    # ---------------------------
    offset_summary = pd.DataFrame()
    if offset_rows:
        odf = pd.DataFrame(offset_rows)

        task_o = (
            odf.groupby(
                ["offset", "endpoint", "task_idx", "task_id"],
                as_index=False,
            )["value"].mean()
        )

        srows = []
        for (offset, endpoint), g in task_o.groupby(["offset", "endpoint"]):
            seed = args.seed + 1000 + abs(int(offset))*31 + sum(map(ord, endpoint))
            mean, lo, hi = bootstrap_ci(g["value"], seed)
            p = sign_flip_p(g["value"], seed + 50000)
            srows.append({
                "offset": int(offset),
                "endpoint": endpoint,
                "mean": mean,
                "ci_low": lo,
                "ci_high": hi,
                "sign_flip_p": p,
                "num_tasks": int(g["task_idx"].nunique()),
            })

        offset_summary = pd.DataFrame(srows)
        offset_summary["bh_q"] = np.nan

        scan_endpoints = [
            "single_token_sufficiency",
            "single_token_necessity_loss",
            "leave_one_out_loss",
        ]
        for endpoint in scan_endpoints:
            mask = (
                (offset_summary["endpoint"] == endpoint)
                & (offset_summary["offset"] < 0)
            )
            if mask.any():
                offset_summary.loc[mask, "bh_q"] = bh_qvalues(
                    offset_summary.loc[mask, "sign_flip_p"].to_numpy()
                )

        # Attach modal decoded token for audit readability.
        token_mode = (
            token_df.groupby(["offset", "decoded_token"])
            .size().reset_index(name="n")
            .sort_values(["offset", "n"], ascending=[True, False])
            .drop_duplicates("offset")
            [["offset", "decoded_token"]]
        )
        offset_summary = offset_summary.merge(token_mode, on="offset", how="left")
        offset_summary.to_csv(out_dir / "offset_summary.csv", index=False)

        # Plot exact scan only.
        plot_df = offset_summary[
            offset_summary["endpoint"].isin(scan_endpoints)
            & (offset_summary["offset"] < 0)
        ].copy()

        fig = plt.figure(figsize=(11, 5.8))
        for endpoint in scan_endpoints:
            d = plot_df[plot_df["endpoint"] == endpoint].sort_values("offset")
            plt.plot(d["offset"], d["mean"], marker="o", label=endpoint)
        plt.axhline(0.0, linestyle="--")
        plt.xlabel("Token offset from generation boundary")
        plt.ylabel("Causal effect")
        plt.title("EXP11 exact KV0×B5 token localization")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "offset_profile.png", dpi=180)
        plt.close(fig)

    # ---------------------------
    # Reader-head inference
    # ---------------------------
    reader_summary = pd.DataFrame()
    if reader_rows:
        rdf = pd.DataFrame(reader_rows)

        task_r = (
            rdf.groupby(
                ["reader", "endpoint", "task_idx", "task_id"],
                as_index=False,
            )
            .agg(
                value=("value", "mean"),
                non_kv0_delta_ratio=("non_kv0_delta_ratio", "mean"),
            )
        )

        rrows = []
        for (reader, endpoint), g in task_r.groupby(["reader", "endpoint"]):
            seed = args.seed + 2000 + sum(map(ord, str(reader)+endpoint))
            mean, lo, hi = bootstrap_ci(g["value"], seed)
            p = sign_flip_p(g["value"], seed + 50000)
            rrows.append({
                "reader": str(reader),
                "endpoint": endpoint,
                "mean": mean,
                "ci_low": lo,
                "ci_high": hi,
                "sign_flip_p": p,
                "num_tasks": int(g["task_idx"].nunique()),
                "mean_non_kv0_delta_ratio": float(
                    g["non_kv0_delta_ratio"].mean()
                ),
            })

        reader_summary = pd.DataFrame(rrows)
        reader_summary["bh_q"] = np.nan

        for endpoint in ["path_sufficiency", "path_necessity_loss"]:
            mask = (
                (reader_summary["endpoint"] == endpoint)
                & (reader_summary["reader"].isin([str(x) for x in reader_heads]))
            )
            if mask.any():
                reader_summary.loc[mask, "bh_q"] = bh_qvalues(
                    reader_summary.loc[mask, "sign_flip_p"].to_numpy()
                )

        # Wording subgroup means for Q0..Q6.
        subgroup = (
            rdf[
                rdf["reader"].isin([str(x) for x in reader_heads])
                if rdf["reader"].dtype == object else
                rdf["reader"].astype(str).isin([str(x) for x in reader_heads])
            ]
            .assign(reader=lambda x: x["reader"].astype(str))
            .groupby(["reader", "endpoint", "recipient_wording"], as_index=False)
            ["value"].mean()
        )
        subgroup_wide = subgroup.pivot_table(
            index=["reader", "endpoint"],
            columns="recipient_wording",
            values="value",
        ).reset_index()

        reader_summary = reader_summary.merge(
            subgroup_wide,
            on=["reader", "endpoint"],
            how="left",
        )

        reader_summary.to_csv(out_dir / "reader_summary.csv", index=False)

        # Plot Q0..Q6.
        qplot = reader_summary[
            reader_summary["reader"].isin([str(x) for x in reader_heads])
        ].copy()
        fig = plt.figure(figsize=(10, 5.8))
        x = np.arange(len(reader_heads))
        width = .36

        suff = (
            qplot[qplot["endpoint"] == "path_sufficiency"]
            .set_index("reader").loc[[str(x) for x in reader_heads]]
        )
        nec = (
            qplot[qplot["endpoint"] == "path_necessity_loss"]
            .set_index("reader").loc[[str(x) for x in reader_heads]]
        )

        plt.bar(x-width/2, suff["mean"], width, label="path sufficiency")
        plt.bar(x+width/2, nec["mean"], width, label="path necessity")
        plt.axhline(0.0, linestyle="--")
        plt.xticks(x, [f"Q{x}" for x in reader_heads])
        plt.ylabel("Causal effect")
        plt.title("EXP11 KV0 reader-head path decomposition")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "reader_profile.png", dpi=180)
        plt.close(fig)

    # ---------------------------
    # Compact summary
    # ---------------------------
    summary = {
        "experiment": "EXP11_exact_value_path",
        "phase": args.phase,
        "source_hidden_index": SOURCE_H,
        "consumer_decoder_block": CONSUMER_BLOCK,
        "kv_head": KV_HEAD,
        "reader_query_heads": reader_heads,
    }

    if not offset_summary.empty:
        scan = offset_summary[
            (offset_summary["offset"] < 0)
            & (offset_summary["endpoint"].isin([
                "single_token_sufficiency",
                "single_token_necessity_loss",
                "leave_one_out_loss",
            ]))
        ]

        both_offsets = []
        endpoints = {
            ep: scan[scan["endpoint"] == ep].set_index("offset")
            for ep in [
                "single_token_sufficiency",
                "single_token_necessity_loss",
                "leave_one_out_loss",
            ]
        }
        common_offsets = sorted(
            set(endpoints["single_token_sufficiency"].index)
            & set(endpoints["single_token_necessity_loss"].index)
        )

        for off in common_offsets:
            s = endpoints["single_token_sufficiency"].loc[off]
            n = endpoints["single_token_necessity_loss"].loc[off]
            if (
                np.isfinite(s["bh_q"]) and np.isfinite(n["bh_q"])
                and float(s["bh_q"]) < .05
                and float(n["bh_q"]) < .05
                and float(s["mean"]) > 0
                and float(n["mean"]) > 0
            ):
                both_offsets.append({
                    "offset": int(off),
                    "decoded_token_modal": str(s.get("decoded_token", "")),
                    "sufficiency_mean": float(s["mean"]),
                    "sufficiency_q": float(s["bh_q"]),
                    "necessity_mean": float(n["mean"]),
                    "necessity_q": float(n["bh_q"]),
                })

        summary["offsets_fdr_positive_in_both"] = both_offsets

        ref = offset_summary[
            (offset_summary["offset"] == 0)
            & (offset_summary["endpoint"] == "full_KV0_B5_sufficiency")
        ]
        if not ref.empty:
            summary["full_KV0_B5_sufficiency_mean"] = float(ref.iloc[0]["mean"])

    if not reader_summary.empty:
        reader_both = []
        for qh in reader_heads:
            ss = reader_summary[
                (reader_summary["reader"] == str(qh))
                & (reader_summary["endpoint"] == "path_sufficiency")
            ]
            nn = reader_summary[
                (reader_summary["reader"] == str(qh))
                & (reader_summary["endpoint"] == "path_necessity_loss")
            ]
            if not ss.empty and not nn.empty:
                s = ss.iloc[0]
                n = nn.iloc[0]
                if (
                    np.isfinite(s["bh_q"]) and np.isfinite(n["bh_q"])
                    and float(s["bh_q"]) < .05
                    and float(n["bh_q"]) < .05
                    and float(s["mean"]) > 0
                    and float(n["mean"]) > 0
                ):
                    reader_both.append({
                        "query_head": qh,
                        "sufficiency_mean": float(s["mean"]),
                        "sufficiency_q": float(s["bh_q"]),
                        "necessity_mean": float(n["mean"]),
                        "necessity_q": float(n["bh_q"]),
                    })

        summary["reader_heads_fdr_positive_in_both"] = reader_both

        def row(reader, endpoint):
            x = reader_summary[
                (reader_summary["reader"] == str(reader))
                & (reader_summary["endpoint"] == endpoint)
            ]
            return None if x.empty else x.iloc[0]

        full_v = row("full_KV0_B5", "verified_V_effect")
        all7s = row("all7", "path_sufficiency")
        all7n = row("all7", "path_necessity_loss")
        nons = row("Q7_Q27", "path_sufficiency")
        nonn = row("Q7_Q27", "path_necessity_loss")

        if full_v is not None:
            summary["verified_KV0_B5_effect_mean"] = float(full_v["mean"])
            summary["mean_non_KV0_reader_delta_ratio"] = float(
                full_v["mean_non_kv0_delta_ratio"]
            )
        if all7s is not None:
            summary["all7_reconstruction_sufficiency_mean"] = float(all7s["mean"])
        if all7n is not None:
            summary["all7_removal_necessity_mean"] = float(all7n["mean"])
        if nons is not None:
            summary["Q7_Q27_control_sufficiency_mean"] = float(nons["mean"])
        if nonn is not None:
            summary["Q7_Q27_control_necessity_mean"] = float(nonn["mean"])

    manifest = {
        "experiment": "EXP11_exact_value_path",
        "git_commit": git_commit(),
        "source_exp01b_git_commit": manifest01b.get("git_commit"),
        "seed": args.seed,
        "phase": args.phase,
        "model_dir": str(model_dir),
        "model_name": model_dir.name,
        "source_hidden_index": SOURCE_H,
        "consumer_decoder_block": CONSUMER_BLOCK,
        "kv_head": KV_HEAD,
        "num_attention_heads": num_q_heads,
        "num_key_value_heads": num_kv_heads,
        "queries_per_kv_group": q_per_kv,
        "reader_query_heads": reader_heads,
        "head_dim": head_dim,
        "num_position_bins": N_BINS,
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
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
