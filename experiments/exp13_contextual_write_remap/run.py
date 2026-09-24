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
EXP12_CODE = REPO_ROOT / "experiments" / "exp12_independent_replication" / "run.py"
DEFAULT_OUT = REPO_ROOT / "outputs" / "exp13_contextual_write_remap"

SOURCE_H = 20
CONSUMER_BLOCK = 20
KV_HEAD = 0

SPLIT_SEED = 5313
DISCOVERY_PER_FAMILY = 8
ANCHOR_WIDTH = 6

ANCHORS = (
    "SKILL_END",
    "SYSTEM_END",
    "ISSUE_END",
    "DETAIL_END",
    "ACTION0_END",
    "ACTION1_END",
    "FINAL_INSTRUCTION_END",
    "USER_END",
    "GENERATION_BOUNDARY",
)

OLD_ABSOLUTE_OFFSETS = (-13, -5, -3, -1)
TARGET_READERS = (0, 3, 5)
NEGATIVE_READERS = (2, 4, 6)
ALL_KV0_READERS = tuple(range(7))
NON_KV0_READERS = tuple(range(7, 28))


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


def load_exp12():
    if not EXP12_CODE.is_file():
        raise FileNotFoundError(
            f"EXP13 requires repository EXP12 implementation: {EXP12_CODE}"
        )

    spec = importlib.util.spec_from_file_location("skills4s_exp12", EXP12_CODE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    required = [
        "make_tasks",
        "make_skill_entries",
        "donor_maps_skill",
        "longest_common_suffix",
        "render_prompt",
        "resolve_model_dir",
        "get_layers",
        "SKILL_TEXT",
    ]
    missing = [name for name in required if not hasattr(mod, name)]
    if missing:
        raise RuntimeError(f"EXP12 API drift; missing: {missing}")
    return mod


def family_split(tasks):
    by_family = {}
    for task in tasks:
        by_family.setdefault(task["family"], []).append(task)

    split = {
        "seed": SPLIT_SEED,
        "discovery_per_family": DISCOVERY_PER_FAMILY,
        "families": {},
    }

    for fi, family in enumerate(sorted(by_family)):
        family_tasks = sorted(by_family[family], key=lambda x: x["task_id"])
        if len(family_tasks) != 16:
            raise RuntimeError(f"Expected 16 tasks in {family}, got {len(family_tasks)}")

        rng = np.random.default_rng(SPLIT_SEED + fi)
        order = rng.permutation(len(family_tasks))
        disc_idx = order[:DISCOVERY_PER_FAMILY]
        conf_idx = order[DISCOVERY_PER_FAMILY:]

        split["families"][family] = {
            "discovery": [family_tasks[int(i)]["task_id"] for i in disc_idx],
            "confirmation": [family_tasks[int(i)]["task_id"] for i in conf_idx],
        }

    return split


def split_ids(split, partition):
    return {
        tid
        for data in split["families"].values()
        for tid in data[partition]
    }


def render_ids(exp12, tok, entry):
    text = exp12.render_prompt(tok, entry["messages"])
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return text, ids


def find_segment(text, segment):
    start = text.rfind(segment)
    if start < 0:
        raise RuntimeError(f"Could not find semantic segment: {segment!r}")
    return start, start + len(segment)


def overlap_token_indices(offset_mapping, start, end):
    out = []
    for i, pair in enumerate(offset_mapping):
        s, e = int(pair[0]), int(pair[1])
        if e <= s:
            continue
        if e > start and s < end:
            out.append(i)
    if not out:
        raise RuntimeError(f"No token overlaps char span [{start},{end})")
    return out


def fixed_end_window(end_idx, prompt_len, width=ANCHOR_WIDTH):
    start = int(end_idx) - width + 1
    if start < 0:
        raise RuntimeError(f"Anchor too near prompt start: end={end_idx}")
    positions = list(range(start, int(end_idx) + 1))
    if len(positions) != width or max(positions) >= prompt_len:
        raise RuntimeError(f"Invalid fixed anchor window: {positions}")
    return positions


def compute_schema(exp12, tok, task, entry):
    text = exp12.render_prompt(tok, entry["messages"])
    enc = tok(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = [int(x) for x in enc["input_ids"]]
    offsets = enc["offset_mapping"]
    n = len(ids)

    skill_text = exp12.SKILL_TEXT[task["family"]][int(entry["label"])][entry["wording"]]

    semantic_segments = {
        "SKILL_END": skill_text,
        "ISSUE_END": task["issue"],
        "DETAIL_END": task["detail"],
        "ACTION0_END": task["action0"],
        "ACTION1_END": task["action1"],
        "FINAL_INSTRUCTION_END": "Choose the single next action now.",
    }

    anchors = {}
    for name, segment in semantic_segments.items():
        start, end = find_segment(text, segment)
        toks = overlap_token_indices(offsets, start, end)
        anchors[name] = fixed_end_window(toks[-1], n)

    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    end_positions = [i for i, tid in enumerate(ids) if tid == im_end_id]
    if len(end_positions) < 2:
        raise RuntimeError("Expected system and user <|im_end|> tokens")

    anchors["SYSTEM_END"] = fixed_end_window(end_positions[0], n)
    anchors["USER_END"] = fixed_end_window(end_positions[-1], n)
    anchors["GENERATION_BOUNDARY"] = list(range(n - ANCHOR_WIDTH, n))

    for name in ANCHORS:
        if name not in anchors or len(anchors[name]) != ANCHOR_WIDTH:
            raise RuntimeError(f"Schema anchor invariant failed: {name}")

    return {"text": text, "ids": ids, "anchors": anchors}


def audit_schema_rows(tok, task, entry, schema, partition):
    rows = []
    for anchor in ANCHORS:
        for j, pos in enumerate(schema["anchors"][anchor]):
            tid = int(schema["ids"][pos])
            rows.append({
                "partition": partition,
                "task_id": task["task_id"],
                "family": task["family"],
                "wording": entry["wording"],
                "label": int(entry["label"]),
                "anchor": anchor,
                "anchor_local_index": j,
                "prompt_position": int(pos),
                "offset_from_generation_boundary": int(pos - len(schema["ids"])),
                "token_id": tid,
                "token": tok.convert_ids_to_tokens(tid),
                "decoded": tok.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False),
            })
    return rows

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
def oproj_replace_heads_hook(o_proj, head_indices, replacements, num_q_heads, head_dim):
    def pre_hook(module, inputs):
        x = inputs[0].clone()
        shape = x.shape
        xh = x.view(*shape[:-1], num_q_heads, head_dim)
        heads = head_indices.to(x.device)
        rep = replacements.to(x.device, dtype=x.dtype)
        xh[:, :, heads, :] = rep
        return (xh.reshape(*shape),) + tuple(inputs[1:])

    handle = o_proj.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


@torch.inference_mode()
def capture_prompt_states(model, tok, layers, exp12, task, entry, num_kv_heads, head_dim):
    device = model.get_input_embeddings().weight.device
    text = exp12.render_prompt(tok, entry["messages"])
    enc = tok(text, add_special_tokens=False, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}

    store = {"residual": None, "v": None}
    handles = []

    def residual_hook(module, inputs, output):
        t = output[0] if isinstance(output, tuple) else output
        store["residual"] = t[0].detach().float().cpu().numpy().astype(np.float32)

    def v_hook(module, inputs, output):
        x = output[0].view(output.shape[1], num_kv_heads, head_dim)
        store["v"] = x.detach().float().cpu().numpy().astype(np.float32)

    handles.append(layers[SOURCE_H - 1].register_forward_hook(residual_hook))
    handles.append(layers[CONSUMER_BLOCK].self_attn.v_proj.register_forward_hook(v_hook))
    try:
        model(**enc, use_cache=False, return_dict=True)
    finally:
        for h in handles:
            h.remove()

    schema = compute_schema(exp12, tok, task, entry)
    if ids != schema["ids"]:
        raise RuntimeError("Tokenizer mismatch between capture and schema")

    return {
        "ids": ids,
        "residual": store["residual"],
        "v": store["v"],
        "schema": schema,
    }


def build_task_cache(model, tok, layers, exp12, task, entries, num_kv_heads, head_dim):
    return [
        capture_prompt_states(model, tok, layers, exp12, task, e, num_kv_heads, head_dim)
        for e in entries
    ]


def anchor_positions(cache, anchor):
    return list(cache["schema"]["anchors"][anchor])


def old_absolute_positions(cache):
    n = len(cache["ids"])
    pos = [n + int(o) for o in OLD_ABSOLUTE_OFFSETS]
    if min(pos) < 0 or max(pos) >= n:
        raise RuntimeError("Old absolute offset out of range")
    return pos


def donor_sign(donor):
    return 1.0 if int(donor["label"]) == 1 else -1.0


def signed_effect(patched_margin, baseline_margin, donor):
    return donor_sign(donor) * (patched_margin - baseline_margin)


def candidate_mean_logprob(logits, input_ids, prompt_len, candidate_len):
    lp = torch.log_softmax(logits.float(), dim=-1)
    pred_pos = torch.arange(prompt_len - 1, prompt_len + candidate_len - 1, device=input_ids.device)
    targ_pos = torch.arange(prompt_len, prompt_len + candidate_len, device=input_ids.device)
    targets = input_ids[0, targ_pos]
    return float(lp[0, pred_pos, targets].mean().cpu())


@torch.inference_mode()
@torch.inference_mode()
def score_margin(
    model,
    tok,
    layers,
    exp12,
    entry,
    rec_cache,
    num_kv_heads,
    head_dim,
    residual_source_cache=None,
    residual_rec_positions=None,
    residual_donor_positions=None,
    v_source_cache=None,
    recipient_positions=None,
    donor_positions=None,
    v_heads=(KV_HEAD,),
):
    device = model.get_input_embeddings().weight.device
    prompt_text = exp12.render_prompt(tok, entry["messages"])
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    if prompt_ids != rec_cache["ids"]:
        raise RuntimeError("Recipient cache mismatch")
    plen = len(prompt_ids)
    scores = []

    for candidate in entry["candidates"]:
        cids = tok(candidate, add_special_tokens=False)["input_ids"]
        input_ids = torch.tensor([prompt_ids + cids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)

        with ExitStack() as stack:
            if residual_source_cache is not None:
                if residual_rec_positions is None or residual_donor_positions is None:
                    raise RuntimeError("Residual patch requires source and target positions")
                rec_r = np.asarray(residual_rec_positions, dtype=np.int64)
                don_r = np.asarray(residual_donor_positions, dtype=np.int64)
                if len(rec_r) != len(don_r):
                    raise RuntimeError("Residual source/target count mismatch")
                values = residual_source_cache["residual"][don_r, :]
                stack.enter_context(
                    residual_patch_hook(
                        layers[SOURCE_H - 1],
                        torch.tensor(rec_r[None, :], dtype=torch.long, device=device),
                        torch.tensor(values[None, :, :], dtype=torch.float32, device=device),
                    )
                )

            if v_source_cache is not None:
                if recipient_positions is None or donor_positions is None:
                    raise RuntimeError("V patch requires source and target positions")
                rec_pos = np.asarray(recipient_positions, dtype=np.int64)
                donor_pos = np.asarray(donor_positions, dtype=np.int64)
                if len(rec_pos) != len(donor_pos):
                    raise RuntimeError("V source/target position count mismatch")
                values = v_source_cache["v"][donor_pos][:, list(v_heads), :]
                stack.enter_context(
                    value_patch_hook(
                        layers[CONSUMER_BLOCK].self_attn.v_proj,
                        torch.tensor(rec_pos[None, :], dtype=torch.long, device=device),
                        torch.tensor(list(v_heads), dtype=torch.long, device=device),
                        torch.tensor(values[None, :, :, :], dtype=torch.float32, device=device),
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

        scores.append(candidate_mean_logprob(out.logits, input_ids, plen, len(cids)))

    return scores[1] - scores[0]


def baseline_margins(model, tok, layers, exp12, entries, cache, num_kv_heads, head_dim):
    return [
        score_margin(model, tok, layers, exp12, e, cache[i], num_kv_heads, head_dim)
        for i, e in enumerate(entries)
    ]


def residual_effect(
    model, tok, layers, exp12, entry, rec_cache, donor, donor_cache,
    baseline_margin, rec_positions, donor_positions, num_kv_heads, head_dim,
):
    margin = score_margin(
        model, tok, layers, exp12, entry, rec_cache, num_kv_heads, head_dim,
        residual_source_cache=donor_cache,
        residual_rec_positions=rec_positions,
        residual_donor_positions=donor_positions,
    )
    return signed_effect(margin, baseline_margin, donor)


def v_suff_nec(
    model, tok, layers, exp12, entry, rec_cache, donor, donor_cache,
    baseline_margin, residual_ref, rec_positions, donor_positions,
    num_kv_heads, head_dim,
):
    suff_margin = score_margin(
        model, tok, layers, exp12, entry, rec_cache, num_kv_heads, head_dim,
        v_source_cache=donor_cache,
        recipient_positions=rec_positions,
        donor_positions=donor_positions,
    )
    suff = signed_effect(suff_margin, baseline_margin, donor)

    # Necessity is local to the same semantic write site: restore donor H20
    # residual at the schema-mapped positions, then clamp KV0 values there
    # back to the recipient baseline.
    clamp_margin = score_margin(
        model, tok, layers, exp12, entry, rec_cache, num_kv_heads, head_dim,
        residual_source_cache=donor_cache,
        residual_rec_positions=rec_positions,
        residual_donor_positions=donor_positions,
        v_source_cache=rec_cache,
        recipient_positions=rec_positions,
        donor_positions=rec_positions,
    )
    retained = signed_effect(clamp_margin, baseline_margin, donor)
    necessity = residual_ref - retained
    return suff, necessity

@torch.inference_mode()
def candidate_capture_oproj(
    model, tok, layers, exp12, entry, candidate, rec_cache,
    num_kv_heads, num_q_heads, head_dim,
    v_source_cache=None, recipient_positions=None, donor_positions=None,
):
    device = model.get_input_embeddings().weight.device
    prompt_text = exp12.render_prompt(tok, entry["messages"])
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    if prompt_ids != rec_cache["ids"]:
        raise RuntimeError("Reader cache mismatch")

    cids = tok(candidate, add_special_tokens=False)["input_ids"]
    plen = len(prompt_ids)
    input_ids = torch.tensor([prompt_ids + cids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    store = {}

    with ExitStack() as stack:
        if v_source_cache is not None:
            rec_pos = np.asarray(recipient_positions, dtype=np.int64)
            donor_pos = np.asarray(donor_positions, dtype=np.int64)
            values = v_source_cache["v"][donor_pos][:, [KV_HEAD], :]
            stack.enter_context(
                value_patch_hook(
                    layers[CONSUMER_BLOCK].self_attn.v_proj,
                    torch.tensor(rec_pos[None, :], dtype=torch.long, device=device),
                    torch.tensor([KV_HEAD], dtype=torch.long, device=device),
                    torch.tensor(values[None, :, :, :], dtype=torch.float32, device=device),
                    num_kv_heads,
                    head_dim,
                )
            )

        stack.enter_context(
            oproj_capture_hook(
                layers[CONSUMER_BLOCK].self_attn.o_proj,
                store,
                "oproj",
                num_q_heads,
                head_dim,
            )
        )

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

    return (
        candidate_mean_logprob(out.logits, input_ids, plen, len(cids)),
        store["oproj"],
    )


@torch.inference_mode()
def candidate_replace_readers(
    model, tok, layers, exp12, entry, candidate, rec_cache,
    num_kv_heads, num_q_heads, head_dim,
    replacement_np, readers,
    upstream_v_cache=None, recipient_positions=None, donor_positions=None,
):
    device = model.get_input_embeddings().weight.device
    prompt_text = exp12.render_prompt(tok, entry["messages"])
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    if prompt_ids != rec_cache["ids"]:
        raise RuntimeError("Reader replacement cache mismatch")

    cids = tok(candidate, add_special_tokens=False)["input_ids"]
    plen = len(prompt_ids)
    input_ids = torch.tensor([prompt_ids + cids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    with ExitStack() as stack:
        if upstream_v_cache is not None:
            rec_pos = np.asarray(recipient_positions, dtype=np.int64)
            donor_pos = np.asarray(donor_positions, dtype=np.int64)
            values = upstream_v_cache["v"][donor_pos][:, [KV_HEAD], :]
            stack.enter_context(
                value_patch_hook(
                    layers[CONSUMER_BLOCK].self_attn.v_proj,
                    torch.tensor(rec_pos[None, :], dtype=torch.long, device=device),
                    torch.tensor([KV_HEAD], dtype=torch.long, device=device),
                    torch.tensor(values[None, :, :, :], dtype=torch.float32, device=device),
                    num_kv_heads,
                    head_dim,
                )
            )

        stack.enter_context(
            oproj_replace_heads_hook(
                layers[CONSUMER_BLOCK].self_attn.o_proj,
                torch.tensor(list(readers), dtype=torch.long, device=device),
                torch.tensor(replacement_np, dtype=torch.float32, device=device),
                num_q_heads,
                head_dim,
            )
        )

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

    return candidate_mean_logprob(out.logits, input_ids, plen, len(cids))


def pair_margin(scores):
    return scores[1] - scores[0]


def reader_path_effects(
    model, tok, layers, exp12, entry, rec_cache, donor, donor_cache,
    rec_positions, donor_positions, baseline_margin,
    num_kv_heads, num_q_heads, head_dim,
):
    base_scores, patch_scores = [], []
    base_caps, patch_caps = [], []

    for candidate in entry["candidates"]:
        bs, bc = candidate_capture_oproj(
            model, tok, layers, exp12, entry, candidate, rec_cache,
            num_kv_heads, num_q_heads, head_dim,
        )
        ps, pc = candidate_capture_oproj(
            model, tok, layers, exp12, entry, candidate, rec_cache,
            num_kv_heads, num_q_heads, head_dim,
            v_source_cache=donor_cache,
            recipient_positions=rec_positions,
            donor_positions=donor_positions,
        )
        base_scores.append(bs)
        patch_scores.append(ps)
        base_caps.append(bc)
        patch_caps.append(pc)

    reader_base_margin = pair_margin(base_scores)
    patched_margin = pair_margin(patch_scores)
    verified_v_effect = signed_effect(patched_margin, reader_base_margin, donor)

    num = 0.0
    den = 0.0
    for bc, pc in zip(base_caps, patch_caps):
        delta = pc - bc
        num += float(np.sum(delta[:, :, list(NON_KV0_READERS), :] ** 2))
        den += float(np.sum(delta[:, :, list(ALL_KV0_READERS), :] ** 2))
    leakage_ratio = float(np.sqrt(num / (den + 1e-30)))

    def group_effect(readers):
        suff_scores, nec_scores = [], []
        for ci, candidate in enumerate(entry["candidates"]):
            bc = base_caps[ci]
            pc = patch_caps[ci]
            suff_scores.append(
                candidate_replace_readers(
                    model, tok, layers, exp12, entry, candidate, rec_cache,
                    num_kv_heads, num_q_heads, head_dim,
                    replacement_np=pc[:, :, list(readers), :],
                    readers=readers,
                )
            )
            nec_scores.append(
                candidate_replace_readers(
                    model, tok, layers, exp12, entry, candidate, rec_cache,
                    num_kv_heads, num_q_heads, head_dim,
                    replacement_np=bc[:, :, list(readers), :],
                    readers=readers,
                    upstream_v_cache=donor_cache,
                    recipient_positions=rec_positions,
                    donor_positions=donor_positions,
                )
            )

        suff = signed_effect(pair_margin(suff_scores), reader_base_margin, donor)
        retained = signed_effect(pair_margin(nec_scores), reader_base_margin, donor)
        nec = verified_v_effect - retained
        return suff, nec

    groups = {
        "target": TARGET_READERS,
        "negative": NEGATIVE_READERS,
        "all7": ALL_KV0_READERS,
        "non_kv0": NON_KV0_READERS,
    }

    out = {
        "verified_v_effect": verified_v_effect,
        "reader_baseline_margin": reader_base_margin,
        "baseline_margin_diff": reader_base_margin - baseline_margin,
        "leakage_ratio": leakage_ratio,
    }
    for name, readers in groups.items():
        suff, nec = group_effect(readers)
        out[f"{name}_sufficiency"] = suff
        out[f"{name}_necessity"] = nec
    return out


def bootstrap_ci(values, seed, n_boot=5000):
    x = np.asarray(values, dtype=np.float64)
    if len(x) == 0:
        raise RuntimeError("Empty bootstrap input")
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
    return bootstrap_ci(a - b, seed)


def add_result(rows, *, phase, task, entry, donor, relation, anchor, metric, value, extra=None):
    row = {
        "phase": phase,
        "task_id": task["task_id"],
        "family": task["family"],
        "recipient_wording": entry["wording"],
        "recipient_label": int(entry["label"]),
        "donor_wording": donor["wording"] if donor is not None else "",
        "donor_label": int(donor["label"]) if donor is not None else -1,
        "relation": relation,
        "anchor": anchor,
        "metric": metric,
        "value": float(value),
    }
    if extra:
        row.update(extra)
    rows.append(row)

def run_discovery(
    model, tok, layers, exp12, tasks, split,
    num_kv_heads, head_dim, out_dir, seed,
):
    rows = []
    audit_rows_all = []
    ids = split_ids(split, "discovery")
    discovery_tasks = [t for t in tasks if t["task_id"] in ids]

    for n, task in enumerate(discovery_tasks, 1):
        entries = exp12.make_skill_entries(task)
        maps = exp12.donor_maps_skill(entries)
        cache = build_task_cache(
            model, tok, layers, exp12, task, entries, num_kv_heads, head_dim
        )
        baselines = baseline_margins(
            model, tok, layers, exp12, entries, cache, num_kv_heads, head_dim
        )

        for i, entry in enumerate(entries):
            audit_rows_all.extend(
                audit_schema_rows(tok, task, entry, cache[i]["schema"], "discovery")
            )

            di = maps[i]["opposite_same_wording"]
            donor = entries[di]

            for anchor in ANCHORS:
                rec_pos = anchor_positions(cache[i], anchor)
                donor_pos = anchor_positions(cache[di], anchor)
                res_eff = residual_effect(
                    model, tok, layers, exp12, entry, cache[i], donor, cache[di],
                    baselines[i], rec_pos, donor_pos, num_kv_heads, head_dim,
                )
                suff, nec = v_suff_nec(
                    model, tok, layers, exp12, entry, cache[i], donor, cache[di],
                    baselines[i], res_eff, rec_pos, donor_pos,
                    num_kv_heads, head_dim,
                )
                add_result(
                    rows,
                    phase="discovery",
                    task=task,
                    entry=entry,
                    donor=donor,
                    relation="opposite_same_wording",
                    anchor=anchor,
                    metric="V_sufficiency",
                    value=suff,
                )
                add_result(
                    rows,
                    phase="discovery",
                    task=task,
                    entry=entry,
                    donor=donor,
                    relation="opposite_same_wording",
                    anchor=anchor,
                    metric="V_necessity_loss",
                    value=nec,
                )

        print(f"[discovery {n:03d}/{len(discovery_tasks):03d}] {task['task_id']} ({task['family']})")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = pd.DataFrame(rows)
    audit_df = pd.DataFrame(audit_rows_all)
    results.to_csv(out_dir / "discovery_results.csv", index=False)
    audit_df.to_csv(out_dir / "schema_audit_discovery.csv", index=False)

    # Task is inferential unit; label remains explicit because bidirectionality
    # is part of the discovery rule.
    task_level = (
        results.groupby(
            ["task_id", "family", "recipient_wording", "recipient_label", "anchor", "metric"],
            as_index=False,
        )["value"].mean()
    )

    summary_rows = []
    selected = {
        "selection_rule": (
            "Per family×wording select anchor maximizing "
            "min(mean_suff_label0, mean_suff_label1, mean_nec_label0, mean_nec_label1)."
        ),
        "anchor_order": list(ANCHORS),
        "strata": {},
    }

    for family in sorted(split["families"]):
        for wording in ("canonical", "paraphrase"):
            stratum_rows = []
            for anchor_order, anchor in enumerate(ANCHORS):
                d = task_level[
                    (task_level["family"] == family)
                    & (task_level["recipient_wording"] == wording)
                    & (task_level["anchor"] == anchor)
                ]
                means = {}
                for label in (0, 1):
                    for metric, short in [
                        ("V_sufficiency", "suff"),
                        ("V_necessity_loss", "nec"),
                    ]:
                        vals = d[
                            (d["recipient_label"] == label)
                            & (d["metric"] == metric)
                        ]["value"].to_numpy(np.float64)
                        m, lo, hi = bootstrap_ci(
                            vals,
                            seed + 1000 + sum(map(ord, f"{family}{wording}{anchor}{label}{metric}")),
                        )
                        means[f"{short}_label{label}"] = m
                        means[f"{short}_label{label}_ci_low"] = lo
                        means[f"{short}_label{label}_ci_high"] = hi

                score = min(
                    means["suff_label0"],
                    means["suff_label1"],
                    means["nec_label0"],
                    means["nec_label1"],
                )
                row = {
                    "family": family,
                    "wording": wording,
                    "anchor": anchor,
                    "anchor_order": anchor_order,
                    "bidirectional_score": score,
                    **means,
                }
                summary_rows.append(row)
                stratum_rows.append(row)

            ranked = sorted(
                stratum_rows,
                key=lambda r: (-r["bidirectional_score"], r["anchor_order"]),
            )
            best, runner = ranked[0], ranked[1]
            key = f"{family}::{wording}"
            selected["strata"][key] = {
                "family": family,
                "wording": wording,
                "selected_anchor": best["anchor"],
                "runner_up_anchor": runner["anchor"],
                "selected_bidirectional_score": float(best["bidirectional_score"]),
                "runner_up_bidirectional_score": float(runner["bidirectional_score"]),
                "selected_discovery_components": {
                    k: float(v) for k, v in best.items()
                    if k.startswith("suff_") or k.startswith("nec_")
                },
            }

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "discovery_summary.csv", index=False)
    (out_dir / "selected_schema.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig = plt.figure(figsize=(14, 7))
    x = np.arange(len(ANCHORS))
    for key, data in selected["strata"].items():
        f = summary_df[
            (summary_df["family"] == data["family"])
            & (summary_df["wording"] == data["wording"])
        ].set_index("anchor").loc[list(ANCHORS)]
        plt.plot(x, f["bidirectional_score"], marker="o", label=key)
    plt.axhline(0.0, linestyle="--")
    plt.xticks(x, ANCHORS, rotation=35, ha="right")
    plt.ylabel("Discovery bidirectional score")
    plt.title("EXP13 contextual writer discovery")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    fig.savefig(out_dir / "discovery_profile.png", dpi=180)
    plt.close(fig)

    return results, summary_df, selected


def selected_for(selected, family, wording):
    return selected["strata"][f"{family}::{wording}"]


def run_confirmation(
    model, tok, layers, exp12, tasks, split, selected,
    num_kv_heads, num_q_heads, head_dim, out_dir,
):
    rows = []
    audit_rows_all = []
    ids = split_ids(split, "confirmation")
    conf_tasks = [t for t in tasks if t["task_id"] in ids]

    for n, task in enumerate(conf_tasks, 1):
        entries = exp12.make_skill_entries(task)
        maps = exp12.donor_maps_skill(entries)
        cache = build_task_cache(
            model, tok, layers, exp12, task, entries, num_kv_heads, head_dim
        )
        baselines = baseline_margins(
            model, tok, layers, exp12, entries, cache, num_kv_heads, head_dim
        )

        for i, entry in enumerate(entries):
            audit_rows_all.extend(
                audit_schema_rows(tok, task, entry, cache[i]["schema"], "confirmation")
            )
            stratum = selected_for(selected, task["family"], entry["wording"])
            selected_anchor = stratum["selected_anchor"]
            runner_anchor = stratum["runner_up_anchor"]

            # Same-wording opposite-state confirmation. Each write-site
            # necessity test uses an H20 residual intervention at that same
            # schema location; no common-suffix assumption is imposed.
            di = maps[i]["opposite_same_wording"]
            donor = entries[di]

            sel_rec = anchor_positions(cache[i], selected_anchor)
            sel_don = anchor_positions(cache[di], selected_anchor)
            sel_res = residual_effect(
                model, tok, layers, exp12, entry, cache[i], donor, cache[di],
                baselines[i], sel_rec, sel_don, num_kv_heads, head_dim,
            )
            sel_suff, sel_nec = v_suff_nec(
                model, tok, layers, exp12, entry, cache[i], donor, cache[di],
                baselines[i], sel_res, sel_rec, sel_don,
                num_kv_heads, head_dim,
            )

            old_rec = old_absolute_positions(cache[i])
            old_don = old_absolute_positions(cache[di])
            old_res = residual_effect(
                model, tok, layers, exp12, entry, cache[i], donor, cache[di],
                baselines[i], old_rec, old_don, num_kv_heads, head_dim,
            )
            old_suff, old_nec = v_suff_nec(
                model, tok, layers, exp12, entry, cache[i], donor, cache[di],
                baselines[i], old_res, old_rec, old_don,
                num_kv_heads, head_dim,
            )

            run_rec = anchor_positions(cache[i], runner_anchor)
            run_don = anchor_positions(cache[di], runner_anchor)
            run_res = residual_effect(
                model, tok, layers, exp12, entry, cache[i], donor, cache[di],
                baselines[i], run_rec, run_don, num_kv_heads, head_dim,
            )
            run_suff, run_nec = v_suff_nec(
                model, tok, layers, exp12, entry, cache[i], donor, cache[di],
                baselines[i], run_res, run_rec, run_don,
                num_kv_heads, head_dim,
            )

            for anchor, metric, value in [
                (selected_anchor, "selected_H20_residual_effect", sel_res),
                (selected_anchor, "selected_V_sufficiency", sel_suff),
                (selected_anchor, "selected_V_necessity_loss", sel_nec),
                ("OLD_ABSOLUTE", "old_absolute_V_sufficiency", old_suff),
                ("OLD_ABSOLUTE", "old_absolute_local_V_necessity_loss", old_nec),
                (runner_anchor, "runner_up_V_sufficiency", run_suff),
                (runner_anchor, "runner_up_V_necessity_loss", run_nec),
            ]:
                add_result(
                    rows, phase="confirmation", task=task, entry=entry, donor=donor,
                    relation="opposite_same_wording", anchor=anchor, metric=metric, value=value,
                    extra={"selected_anchor": selected_anchor, "runner_up_anchor": runner_anchor},
                )

            reader = reader_path_effects(
                model, tok, layers, exp12, entry, cache[i], donor, cache[di],
                sel_rec, sel_don, baselines[i],
                num_kv_heads, num_q_heads, head_dim,
            )
            for metric in [
                "verified_v_effect", "target_sufficiency", "target_necessity",
                "negative_sufficiency", "negative_necessity",
                "all7_sufficiency", "all7_necessity",
                "non_kv0_sufficiency", "non_kv0_necessity",
            ]:
                add_result(
                    rows, phase="confirmation", task=task, entry=entry, donor=donor,
                    relation="opposite_same_wording", anchor=selected_anchor,
                    metric=f"reader_{metric}", value=reader[metric],
                    extra={
                        "selected_anchor": selected_anchor,
                        "runner_up_anchor": runner_anchor,
                        "reader_leakage_ratio": reader["leakage_ratio"],
                        "reader_baseline_margin_diff": reader["baseline_margin_diff"],
                        "reader_verified_v_diff": reader["verified_v_effect"] - sel_suff,
                    },
                )

            # Cross-wording transfer maps recipient and donor using their own
            # pre-frozen schema anchors.
            cdi = maps[i]["opposite_cross_wording"]
            cross_donor = entries[cdi]
            donor_stratum = selected_for(selected, task["family"], cross_donor["wording"])
            cross_rec_pos = anchor_positions(cache[i], selected_anchor)
            cross_don_pos = anchor_positions(cache[cdi], donor_stratum["selected_anchor"])
            cross_res = residual_effect(
                model, tok, layers, exp12, entry, cache[i], cross_donor, cache[cdi],
                baselines[i], cross_rec_pos, cross_don_pos, num_kv_heads, head_dim,
            )
            cross_suff, cross_nec = v_suff_nec(
                model, tok, layers, exp12, entry, cache[i], cross_donor, cache[cdi],
                baselines[i], cross_res, cross_rec_pos, cross_don_pos,
                num_kv_heads, head_dim,
            )
            add_result(
                rows, phase="confirmation", task=task, entry=entry, donor=cross_donor,
                relation="opposite_cross_wording", anchor=selected_anchor,
                metric="cross_selected_V_sufficiency", value=cross_suff,
                extra={"donor_selected_anchor": donor_stratum["selected_anchor"]},
            )
            add_result(
                rows, phase="confirmation", task=task, entry=entry, donor=cross_donor,
                relation="opposite_cross_wording", anchor=selected_anchor,
                metric="cross_selected_V_necessity_loss", value=cross_nec,
                extra={"donor_selected_anchor": donor_stratum["selected_anchor"]},
            )

            # Same-state cross-wording semantic control with wording-specific mapping.
            sdi = maps[i]["same_state_cross_wording"]
            same_donor = entries[sdi]
            same_donor_stratum = selected_for(selected, task["family"], same_donor["wording"])
            same_margin = score_margin(
                model, tok, layers, exp12, entry, cache[i], num_kv_heads, head_dim,
                v_source_cache=cache[sdi],
                recipient_positions=anchor_positions(cache[i], selected_anchor),
                donor_positions=anchor_positions(cache[sdi], same_donor_stratum["selected_anchor"]),
            )
            same_effect = (1.0 if int(entry["label"]) == 1 else -1.0) * (same_margin - baselines[i])
            add_result(
                rows, phase="confirmation", task=task, entry=entry, donor=same_donor,
                relation="same_state_cross_wording", anchor=selected_anchor,
                metric="same_state_selected_V_control", value=same_effect,
                extra={"donor_selected_anchor": same_donor_stratum["selected_anchor"]},
            )

        print(f"[confirmation {n:03d}/{len(conf_tasks):03d}] {task['task_id']} ({task['family']})")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = pd.DataFrame(rows)
    audit_df = pd.DataFrame(audit_rows_all)
    results.to_csv(out_dir / "confirmation_results.csv", index=False)
    audit_df.to_csv(out_dir / "schema_audit_confirmation.csv", index=False)
    return results

def summarize_values(values, seed):
    m, lo, hi = bootstrap_ci(values, seed)
    return {"mean": m, "ci_low": lo, "ci_high": hi, "n": len(values)}


def confirmation_inference(results, selected, out_dir, seed):
    # Pooled task-level inference averages the four within-task entries.
    task_level = (
        results.groupby(["task_id", "family", "relation", "metric"], as_index=False)["value"].mean()
    )

    summary_rows = []
    for (relation, metric), g in task_level.groupby(["relation", "metric"]):
        vals = g["value"].to_numpy(np.float64)
        m, lo, hi = bootstrap_ci(vals, seed + 10000 + sum(map(ord, relation + metric)))
        summary_rows.append({
            "relation": relation,
            "metric": metric,
            "mean": m,
            "ci_low": lo,
            "ci_high": hi,
            "num_tasks": int(len(vals)),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "confirmation_summary.csv", index=False)

    # Directional analysis: average wordings within each task but preserve label.
    directional_task = (
        results.groupby(
            ["task_id", "family", "recipient_label", "relation", "metric"],
            as_index=False,
        )["value"].mean()
    )
    directional_rows = []
    for (label, relation, metric), g in directional_task.groupby(
        ["recipient_label", "relation", "metric"]
    ):
        vals = g["value"].to_numpy(np.float64)
        m, lo, hi = bootstrap_ci(
            vals, seed + 20000 + int(label) * 997 + sum(map(ord, relation + metric))
        )
        directional_rows.append({
            "recipient_label": int(label),
            "relation": relation,
            "metric": metric,
            "mean": m,
            "ci_low": lo,
            "ci_high": hi,
            "num_tasks": int(len(vals)),
        })
    directional_df = pd.DataFrame(directional_rows)
    directional_df.to_csv(out_dir / "directional_confirmation.csv", index=False)

    # Fine diagnostic, secondary only: family × wording × label.
    fine_task = (
        results.groupby(
            ["task_id", "family", "recipient_wording", "recipient_label", "relation", "metric"],
            as_index=False,
        )["value"].mean()
    )
    fine_rows = []
    keep_metrics = {
        "selected_V_sufficiency",
        "selected_V_necessity_loss",
        "old_absolute_V_sufficiency",
        "old_absolute_local_V_necessity_loss",
        "runner_up_V_sufficiency",
        "runner_up_V_necessity_loss",
        "reader_target_sufficiency",
        "reader_target_necessity",
        "reader_negative_sufficiency",
        "reader_negative_necessity",
    }
    fine = fine_task[
        (fine_task["relation"] == "opposite_same_wording")
        & (fine_task["metric"].isin(keep_metrics))
    ]
    for (family, wording, label, metric), g in fine.groupby(
        ["family", "recipient_wording", "recipient_label", "metric"]
    ):
        vals = g["value"].to_numpy(np.float64)
        m, lo, hi = bootstrap_ci(
            vals,
            seed + 30000 + sum(map(ord, family + wording + metric)) + int(label) * 31,
        )
        fine_rows.append({
            "family": family,
            "wording": wording,
            "recipient_label": int(label),
            "metric": metric,
            "mean": m,
            "ci_low": lo,
            "ci_high": hi,
            "num_tasks": int(len(vals)),
            "selected_anchor": selected_for(selected, family, wording)["selected_anchor"],
            "runner_up_anchor": selected_for(selected, family, wording)["runner_up_anchor"],
        })
    fine_df = pd.DataFrame(fine_rows)
    fine_df.to_csv(out_dir / "family_wording_confirmation.csv", index=False)

    def task_values(metric, relation="opposite_same_wording"):
        d = task_level[(task_level["metric"] == metric) & (task_level["relation"] == relation)]
        return d.sort_values("task_id")[["task_id", "value"]]

    def paired(metric_a, metric_b, name, offset):
        a = task_values(metric_a).rename(columns={"value": "a"})
        b = task_values(metric_b).rename(columns={"value": "b"})
        m = a.merge(b, on="task_id", how="inner")
        mean, lo, hi = paired_ci(m["a"], m["b"], seed + 40000 + offset)
        return {
            "contrast": name,
            "metric_a": metric_a,
            "metric_b": metric_b,
            "mean_difference": mean,
            "ci_low": lo,
            "ci_high": hi,
            "num_tasks": int(len(m)),
        }

    contrasts = [
        paired(
            "selected_V_sufficiency", "old_absolute_V_sufficiency",
            "selected_minus_old_absolute_sufficiency", 1,
        ),
        paired(
            "selected_V_necessity_loss", "old_absolute_local_V_necessity_loss",
            "selected_minus_old_absolute_local_necessity", 2,
        ),
        paired(
            "selected_V_sufficiency", "runner_up_V_sufficiency",
            "selected_minus_runner_up_sufficiency", 3,
        ),
        paired(
            "selected_V_necessity_loss", "runner_up_V_necessity_loss",
            "selected_minus_runner_up_necessity", 4,
        ),
        paired(
            "reader_target_sufficiency", "reader_negative_sufficiency",
            "target_minus_negative_reader_sufficiency", 5,
        ),
        paired(
            "reader_target_necessity", "reader_negative_necessity",
            "target_minus_negative_reader_necessity", 6,
        ),
    ]
    contrast_df = pd.DataFrame(contrasts)
    contrast_df.to_csv(out_dir / "paired_contrasts.csv", index=False)

    def get_row(metric, relation=None):
        d = summary_df[summary_df["metric"] == metric]
        if relation is not None:
            d = d[d["relation"] == relation]
        return None if d.empty else d.iloc[0]

    def ci_positive(metric, relation=None):
        r = get_row(metric, relation)
        return bool(r is not None and float(r["ci_low"]) > 0)

    endpoint_flags = {
        "selected_V_sufficiency": ci_positive("selected_V_sufficiency", "opposite_same_wording"),
        "selected_V_necessity_loss": ci_positive("selected_V_necessity_loss", "opposite_same_wording"),
        "reader_target_sufficiency": ci_positive("reader_target_sufficiency", "opposite_same_wording"),
        "reader_target_necessity": ci_positive("reader_target_necessity", "opposite_same_wording"),
        "cross_selected_V_sufficiency": ci_positive("cross_selected_V_sufficiency", "opposite_cross_wording"),
        "cross_selected_V_necessity_loss": ci_positive("cross_selected_V_necessity_loss", "opposite_cross_wording"),
    }
    contrast_flags = {c["contrast"]: bool(float(c["ci_low"]) > 0) for c in contrasts}

    # Explicit bidirectionality gate using held-out tasks.
    bidir = {}
    bidir_ok = True
    for label in (0, 1):
        for metric in ("selected_V_sufficiency", "selected_V_necessity_loss"):
            d = directional_df[
                (directional_df["recipient_label"] == label)
                & (directional_df["relation"] == "opposite_same_wording")
                & (directional_df["metric"] == metric)
            ]
            key = f"label{label}_{metric}"
            if d.empty:
                bidir[key] = None
                bidir_ok = False
            else:
                row = d.iloc[0]
                bidir[key] = {
                    "mean": float(row["mean"]),
                    "ci_low": float(row["ci_low"]),
                    "ci_high": float(row["ci_high"]),
                }
                bidir_ok = bidir_ok and float(row["ci_low"]) > 0

    same = results[results["relation"] == "opposite_same_wording"]
    def task_metric(metric):
        return same[same["metric"] == metric].groupby("task_id")["value"].mean()

    verified = task_metric("reader_verified_v_effect")
    all7_s = task_metric("reader_all7_sufficiency")
    all7_n = task_metric("reader_all7_necessity")
    non_s = task_metric("reader_non_kv0_sufficiency")
    non_n = task_metric("reader_non_kv0_necessity")

    sanity = {
        "all7_minus_verified_suff_mean": float((all7_s - verified).mean()),
        "all7_minus_verified_nec_mean": float((all7_n - verified).mean()),
        "non_kv0_sufficiency_mean": float(non_s.mean()),
        "non_kv0_necessity_mean": float(non_n.mean()),
        "mean_reader_leakage_ratio": float(same["reader_leakage_ratio"].dropna().mean()),
        "max_abs_reader_baseline_margin_diff": float(same["reader_baseline_margin_diff"].dropna().abs().max()),
        "max_abs_reader_verified_v_diff": float(same["reader_verified_v_diff"].dropna().abs().max()),
    }
    tol = 5e-5
    sanity_pass = bool(
        abs(sanity["all7_minus_verified_suff_mean"]) < tol
        and abs(sanity["all7_minus_verified_nec_mean"]) < tol
        and abs(sanity["non_kv0_sufficiency_mean"]) < tol
        and abs(sanity["non_kv0_necessity_mean"]) < tol
        and sanity["mean_reader_leakage_ratio"] < 1e-6
        and sanity["max_abs_reader_baseline_margin_diff"] < tol
        and sanity["max_abs_reader_verified_v_diff"] < tol
    )

    writer_pass = bool(
        endpoint_flags["selected_V_sufficiency"]
        and endpoint_flags["selected_V_necessity_loss"]
        and endpoint_flags["cross_selected_V_sufficiency"]
        and endpoint_flags["cross_selected_V_necessity_loss"]
        and contrast_flags["selected_minus_runner_up_sufficiency"]
        and contrast_flags["selected_minus_runner_up_necessity"]
        and bidir_ok
    )
    reader_pass = bool(
        endpoint_flags["reader_target_sufficiency"]
        and endpoint_flags["reader_target_necessity"]
        and contrast_flags["target_minus_negative_reader_sufficiency"]
        and contrast_flags["target_minus_negative_reader_necessity"]
    )
    confirmatory_pass = bool(writer_pass and reader_pass and sanity_pass)

    selected_anchors = {
        key: data["selected_anchor"] for key, data in selected["strata"].items()
    }
    unique_count = len(set(selected_anchors.values()))

    if confirmatory_pass:
        pattern = (
            "contextual_writer_plus_stable_reader"
            if unique_count > 1
            else "schema_stable_writer_plus_stable_reader"
        )
    elif reader_pass and not writer_pass:
        pattern = "portable_writer_not_confirmed_stable_reader_survives"
    elif writer_pass and not reader_pass:
        pattern = "writer_confirms_reader_register_fails"
    else:
        pattern = "neither_writer_nor_reader_confirmed"

    same_state_row = get_row("same_state_selected_V_control", "same_state_cross_wording")

    result = {
        "confirmatory_pass": confirmatory_pass,
        "writer_pass": writer_pass,
        "reader_pass": reader_pass,
        "bidirectional_writer_pass": bool(bidir_ok),
        "sanity_pass": sanity_pass,
        "mechanism_pattern": pattern,
        "selected_anchors": selected_anchors,
        "unique_selected_anchor_count": unique_count,
        "endpoint_flags": endpoint_flags,
        "contrast_flags": contrast_flags,
        "directional_endpoints": bidir,
        "same_state_selected_V_control_mean": (
            None if same_state_row is None else float(same_state_row["mean"])
        ),
        "sanity": sanity,
        "overall_metrics": summary_df.to_dict(orient="records"),
        "paired_contrasts": contrasts,
    }

    # Plot held-out target/control effects.
    plot_metrics = [
        "selected_V_sufficiency", "selected_V_necessity_loss",
        "old_absolute_V_sufficiency", "old_absolute_local_V_necessity_loss",
        "runner_up_V_sufficiency", "runner_up_V_necessity_loss",
        "reader_target_sufficiency", "reader_target_necessity",
        "reader_negative_sufficiency", "reader_negative_necessity",
    ]
    labels, vals = [], []
    for metric in plot_metrics:
        r = get_row(metric, "opposite_same_wording")
        if r is not None:
            labels.append(metric)
            vals.append(float(r["mean"]))
    fig = plt.figure(figsize=(13, 6))
    x = np.arange(len(labels))
    plt.bar(x, vals)
    plt.axhline(0.0, linestyle="--")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("Held-out causal effect")
    plt.title("EXP13 held-out contextual writer + frozen reader")
    plt.tight_layout()
    fig.savefig(out_dir / "confirmation_profile.png", dpi=180)
    plt.close(fig)

    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=5313)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--phase",
        choices=["all", "discovery", "confirmation"],
        default="all",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    exp12 = load_exp12()
    tasks = exp12.make_tasks()
    if len(tasks) != 64:
        raise RuntimeError(f"Expected exact EXP12 64-task set, got {len(tasks)}")
    split = family_split(tasks)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "split.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    model_dir = exp12.resolve_model_dir(args.model)
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
    layers = exp12.get_layers(model)

    num_q_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    hidden_size = int(model.config.hidden_size)
    head_dim = int(getattr(model.config, "head_dim", hidden_size // num_q_heads))

    if num_q_heads != 28 or num_kv_heads != 4 or num_q_heads // num_kv_heads != 7:
        raise RuntimeError(
            f"Frozen architecture invalid: Q={num_q_heads}, KV={num_kv_heads}"
        )

    print("=" * 92)
    print("EXP13: Context-Conditioned Schema Write-Site Remapping")
    print("EXP12 code:", EXP12_CODE)
    print("EXP12 sha256:", sha256_file(EXP12_CODE))
    print("Model:", model_dir)
    print("Tasks:", len(tasks))
    print("Split seed:", SPLIT_SEED)
    print("Anchors:", ANCHORS)
    print("Frozen readers:", TARGET_READERS)
    print("Phase:", args.phase)
    print("=" * 92)

    selected = None
    discovery_summary = None

    if args.phase in ("all", "discovery"):
        _, disc_summary, selected = run_discovery(
            model, tok, layers, exp12, tasks, split,
            num_kv_heads, head_dim, out_dir, args.seed,
        )
        discovery_summary = {
            "selected_schema": selected,
            "num_strata": 8,
            "num_candidate_anchors": len(ANCHORS),
        }

    if args.phase == "confirmation":
        selected_path = out_dir / "selected_schema.json"
        if not selected_path.is_file():
            raise FileNotFoundError(
                f"Confirmation-only mode requires frozen {selected_path}"
            )
        selected = json.loads(selected_path.read_text(encoding="utf-8"))

    confirmation_summary = None
    if args.phase in ("all", "confirmation"):
        conf = run_confirmation(
            model, tok, layers, exp12, tasks, split, selected,
            num_kv_heads, num_q_heads, head_dim, out_dir,
        )
        confirmation_summary = confirmation_inference(
            conf, selected, out_dir, args.seed
        )

    # Merge audit files.
    audit_frames = []
    for path in [
        out_dir / "schema_audit_discovery.csv",
        out_dir / "schema_audit_confirmation.csv",
    ]:
        if path.is_file():
            df = pd.read_csv(path)
            df["audit_source"] = path.stem
            audit_frames.append(df)
    if audit_frames:
        pd.concat(audit_frames, ignore_index=True).to_csv(
            out_dir / "schema_audit.csv", index=False
        )

    manifest = {
        "experiment": "EXP13_contextual_write_remap",
        "git_commit": git_commit(),
        "design_basis_remote_head_reported": "5c63817",
        "seed": args.seed,
        "split_seed": SPLIT_SEED,
        "phase": args.phase,
        "exp12_code": str(EXP12_CODE),
        "exp12_code_sha256": sha256_file(EXP12_CODE),
        "model_dir": str(model_dir),
        "model_name": model_dir.name,
        "num_tasks": len(tasks),
        "source_hidden_index": SOURCE_H,
        "consumer_decoder_block": CONSUMER_BLOCK,
        "kv_head": KV_HEAD,
        "schema_anchors": list(ANCHORS),
        "anchor_width": ANCHOR_WIDTH,
        "discovery_strata": "family x wording",
        "selection_rule": "max min(suff0,suff1,nec0,nec1)",
        "old_absolute_offsets": list(OLD_ABSOLUTE_OFFSETS),
        "target_readers": list(TARGET_READERS),
        "negative_readers": list(NEGATIVE_READERS),
        "num_attention_heads": num_q_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
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
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "experiment": "EXP13_contextual_write_remap",
        "phase": args.phase,
        "exp12_output_audit_applied": True,
        "discovery": discovery_summary,
        "confirmation": confirmation_summary,
        "decision_rule": (
            "Writer claim requires held-out positive sufficiency/necessity, "
            "bidirectionality, superiority to old absolute and frozen runner-up, "
            "cross-wording mapping, plus the frozen Q0/Q3/Q5 reader register "
            "outperforming Q2/Q4/Q6 under exact decomposition sanity."
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
