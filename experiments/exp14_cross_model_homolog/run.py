#!/usr/bin/env python3
"""EXP14 — Cross-Model Functional Homolog Replication.

Freeze the mechanism definition M (schema-anchored writer -> single-block V(KV)
interface -> positive/inhibitory reader register -> behavior); do NOT freeze the
Qwen-Coder indices H20 / KV0 / Q0,Q3,Q5 / Q2,Q4,Q6.  Search functional homologs on
Model B's discovery split; freeze everything for held-out confirmation.

Reuses EXP12 task infrastructure (importlib, no template duplication) and the
EXP13 capture/patch/bootstrap machinery, parameterized by (source block L, KV
head k).  All capture and patch sites are consistent: residual at layers[L-1]
output (= input to block L), V at layers[L].self_attn.v_proj, readers at
layers[L].self_attn.o_proj pre-conv.  Candidate scoring is batch=1.
"""
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
import subprocess
from contextlib import ExitStack, contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP12_CODE = REPO_ROOT / "experiments" / "exp12_independent_replication" / "run.py"
DEFAULT_OUT = REPO_ROOT / "outputs" / "exp14_cross_model_homolog"
MODEL_A_DIR = REPO_ROOT / "models" / "Qwen2.5-Coder-7B-Instruct"
DEFAULT_MODEL_B = Path("/data/mzb/ar2_scratch/models/Qwen2-7B-Instruct")

SPLIT_SEED = 5313
DISCOVERY_PER_FAMILY = 8
ANCHOR_WIDTH = 6
READER_SET_SIZE = 3
LAYER_SWEEP = tuple(range(16, 28))        # 16..27 (deep band, 12 layers)
KV_SWEEP = (0, 1, 2, 3)
PHASE_A_ANCHORS = ("USER_END", "FINAL_INSTRUCTION_END")

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
INSTRUCTION_FAMILY = ("USER_END", "FINAL_INSTRUCTION_END", "SYSTEM_END", "GENERATION_BOUNDARY")

OLD_ABSOLUTE_OFFSETS = (-13, -5, -3, -1)
GATE_SEED = 9001
N_BOOT = 5000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def load_exp12():
    if not EXP12_CODE.is_file():
        raise FileNotFoundError(f"EXP14 requires repository EXP12 implementation: {EXP12_CODE}")
    spec = importlib.util.spec_from_file_location("skills4s_exp12", EXP12_CODE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    required = [
        "make_tasks", "make_skill_entries", "make_direct_entries", "donor_maps_skill",
        "render_prompt", "resolve_model_dir", "get_layers", "SKILL_TEXT",
    ]
    missing = [name for name in required if not hasattr(mod, name)]
    if missing:
        raise RuntimeError(f"EXP12 API drift; missing: {missing}")
    return mod


def family_split(tasks):
    by_family = {}
    for task in tasks:
        by_family.setdefault(task["family"], []).append(task)
    split = {"seed": SPLIT_SEED, "discovery_per_family": DISCOVERY_PER_FAMILY, "families": {}}
    for fi, family in enumerate(sorted(by_family)):
        family_tasks = sorted(by_family[family], key=lambda x: x["task_id"])
        if len(family_tasks) != 16:
            raise RuntimeError(f"Expected 16 tasks in {family}, got {len(family_tasks)}")
        rng = np.random.default_rng(SPLIT_SEED + fi)
        order = rng.permutation(len(family_tasks))
        split["families"][family] = {
            "discovery": [family_tasks[int(i)]["task_id"] for i in order[:DISCOVERY_PER_FAMILY]],
            "confirmation": [family_tasks[int(i)]["task_id"] for i in order[DISCOVERY_PER_FAMILY:]],
        }
    return split


def split_ids(split, partition):
    return {tid for data in split["families"].values() for tid in data[partition]}


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
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(x) for x in enc["input_ids"]]
    offsets = enc["offset_mapping"]
    n = len(ids)

    wording = entry["wording"]
    if wording != "direct":
        skill_text = exp12.SKILL_TEXT[task["family"]][int(entry["label"])][wording]
    else:
        skill_text = None
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
        if segment is None:
            continue
        try:
            start, end = find_segment(text, segment)
        except RuntimeError:
            continue  # segment absent from this prompt (e.g. SKILL_END in direct)
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
            if wording == "direct" and name == "SKILL_END":
                continue  # direct prompts have no skill text
            raise RuntimeError(f"Schema anchor invariant failed: {name}")
    return {"text": text, "ids": ids, "anchors": anchors}


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
        store[key] = x.view(b, s, num_q_heads, head_dim).detach().float().cpu().numpy().astype(np.float32)

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
def capture_prompt_states(model, tok, layers, exp12, task, entry, L, num_kv_heads, head_dim):
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

    handles.append(layers[L - 1].register_forward_hook(residual_hook))
    handles.append(layers[L].self_attn.v_proj.register_forward_hook(v_hook))
    try:
        model(**enc, use_cache=False, return_dict=True)
    finally:
        for h in handles:
            h.remove()

    schema = compute_schema(exp12, tok, task, entry)
    if ids != schema["ids"]:
        raise RuntimeError("Tokenizer mismatch between capture and schema")
    return {"ids": ids, "residual": store["residual"], "v": store["v"], "schema": schema}


def build_task_cache(model, tok, layers, exp12, task, entries, L, num_kv_heads, head_dim):
    return [
        capture_prompt_states(model, tok, layers, exp12, task, e, L, num_kv_heads, head_dim)
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
def score_margin(
    model, tok, layers, exp12, entry, rec_cache, L, num_kv_heads, head_dim,
    residual_source_cache=None, residual_rec_positions=None, residual_donor_positions=None,
    v_source_cache=None, recipient_positions=None, donor_positions=None,
    v_heads=(0,),
):
    device = model.get_input_embeddings().weight.device
    prompt_text = exp12.render_prompt(tok, entry["messages"])
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    if rec_cache is not None and prompt_ids != rec_cache["ids"]:
        raise RuntimeError("Recipient cache mismatch")
    plen = len(prompt_ids)
    scores = []

    for candidate in entry["candidates"]:
        cids = tok(candidate, add_special_tokens=False)["input_ids"]
        input_ids = torch.tensor([prompt_ids + cids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)

        with ExitStack() as stack:
            if residual_source_cache is not None:
                rec_r = np.asarray(residual_rec_positions, dtype=np.int64)
                don_r = np.asarray(residual_donor_positions, dtype=np.int64)
                if len(rec_r) != len(don_r):
                    raise RuntimeError("Residual source/target count mismatch")
                values = residual_source_cache["residual"][don_r, :]
                stack.enter_context(residual_patch_hook(
                    layers[L - 1],
                    torch.tensor(rec_r[None, :], dtype=torch.long, device=device),
                    torch.tensor(values[None, :, :], dtype=torch.float32, device=device),
                ))

            if v_source_cache is not None:
                rec_pos = np.asarray(recipient_positions, dtype=np.int64)
                donor_pos = np.asarray(donor_positions, dtype=np.int64)
                if len(rec_pos) != len(donor_pos):
                    raise RuntimeError("V source/target position count mismatch")
                values = v_source_cache["v"][donor_pos][:, list(v_heads), :]
                stack.enter_context(value_patch_hook(
                    layers[L].self_attn.v_proj,
                    torch.tensor(rec_pos[None, :], dtype=torch.long, device=device),
                    torch.tensor(list(v_heads), dtype=torch.long, device=device),
                    torch.tensor(values[None, :, :, :], dtype=torch.float32, device=device),
                    num_kv_heads, head_dim,
                ))

            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        use_cache=False, return_dict=True)
        scores.append(candidate_mean_logprob(out.logits, input_ids, plen, len(cids)))
    return scores[1] - scores[0]


def baseline_margins(model, tok, layers, exp12, entries, cache, L, num_kv_heads, head_dim):
    return [
        score_margin(model, tok, layers, exp12, e, cache[i], L, num_kv_heads, head_dim)
        for i, e in enumerate(entries)
    ]


def residual_effect(model, tok, layers, exp12, entry, rec_cache, donor, donor_cache,
                    baseline_margin, L, rec_positions, donor_positions,
                    num_kv_heads, head_dim):
    margin = score_margin(
        model, tok, layers, exp12, entry, rec_cache, L, num_kv_heads, head_dim,
        residual_source_cache=donor_cache,
        residual_rec_positions=rec_positions,
        residual_donor_positions=donor_positions,
    )
    return signed_effect(margin, baseline_margin, donor)


def v_suff_nec(model, tok, layers, exp12, entry, rec_cache, donor, donor_cache,
               baseline_margin, residual_ref, L, rec_positions, donor_positions,
               num_kv_heads, head_dim, v_heads=(0,)):
    suff_margin = score_margin(
        model, tok, layers, exp12, entry, rec_cache, L, num_kv_heads, head_dim,
        v_source_cache=donor_cache,
        recipient_positions=rec_positions,
        donor_positions=donor_positions,
        v_heads=v_heads,
    )
    suff = signed_effect(suff_margin, baseline_margin, donor)

    if residual_ref is None:
        # suff-only mode (Phase A / cross / controls): necessity not requested
        return suff, np.nan

    clamp_margin = score_margin(
        model, tok, layers, exp12, entry, rec_cache, L, num_kv_heads, head_dim,
        residual_source_cache=donor_cache,
        residual_rec_positions=rec_positions,
        residual_donor_positions=donor_positions,
        v_source_cache=rec_cache,
        recipient_positions=rec_positions,
        donor_positions=rec_positions,
        v_heads=v_heads,
    )
    retained = signed_effect(clamp_margin, baseline_margin, donor)
    necessity = residual_ref - retained
    return suff, necessity


@torch.inference_mode()
def candidate_capture_oproj(model, tok, layers, exp12, entry, candidate, rec_cache,
                            L, num_kv_heads, num_q_heads, head_dim,
                            v_source_cache=None, recipient_positions=None,
                            donor_positions=None, v_heads=(0,)):
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
            values = v_source_cache["v"][donor_pos][:, list(v_heads), :]
            stack.enter_context(value_patch_hook(
                layers[L].self_attn.v_proj,
                torch.tensor(rec_pos[None, :], dtype=torch.long, device=device),
                torch.tensor(list(v_heads), dtype=torch.long, device=device),
                torch.tensor(values[None, :, :, :], dtype=torch.float32, device=device),
                num_kv_heads, head_dim,
            ))
        stack.enter_context(oproj_capture_hook(
            layers[L].self_attn.o_proj, store, "oproj", num_q_heads, head_dim,
        ))
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    use_cache=False, return_dict=True)
    return candidate_mean_logprob(out.logits, input_ids, plen, len(cids)), store["oproj"]


@torch.inference_mode()
def candidate_replace_readers(model, tok, layers, exp12, entry, candidate, rec_cache,
                              L, num_kv_heads, num_q_heads, head_dim,
                              replacement_np, readers,
                              upstream_v_cache=None, recipient_positions=None,
                              donor_positions=None, v_heads=(0,)):
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
            values = upstream_v_cache["v"][donor_pos][:, list(v_heads), :]
            stack.enter_context(value_patch_hook(
                layers[L].self_attn.v_proj,
                torch.tensor(rec_pos[None, :], dtype=torch.long, device=device),
                torch.tensor(list(v_heads), dtype=torch.long, device=device),
                torch.tensor(values[None, :, :, :], dtype=torch.float32, device=device),
                num_kv_heads, head_dim,
            ))
        stack.enter_context(oproj_replace_heads_hook(
            layers[L].self_attn.o_proj,
            torch.tensor(list(readers), dtype=torch.long, device=device),
            torch.tensor(replacement_np, dtype=torch.float32, device=device),
            num_q_heads, head_dim,
        ))
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    use_cache=False, return_dict=True)
    return candidate_mean_logprob(out.logits, input_ids, plen, len(cids))


def pair_margin(scores):
    return scores[1] - scores[0]


def reader_path_effects(model, tok, layers, exp12, entry, rec_cache, donor, donor_cache,
                        rec_positions, donor_positions, baseline_margin,
                        L, v_heads, num_kv_heads, num_q_heads, head_dim,
                        leak_register_heads=None, include_nec=True):
    base_scores, patch_scores = [], []
    base_caps, patch_caps = [], []
    for candidate in entry["candidates"]:
        bs, bc = candidate_capture_oproj(
            model, tok, layers, exp12, entry, candidate, rec_cache,
            L, num_kv_heads, num_q_heads, head_dim, v_heads=v_heads,
        )
        ps, pc = candidate_capture_oproj(
            model, tok, layers, exp12, entry, candidate, rec_cache,
            L, num_kv_heads, num_q_heads, head_dim,
            v_source_cache=donor_cache, recipient_positions=rec_positions,
            donor_positions=donor_positions, v_heads=v_heads,
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
    if leak_register_heads is not None:
        reg_set = set(leak_register_heads)
        for bc, pc in zip(base_caps, patch_caps):
            delta = pc - bc
            all_heads = tuple(range(num_q_heads))
            outside = tuple(qh for qh in all_heads if qh not in reg_set)
            num += float(np.sum(delta[:, :, list(outside), :] ** 2))
            den += float(np.sum(delta[:, :, list(reg_set), :] ** 2))
    leakage_ratio = float(np.sqrt(num / (den + 1e-30)))

    def group_effect(readers):
        suff_scores, nec_scores = [], []
        for ci, candidate in enumerate(entry["candidates"]):
            bc = base_caps[ci]
            pc = patch_caps[ci]
            suff_scores.append(candidate_replace_readers(
                model, tok, layers, exp12, entry, candidate, rec_cache,
                L, num_kv_heads, num_q_heads, head_dim,
                replacement_np=pc[:, :, list(readers), :], readers=readers, v_heads=v_heads,
            ))
            if include_nec:
                nec_scores.append(candidate_replace_readers(
                    model, tok, layers, exp12, entry, candidate, rec_cache,
                    L, num_kv_heads, num_q_heads, head_dim,
                    replacement_np=bc[:, :, list(readers), :], readers=readers,
                    upstream_v_cache=donor_cache, recipient_positions=rec_positions,
                    donor_positions=donor_positions, v_heads=v_heads,
                ))
        suff = signed_effect(pair_margin(suff_scores), reader_base_margin, donor)
        if not include_nec:
            return suff, np.nan
        retained = signed_effect(pair_margin(nec_scores), reader_base_margin, donor)
        nec = verified_v_effect - retained
        return suff, nec

    return {
        "verified_v_effect": verified_v_effect,
        "reader_baseline_margin": reader_base_margin,
        "baseline_margin_diff": reader_base_margin - baseline_margin,
        "leakage_ratio": leakage_ratio,
        "groups": group_effect,
    }


def non_reader_heads(pos_set=(), neg_set=()):
    excluded = set(pos_set) | set(neg_set)
    return tuple(q for q in range(28) if q not in excluded)


def bootstrap_ci(values, seed, n_boot=N_BOOT):
    x = np.asarray(values, dtype=np.float64)
    if len(x) == 0:
        raise RuntimeError("Empty bootstrap input")
    rng = np.random.default_rng(seed)
    boots = np.asarray([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)])
    return float(x.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def paired_ci(a, b, seed):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) != len(b):
        raise RuntimeError("Paired arrays differ in length")
    return bootstrap_ci(a - b, seed)


def add_result(rows, *, phase, task, entry, donor, relation, anchor, metric, value,
               L=None, k=None, readers=None, extra=None):
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
        "source_L": L,
        "kv_head": k,
        "readers": readers,
    }
    if extra:
        row.update(extra)
    rows.append(row)


# ---------------------------------------------------------------- phases

def load_model(model_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    layers = model.model.layers
    if len(layers) != 28:
        raise RuntimeError(f"EXP14 currently validates on 28-layer Qwen2; got {len(layers)}")
    return model, tok, layers


def run_gate(model, tok, layers, exp12, tasks, num_kv_heads, head_dim, out_dir, seed):
    rows = []
    for task in tasks:
        entries = exp12.make_skill_entries(task)
        for e in entries:
            margin = score_margin(
                model, tok, layers, exp12, e, None, None, num_kv_heads, head_dim,
            )
            rows.append({
                "task_id": task["task_id"], "family": task["family"],
                "wording": e["wording"], "label": int(e["label"]),
                "margin": margin, "signed": margin if e["label"] == 1 else -margin,
            })
    df = pd.DataFrame(rows)
    pooled = bootstrap_ci(df["signed"].values, GATE_SEED)
    per_family = {
        fam: bootstrap_ci(g["signed"].values, GATE_SEED + i)
        for i, (fam, g) in enumerate(df.groupby("family"))
    }
    family_sign_ok = all(m[0] > 0 for m in per_family.values())
    pooled_ok = pooled[0] > 0 and pooled[1] > 0  # CI_low > 0
    gate_pass = bool(pooled_ok and family_sign_ok)

    report = {
        "gate_pass": gate_pass,
        "pooled_signed_baseline": {"mean": pooled[0], "ci_low": pooled[1], "ci_high": pooled[2]},
        "per_family": {k: {"mean": v[0], "ci_low": v[1], "ci_high": v[2]} for k, v in per_family.items()},
        "per_label": {
            str(l): {"mean": m[0], "ci_low": m[1], "ci_high": m[2]}
            for l, m in df.groupby("label")["signed"].apply(lambda s: bootstrap_ci(s.values, GATE_SEED + 10)).items()
        },
    }
    df.to_csv(out_dir / "gate_report.csv", index=False)
    out_dir.joinpath("gate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report, df


def run_discovery(model, tok, layers, exp12, tasks, split, num_kv_heads, head_dim,
                  out_dir, seed):
    disc_ids = split_ids(split, "discovery")
    disc_tasks = [t for t in tasks if t["task_id"] in disc_ids]
    rows = []

    # baseline margins per discovery entry (index-independent; compute once)
    entry_keys = []          # (task, entry, cache_i)
    baselines = {}           # (task_id, i) -> margin
    for task in disc_tasks:
        entries = exp12.make_skill_entries(task)
        for i in range(len(entries)):
            base = score_margin(model, tok, layers, exp12, entries[i], None,
                                None, num_kv_heads, head_dim)
            baselines[(task["task_id"], i)] = base

    # ---- Phase A: (L, k) coarse selection at {USER_END, FINAL_INSTRUCTION_END}
    a_scores = {}   # (L, k) -> score
    a_rows = []
    best_phase_a = None
    for L in LAYER_SWEEP:
        # cache per task at this L
        task_caches = {}
        for task in disc_tasks:
            task_caches[task["task_id"]] = build_task_cache(
                model, tok, layers, exp12, task,
                exp12.make_skill_entries(task), L, num_kv_heads, head_dim)
        for k in KV_SWEEP:
            suff_by_label = {0: [], 1: []}
            for task in disc_tasks:
                entries = exp12.make_skill_entries(task)
                maps = exp12.donor_maps_skill(entries)
                for i, entry in enumerate(entries):
                    donor = entries[maps[i]["opposite_same_wording"]]
                    for anchor in PHASE_A_ANCHORS:
                        rec_pos = anchor_positions(task_caches[task["task_id"]][i], anchor)
                        don_pos = anchor_positions(task_caches[task["task_id"]][maps[i]["opposite_same_wording"]], anchor)
                        suff = v_suff_nec(
                            model, tok, layers, exp12, entry,
                            task_caches[task["task_id"]][i],
                            donor, task_caches[task["task_id"]][maps[i]["opposite_same_wording"]],
                            baselines[(task["task_id"], i)], None,
                            L, rec_pos, don_pos, num_kv_heads, head_dim, v_heads=(k,),
                        )[0]
                        suff_by_label[int(entry["label"])].append(suff)
                        a_rows.append({
                            "phase": "discovery_A", "task_id": task["task_id"],
                            "family": task["family"], "label": int(entry["label"]),
                            "anchor": anchor, "L": L, "k": k, "suff": suff,
                        })
            score = min(float(np.mean(suff_by_label[0])), float(np.mean(suff_by_label[1])))
            a_scores[(L, k)] = score
            print(f"[phaseA] L={L} k={k} score={score:.5f}")
        del task_caches
        torch.cuda.empty_cache()

    best = max(a_scores, key=a_scores.get)
    L_star, k_star = int(best[0]), int(best[1])
    phase_a_df = pd.DataFrame(a_rows)
    phase_a_df.to_csv(out_dir / "discovery_phaseA.csv", index=False)

    # ---- Phase B: per-stratum anchor selection at (L*, k*)
    # rebuild cache at L* (needed for confirmation later anyway via selected file)
    task_caches = {}
    for task in disc_tasks:
        task_caches[task["task_id"]] = build_task_cache(
            model, tok, layers, exp12, task, exp12.make_skill_entries(task),
            L_star, num_kv_heads, head_dim)

    strata = {}
    for task in disc_tasks:
        entries = exp12.make_skill_entries(task)
        maps = exp12.donor_maps_skill(entries)
        for i, entry in enumerate(entries):
            key = (task["family"], entry["wording"])
            strata.setdefault(key, []).append((task, entries, maps, i, entry))

    selected = {}
    for (family, wording), members in strata.items():
        anchor_metrics = {}
        for anchor in ANCHORS:
            s0, s1, n0, n1 = [], [], [], []
            for task, entries, maps, i, entry in members:
                donor = entries[maps[i]["opposite_same_wording"]]
                rec_cache = task_caches[task["task_id"]][i]
                don_cache = task_caches[task["task_id"]][maps[i]["opposite_same_wording"]]
                rec_pos = anchor_positions(rec_cache, anchor)
                don_pos = anchor_positions(don_cache, anchor)
                res_ref = residual_effect(
                    model, tok, layers, exp12, entry, rec_cache, donor, don_cache,
                    baselines[(task["task_id"], i)], L_star, rec_pos, don_pos,
                    num_kv_heads, head_dim,
                )
                suff, nec = v_suff_nec(
                    model, tok, layers, exp12, entry, rec_cache, donor, don_cache,
                    baselines[(task["task_id"], i)], res_ref, L_star, rec_pos, don_pos,
                    num_kv_heads, head_dim, v_heads=(k_star,),
                )
                if int(entry["label"]) == 0:
                    s0.append(suff); n0.append(nec)
                else:
                    s1.append(suff); n1.append(nec)
                add_result(rows, phase="discovery", task=task, entry=entry, donor=donor,
                           relation="opposite_same_wording", anchor=anchor,
                           metric="V_sufficiency", value=suff, L=L_star, k=k_star)
                add_result(rows, phase="discovery", task=task, entry=entry, donor=donor,
                           relation="opposite_same_wording", anchor=anchor,
                           metric="V_necessity_loss", value=nec, L=L_star, k=k_star)
            s0m, s1m, n0m, n1m = np.mean(s0), np.mean(s1), np.mean(n0), np.mean(n1)
            anchor_metrics[anchor] = {
                "suff0": float(s0m), "suff1": float(s1m),
                "nec0": float(n0m), "nec1": float(n1m),
                "bidirectional_score": float(min(s0m, s1m, n0m, n1m)),
            }
        order = sorted(anchor_metrics, key=lambda a: (-anchor_metrics[a]["bidirectional_score"], ANCHORS.index(a)))
        sel, run = order[0], order[1]
        selected[f"{family}::{wording}"] = {
            "family": family, "wording": wording,
            "selected_anchor": sel, "runner_up_anchor": run,
            "selected_score": anchor_metrics[sel]["bidirectional_score"],
            "runner_up_score": anchor_metrics[run]["bidirectional_score"],
            "anchors": anchor_metrics,
        }
        print(f"[phaseB] {family}::{wording} -> {sel} (score {anchor_metrics[sel]['bidirectional_score']:.5f})")

    # ---- Phase C: reader register search at (L*, k*) with per-stratum anchors
    head_rows = []
    per_head = {}
    for task in disc_tasks:
        entries = exp12.make_skill_entries(task)
        maps = exp12.donor_maps_skill(entries)
        for i, entry in enumerate(entries):
            key = f"{task['family']}::{entry['wording']}"
            anchor = selected[key]["selected_anchor"]
            donor = entries[maps[i]["opposite_same_wording"]]
            rec_cache = task_caches[task["task_id"]][i]
            don_cache = task_caches[task["task_id"]][maps[i]["opposite_same_wording"]]
            rec_pos = anchor_positions(rec_cache, anchor)
            don_pos = anchor_positions(don_cache, anchor)
            base_margin = baselines[(task["task_id"], i)]
            rp = reader_path_effects(
                model, tok, layers, exp12, entry, rec_cache, donor, don_cache,
                rec_pos, don_pos, base_margin, L_star, (k_star,),
                num_kv_heads, model.config.num_attention_heads, head_dim,
                include_nec=False,
            )
            for q in range(model.config.num_attention_heads):
                suff, _ = rp["groups"]([q])
                per_head.setdefault(q, []).append(suff)
                head_rows.append({
                    "task_id": task["task_id"], "family": task["family"],
                    "wording": entry["wording"], "label": int(entry["label"]),
                    "head": q, "suff": suff,
                })
    head_df = pd.DataFrame(head_rows)
    head_df.to_csv(out_dir / "discovery_reader_heads.csv", index=False)
    head_means = {q: float(np.mean(v)) for q, v in per_head.items()}
    sorted_pos = sorted(head_means, key=lambda q: -head_means[q])
    pos_set = tuple(sorted(sorted_pos[:READER_SET_SIZE]))
    neg_set = tuple(sorted(sorted_pos[-READER_SET_SIZE:]))
    print(f"[phaseC] pos_set={pos_set} neg_set={neg_set}")

    result = {
        "L_star": L_star, "k_star": k_star,
        "phase_a_best_score": a_scores[best],
        "reader_pos_set": list(pos_set), "reader_neg_set": list(neg_set),
        "head_means": head_means,
        "strata": selected,
    }
    out_dir.joinpath("selected_homolog.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_dir / "discovery_results.csv", index=False)
    print(json.dumps({"L_star": L_star, "k_star": k_star, "pos": list(pos_set), "neg": list(neg_set)},
                     ensure_ascii=False))
    return result


def run_confirmation(model, tok, layers, exp12, tasks, split, selected,
                     num_kv_heads, num_q_heads, head_dim, out_dir, seed):
    L_star, k_star = selected["L_star"], selected["k_star"]
    pos_set, neg_set = tuple(selected["reader_pos_set"]), tuple(selected["reader_neg_set"])
    non_set = non_reader_heads(pos_set, neg_set)
    all_set = tuple(range(num_q_heads))
    conf_ids = split_ids(split, "confirmation")
    conf_tasks = [t for t in tasks if t["task_id"] in conf_ids]
    rows = []

    for n, task in enumerate(conf_tasks, 1):
        entries = exp12.make_skill_entries(task)
        maps = exp12.donor_maps_skill(entries)
        cache = build_task_cache(model, tok, layers, exp12, task, entries,
                                 L_star, num_kv_heads, head_dim)
        baselines = baseline_margins(model, tok, layers, exp12, entries, cache,
                                     L_star, num_kv_heads, head_dim)

        for i, entry in enumerate(entries):
            stratum = f"{task['family']}::{entry['wording']}"
            anchor = selected["strata"][stratum]["selected_anchor"]
            runner_up = selected["strata"][stratum]["runner_up_anchor"]
            donor_i = maps[i]["opposite_same_wording"]
            donor = entries[donor_i]
            cross_i = maps[i]["opposite_cross_wording"]
            cross_donor = entries[cross_i]
            same_i = maps[i]["same_state_cross_wording"]
            same_donor = entries[same_i]

            rec_cache = cache[i]
            don_cache = cache[donor_i]
            cross_cache = cache[cross_i]
            same_cache = cache[same_i]
            base = baselines[i]

            rec_pos = anchor_positions(rec_cache, anchor)
            don_pos = anchor_positions(don_cache, anchor)
            cross_rec_pos = anchor_positions(rec_cache, anchor)
            cross_don_pos = anchor_positions(cross_cache,
                selected["strata"][f"{task['family']}::{cross_donor['wording']}"]["selected_anchor"])
            same_pos = anchor_positions(rec_cache, anchor)
            same_don_pos = anchor_positions(same_cache,
                selected["strata"][f"{task['family']}::{same_donor['wording']}"]["selected_anchor"])
            neg_pos = anchor_positions(rec_cache, "ISSUE_END")
            neg_don_pos = anchor_positions(don_cache, "ISSUE_END")
            old_pos = old_absolute_positions(rec_cache)
            old_don_pos = old_absolute_positions(don_cache)

            res_ref = residual_effect(
                model, tok, layers, exp12, entry, rec_cache, donor, don_cache, base,
                L_star, rec_pos, don_pos, num_kv_heads, head_dim)
            suff, nec = v_suff_nec(
                model, tok, layers, exp12, entry, rec_cache, donor, don_cache, base,
                res_ref, L_star, rec_pos, don_pos, num_kv_heads, head_dim, v_heads=(k_star,))
            cross_suff, _ = v_suff_nec(
                model, tok, layers, exp12, entry, rec_cache, cross_donor, cross_cache,
                base, None, L_star, cross_rec_pos, cross_don_pos,
                num_kv_heads, head_dim, v_heads=(k_star,))
            same_ctl, _ = v_suff_nec(
                model, tok, layers, exp12, entry, rec_cache, same_donor, same_cache,
                base, None, L_star, same_pos, same_don_pos,
                num_kv_heads, head_dim, v_heads=(k_star,))
            neg_ctl, _ = v_suff_nec(
                model, tok, layers, exp12, entry, rec_cache, donor, don_cache,
                base, None, L_star, neg_pos, neg_don_pos,
                num_kv_heads, head_dim, v_heads=(k_star,))
            old_suff, _ = v_suff_nec(
                model, tok, layers, exp12, entry, rec_cache, donor, don_cache,
                base, None, L_star, old_pos, old_don_pos,
                num_kv_heads, head_dim, v_heads=(k_star,))

            for metric, value in [
                ("selected_V_sufficiency", suff), ("selected_V_necessity_loss", nec),
                ("cross_selected_V_sufficiency", cross_suff),
                ("same_state_selected_V_control", same_ctl),
                ("matched_negative_ISSUE_V_sufficiency", neg_ctl),
                ("old_absolute_V_sufficiency", old_suff),
            ]:
                add_result(rows, phase="confirmation", task=task, entry=entry, donor=donor,
                           relation="opposite_same_wording", anchor=anchor,
                           metric=metric, value=value, L=L_star, k=k_star)

            # runner-up control (paired, per entry)
            run_rec_pos = anchor_positions(rec_cache, runner_up)
            run_don_pos = anchor_positions(don_cache, runner_up)
            run_suff, run_nec = v_suff_nec(
                model, tok, layers, exp12, entry, rec_cache,
                donor, don_cache, base, res_ref, L_star,
                run_rec_pos, run_don_pos, num_kv_heads, head_dim, v_heads=(k_star,))
            for metric, value in [
                ("runner_up_V_sufficiency", run_suff),
                ("runner_up_V_necessity_loss", run_nec),
            ]:
                add_result(rows, phase="confirmation", task=task, entry=entry, donor=donor,
                           relation="opposite_same_wording", anchor=runner_up,
                           metric=metric, value=value, L=L_star, k=k_star)

            rp = reader_path_effects(
                model, tok, layers, exp12, entry, rec_cache, donor, don_cache,
                rec_pos, don_pos, base, L_star, (k_star,),
                num_kv_heads, num_q_heads, head_dim,
                leak_register_heads=tuple(sorted(set(pos_set) | set(neg_set))),
            )
            groups = {
                "target_pos": pos_set, "negative_neg": neg_set,
                "non_set": non_set, "all28": all_set,
            }
            for gname, readers in groups.items():
                gsuff, gnec = rp["groups"](readers)
                add_result(rows, phase="confirmation", task=task, entry=entry, donor=donor,
                           relation="opposite_same_wording", anchor=anchor,
                           metric=f"reader_{gname}_sufficiency", value=gsuff,
                           L=L_star, k=k_star, readers=",".join(map(str, readers)))
                add_result(rows, phase="confirmation", task=task, entry=entry, donor=donor,
                           relation="opposite_same_wording", anchor=anchor,
                           metric=f"reader_{gname}_necessity", value=gnec,
                           L=L_star, k=k_star, readers=",".join(map(str, readers)))
            add_result(rows, phase="confirmation", task=task, entry=entry, donor=donor,
                       relation="opposite_same_wording", anchor=anchor,
                       metric="reader_verified_v_effect", value=rp["verified_v_effect"],
                       L=L_star, k=k_star)
            add_result(rows, phase="confirmation", task=task, entry=entry, donor=donor,
                       relation="opposite_same_wording", anchor=anchor,
                       metric="reader_leakage_ratio", value=rp["leakage_ratio"],
                       L=L_star, k=k_star)

        # procedural specificity: direct-condition V suff @ USER_END,
        # donor = opposite-label direct entry (mirrors skill opposite-label donor)
        direct_entries = exp12.make_direct_entries(task)
        d_cache = build_task_cache(model, tok, layers, exp12, task, direct_entries,
                                   L_star, num_kv_heads, head_dim)
        d_bases = baseline_margins(model, tok, layers, exp12, direct_entries, d_cache,
                                   L_star, num_kv_heads, head_dim)
        for di, direct in enumerate(direct_entries):
            dopp = 1 - di  # opposite label
            d_pos = anchor_positions(d_cache[di], "USER_END")
            d_don_pos = anchor_positions(d_cache[dopp], "USER_END")
            d_suff, _ = v_suff_nec(
                model, tok, layers, exp12, direct, d_cache[di],
                direct_entries[dopp], d_cache[dopp], d_bases[di], None, L_star,
                d_pos, d_don_pos, num_kv_heads, head_dim, v_heads=(k_star,))
            add_result(rows, phase="confirmation", task=task, entry=direct,
                       donor=direct_entries[dopp],
                       relation="direct_opposite_label", anchor="USER_END",
                       metric="direct_V_sufficiency", value=d_suff, L=L_star, k=k_star)

        print(f"[confirmation {n:03d}/{len(conf_tasks):03d}] {task['task_id']} ({task['family']})")
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "confirmation_results.csv", index=False)
    return df


def summarize(df, phase, seed):
    task_level = df[df["phase"] == phase].groupby(["relation", "metric"]).agg(
        mean=("value", "mean"),
        num_tasks=("value", "count"),
    ).reset_index()
    rows = []
    for (relation, metric), g in df[df["phase"] == phase].groupby(["relation", "metric"]):
        by_task = g.groupby("task_id")["value"].agg("mean").values
        m, lo, hi = bootstrap_ci(by_task, seed)
        rows.append({"relation": relation, "metric": metric,
                     "mean": m, "ci_low": lo, "ci_high": hi, "num_tasks": len(by_task)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_B))
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--phase", choices=("gate", "discovery", "confirmation", "all"), default="all")
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = ap.parse_args()

    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp12 = load_exp12()
    tasks = exp12.make_tasks()
    if len(tasks) != 64:
        raise RuntimeError(f"Expected 64 tasks, got {len(tasks)}")
    split = family_split(tasks)

    print("=" * 92)
    print("EXP14: Cross-Model Functional Homolog Replication")
    print("Model B:", model_dir)
    print("Model A reference:", MODEL_A_DIR)
    print("Tasks:", len(tasks), "| split seed:", SPLIT_SEED, "| phase:", args.phase)
    print("Layer sweep:", LAYER_SWEEP, "| KV sweep:", KV_SWEEP)
    print("=" * 92)

    model, tok, layers = load_model(model_dir)
    num_q_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // num_q_heads

    selected_path = out_dir / "selected_homolog.json"
    if args.phase in ("confirmation",) and not selected_path.is_file():
        raise FileNotFoundError(f"Confirmation requires frozen {selected_path}")

    if args.phase in ("gate", "all"):
        gate_report, _ = run_gate(model, tok, layers, exp12, tasks,
                                  num_kv_heads, head_dim, out_dir, args.seed)
        if not gate_report["gate_pass"]:
            print("GATE FAILED — stopping before discovery (preregistered null on engagement axis).")
            if args.phase == "gate":
                return
            raise SystemExit(1)

    if args.phase in ("discovery", "all"):
        selected = run_discovery(model, tok, layers, exp12, tasks, split,
                                 num_kv_heads, head_dim, out_dir, args.seed)

    if args.phase in ("confirmation", "all"):
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        conf = run_confirmation(model, tok, layers, exp12, tasks, split, selected,
                                num_kv_heads, num_q_heads, head_dim, out_dir, args.seed)

        summary = summarize(conf, "confirmation", args.seed + 5)
        summary.to_csv(out_dir / "confirmation_summary.csv", index=False)
        print(summary.to_string(index=False))

        manifest = {
            "experiment": "EXP14_cross_model_homolog",
            "git_commit": git_commit(),
            "seed": args.seed,
            "split_seed": SPLIT_SEED,
            "phase": args.phase,
            "model_b_dir": str(model_dir),
            "model_b_name": model_dir.name,
            "model_b_config_sha256": sha256_file(model_dir / "config.json"),
            "model_a_dir": str(MODEL_A_DIR),
            "exp12_code": str(EXP12_CODE),
            "exp12_code_sha256": sha256_file(EXP12_CODE),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "layer_sweep": list(LAYER_SWEEP),
            "kv_sweep": list(KV_SWEEP),
            "phase_a_anchors": list(PHASE_A_ANCHORS),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "offline_only": True,
        }
        out_dir.joinpath("run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def head_dim_of(model_dir):
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    return cfg["hidden_size"] // cfg["num_attention_heads"]


if __name__ == "__main__":
    main()
