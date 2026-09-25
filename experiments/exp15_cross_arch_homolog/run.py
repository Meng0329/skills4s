#!/usr/bin/env python3
"""EXP15 — Cross-Architecture Functional Homolog Replication.

This runner imports the tested EXP14 intervention machinery, removes Qwen-specific
architecture assumptions, and strengthens discovery:

- semantic anchors are chat-template agnostic;
- layer sweep is relative-depth coarse -> local refinement;
- writer selection uses sufficiency + necessity + both labels;
- reader selection uses sufficiency + necessity;
- layer/head/KV counts are read from config.

Important scope: EXP15 is a preregistered test of the Qwen-family *V/KV-mediated*
functional organization in another architecture. A negative result means that
this V-mediated homolog did not replicate; it does not prove that the target
model lacks all procedural-control circuitry.
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
import math
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP14_CODE = REPO_ROOT / "experiments" / "exp14_cross_model_homolog" / "run.py"
EXP12_CODE = REPO_ROOT / "experiments" / "exp12_independent_replication" / "run.py"
DEFAULT_OUT = REPO_ROOT / "outputs" / "exp15_cross_arch_homolog"

SPLIT_SEED = 5515
DISCOVERY_PER_FAMILY = 8
ANCHOR_WIDTH = 6
READER_SET_SIZE = 3
COARSE_FRACTIONS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.96)
REFINE_RADIUS = 2
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

DEFAULT_INVENTORY_ROOTS = (
    REPO_ROOT / "models",
    Path("/data/mzb/ar2_scratch/models"),
)


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


def load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def nested_attr(obj, path):
    cur = obj
    for part in path:
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def resolve_layers(model):
    candidates = (
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("transformer", "h"),
        ("language_model", "model", "layers"),
        ("model", "language_model", "layers"),
    )
    for path in candidates:
        layers = nested_attr(model, path)
        if layers is not None and hasattr(layers, "__len__") and len(layers) > 0:
            return layers, ".".join(path)
    raise RuntimeError(
        "Unsupported decoder stack. Expected one of: "
        + ", ".join(".".join(x) for x in candidates)
    )


def attention_modules(layers):
    sample = layers[0]
    attn = getattr(sample, "self_attn", None)
    if attn is None:
        raise RuntimeError("Unsupported block: missing layer.self_attn")
    if not hasattr(attn, "v_proj") or not hasattr(attn, "o_proj"):
        raise RuntimeError(
            "Unsupported attention module: EXP15 requires self_attn.v_proj and "
            "self_attn.o_proj at consistent hook sites."
        )
    return attn


def config_geometry(config):
    q = int(config.num_attention_heads)
    kv = int(getattr(config, "num_key_value_heads", q) or q)
    hidden = int(config.hidden_size)
    head_dim = int(getattr(config, "head_dim", hidden // q))
    if q % kv != 0:
        raise RuntimeError(f"num_attention_heads={q} not divisible by num_key_value_heads={kv}")
    if hidden != q * head_dim:
        # Some architectures expose a separate head_dim; q_proj may still use q*head_dim.
        # We allow it, but report it explicitly.
        pass
    kind = "MHA" if kv == q else ("MQA" if kv == 1 else "GQA")
    return q, kv, head_dim, kind


def is_qwen2_family(config_dict):
    mt = str(config_dict.get("model_type", "")).lower()
    arch = " ".join(config_dict.get("architectures") or []).lower()
    return "qwen2" in mt or "qwen2" in arch


def inventory_models(roots):
    found = []
    seen = set()
    for root in roots:
        root = Path(root).expanduser()
        if not root.exists():
            continue
        for cfg_path in root.rglob("config.json"):
            model_dir = cfg_path.parent.resolve()
            if model_dir in seen:
                continue
            seen.add(model_dir)
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                q = cfg.get("num_attention_heads")
                kv = cfg.get("num_key_value_heads", q)
                hidden = cfg.get("hidden_size")
                head_dim = cfg.get("head_dim")
                if head_dim is None and q and hidden:
                    head_dim = int(hidden) // int(q)
                if q and kv:
                    attn_kind = "MHA" if int(q) == int(kv) else ("MQA" if int(kv) == 1 else "GQA")
                else:
                    attn_kind = None
                found.append({
                    "model_dir": str(model_dir),
                    "name": model_dir.name,
                    "model_type": cfg.get("model_type"),
                    "architectures": cfg.get("architectures"),
                    "num_hidden_layers": cfg.get("num_hidden_layers"),
                    "hidden_size": hidden,
                    "num_attention_heads": q,
                    "num_key_value_heads": kv,
                    "head_dim": head_dim,
                    "attention_kind": attn_kind,
                    "qwen2_family": is_qwen2_family(cfg),
                    "config_sha256": sha256_file(cfg_path),
                })
            except Exception as e:
                found.append({
                    "model_dir": str(model_dir),
                    "error": f"{type(e).__name__}: {e}",
                })
    return found


def find_segment(text: str, segment: str):
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
        raise RuntimeError(f"Invalid anchor window: {positions}")
    return positions


def generic_compute_schema(exp12, tok, task, entry):
    text = exp12.render_prompt(tok, entry["messages"])
    enc = tok(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = [int(x) for x in enc["input_ids"]]
    offsets = enc["offset_mapping"]
    n = len(ids)

    wording = entry["wording"]
    skill_text = None
    if wording != "direct":
        try:
            skill_text = exp12.SKILL_TEXT[task["family"]][int(entry["label"])][wording]
        except Exception:
            skill_text = None

    system_content = entry["messages"][0]["content"]
    user_content = entry["messages"][-1]["content"]

    semantic_segments = {
        "SKILL_END": skill_text,
        "SYSTEM_END": system_content,
        "ISSUE_END": task["issue"],
        "DETAIL_END": task["detail"],
        "ACTION0_END": task["action0"],
        "ACTION1_END": task["action1"],
        "FINAL_INSTRUCTION_END": "Choose the single next action now.",
        "USER_END": user_content,
    }

    anchors = {}
    for name, segment in semantic_segments.items():
        if segment is None:
            continue
        start, end = find_segment(text, segment)
        toks = overlap_token_indices(offsets, start, end)
        anchors[name] = fixed_end_window(toks[-1], n)

    anchors["GENERATION_BOUNDARY"] = list(range(n - ANCHOR_WIDTH, n))

    required = set(ANCHORS)
    if wording == "direct":
        required.discard("SKILL_END")
    missing = sorted(required - set(anchors))
    if missing:
        raise RuntimeError(f"Missing schema anchors: {missing}")

    return {"text": text, "ids": ids, "anchors": anchors}


def coarse_layers(n_layers):
    valid_max = n_layers - 1  # consumer block L must have a preceding layer L-1
    if valid_max < 2:
        raise RuntimeError(f"Too few decoder layers: {n_layers}")
    vals = []
    for frac in COARSE_FRACTIONS:
        L = int(round(frac * valid_max))
        L = min(max(L, 1), valid_max)
        vals.append(L)
    return tuple(sorted(set(vals)))


def refined_layers(best_L, n_layers):
    valid_max = n_layers - 1
    lo = max(1, best_L - REFINE_RADIUS)
    hi = min(valid_max, best_L + REFINE_RADIUS)
    return tuple(range(lo, hi + 1))


def load_cross_arch_model(model_dir: Path, trust_remote_code=False):
    model_dir = Path(model_dir).resolve()
    cfg = AutoConfig.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    tok = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=trust_remote_code,
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    layers, layer_path = resolve_layers(model)
    attention_modules(layers)
    q, kv, head_dim, kind = config_geometry(cfg)
    return model, tok, layers, cfg, {
        "layer_path": layer_path,
        "num_layers": len(layers),
        "num_attention_heads": q,
        "num_key_value_heads": kv,
        "head_dim": head_dim,
        "attention_kind": kind,
    }


def family_split(exp14, tasks):
    # Keep EXP14's controlled model-comparison split, but record EXP15 seed separately.
    return exp14.family_split(tasks)


def phase_a_scores(
    exp14, model, tok, layers, exp12, disc_tasks, baselines,
    layers_to_test, kv_heads, num_kv_heads, head_dim, out_rows
):
    # GREEN-ZONE engineering optimization (EXP15 run note 1): residual_effect is
    # k-independent (residual donor patch at layers[L-1], no v_heads involved), so
    # it is computed once per (L, task, entry, anchor) and reused across all KV
    # heads. Result-identical (deterministic inference, same inputs -> same float).
    # For 8 KV heads this removes 8x redundant residual forwards in Phase A.
    scores = {}
    for L in layers_to_test:
        task_caches = {}
        for task in disc_tasks:
            task_caches[task["task_id"]] = exp14.build_task_cache(
                model, tok, layers, exp12, task,
                exp12.make_skill_entries(task), L, num_kv_heads, head_dim,
            )

        res_refs = {}
        for task in disc_tasks:
            entries = exp12.make_skill_entries(task)
            maps = exp12.donor_maps_skill(entries)
            for i, entry in enumerate(entries):
                donor_i = maps[i]["opposite_same_wording"]
                donor = entries[donor_i]
                rec_cache = task_caches[task["task_id"]][i]
                don_cache = task_caches[task["task_id"]][donor_i]
                for anchor in PHASE_A_ANCHORS:
                    rec_pos = exp14.anchor_positions(rec_cache, anchor)
                    don_pos = exp14.anchor_positions(don_cache, anchor)
                    res_ref = exp14.residual_effect(
                        model, tok, layers, exp12,
                        entry, rec_cache, donor, don_cache,
                        baselines[(task["task_id"], i)],
                        L, rec_pos, don_pos,
                        num_kv_heads, head_dim,
                    )
                    res_refs[(task["task_id"], i, anchor)] = (res_ref, rec_pos, don_pos)

        for k in kv_heads:
            buckets = {
                "suff0": [], "suff1": [],
                "nec0": [], "nec1": [],
            }
            for task in disc_tasks:
                entries = exp12.make_skill_entries(task)
                maps = exp12.donor_maps_skill(entries)
                for i, entry in enumerate(entries):
                    donor_i = maps[i]["opposite_same_wording"]
                    donor = entries[donor_i]
                    rec_cache = task_caches[task["task_id"]][i]
                    don_cache = task_caches[task["task_id"]][donor_i]
                    for anchor in PHASE_A_ANCHORS:
                        res_ref, rec_pos, don_pos = res_refs[(task["task_id"], i, anchor)]
                        suff, nec = exp14.v_suff_nec(
                            model, tok, layers, exp12,
                            entry, rec_cache, donor, don_cache,
                            baselines[(task["task_id"], i)],
                            res_ref, L, rec_pos, don_pos,
                            num_kv_heads, head_dim, v_heads=(k,),
                        )
                        label = int(entry["label"])
                        buckets[f"suff{label}"].append(suff)
                        buckets[f"nec{label}"].append(nec)
                        out_rows.append({
                            "task_id": task["task_id"],
                            "family": task["family"],
                            "wording": entry["wording"],
                            "label": label,
                            "anchor": anchor,
                            "L": L,
                            "k": k,
                            "suff": suff,
                            "nec": nec,
                        })
            means = {name: float(np.mean(vals)) for name, vals in buckets.items()}
            score = min(means.values())
            scores[(L, k)] = {"score": score, **means}
            print(
                f"[phaseA] L={L} k={k} score={score:.6f} "
                f"s0={means['suff0']:.5f} s1={means['suff1']:.5f} "
                f"n0={means['nec0']:.5f} n1={means['nec1']:.5f}"
            )
        del task_caches, res_refs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return scores


def run_discovery(
    exp14, model, tok, layers, exp12, tasks, split,
    num_kv_heads, num_q_heads, head_dim, out_dir, seed
):
    disc_ids = exp14.split_ids(split, "discovery")
    disc_tasks = [t for t in tasks if t["task_id"] in disc_ids]

    baselines = {}
    for task in disc_tasks:
        entries = exp12.make_skill_entries(task)
        for i, entry in enumerate(entries):
            baselines[(task["task_id"], i)] = exp14.score_margin(
                model, tok, layers, exp12, entry, None,
                None, num_kv_heads, head_dim,
            )

    phase_a_rows = []
    kv_heads = tuple(range(num_kv_heads))
    coarse = coarse_layers(len(layers))
    coarse_scores = phase_a_scores(
        exp14, model, tok, layers, exp12, disc_tasks, baselines,
        coarse, kv_heads, num_kv_heads, head_dim, phase_a_rows
    )
    coarse_best = max(coarse_scores, key=lambda x: coarse_scores[x]["score"])
    refine = refined_layers(coarse_best[0], len(layers))

    refine_scores = phase_a_scores(
        exp14, model, tok, layers, exp12, disc_tasks, baselines,
        refine, kv_heads, num_kv_heads, head_dim, phase_a_rows
    )
    best = max(refine_scores, key=lambda x: refine_scores[x]["score"])
    L_star, k_star = map(int, best)

    pd.DataFrame(phase_a_rows).to_csv(
        out_dir / "discovery_phaseA.csv", index=False
    )

    # Phase B: semantic writer anchors.
    rows = []
    task_caches = {}
    for task in disc_tasks:
        task_caches[task["task_id"]] = exp14.build_task_cache(
            model, tok, layers, exp12, task,
            exp12.make_skill_entries(task), L_star, num_kv_heads, head_dim,
        )

    strata = {}
    for task in disc_tasks:
        entries = exp12.make_skill_entries(task)
        maps = exp12.donor_maps_skill(entries)
        for i, entry in enumerate(entries):
            strata.setdefault(
                (task["family"], entry["wording"]), []
            ).append((task, entries, maps, i, entry))

    selected = {}
    for (family, wording), members in strata.items():
        anchor_metrics = {}
        for anchor in ANCHORS:
            buckets = {"suff0": [], "suff1": [], "nec0": [], "nec1": []}
            for task, entries, maps, i, entry in members:
                donor_i = maps[i]["opposite_same_wording"]
                donor = entries[donor_i]
                rec_cache = task_caches[task["task_id"]][i]
                don_cache = task_caches[task["task_id"]][donor_i]
                rec_pos = exp14.anchor_positions(rec_cache, anchor)
                don_pos = exp14.anchor_positions(don_cache, anchor)
                res_ref = exp14.residual_effect(
                    model, tok, layers, exp12,
                    entry, rec_cache, donor, don_cache,
                    baselines[(task["task_id"], i)],
                    L_star, rec_pos, don_pos,
                    num_kv_heads, head_dim,
                )
                suff, nec = exp14.v_suff_nec(
                    model, tok, layers, exp12,
                    entry, rec_cache, donor, don_cache,
                    baselines[(task["task_id"], i)],
                    res_ref, L_star, rec_pos, don_pos,
                    num_kv_heads, head_dim, v_heads=(k_star,),
                )
                label = int(entry["label"])
                buckets[f"suff{label}"].append(suff)
                buckets[f"nec{label}"].append(nec)
                exp14.add_result(
                    rows, phase="discovery", task=task, entry=entry, donor=donor,
                    relation="opposite_same_wording", anchor=anchor,
                    metric="V_sufficiency", value=suff, L=L_star, k=k_star,
                )
                exp14.add_result(
                    rows, phase="discovery", task=task, entry=entry, donor=donor,
                    relation="opposite_same_wording", anchor=anchor,
                    metric="V_necessity_loss", value=nec, L=L_star, k=k_star,
                )
            means = {name: float(np.mean(vals)) for name, vals in buckets.items()}
            anchor_metrics[anchor] = {
                **means,
                "bidirectional_score": min(means.values()),
            }

        order = sorted(
            anchor_metrics,
            key=lambda a: (-anchor_metrics[a]["bidirectional_score"], ANCHORS.index(a))
        )
        sel, runner = order[0], order[1]
        selected[f"{family}::{wording}"] = {
            "family": family,
            "wording": wording,
            "selected_anchor": sel,
            "runner_up_anchor": runner,
            "selected_score": anchor_metrics[sel]["bidirectional_score"],
            "runner_up_score": anchor_metrics[runner]["bidirectional_score"],
            "anchors": anchor_metrics,
        }
        print(
            f"[phaseB] {family}::{wording} -> {sel} "
            f"score={anchor_metrics[sel]['bidirectional_score']:.6f}"
        )

    # Phase C: reader-head discovery with sufficiency + necessity.
    head_rows = []
    per_head_suff = {q: [] for q in range(num_q_heads)}
    per_head_nec = {q: [] for q in range(num_q_heads)}

    for task in disc_tasks:
        entries = exp12.make_skill_entries(task)
        maps = exp12.donor_maps_skill(entries)
        for i, entry in enumerate(entries):
            key = f"{task['family']}::{entry['wording']}"
            anchor = selected[key]["selected_anchor"]
            donor_i = maps[i]["opposite_same_wording"]
            donor = entries[donor_i]
            rec_cache = task_caches[task["task_id"]][i]
            don_cache = task_caches[task["task_id"]][donor_i]
            rec_pos = exp14.anchor_positions(rec_cache, anchor)
            don_pos = exp14.anchor_positions(don_cache, anchor)
            rp = exp14.reader_path_effects(
                model, tok, layers, exp12,
                entry, rec_cache, donor, don_cache,
                rec_pos, don_pos, baselines[(task["task_id"], i)],
                L_star, (k_star,), num_kv_heads, num_q_heads, head_dim,
                include_nec=True,
            )
            for q in range(num_q_heads):
                suff, nec = rp["groups"]((q,))
                per_head_suff[q].append(suff)
                per_head_nec[q].append(nec)
                head_rows.append({
                    "task_id": task["task_id"],
                    "family": task["family"],
                    "wording": entry["wording"],
                    "label": int(entry["label"]),
                    "head": q,
                    "suff": suff,
                    "nec": nec,
                })

    head_summary = {}
    for q in range(num_q_heads):
        ms = float(np.mean(per_head_suff[q]))
        mn = float(np.mean(per_head_nec[q]))
        head_summary[q] = {
            "mean_suff": ms,
            "mean_nec": mn,
            "positive_score": min(ms, mn),
            "negative_score": max(ms, mn),
        }

    pos_rank = sorted(head_summary, key=lambda q: -head_summary[q]["positive_score"])
    pos_set = tuple(sorted(pos_rank[:min(READER_SET_SIZE, num_q_heads)]))
    neg_candidates = [q for q in head_summary if q not in pos_set]
    neg_rank = sorted(neg_candidates, key=lambda q: head_summary[q]["negative_score"])
    neg_set = tuple(sorted(neg_rank[:min(READER_SET_SIZE, len(neg_rank))]))

    pd.DataFrame(head_rows).to_csv(
        out_dir / "discovery_reader_heads.csv", index=False
    )
    pd.DataFrame(rows).to_csv(
        out_dir / "discovery_results.csv", index=False
    )

    result = {
        "L_star": L_star,
        "k_star": k_star,
        "coarse_layers": list(coarse),
        "coarse_best": {
            "L": int(coarse_best[0]),
            "k": int(coarse_best[1]),
            **coarse_scores[coarse_best],
        },
        "refine_layers": list(refine),
        "phase_a_best": {
            "L": L_star,
            "k": k_star,
            **refine_scores[best],
        },
        "reader_pos_set": list(pos_set),
        "reader_neg_set": list(neg_set),
        "head_summary": {str(k): v for k, v in head_summary.items()},
        "strata": selected,
    }
    (out_dir / "selected_homolog.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(
        {"L_star": L_star, "k_star": k_star, "pos": list(pos_set), "neg": list(neg_set)},
        ensure_ascii=False
    ))
    return result


def task_metric(df, metric):
    d = df[df["metric"] == metric]
    return d.groupby("task_id")["value"].mean().sort_index()


def bootstrap_ci(exp14, vals, seed):
    vals = np.asarray(vals, dtype=np.float64)
    return exp14.bootstrap_ci(vals, seed)


def make_verdict(exp14, conf_df, summary_df, selected, seed):
    def metric_row(metric):
        d = summary_df[summary_df["metric"] == metric]
        return None if d.empty else d.iloc[0]

    required_positive = (
        "selected_V_sufficiency",
        "selected_V_necessity_loss",
        "cross_selected_V_sufficiency",
        "reader_target_pos_sufficiency",
        "reader_target_pos_necessity",
    )
    endpoint_flags = {}
    for m in required_positive:
        r = metric_row(m)
        endpoint_flags[m] = bool(r is not None and float(r["ci_low"]) > 0)

    pos_s = task_metric(conf_df, "reader_target_pos_sufficiency")
    neg_s = task_metric(conf_df, "reader_negative_neg_sufficiency")
    pos_n = task_metric(conf_df, "reader_target_pos_necessity")
    neg_n = task_metric(conf_df, "reader_negative_neg_necessity")

    common_s = pos_s.index.intersection(neg_s.index)
    common_n = pos_n.index.intersection(neg_n.index)
    c_s = exp14.bootstrap_ci(
        (pos_s.loc[common_s] - neg_s.loc[common_s]).values, seed + 101
    )
    c_n = exp14.bootstrap_ci(
        (pos_n.loc[common_n] - neg_n.loc[common_n]).values, seed + 102
    )

    contrast_flags = {
        "target_minus_negative_reader_sufficiency": c_s[1] > 0,
        "target_minus_negative_reader_necessity": c_n[1] > 0,
    }

    same = metric_row("same_state_selected_V_control")
    leak = metric_row("reader_leakage_ratio")
    runner_s = metric_row("runner_up_V_sufficiency")
    runner_n = metric_row("runner_up_V_necessity_loss")
    selected_s = metric_row("selected_V_sufficiency")
    selected_n = metric_row("selected_V_necessity_loss")

    fidelity = {
        "same_state_control": None if same is None else {
            "mean": float(same["mean"]),
            "ci_low": float(same["ci_low"]),
            "ci_high": float(same["ci_high"]),
        },
        "reader_leakage_ratio": None if leak is None else float(leak["mean"]),
        "selected_minus_runner_up_sufficiency": (
            None if selected_s is None or runner_s is None
            else float(selected_s["mean"] - runner_s["mean"])
        ),
        "selected_minus_runner_up_necessity": (
            None if selected_n is None or runner_n is None
            else float(selected_n["mean"] - runner_n["mean"])
        ),
    }

    strong = all(endpoint_flags.values()) and all(contrast_flags.values())
    if strong:
        # Fidelity deviations do not erase causal homolog evidence; they downgrade wording.
        contaminated = (
            same is not None
            and not (float(same["ci_low"]) <= 0 <= float(same["ci_high"]))
        )
        high_leak = leak is not None and float(leak["mean"]) > 0.10
        verdict = "PARTIAL / ALGORITHMIC HOMOLOG" if (contaminated or high_leak) else "STRONG HOMOLOG"
    else:
        # If some major levels survive, retain partial rather than forcing binary failure.
        passed = sum(endpoint_flags.values()) + sum(contrast_flags.values())
        verdict = "PARTIAL / ALGORITHMIC HOMOLOG" if passed >= 3 else "NO V-MEDIATED HOMOLOG"

    return {
        "verdict": verdict,
        "endpoint_flags": endpoint_flags,
        "contrast_flags": contrast_flags,
        "reader_contrasts": {
            "sufficiency": {"mean": c_s[0], "ci_low": c_s[1], "ci_high": c_s[2]},
            "necessity": {"mean": c_n[0], "ci_low": c_n[1], "ci_high": c_n[2]},
        },
        "fidelity": fidelity,
        "selected": {
            "L_star": selected["L_star"],
            "k_star": selected["k_star"],
            "reader_pos_set": selected["reader_pos_set"],
            "reader_neg_set": selected["reader_neg_set"],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument(
        "--phase",
        choices=("inventory", "gate", "discovery", "confirmation", "all"),
        default="inventory",
    )
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    ap.add_argument(
        "--inventory_root",
        action="append",
        default=None,
        help="May be repeated. Defaults to repo/models and /data/mzb/ar2_scratch/models.",
    )
    ap.add_argument("--allow_same_family", action="store_true")
    ap.add_argument("--trust_remote_code", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    roots = [Path(x) for x in args.inventory_root] if args.inventory_root else list(DEFAULT_INVENTORY_ROOTS)
    inventory = inventory_models(roots)
    (out_dir / "model_inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.phase == "inventory":
        print(json.dumps(inventory, ensure_ascii=False, indent=2))
        return

    if not args.model_dir:
        raise SystemExit("--model_dir is required for gate/discovery/confirmation/all")

    model_dir = Path(args.model_dir).resolve()
    cfg_dict = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if is_qwen2_family(cfg_dict) and not args.allow_same_family:
        raise RuntimeError(
            "EXP15 requires a genuinely cross-architecture model. "
            "This config is Qwen2-family. Use EXP14 for same-family replication."
        )

    exp14 = load_module(EXP14_CODE, "skills4s_exp14")
    exp12 = exp14.load_exp12()

    # Replace Qwen-specialized schema parsing and hard-coded head range.
    exp14.compute_schema = generic_compute_schema

    model, tok, layers, cfg, arch = load_cross_arch_model(
        model_dir, trust_remote_code=args.trust_remote_code
    )
    num_q_heads = arch["num_attention_heads"]
    num_kv_heads = arch["num_key_value_heads"]
    head_dim = arch["head_dim"]

    exp14.non_reader_heads = lambda pos_set=(), neg_set=(): tuple(
        q for q in range(num_q_heads)
        if q not in (set(pos_set) | set(neg_set))
    )

    tasks = exp12.make_tasks()
    if len(tasks) != 64:
        raise RuntimeError(f"Expected 64 controlled tasks, got {len(tasks)}")
    split = family_split(exp14, tasks)
    (out_dir / "split.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    arch_sanity = {
        **arch,
        "model_type": getattr(cfg, "model_type", None),
        "architectures": getattr(cfg, "architectures", None),
        "qwen2_family": is_qwen2_family(cfg_dict),
        "self_attn_has_v_proj": True,
        "self_attn_has_o_proj": True,
        "coarse_layers": list(coarse_layers(len(layers))),
    }
    (out_dir / "architecture_sanity.json").write_text(
        json.dumps(arch_sanity, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 92)
    print("EXP15: Cross-Architecture Functional Homolog Replication")
    print("Model:", model_dir)
    print("Architecture:", arch_sanity)
    print("Phase:", args.phase)
    print("=" * 92)

    selected_path = out_dir / "selected_homolog.json"
    if args.phase == "confirmation" and not selected_path.is_file():
        raise FileNotFoundError(f"Confirmation requires frozen {selected_path}")

    if args.phase in ("gate", "all"):
        gate, _ = exp14.run_gate(
            model, tok, layers, exp12, tasks,
            num_kv_heads, head_dim, out_dir, args.seed,
        )
        if not gate["gate_pass"]:
            print("BEHAVIORAL GATE FAILED — stop before mechanistic discovery.")
            if args.phase == "gate":
                return
            raise SystemExit(2)

    if args.phase in ("discovery", "all"):
        selected = run_discovery(
            exp14, model, tok, layers, exp12, tasks, split,
            num_kv_heads, num_q_heads, head_dim, out_dir, args.seed,
        )

    if args.phase in ("confirmation", "all"):
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        conf = exp14.run_confirmation(
            model, tok, layers, exp12, tasks, split, selected,
            num_kv_heads, num_q_heads, head_dim, out_dir, args.seed,
        )
        summary_df = exp14.summarize(conf, "confirmation", args.seed + 5)
        summary_df.to_csv(out_dir / "confirmation_summary.csv", index=False)

        verdict = make_verdict(exp14, conf, summary_df, selected, args.seed)
        summary = {
            "experiment": "EXP15_cross_arch_homolog",
            "model_dir": str(model_dir),
            "architecture": arch_sanity,
            "verdict": verdict,
            "claim_boundary": (
                "A negative result rules out replication of the Qwen-family "
                "V/KV-mediated functional organization in this model; it does "
                "not prove absence of all procedural-control mechanisms."
            ),
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        manifest = {
            "experiment": "EXP15_cross_arch_homolog",
            "git_commit": git_commit(),
            "seed": args.seed,
            "model_dir": str(model_dir),
            "model_config_sha256": sha256_file(model_dir / "config.json"),
            "exp14_code_sha256": sha256_file(EXP14_CODE),
            "exp12_code_sha256": sha256_file(EXP12_CODE),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "architecture": arch_sanity,
            "coarse_fractions": list(COARSE_FRACTIONS),
            "refine_radius": REFINE_RADIUS,
            "phase_a_anchors": list(PHASE_A_ANCHORS),
            "anchors": list(ANCHORS),
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
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
