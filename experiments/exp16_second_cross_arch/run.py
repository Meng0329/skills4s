#!/usr/bin/env python3
"""EXP16 — Second Cross-Architecture Homolog + Anchor-Conditioned Interface Discovery."""
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import argparse
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

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP15_CODE = REPO_ROOT / "experiments" / "exp15_cross_arch_homolog" / "run.py"
DEFAULT_OUT = REPO_ROOT / "outputs" / "exp16_second_cross_arch"
SEED = 5616
TOP_A1 = 2
TOP_KV = 4
REFINE_RADIUS = 2
MAX_REGISTER = 4
POS_ABS_THRESHOLD = 0.002
NEG_ABS_THRESHOLD = -0.002
REL_THRESHOLD = 0.05
ANCHORS = (
    "SKILL_END", "SYSTEM_END", "ISSUE_END", "DETAIL_END",
    "ACTION0_END", "ACTION1_END", "FINAL_INSTRUCTION_END",
    "USER_END", "GENERATION_BOUNDARY",
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def prior_family(config):
    mt = str(getattr(config, "model_type", "")).lower()
    arch = " ".join(getattr(config, "architectures", None) or []).lower()
    return any(x in mt or x in arch for x in ("qwen2", "mistral"))


def baseline_map(exp14, model, tok, layers, exp12, tasks, num_kv_heads, head_dim):
    out = {}
    for task in tasks:
        entries = exp12.make_skill_entries(task)
        for i, entry in enumerate(entries):
            out[(task["task_id"], i)] = exp14.score_margin(
                model, tok, layers, exp12, entry, None, None,
                num_kv_heads, head_dim,
            )
    return out


def task_cache(exp14, model, tok, layers, exp12, tasks, L, num_kv_heads, head_dim):
    return {
        task["task_id"]: exp14.build_task_cache(
            model, tok, layers, exp12, task, exp12.make_skill_entries(task),
            L, num_kv_heads, head_dim,
        )
        for task in tasks
    }


def aggregate_four(buckets):
    means = {k: float(np.mean(v)) for k, v in buckets.items()}
    return min(means.values()), means


def eval_residual_layer(exp14, model, tok, layers, exp12, tasks, baselines,
                        L, num_kv_heads, head_dim, rows):
    caches = task_cache(exp14, model, tok, layers, exp12, tasks, L, num_kv_heads, head_dim)
    buckets = {a: {"l0": [], "l1": []} for a in ANCHORS}
    for task in tasks:
        entries = exp12.make_skill_entries(task)
        maps = exp12.donor_maps_skill(entries)
        for i, entry in enumerate(entries):
            donor_i = maps[i]["opposite_same_wording"]
            donor = entries[donor_i]
            rec = caches[task["task_id"]][i]
            don = caches[task["task_id"]][donor_i]
            for anchor in ANCHORS:
                rp = exp14.anchor_positions(rec, anchor)
                dp = exp14.anchor_positions(don, anchor)
                eff = exp14.residual_effect(
                    model, tok, layers, exp12, entry, rec, donor, don,
                    baselines[(task["task_id"], i)], L, rp, dp,
                    num_kv_heads, head_dim,
                )
                label = int(entry["label"])
                buckets[anchor][f"l{label}"].append(eff)
                rows.append({
                    "task_id": task["task_id"], "family": task["family"],
                    "wording": entry["wording"], "label": label,
                    "L": L, "anchor": anchor, "residual_effect": eff,
                })
    scores = {}
    for anchor in ANCHORS:
        m0 = float(np.mean(buckets[anchor]["l0"]))
        m1 = float(np.mean(buckets[anchor]["l1"]))
        scores[(L, anchor)] = {"score": min(m0, m1), "label0": m0, "label1": m1}
    del caches
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores


def eval_v_config(exp14, model, tok, layers, exp12, tasks, baselines,
                  L, k, num_kv_heads, head_dim, fixed_anchor=None,
                  strata_anchors=None, rows=None, stage=""):
    caches = task_cache(exp14, model, tok, layers, exp12, tasks, L, num_kv_heads, head_dim)
    buckets = {"suff0": [], "suff1": [], "nec0": [], "nec1": []}
    for task in tasks:
        entries = exp12.make_skill_entries(task)
        maps = exp12.donor_maps_skill(entries)
        for i, entry in enumerate(entries):
            donor_i = maps[i]["opposite_same_wording"]
            donor = entries[donor_i]
            rec = caches[task["task_id"]][i]
            don = caches[task["task_id"]][donor_i]
            if strata_anchors is not None:
                key = f"{task['family']}::{entry['wording']}"
                anchor = strata_anchors[key]["selected_anchor"]
            else:
                anchor = fixed_anchor
            rp = exp14.anchor_positions(rec, anchor)
            dp = exp14.anchor_positions(don, anchor)
            res_ref = exp14.residual_effect(
                model, tok, layers, exp12, entry, rec, donor, don,
                baselines[(task["task_id"], i)], L, rp, dp,
                num_kv_heads, head_dim,
            )
            suff, nec = exp14.v_suff_nec(
                model, tok, layers, exp12, entry, rec, donor, don,
                baselines[(task["task_id"], i)], res_ref, L, rp, dp,
                num_kv_heads, head_dim, v_heads=(k,),
            )
            label = int(entry["label"])
            buckets[f"suff{label}"].append(suff)
            buckets[f"nec{label}"].append(nec)
            if rows is not None:
                rows.append({
                    "stage": stage, "task_id": task["task_id"],
                    "family": task["family"], "wording": entry["wording"],
                    "label": label, "L": L, "k": k, "anchor": anchor,
                    "suff": suff, "nec": nec,
                })
    score, means = aggregate_four(buckets)
    del caches
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"score": score, **means}


def select_contextual_anchors(exp14, model, tok, layers, exp12, tasks, baselines,
                              L, k, num_kv_heads, head_dim, out_rows):
    caches = task_cache(exp14, model, tok, layers, exp12, tasks, L, num_kv_heads, head_dim)
    strata = {}
    members = {}
    for task in tasks:
        entries = exp12.make_skill_entries(task)
        maps = exp12.donor_maps_skill(entries)
        for i, entry in enumerate(entries):
            members.setdefault((task["family"], entry["wording"]), []).append(
                (task, entries, maps, i, entry)
            )
    for (family, wording), group in members.items():
        anchor_stats = {}
        for anchor in ANCHORS:
            buckets = {"suff0": [], "suff1": [], "nec0": [], "nec1": []}
            for task, entries, maps, i, entry in group:
                donor_i = maps[i]["opposite_same_wording"]
                donor = entries[donor_i]
                rec = caches[task["task_id"]][i]
                don = caches[task["task_id"]][donor_i]
                rp = exp14.anchor_positions(rec, anchor)
                dp = exp14.anchor_positions(don, anchor)
                res_ref = exp14.residual_effect(
                    model, tok, layers, exp12, entry, rec, donor, don,
                    baselines[(task["task_id"], i)], L, rp, dp,
                    num_kv_heads, head_dim,
                )
                suff, nec = exp14.v_suff_nec(
                    model, tok, layers, exp12, entry, rec, donor, don,
                    baselines[(task["task_id"], i)], res_ref, L, rp, dp,
                    num_kv_heads, head_dim, v_heads=(k,),
                )
                label = int(entry["label"])
                buckets[f"suff{label}"].append(suff)
                buckets[f"nec{label}"].append(nec)
                out_rows.append({
                    "task_id": task["task_id"], "family": family,
                    "wording": wording, "label": label, "anchor": anchor,
                    "L": L, "k": k, "suff": suff, "nec": nec,
                })
            score, means = aggregate_four(buckets)
            anchor_stats[anchor] = {"bidirectional_score": score, **means}
        ordered = sorted(anchor_stats, key=lambda a: (-anchor_stats[a]["bidirectional_score"], ANCHORS.index(a)))
        strata[f"{family}::{wording}"] = {
            "family": family, "wording": wording,
            "selected_anchor": ordered[0], "runner_up_anchor": ordered[1],
            "selected_score": anchor_stats[ordered[0]]["bidirectional_score"],
            "runner_up_score": anchor_stats[ordered[1]]["bidirectional_score"],
            "anchors": anchor_stats,
        }
    del caches
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return strata


def discover_readers(exp14, model, tok, layers, exp12, tasks, baselines,
                     L, k, strata, num_kv_heads, num_q_heads, head_dim, rows):
    caches = task_cache(exp14, model, tok, layers, exp12, tasks, L, num_kv_heads, head_dim)
    suffs = {q: [] for q in range(num_q_heads)}
    necs = {q: [] for q in range(num_q_heads)}
    for task in tasks:
        entries = exp12.make_skill_entries(task)
        maps = exp12.donor_maps_skill(entries)
        for i, entry in enumerate(entries):
            key = f"{task['family']}::{entry['wording']}"
            anchor = strata[key]["selected_anchor"]
            donor_i = maps[i]["opposite_same_wording"]
            donor = entries[donor_i]
            rec = caches[task["task_id"]][i]
            don = caches[task["task_id"]][donor_i]
            rp = exp14.anchor_positions(rec, anchor)
            dp = exp14.anchor_positions(don, anchor)
            path = exp14.reader_path_effects(
                model, tok, layers, exp12, entry, rec, donor, don,
                rp, dp, baselines[(task["task_id"], i)], L, (k,),
                num_kv_heads, num_q_heads, head_dim,
                leak_register_heads=tuple(range(num_q_heads)),
            )
            for q in range(num_q_heads):
                s, n = path["groups"]((q,))
                suffs[q].append(s); necs[q].append(n)
                rows.append({
                    "task_id": task["task_id"], "family": task["family"],
                    "wording": entry["wording"], "label": int(entry["label"]),
                    "head": q, "suff": s, "nec": n,
                })
    hs = {}
    for q in range(num_q_heads):
        ms, mn = float(np.mean(suffs[q])), float(np.mean(necs[q]))
        hs[q] = {
            "mean_suff": ms, "mean_nec": mn,
            "positive_score": min(ms, mn),
            "inhibitory_score": max(ms, mn),
        }
    best_pos = max(v["positive_score"] for v in hs.values())
    pos_thr = max(POS_ABS_THRESHOLD, REL_THRESHOLD * best_pos)
    pos = [q for q, v in hs.items() if v["positive_score"] >= pos_thr and v["positive_score"] > 0]
    pos = sorted(pos, key=lambda q: -hs[q]["positive_score"])[:MAX_REGISTER]

    remaining = [q for q in hs if q not in pos]
    best_neg = min((hs[q]["inhibitory_score"] for q in remaining), default=0.0)
    neg_thr = min(NEG_ABS_THRESHOLD, REL_THRESHOLD * best_neg)
    neg = [q for q in remaining if hs[q]["inhibitory_score"] <= neg_thr and hs[q]["inhibitory_score"] < 0]
    neg = sorted(neg, key=lambda q: hs[q]["inhibitory_score"])[:MAX_REGISTER]

    del caches
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return tuple(sorted(pos)), tuple(sorted(neg)), hs, pos_thr, neg_thr


def run_discovery(exp15, exp14, model, tok, layers, exp12, tasks, split,
                  num_kv_heads, num_q_heads, head_dim, out_dir):
    disc_ids = exp14.split_ids(split, "discovery")
    disc = [t for t in tasks if t["task_id"] in disc_ids]
    baselines = baseline_map(exp14, model, tok, layers, exp12, disc, num_kv_heads, head_dim)

    # A1
    a1_rows, a1_scores = [], {}
    for L in exp15.coarse_layers(len(layers)):
        a1_scores.update(eval_residual_layer(
            exp14, model, tok, layers, exp12, disc, baselines,
            L, num_kv_heads, head_dim, a1_rows,
        ))
        print(f"[A1] L={L} done")
    top_a1 = sorted(a1_scores, key=lambda x: -a1_scores[x]["score"])[:TOP_A1]
    pd.DataFrame(a1_rows).to_csv(out_dir / "discovery_A1_residual.csv", index=False)

    # A2: all KV at candidate L, top4, then local refine
    a2_rows, a2_all = [], {}
    for L0, anchor in top_a1:
        exact = {}
        for k in range(num_kv_heads):
            exact[k] = eval_v_config(
                exp14, model, tok, layers, exp12, disc, baselines,
                L0, k, num_kv_heads, head_dim, fixed_anchor=anchor,
                rows=a2_rows, stage="A2_prefilter",
            )
        topk = sorted(exact, key=lambda k: -exact[k]["score"])[:min(TOP_KV, num_kv_heads)]
        for L in exp15.refined_layers(L0, len(layers)):
            for k in topk:
                stat = eval_v_config(
                    exp14, model, tok, layers, exp12, disc, baselines,
                    L, k, num_kv_heads, head_dim, fixed_anchor=anchor,
                    rows=a2_rows, stage="A2_refine",
                )
                a2_all[(L, k, anchor)] = stat
    provisional = max(a2_all, key=lambda x: a2_all[x]["score"])
    pL, pk, panchor = provisional
    pd.DataFrame(a2_rows).to_csv(out_dir / "discovery_A2_interface.csv", index=False)

    # B contextual anchors at provisional interface
    b_rows = []
    strata = select_contextual_anchors(
        exp14, model, tok, layers, exp12, disc, baselines,
        pL, pk, num_kv_heads, head_dim, b_rows,
    )
    pd.DataFrame(b_rows).to_csv(out_dir / "discovery_B_schema.csv", index=False)

    # C final interface using frozen selected anchors
    c_rows, c_pref = [], {}
    for k in range(num_kv_heads):
        c_pref[k] = eval_v_config(
            exp14, model, tok, layers, exp12, disc, baselines,
            pL, k, num_kv_heads, head_dim, strata_anchors=strata,
            rows=c_rows, stage="C_prefilter",
        )
    topk_c = sorted(c_pref, key=lambda k: -c_pref[k]["score"])[:min(TOP_KV, num_kv_heads)]
    c_all = {}
    for L in exp15.refined_layers(pL, len(layers)):
        for k in topk_c:
            c_all[(L, k)] = eval_v_config(
                exp14, model, tok, layers, exp12, disc, baselines,
                L, k, num_kv_heads, head_dim, strata_anchors=strata,
                rows=c_rows, stage="C_refine",
            )
    final_L, final_k = max(c_all, key=lambda x: c_all[x]["score"])
    pd.DataFrame(c_rows).to_csv(out_dir / "discovery_C_final_interface.csv", index=False)

    # D readers at final interface; anchors remain frozen from B
    d_rows = []
    pos, neg, head_summary, pos_thr, neg_thr = discover_readers(
        exp14, model, tok, layers, exp12, disc, baselines,
        final_L, final_k, strata, num_kv_heads, num_q_heads, head_dim, d_rows,
    )
    pd.DataFrame(d_rows).to_csv(out_dir / "discovery_D_readers.csv", index=False)

    selected = {
        "L_star": int(final_L), "k_star": int(final_k),
        "reader_pos_set": list(pos), "reader_neg_set": list(neg),
        "reader_positive_threshold": pos_thr,
        "reader_inhibitory_threshold": neg_thr,
        "head_summary": {str(q): v for q, v in head_summary.items()},
        "strata": strata,
        "A1_top_candidates": [
            {"L": int(L), "anchor": a, **a1_scores[(L, a)]} for L, a in top_a1
        ],
        "A2_provisional": {"L": int(pL), "k": int(pk), "anchor": panchor, **a2_all[provisional]},
        "C_final": {"L": int(final_L), "k": int(final_k), **c_all[(final_L, final_k)]},
    }
    (out_dir / "selected_homolog.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "A1": selected["A1_top_candidates"],
        "provisional": selected["A2_provisional"],
        "final": selected["C_final"],
        "positive_readers": list(pos), "inhibitory_readers": list(neg),
    }, ensure_ascii=False, indent=2))
    return selected


def safe_group(path, readers):
    if not readers:
        return 0.0, 0.0
    return path["groups"](tuple(readers))


def run_confirmation_fixed(exp14, model, tok, layers, exp12, tasks, split, selected,
                           num_kv_heads, num_q_heads, head_dim, out_dir):
    L, k = int(selected["L_star"]), int(selected["k_star"])
    pos, neg = tuple(selected["reader_pos_set"]), tuple(selected["reader_neg_set"])
    register = set(pos) | set(neg)
    non_set = tuple(q for q in range(num_q_heads) if q not in register)
    all_set = tuple(range(num_q_heads))
    conf_ids = exp14.split_ids(split, "confirmation")
    conf = [t for t in tasks if t["task_id"] in conf_ids]
    rows = []

    for n, task in enumerate(conf, 1):
        entries = exp12.make_skill_entries(task)
        maps = exp12.donor_maps_skill(entries)
        cache = exp14.build_task_cache(model, tok, layers, exp12, task, entries, L, num_kv_heads, head_dim)
        bases = exp14.baseline_margins(model, tok, layers, exp12, entries, cache, L, num_kv_heads, head_dim)
        for i, entry in enumerate(entries):
            key = f"{task['family']}::{entry['wording']}"
            anchor = selected["strata"][key]["selected_anchor"]
            runner = selected["strata"][key]["runner_up_anchor"]
            donor_i = maps[i]["opposite_same_wording"]
            cross_i = maps[i]["opposite_cross_wording"]
            same_i = maps[i]["same_state_cross_wording"]
            donor, cross_donor, same_donor = entries[donor_i], entries[cross_i], entries[same_i]
            rec, don, cross, same = cache[i], cache[donor_i], cache[cross_i], cache[same_i]
            base = bases[i]

            rp = exp14.anchor_positions(rec, anchor); dp = exp14.anchor_positions(don, anchor)
            res_ref = exp14.residual_effect(model, tok, layers, exp12, entry, rec, donor, don,
                                            base, L, rp, dp, num_kv_heads, head_dim)
            suff, nec = exp14.v_suff_nec(model, tok, layers, exp12, entry, rec, donor, don,
                                         base, res_ref, L, rp, dp, num_kv_heads, head_dim,
                                         v_heads=(k,))

            cross_anchor = selected["strata"][f"{task['family']}::{cross_donor['wording']}"]["selected_anchor"]
            cp = exp14.anchor_positions(cross, cross_anchor)
            cross_suff, _ = exp14.v_suff_nec(model, tok, layers, exp12, entry, rec, cross_donor, cross,
                                              base, None, L, rp, cp, num_kv_heads, head_dim, v_heads=(k,))
            same_anchor = selected["strata"][f"{task['family']}::{same_donor['wording']}"]["selected_anchor"]
            sp = exp14.anchor_positions(same, same_anchor)
            same_ctl, _ = exp14.v_suff_nec(model, tok, layers, exp12, entry, rec, same_donor, same,
                                            base, None, L, rp, sp, num_kv_heads, head_dim, v_heads=(k,))
            # Prespecified matched-negative fallback: ISSUE_END unless it is the selected anchor,
            # then DETAIL_END. This keeps the control structurally separate.
            negative_anchor = "ISSUE_END" if anchor != "ISSUE_END" else "DETAIL_END"
            nip = exp14.anchor_positions(rec, negative_anchor); nidp = exp14.anchor_positions(don, negative_anchor)
            neg_ctl, _ = exp14.v_suff_nec(model, tok, layers, exp12, entry, rec, donor, don,
                                           base, None, L, nip, nidp, num_kv_heads, head_dim, v_heads=(k,))

            for metric, value in [
                ("selected_V_sufficiency", suff), ("selected_V_necessity_loss", nec),
                ("cross_selected_V_sufficiency", cross_suff),
                ("same_state_selected_V_control", same_ctl),
("matched_negative_anchor_V_sufficiency", neg_ctl),
            ]:
                exp14.add_result(rows, phase="confirmation", task=task, entry=entry, donor=donor,
                                 relation="opposite_same_wording", anchor=anchor, metric=metric,
                                 value=value, L=L, k=k)

            # Correct runner-up control: runner-up gets its OWN residual reference.
            rrp = exp14.anchor_positions(rec, runner); rdp = exp14.anchor_positions(don, runner)
            runner_res_ref = exp14.residual_effect(model, tok, layers, exp12, entry, rec, donor, don,
                                                   base, L, rrp, rdp, num_kv_heads, head_dim)
            rs, rn = exp14.v_suff_nec(model, tok, layers, exp12, entry, rec, donor, don,
                                      base, runner_res_ref, L, rrp, rdp, num_kv_heads, head_dim,
                                      v_heads=(k,))
            for metric, value in [("runner_up_V_sufficiency", rs), ("runner_up_V_necessity_loss", rn)]:
                exp14.add_result(rows, phase="confirmation", task=task, entry=entry, donor=donor,
                                 relation="opposite_same_wording", anchor=runner, metric=metric,
                                 value=value, L=L, k=k)

            path = exp14.reader_path_effects(
                model, tok, layers, exp12, entry, rec, donor, don, rp, dp, base,
                L, (k,), num_kv_heads, num_q_heads, head_dim,
                leak_register_heads=tuple(sorted(register)),
            )
            groups = {
                "target_pos": pos, "negative_neg": neg,
                "non_set": non_set, "all_readers": all_set,
            }
            for name, readers in groups.items():
                gs, gn = safe_group(path, readers)
                exp14.add_result(rows, phase="confirmation", task=task, entry=entry, donor=donor,
                                 relation="opposite_same_wording", anchor=anchor,
                                 metric=f"reader_{name}_sufficiency", value=gs,
                                 L=L, k=k, readers=",".join(map(str, readers)))
                exp14.add_result(rows, phase="confirmation", task=task, entry=entry, donor=donor,
                                 relation="opposite_same_wording", anchor=anchor,
                                 metric=f"reader_{name}_necessity", value=gn,
                                 L=L, k=k, readers=",".join(map(str, readers)))
            exp14.add_result(rows, phase="confirmation", task=task, entry=entry, donor=donor,
                             relation="opposite_same_wording", anchor=anchor,
                             metric="reader_verified_v_effect", value=path["verified_v_effect"], L=L, k=k)
            exp14.add_result(rows, phase="confirmation", task=task, entry=entry, donor=donor,
                             relation="opposite_same_wording", anchor=anchor,
                             metric="reader_leakage_ratio", value=path["leakage_ratio"], L=L, k=k)
        print(f"[confirmation {n:03d}/{len(conf):03d}] {task['task_id']}")
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "confirmation_results.csv", index=False)
    return df


def task_series(df, metric):
    d = df[df["metric"] == metric]
    return d.groupby("task_id")["value"].mean().sort_index()


def summarize_extra(exp14, df, out_dir, seed):
    summary = exp14.summarize(df, "confirmation", seed)
    summary.to_csv(out_dir / "confirmation_summary.csv", index=False)

    # Family and directional summaries.
    task = df.groupby(["task_id", "family", "recipient_label", "metric"], as_index=False)["value"].mean()
    fam_rows, dir_rows = [], []
    for (family, metric), g in task.groupby(["family", "metric"]):
        m, lo, hi = exp14.bootstrap_ci(g["value"].values, seed + sum(map(ord, family + metric)))
        fam_rows.append({"family": family, "metric": metric, "mean": m, "ci_low": lo, "ci_high": hi,
                         "num_tasks": g["task_id"].nunique()})
    for (label, metric), g in task.groupby(["recipient_label", "metric"]):
        m, lo, hi = exp14.bootstrap_ci(g["value"].values, seed + 1000 + int(label) + sum(map(ord, metric)))
        dir_rows.append({"label": int(label), "metric": metric, "mean": m, "ci_low": lo, "ci_high": hi,
                         "num_tasks": g["task_id"].nunique()})
    pd.DataFrame(fam_rows).to_csv(out_dir / "family_summary.csv", index=False)
    pd.DataFrame(dir_rows).to_csv(out_dir / "directional_summary.csv", index=False)

    contrasts = []
    def paired(a, b, name, offs):
        sa, sb = task_series(df, a), task_series(df, b)
        idx = sa.index.intersection(sb.index)
        vals = (sa.loc[idx] - sb.loc[idx]).values
        m, lo, hi = exp14.bootstrap_ci(vals, seed + offs)
        contrasts.append({"contrast": name, "mean": m, "ci_low": lo, "ci_high": hi, "num_tasks": len(idx)})
    paired("selected_V_sufficiency", "runner_up_V_sufficiency", "selected_minus_runner_suff", 2001)
    paired("selected_V_necessity_loss", "runner_up_V_necessity_loss", "selected_minus_runner_nec", 2002)
    if not task_series(df, "reader_negative_neg_sufficiency").empty:
        paired("reader_target_pos_sufficiency", "reader_negative_neg_sufficiency", "positive_minus_inhibitory_suff", 2003)
        paired("reader_target_pos_necessity", "reader_negative_neg_necessity", "positive_minus_inhibitory_nec", 2004)
    pd.DataFrame(contrasts).to_csv(out_dir / "paired_contrasts.csv", index=False)
    return summary, contrasts, pd.DataFrame(dir_rows)


def get_summary_row(summary_df, metric):
    d = summary_df[summary_df["metric"] == metric]
    return None if d.empty else d.iloc[0]


def build_verdict(summary_df, contrasts, directional_df, selected):
    core = {}
    for metric in [
        "selected_V_sufficiency", "selected_V_necessity_loss",
        "cross_selected_V_sufficiency", "reader_target_pos_sufficiency",
        "reader_target_pos_necessity",
    ]:
        r = get_summary_row(summary_df, metric)
        core[metric] = bool(r is not None and float(r["ci_low"]) > 0)

    labels = {}
    for label in (0, 1):
        d = directional_df[(directional_df["label"] == label) &
                           (directional_df["metric"] == "selected_V_sufficiency")]
        labels[str(label)] = bool(not d.empty and float(d.iloc[0]["ci_low"]) > 0)

    neg_exists = bool(selected["reader_neg_set"])
    neg_pass = False
    if neg_exists:
        rs = get_summary_row(summary_df, "reader_negative_neg_sufficiency")
        rn = get_summary_row(summary_df, "reader_negative_neg_necessity")
        neg_pass = bool(rs is not None and rn is not None and float(rs["ci_high"]) < 0 and float(rn["ci_high"]) < 0)

    contrast_map = {x["contrast"]: x for x in contrasts}
    reader_contrast_pass = True
    if neg_exists:
        reader_contrast_pass = all(
            contrast_map.get(name, {}).get("ci_low", -1) > 0
            for name in ("positive_minus_inhibitory_suff", "positive_minus_inhibitory_nec")
        )

    same = get_summary_row(summary_df, "same_state_selected_V_control")
    leak = get_summary_row(summary_df, "reader_leakage_ratio")
    matched = get_summary_row(summary_df, "matched_negative_anchor_V_sufficiency")
    verified = get_summary_row(summary_df, "reader_verified_v_effect")
    all_s = get_summary_row(summary_df, "reader_all_readers_sufficiency")
    all_n = get_summary_row(summary_df, "reader_all_readers_necessity")
    same_clean = bool(same is not None and float(same["ci_low"]) <= 0 <= float(same["ci_high"]))
    leak_clean = bool(leak is not None and float(leak["mean"]) <= 0.10)
    matched_clean = bool(matched is not None and float(matched["ci_low"]) <= 0 <= float(matched["ci_high"]))
    reconstruction_clean = bool(
        verified is not None and all_s is not None and all_n is not None
        and abs(float(all_s["mean"]) - float(verified["mean"])) < 5e-5
        and abs(float(all_n["mean"]) - float(verified["mean"])) < 5e-5
    )
    runner_clean = all(
        contrast_map.get(name, {}).get("ci_low", -1) > 0
        for name in ("selected_minus_runner_suff", "selected_minus_runner_nec")
    )

    core_pass = all(core.values()) and all(labels.values()) and reader_contrast_pass
    if core_pass and neg_exists and neg_pass and same_clean and leak_clean and matched_clean and reconstruction_clean and runner_clean:
        verdict = "STRONG RECURRING HOMOLOG"
    elif core_pass:
        verdict = "PARTIAL / ALGORITHMIC HOMOLOG"
    else:
        verdict = "ARCHITECTURE-DIVERGENT"

    return {
        "verdict": verdict,
        "core_endpoint_flags": core,
        "directional_flags": labels,
        "inhibitory_register_exists": neg_exists,
        "inhibitory_register_pass": neg_pass,
        "reader_contrast_pass": reader_contrast_pass,
        "fidelity": {
            "same_state_clean": same_clean,
            "leakage_clean": leak_clean,
            "selected_over_runner_clean": runner_clean,
            "matched_negative_clean": matched_clean,
            "all_reader_reconstruction_clean": reconstruction_clean,
            "same_state": None if same is None else same.to_dict(),
            "reader_leakage": None if leak is None else leak.to_dict(),
            "matched_negative": None if matched is None else matched.to_dict(),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--phase", choices=("inventory", "gate", "discovery", "confirmation", "all"), default="inventory")
    ap.add_argument("--allow_prior_family", action="store_true")
    ap.add_argument("--trust_remote_code", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    exp15 = load_module(EXP15_CODE, "skills4s_exp15")
    exp14 = exp15.load_module(exp15.EXP14_CODE, "skills4s_exp14_for16")
    exp12 = exp14.load_exp12()
    exp14.compute_schema = exp15.generic_compute_schema

    inventory = exp15.inventory_models(exp15.DEFAULT_INVENTORY_ROOTS)
    (out_dir / "model_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.phase == "inventory":
        print(json.dumps(inventory, ensure_ascii=False, indent=2)); return
    if not args.model_dir:
        raise SystemExit("--model_dir required")

    model, tok, layers, cfg, arch = exp15.load_cross_arch_model(Path(args.model_dir), args.trust_remote_code)
    if prior_family(cfg) and not args.allow_prior_family:
        raise RuntimeError("EXP16 Model D must not be Qwen2 or Mistral family.")
    if arch["attention_kind"] != "GQA":
        raise RuntimeError(
            "EXP16 confirmatory geometry is preregistered for GQA. "
            f"Got {arch['attention_kind']}. Use an architecture-adapted exploratory branch for MHA/MQA."
        )
    num_q_heads, num_kv_heads, head_dim = arch["num_attention_heads"], arch["num_key_value_heads"], arch["head_dim"]
    exp14.non_reader_heads = lambda pos_set=(), neg_set=(): tuple(
        q for q in range(num_q_heads) if q not in (set(pos_set) | set(neg_set))
    )

    tasks = exp12.make_tasks(); split = exp15.family_split(exp14, tasks)
    (out_dir / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
    arch_info = {**arch, "model_type": getattr(cfg, "model_type", None), "architectures": getattr(cfg, "architectures", None)}
    (out_dir / "architecture_sanity.json").write_text(json.dumps(arch_info, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.phase in ("gate", "all"):
        gate, _ = exp14.run_gate(model, tok, layers, exp12, tasks, num_kv_heads, head_dim, out_dir, SEED)
        if not gate["gate_pass"]:
            summary = {"experiment": "EXP16_second_cross_arch", "verdict": "BEHAVIORAL GATE FAILURE", "gate": gate, "architecture": arch_info}
            (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(summary, ensure_ascii=False, indent=2)); return

    selected_path = out_dir / "selected_homolog.json"
    if args.phase in ("discovery", "all"):
        run_discovery(exp15, exp14, model, tok, layers, exp12, tasks, split,
                      num_kv_heads, num_q_heads, head_dim, out_dir)

    if args.phase in ("confirmation", "all"):
        if not selected_path.is_file(): raise FileNotFoundError(selected_path)
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        conf = run_confirmation_fixed(exp14, model, tok, layers, exp12, tasks, split, selected,
                                      num_kv_heads, num_q_heads, head_dim, out_dir)
        summary_df, contrasts, directional_df = summarize_extra(exp14, conf, out_dir, SEED + 5)
        verdict = build_verdict(summary_df, contrasts, directional_df, selected)
        summary = {
            "experiment": "EXP16_second_cross_arch",
            "model_dir": str(Path(args.model_dir).resolve()),
            "architecture": arch_info,
            "selected": selected,
            "verdict": verdict,
            "methodological_repairs": [
                "final (L,KV) is localized using the actually selected stratum-specific anchors",
                "reader register uses causal-effect thresholds rather than forced zero-score quota filling",
                "runner-up necessity uses runner-up-specific residual reference",
            ],
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest = {
            "experiment": "EXP16_second_cross_arch", "git_commit": git_commit(), "seed": SEED,
            "model_dir": str(Path(args.model_dir).resolve()), "architecture": arch_info,
            "exp15_sha256": exp15.sha256_file(EXP15_CODE),
            "script_sha256": exp15.sha256_file(Path(__file__).resolve()),
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "offline_experiment": True,
        }
        (out_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
