#!/usr/bin/env python3
"""Recompute EXP15 runner-up V necessity using runner-up-specific residual reference.

Non-destructive: writes under outputs/exp15_cross_arch_homolog/diagnostics/.
"""
from __future__ import annotations
import importlib.util, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP15 = REPO_ROOT / "experiments" / "exp15_cross_arch_homolog" / "run.py"
OUT15 = REPO_ROOT / "outputs" / "exp15_cross_arch_homolog"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None; spec.loader.exec_module(mod); return mod


def main():
    exp15 = load(EXP15, "exp15_diag")
    exp14 = exp15.load_module(exp15.EXP14_CODE, "exp14_diag")
    exp12 = exp14.load_exp12(); exp14.compute_schema = exp15.generic_compute_schema
    summary = json.loads((OUT15 / "summary.json").read_text(encoding="utf-8"))
    selected = json.loads((OUT15 / "selected_homolog.json").read_text(encoding="utf-8"))
    model_dir = Path(summary["model_dir"])
    model, tok, layers, cfg, arch = exp15.load_cross_arch_model(model_dir)
    nq, nk, hd = arch["num_attention_heads"], arch["num_key_value_heads"], arch["head_dim"]
    tasks = exp12.make_tasks(); split = exp15.family_split(exp14, tasks)
    conf_ids = exp14.split_ids(split, "confirmation"); conf = [t for t in tasks if t["task_id"] in conf_ids]
    L, k = selected["L_star"], selected["k_star"]
    rows = []
    for n, task in enumerate(conf, 1):
        entries = exp12.make_skill_entries(task); maps = exp12.donor_maps_skill(entries)
        cache = exp14.build_task_cache(model, tok, layers, exp12, task, entries, L, nk, hd)
        bases = exp14.baseline_margins(model, tok, layers, exp12, entries, cache, L, nk, hd)
        for i, entry in enumerate(entries):
            donor_i = maps[i]["opposite_same_wording"]; donor = entries[donor_i]
            rec, don = cache[i], cache[donor_i]
            key = f"{task['family']}::{entry['wording']}"; runner = selected["strata"][key]["runner_up_anchor"]
            rp = exp14.anchor_positions(rec, runner); dp = exp14.anchor_positions(don, runner)
            runner_res = exp14.residual_effect(model, tok, layers, exp12, entry, rec, donor, don,
                                                bases[i], L, rp, dp, nk, hd)
            suff, nec = exp14.v_suff_nec(model, tok, layers, exp12, entry, rec, donor, don,
                                         bases[i], runner_res, L, rp, dp, nk, hd, v_heads=(k,))
            rows.append({"task_id": task["task_id"], "family": task["family"], "wording": entry["wording"],
                         "label": int(entry["label"]), "runner_up_anchor": runner,
                         "runner_residual_effect": runner_res, "runner_up_V_sufficiency_corrected": suff,
                         "runner_up_V_necessity_corrected": nec})
        print(f"[{n:02d}/{len(conf):02d}] {task['task_id']}")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    diag = OUT15 / "diagnostics"; diag.mkdir(exist_ok=True)
    df.to_csv(diag / "runnerup_corrected_results.csv", index=False)
    taskdf = df.groupby("task_id", as_index=False).agg(
        suff=("runner_up_V_sufficiency_corrected", "mean"),
        nec=("runner_up_V_necessity_corrected", "mean"),
        residual=("runner_residual_effect", "mean"),
    )
    result = {}
    for name in ("suff", "nec", "residual"):
        m, lo, hi = exp14.bootstrap_ci(taskdf[name].values, 5915 + sum(map(ord, name)))
        result[name] = {"mean": m, "ci_low": lo, "ci_high": hi}
    original = pd.read_csv(OUT15 / "confirmation_summary.csv")
    old = original[original["metric"].isin(["runner_up_V_sufficiency", "runner_up_V_necessity_loss"])].to_dict("records")
    summary_out = {
        "diagnostic": "EXP15 runner-up residual-reference correction",
        "bug": "original runner-up necessity reused selected-anchor res_ref",
        "original_summary_rows": old,
        "corrected": result,
        "primary_exp15_endpoints_affected": False,
        "note": "Update EXPERIMENTS.md: original runner-up necessity is invalid; use this diagnostic instead.",
    }
    (diag / "runnerup_corrected_summary.json").write_text(json.dumps(summary_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary_out, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
