#!/usr/bin/env python3
"""EXP14 post-processing: generate preregistered summary outputs from the raw
confirmation/discovery CSVs. Pure pandas — no model required."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "outputs" / "exp14_cross_model_homolog"
N_BOOT = 5000
SEED = 5313


def bootstrap_ci(values, seed, n_boot=N_BOOT):
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = np.asarray([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)])
    return float(x.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def task_level_summary(df, metrics):
    rows = []
    for (relation, metric), g in df[df["metric"].isin(metrics)].groupby(["relation", "metric"]):
        by_task = g.groupby("task_id")["value"].agg("mean").values
        m, lo, hi = bootstrap_ci(by_task, SEED)
        rows.append({"relation": relation, "metric": metric,
                     "mean": m, "ci_low": lo, "ci_high": hi, "num_tasks": len(by_task)})
    return pd.DataFrame(rows)


def main():
    conf = pd.read_csv(OUT / "confirmation_results.csv")
    selected = json.loads((OUT / "selected_homolog.json").read_text(encoding="utf-8"))
    L_star, k_star = selected["L_star"], selected["k_star"]

    # --- paired contrasts (task-level paired bootstrap) ---
    def task_mean(metric):
        sub = conf[conf["metric"] == metric]
        return sub.groupby("task_id")["value"].agg("mean")

    contrasts = [
        ("selected_minus_old_absolute_sufficiency", "selected_V_sufficiency", "old_absolute_V_sufficiency"),
        ("selected_minus_old_absolute_necessity", "selected_V_necessity_loss", "old_absolute_V_necessity_loss"),
        ("selected_minus_runner_up_sufficiency", "selected_V_sufficiency", "runner_up_V_sufficiency"),
        ("selected_minus_runner_up_necessity", "selected_V_necessity_loss", "runner_up_V_necessity_loss"),
        ("target_minus_negative_reader_sufficiency", "reader_target_pos_sufficiency", "reader_negative_neg_sufficiency"),
        ("target_minus_negative_reader_necessity", "reader_target_pos_necessity", "reader_negative_neg_necessity"),
    ]
    if "old_absolute_V_necessity_loss" not in conf["metric"].unique():
        # necessity of old-absolute was not separately measured; skip that contrast
        contrasts = [c for c in contrasts if c[2] != "old_absolute_V_necessity_loss"]
    crows = []
    for name, a, b in contrasts:
        ta, tb = task_mean(a), task_mean(b)
        common = ta.index.intersection(tb.index)
        m, lo, hi = bootstrap_ci((ta.loc[common] - tb.loc[common]).values, SEED)
        crows.append({"contrast": name, "metric_a": a, "metric_b": b,
                      "mean_difference": m, "ci_low": lo, "ci_high": hi,
                      "num_tasks": len(common)})
    pd.DataFrame(crows).to_csv(OUT / "paired_contrasts.csv", index=False)

    # --- directional (label 0 / 1) selected V + reader ---
    drows = []
    for metric in ["selected_V_sufficiency", "selected_V_necessity_loss",
                   "reader_target_pos_sufficiency", "reader_negative_neg_sufficiency"]:
        for label in (0, 1):
            sub = conf[(conf["metric"] == metric) & (conf["recipient_label"] == label)]
            by_task = sub.groupby("task_id")["value"].agg("mean").values
            m, lo, hi = bootstrap_ci(by_task, SEED)
            drows.append({"metric": metric, "recipient_label": label,
                          "mean": m, "ci_low": lo, "ci_high": hi, "num_tasks": len(by_task)})
    pd.DataFrame(drows).to_csv(OUT / "directional_confirmation.csv", index=False)

    # --- family x wording x label ---
    frows = []
    for metric in ["selected_V_sufficiency", "selected_V_necessity_loss"]:
        for (family, wording, label), g in conf[conf["metric"] == metric].groupby(
                ["family", "recipient_wording", "recipient_label"]):
            by_task = g.groupby("task_id")["value"].agg("mean").values
            m, lo, hi = bootstrap_ci(by_task, SEED)
            frows.append({"family": family, "wording": wording, "recipient_label": label,
                          "metric": metric, "mean": m, "ci_low": lo, "ci_high": hi,
                          "num_tasks": len(by_task)})
    pd.DataFrame(frows).to_csv(OUT / "family_wording_confirmation.csv", index=False)

    # --- discovery summary (selected anchor table) ---
    disc = pd.read_csv(OUT / "discovery_results.csv")
    dsum = []
    for (family, wording), g in disc.groupby(["family", "recipient_wording"]):
        for metric in ["V_sufficiency", "V_necessity_loss"]:
            sub = g[g["metric"] == metric]
            by_task = sub.groupby("task_id")["value"].agg("mean").values
            m, lo, hi = bootstrap_ci(by_task, SEED)
            dsum.append({"family": family, "wording": wording, "metric": metric,
                         "mean": m, "ci_low": lo, "ci_high": hi, "num_tasks": len(by_task)})
    pd.DataFrame(dsum).to_csv(OUT / "discovery_summary.csv", index=False)

    # --- verdict flags against preregistered hard endpoints ---
    def flag(metric, direction):
        sub = conf[conf["metric"] == metric]
        by_task = sub.groupby("task_id")["value"].agg("mean").values
        m, lo, hi = bootstrap_ci(by_task, SEED)
        ok = (lo > 0) if direction == ">" else ((hi < 0) if direction == "<" else (lo < 0 < hi))
        return {"mean": m, "ci_low": lo, "ci_high": hi, "pass": bool(ok)}

    selected_suff = flag("selected_V_sufficiency", ">")
    selected_nec = flag("selected_V_necessity_loss", ">")
    cross_suff = flag("cross_selected_V_sufficiency", ">")
    same_state = flag("same_state_selected_V_control", "0")   # expected ≈0 (CI contains 0)
    reader_pos = flag("reader_target_pos_sufficiency", ">")
    reader_neg = flag("reader_negative_neg_sufficiency", "<")
    reader_non = flag("reader_non_set_sufficiency", "0")
    leakage = conf[conf["metric"] == "reader_leakage_ratio"].groupby("task_id")["value"].agg("mean").values
    leak_mean = float(np.mean(leakage))

    dir_df = pd.read_csv(OUT / "directional_confirmation.csv")
    lab0 = dir_df[(dir_df["metric"] == "selected_V_sufficiency") & (dir_df["recipient_label"] == 0)].iloc[0]
    lab1 = dir_df[(dir_df["metric"] == "selected_V_sufficiency") & (dir_df["recipient_label"] == 1)].iloc[0]
    bidir = bool(lab0["ci_low"] > 0 and lab1["ci_low"] > 0)

    all28_eq = np.allclose(
        conf[conf["metric"] == "reader_all28_sufficiency"].groupby("task_id")["value"].agg("mean").values,
        conf[conf["metric"] == "reader_verified_v_effect"].groupby("task_id")["value"].agg("mean").values,
        atol=1e-6,
    )

    # writer-anchor functional family check
    instr_family = {"USER_END", "FINAL_INSTRUCTION_END", "SYSTEM_END", "GENERATION_BOUNDARY"}
    strata = selected["strata"]
    anchor_family_ok = all(strata[k]["selected_anchor"] in instr_family for k in strata)

    hard = {
        "selected_V_sufficiency": selected_suff,
        "selected_V_necessity_loss": selected_nec,
        "cross_selected_V_sufficiency": cross_suff,
        "same_state_selected_V_control_approx0": same_state,
        "reader_pos_set_sufficiency": reader_pos,
        "reader_neg_set_sufficiency": reader_neg,
        "reader_non_set_approx0": reader_non,
        "all28_eq_verified_exact": all28_eq,
        "bidirectional_label0_and_label1": bidir,
        "writer_anchor_in_functional_family": anchor_family_ok,
    }
    hard_pass = all(v["pass"] if isinstance(v, dict) else v for v in hard.values())
    leak_ok = leak_mean < 0.05

    if hard_pass and leak_ok:
        verdict = "SUCCESS"
    elif hard_pass:
        verdict = "PARTIAL_SUCC_LEAKAGE"
    else:
        verdict = "PARTIAL" if (hard["reader_pos_set_sufficiency"]["pass"]
                                and hard["reader_neg_set_sufficiency"]["pass"]) else "FAIL"

    summary = {
        "experiment": "EXP14_cross_model_homolog",
        "model_b": "Qwen2-7B-Instruct",
        "model_a_reference": "Qwen2.5-Coder-7B-Instruct",
        "frozen_homolog": {"L_star": L_star, "k_star": k_star,
                           "reader_pos_set": selected["reader_pos_set"],
                           "reader_neg_set": selected["reader_neg_set"]},
        "verdict": verdict,
        "hard_endpoint_flags": hard,
        "leakage_ratio_mean": leak_mean,
        "contrasts": {c["contrast"]: {"mean": c["mean_difference"], "ci_low": c["ci_low"],
                                      "ci_high": c["ci_high"]} for c in crows},
        "directional": {f"label{int(r['recipient_label'])}_{r['metric']}":
                        {"mean": r["mean"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]}
                        for _, r in dir_df.iterrows()},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
