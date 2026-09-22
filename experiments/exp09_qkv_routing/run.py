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
DEFAULT_OUT = REPO_ROOT / "outputs" / "exp09_qkv_routing"

LABEL_TEST = 0
LABEL_IMPL = 1

SOURCE_HIDDEN = [18, 19, 20]
COMPONENTS = {
    "Q": ("q",),
    "K": ("k",),
    "V": ("v",),
    "KV": ("k", "v"),
    "QKV": ("q", "k", "v"),
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
def projection_patch_hook(module, positions, replacements):
    """Patch q_proj/k_proj/v_proj output before reshape and RoPE."""
    def hook_fn(mod, inputs, output):
        y = output.clone()
        b = torch.arange(y.shape[0], device=y.device).unsqueeze(1)
        y[b, positions.to(y.device), :] = replacements.to(y.device, dtype=y.dtype)
        return y

    handle = module.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


def tokenize_entry(tok, entry):
    text = render_prompt(tok, entry["messages"])
    return tok(text, add_special_tokens=False)["input_ids"]


@torch.inference_mode()
def capture_prompt(model, tok, layers, entry, max_suffix):
    """Capture exact patch sites in float32.

    residual[Hk] = decoder block k-1 output
    q/k/v[Hk]    = q_proj/k_proj/v_proj output in consumer block k
    """
    device = model.get_input_embeddings().weight.device
    text = render_prompt(tok, entry["messages"])
    enc = tok(text, add_special_tokens=False, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}

    store = {"residual": {}, "q": {}, "k": {}, "v": {}}
    handles = []

    def residual_hook(hidx):
        def fn(module, inputs, output):
            t = output[0] if isinstance(output, tuple) else output
            take = min(max_suffix, t.shape[1])
            store["residual"][hidx] = (
                t[0, -take:, :].detach().float().cpu().numpy().astype(np.float32)
            )
        return fn

    def proj_hook(kind, hidx):
        def fn(module, inputs, output):
            take = min(max_suffix, output.shape[1])
            store[kind][hidx] = (
                output[0, -take:, :].detach().float().cpu().numpy().astype(np.float32)
            )
        return fn

    for hidx in SOURCE_HIDDEN:
        handles.append(
            layers[hidx - 1].register_forward_hook(residual_hook(hidx))
        )
        attn = layers[hidx].self_attn
        handles.append(attn.q_proj.register_forward_hook(proj_hook("q", hidx)))
        handles.append(attn.k_proj.register_forward_hook(proj_hook("k", hidx)))
        handles.append(attn.v_proj.register_forward_hook(proj_hook("v", hidx)))

    try:
        model(**enc, use_cache=False, return_dict=True)
    finally:
        for h in handles:
            h.remove()

    return {"ids": ids, **store}


def prepare_task_cache(model, tok, layers, entries):
    maps = donor_maps(entries)
    ids = [tokenize_entry(tok, e) for e in entries]

    max_suffix = []
    for i in range(len(entries)):
        ks = [
            longest_common_suffix(ids[i], ids[j])
            for j in maps[i].values()
        ]
        max_suffix.append(max(ks))

    cache = [
        capture_prompt(model, tok, layers, entries[i], max_suffix[i])
        for i in range(len(entries))
    ]
    return cache, maps


def get_slice(cache, kind, hidx, k):
    arr = cache[kind][hidx]
    if arr.shape[0] < k:
        raise RuntimeError(
            f"{kind} H{hidx}: captured {arr.shape[0]} tokens but need {k}"
        )
    return arr[-k:, :].astype(np.float32)


def install_projection_patch(
    stack,
    layers,
    source_h,
    components,
    positions,
    source_cache,
    k,
    device,
):
    attn = layers[source_h].self_attn
    module_map = {"q": attn.q_proj, "k": attn.k_proj, "v": attn.v_proj}

    for kind in components:
        values = get_slice(source_cache, kind, source_h, k)
        replacement = torch.tensor(
            values[None, :, :],
            dtype=torch.float32,
            device=device,
        )
        stack.enter_context(
            projection_patch_hook(module_map[kind], positions, replacement)
        )


@torch.inference_mode()
def score(
    model,
    tok,
    layers,
    recipient,
    recipient_cache,
    donor_cache=None,
    common_k=None,
    residual_source_h=None,
    projection_specs=None,
):
    """Score TEST and IMPL candidates one-by-one."""
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
            if donor_cache is not None and common_k is not None:
                k = int(common_k)
                pos_np = np.arange(plen-k, plen, dtype=np.int64)
                positions = torch.tensor(
                    pos_np[None, :], dtype=torch.long, device=device
                )

                if residual_source_h is not None:
                    values = get_slice(
                        donor_cache, "residual", residual_source_h, k
                    )
                    replacement = torch.tensor(
                        values[None, :, :],
                        dtype=torch.float32,
                        device=device,
                    )
                    stack.enter_context(
                        residual_patch_hook(
                            layers[residual_source_h - 1],
                            positions,
                            replacement,
                        )
                    )

                if projection_specs:
                    for spec in projection_specs:
                        install_projection_patch(
                            stack=stack,
                            layers=layers,
                            source_h=spec["source_h"],
                            components=spec["components"],
                            positions=positions,
                            source_cache=spec["cache"],
                            k=k,
                            device=device,
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


def paired_ci(a, b, seed):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) != len(b):
        raise RuntimeError("Paired contrast length mismatch")
    return bootstrap_ci(a-b, seed)


def append_result(
    rows,
    *,
    ti,
    rec,
    donor,
    relation,
    source_h,
    config,
    component,
    effect_type,
    value,
    residual_reference=np.nan,
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
        "source_h": source_h if source_h is not None else -1,
        "consumer_decoder_block": source_h if source_h is not None else -1,
        "config": config,
        "component": component,
        "effect_type": effect_type,
        "value": float(value),
        "residual_reference": float(residual_reference)
        if np.isfinite(residual_reference) else np.nan,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=4909)
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

    if max(SOURCE_HIDDEN) >= len(layers):
        raise RuntimeError("Consumer block for H20 is unavailable")

    print("=" * 82)
    print("EXP09: Residual -> Q/K/V Routing Mediation")
    print("Model:", model_dir)
    print("Tasks:", len(by_task))
    print("Sources H18/H19/H20 -> consumer blocks 18/19/20")
    print("=" * 82)

    rows = []

    for task_no, ti in enumerate(sorted(by_task), start=1):
        entries = by_task[ti]
        cache, maps = prepare_task_cache(model, tok, layers, entries)

        baselines = [
            score(model, tok, layers, rec, cache[i])
            for i, rec in enumerate(entries)
        ]

        # Main same-wording opposite-state experiments.
        for i, rec in enumerate(entries):
            di = maps[i]["opposite_same_wording"]
            donor = entries[di]
            k_common = longest_common_suffix(cache[i]["ids"], cache[di]["ids"])
            sign = 1.0 if donor["label"] == LABEL_IMPL else -1.0
            base = baselines[i]["margin"]

            for source_h in SOURCE_HIDDEN:
                residual = score(
                    model, tok, layers, rec, cache[i],
                    donor_cache=cache[di],
                    common_k=k_common,
                    residual_source_h=source_h,
                )
                e_res = sign * (residual["margin"] - base)

                append_result(
                    rows, ti=ti, rec=rec, donor=donor,
                    relation="opposite_same_wording",
                    source_h=source_h,
                    config=f"H{source_h}_residual",
                    component="RESIDUAL",
                    effect_type="residual_reference",
                    value=e_res,
                )

                for component_name, component_tuple in COMPONENTS.items():
                    suff = score(
                        model, tok, layers, rec, cache[i],
                        donor_cache=cache[di],
                        common_k=k_common,
                        projection_specs=[{
                            "source_h": source_h,
                            "components": component_tuple,
                            "cache": cache[di],
                        }],
                    )
                    e_suff = sign * (suff["margin"] - base)

                    append_result(
                        rows, ti=ti, rec=rec, donor=donor,
                        relation="opposite_same_wording",
                        source_h=source_h,
                        config=f"H{source_h}_{component_name}_sufficiency",
                        component=component_name,
                        effect_type="sufficiency",
                        value=e_suff,
                        residual_reference=e_res,
                    )

                    nec = score(
                        model, tok, layers, rec, cache[i],
                        donor_cache=cache[di],
                        common_k=k_common,
                        residual_source_h=source_h,
                        projection_specs=[{
                            "source_h": source_h,
                            "components": component_tuple,
                            "cache": cache[i],
                        }],
                    )
                    retained = sign * (nec["margin"] - base)
                    loss = e_res - retained

                    append_result(
                        rows, ti=ti, rec=rec, donor=donor,
                        relation="opposite_same_wording",
                        source_h=source_h,
                        config=f"H{source_h}_{component_name}_necessity_loss",
                        component=component_name,
                        effect_type="necessity_loss",
                        value=loss,
                        residual_reference=e_res,
                    )

            # Exploratory coordinated consumer routing.
            for name, component_tuple in [
                ("chain_KV_sufficiency", COMPONENTS["KV"]),
                ("chain_QKV_sufficiency", COMPONENTS["QKV"]),
            ]:
                chain = score(
                    model, tok, layers, rec, cache[i],
                    donor_cache=cache[di],
                    common_k=k_common,
                    projection_specs=[
                        {
                            "source_h": h,
                            "components": component_tuple,
                            "cache": cache[di],
                        }
                        for h in SOURCE_HIDDEN
                    ],
                )
                e_chain = sign * (chain["margin"] - base)
                append_result(
                    rows, ti=ti, rec=rec, donor=donor,
                    relation="opposite_same_wording",
                    source_h=None,
                    config=name,
                    component="QKV" if "QKV" in name else "KV",
                    effect_type="chain_sufficiency",
                    value=e_chain,
                )

        # QKV controls.
        for relation in [
            "self",
            "same_state_cross_wording",
            "opposite_cross_wording",
        ]:
            for i, rec in enumerate(entries):
                di = maps[i][relation]
                donor = entries[di]
                k_common = longest_common_suffix(cache[i]["ids"], cache[di]["ids"])
                base = baselines[i]["margin"]
                sign = 1.0 if relation == "self" else (
                    1.0 if donor["label"] == LABEL_IMPL else -1.0
                )

                for source_h in SOURCE_HIDDEN:
                    qkv = score(
                        model, tok, layers, rec, cache[i],
                        donor_cache=cache[di],
                        common_k=k_common,
                        projection_specs=[{
                            "source_h": source_h,
                            "components": COMPONENTS["QKV"],
                            "cache": cache[di],
                        }],
                    )
                    qkv_effect = sign * (qkv["margin"] - base)

                    append_result(
                        rows, ti=ti, rec=rec, donor=donor,
                        relation=relation,
                        source_h=source_h,
                        config=f"H{source_h}_QKV_control_sufficiency",
                        component="QKV",
                        effect_type="control_sufficiency",
                        value=qkv_effect,
                    )

                    if relation == "opposite_cross_wording":
                        residual = score(
                            model, tok, layers, rec, cache[i],
                            donor_cache=cache[di],
                            common_k=k_common,
                            residual_source_h=source_h,
                        )
                        e_res = sign * (residual["margin"] - base)

                        nec = score(
                            model, tok, layers, rec, cache[i],
                            donor_cache=cache[di],
                            common_k=k_common,
                            residual_source_h=source_h,
                            projection_specs=[{
                                "source_h": source_h,
                                "components": COMPONENTS["QKV"],
                                "cache": cache[i],
                            }],
                        )
                        retained = sign * (nec["margin"] - base)
                        loss = e_res - retained

                        append_result(
                            rows, ti=ti, rec=rec, donor=donor,
                            relation=relation,
                            source_h=source_h,
                            config=f"H{source_h}_QKV_control_necessity_loss",
                            component="QKV",
                            effect_type="control_necessity_loss",
                            value=loss,
                            residual_reference=e_res,
                        )

        print(f"[{task_no:03d}/{len(by_task):03d}] task {ti}")
        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "intervention_results.csv", index=False)

    task_level = (
        results.groupby(
            [
                "config", "relation", "source_h",
                "component", "effect_type",
                "task_idx", "task_id",
            ],
            as_index=False,
        )["value"].mean()
    )

    summary_rows = []
    for keys, g in task_level.groupby(
        ["config", "relation", "source_h", "component", "effect_type"]
    ):
        config, relation, source_h, component, effect_type = keys
        seed = args.seed + sum(
            ord(c) for c in (config + relation + component + effect_type)
        )
        mean, lo, hi = bootstrap_ci(g["value"], seed)
        summary_rows.append({
            "config": config,
            "relation": relation,
            "source_h": int(source_h),
            "component": component,
            "effect_type": effect_type,
            "mean": mean,
            "ci_low": lo,
            "ci_high": hi,
            "num_tasks": int(g["task_idx"].nunique()),
        })

    summary_df = pd.DataFrame(summary_rows)
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

    contrast_defs = [
        ("H20_KV_suff_minus_Q_suff", "H20_KV_sufficiency", "H20_Q_sufficiency"),
        ("H20_KV_nec_minus_Q_nec", "H20_KV_necessity_loss", "H20_Q_necessity_loss"),
        ("H20_QKV_suff_minus_KV_suff", "H20_QKV_sufficiency", "H20_KV_sufficiency"),
        ("H20_QKV_nec_minus_KV_nec", "H20_QKV_necessity_loss", "H20_KV_necessity_loss"),
    ]

    contrast_rows = []
    for name, a_cfg, b_cfg in contrast_defs:
        mean, lo, hi = paired_ci(
            task_values(a_cfg),
            task_values(b_cfg),
            args.seed + sum(ord(c) for c in name),
        )
        contrast_rows.append({
            "contrast": name,
            "mean_difference": mean,
            "ci_low": lo,
            "ci_high": hi,
        })

    contrast_df = pd.DataFrame(contrast_rows)
    contrast_df.to_csv(out_dir / "paired_contrasts.csv", index=False)

    def get_summary(config, relation="opposite_same_wording"):
        return summary_df[
            (summary_df["config"] == config)
            & (summary_df["relation"] == relation)
        ].iloc[0]

    profile_rows = []
    for h in SOURCE_HIDDEN:
        residual_mean = float(get_summary(f"H{h}_residual")["mean"])
        for component in COMPONENTS:
            for effect_type, suffix in [
                ("sufficiency", "sufficiency"),
                ("necessity_loss", "necessity_loss"),
            ]:
                row = get_summary(f"H{h}_{component}_{suffix}")
                mean = float(row["mean"])
                profile_rows.append({
                    "source_h": h,
                    "component": component,
                    "effect_type": effect_type,
                    "mean": mean,
                    "ci_low": float(row["ci_low"]),
                    "ci_high": float(row["ci_high"]),
                    "residual_reference": residual_mean,
                    "fraction_of_residual": (
                        mean / residual_mean if abs(residual_mean) > 1e-12 else np.nan
                    ),
                })

    profile = pd.DataFrame(profile_rows)
    profile.to_csv(out_dir / "routing_profile.csv", index=False)

    # Plot primary H20.
    h20 = profile[profile["source_h"] == 20].copy()
    comps = list(COMPONENTS)
    x = np.arange(len(comps))
    width = 0.36

    suff = (
        h20[h20["effect_type"] == "sufficiency"]
        .set_index("component").loc[comps]
    )
    nec = (
        h20[h20["effect_type"] == "necessity_loss"]
        .set_index("component").loc[comps]
    )

    fig = plt.figure(figsize=(10, 6))
    plt.bar(x-width/2, suff["mean"], width, label="sufficiency")
    plt.bar(x+width/2, nec["mean"], width, label="necessity loss")
    plt.axhline(0.0, linestyle="--")
    plt.xticks(x, comps)
    plt.ylabel("Causal effect")
    plt.title("EXP09 H20 consumer Q/K/V mediation")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "qkv_routing.png", dpi=180)
    plt.close(fig)

    # Primary summary.
    h20_res = get_summary("H20_residual")
    h20_kv_s = get_summary("H20_KV_sufficiency")
    h20_kv_n = get_summary("H20_KV_necessity_loss")
    h20_qkv_s = get_summary("H20_QKV_sufficiency")
    h20_qkv_n = get_summary("H20_QKV_necessity_loss")
    h20_self = get_summary("H20_QKV_control_sufficiency", "self")
    h20_same = get_summary(
        "H20_QKV_control_sufficiency", "same_state_cross_wording"
    )
    h20_cross_s = get_summary(
        "H20_QKV_control_sufficiency", "opposite_cross_wording"
    )
    h20_cross_n = get_summary(
        "H20_QKV_control_necessity_loss", "opposite_cross_wording"
    )
    chain_kv = get_summary("chain_KV_sufficiency")
    chain_qkv = get_summary("chain_QKV_sufficiency")

    manifest = {
        "experiment": "EXP09_qkv_routing",
        "git_commit": git_commit(),
        "source_exp01b_git_commit": manifest01b.get("git_commit"),
        "seed": args.seed,
        "model_dir": str(model_dir),
        "model_name": model_dir.name,
        "source_hidden_indices": SOURCE_HIDDEN,
        "source_to_consumer_decoder_block": {str(h): h for h in SOURCE_HIDDEN},
        "projection_components": {k: list(v) for k, v in COMPONENTS.items()},
        "q_proj_out_features": int(layers[20].self_attn.q_proj.out_features),
        "k_proj_out_features": int(layers[20].self_attn.k_proj.out_features),
        "v_proj_out_features": int(layers[20].self_attn.v_proj.out_features),
        "num_attention_heads": int(model.config.num_attention_heads),
        "num_key_value_heads": int(model.config.num_key_value_heads),
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
        "experiment": "EXP09_qkv_routing",
        "primary_source": "H20",
        "H20_residual_reference_mean": float(h20_res["mean"]),
        "H20_residual_reference_95ci": ci(h20_res),
        "H20_KV_sufficiency_mean": float(h20_kv_s["mean"]),
        "H20_KV_sufficiency_95ci": ci(h20_kv_s),
        "H20_KV_necessity_loss_mean": float(h20_kv_n["mean"]),
        "H20_KV_necessity_loss_95ci": ci(h20_kv_n),
        "H20_QKV_sufficiency_mean": float(h20_qkv_s["mean"]),
        "H20_QKV_sufficiency_95ci": ci(h20_qkv_s),
        "H20_QKV_necessity_loss_mean": float(h20_qkv_n["mean"]),
        "H20_QKV_necessity_loss_95ci": ci(h20_qkv_n),
        "H20_QKV_self_control_mean": float(h20_self["mean"]),
        "H20_QKV_same_state_cross_wording_mean": float(h20_same["mean"]),
        "H20_QKV_opposite_cross_wording_sufficiency_mean": float(
            h20_cross_s["mean"]
        ),
        "H20_QKV_opposite_cross_wording_necessity_loss_mean": float(
            h20_cross_n["mean"]
        ),
        "chain_KV_sufficiency_mean": float(chain_kv["mean"]),
        "chain_KV_sufficiency_95ci": ci(chain_kv),
        "chain_QKV_sufficiency_mean": float(chain_qkv["mean"]),
        "chain_QKV_sufficiency_95ci": ci(chain_qkv),
        "H20_paired_contrasts": contrast_df.to_dict(orient="records"),
        "routing_profile": profile.to_dict(orient="records"),
        "decision_rule": (
            "KV/QKV routing mediation is supported when H20 sufficiency and "
            "necessity are positive with CIs excluding zero, controls are small, "
            "and cross-wording preserves sign. Necessity without sufficiency "
            "supports context-conditioned routing rather than a portable state."
        ),
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
