#!/usr/bin/env python3
"""Verify: batch=1 no-padding score_one recovers self-patch ~0 and keeps real transfer."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys, numpy as np, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from exp04_distributed_restoration import run as exp04
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import ExitStack

model_dir = Path("/data/mzb/skills4s/models/Qwen2.5-Coder-7B-Instruct")
tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    str(model_dir), local_files_only=True, trust_remote_code=False,
    torch_dtype="auto", device_map="auto", low_cpu_mem_usage=True,
)
model.eval()
layers = exp04.get_layers(model)

exp01b = exp04.load_exp01b()
manifest = exp04.load_manifest()
by_task = exp04.recreate_entries(exp01b, manifest)


def capture_f32(model, tok, entries, hidden_indices):
    device = model.get_input_embeddings().weight.device
    cache = []
    for e in entries:
        text = exp04.render_prompt(tok, e["messages"])
        enc = tok(text, add_special_tokens=False, return_tensors="pt")
        ids = enc["input_ids"][0].tolist()
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
        states = {
            h: out.hidden_states[h][0].detach().float().cpu().numpy().astype(np.float32)
            for h in hidden_indices
        }
        cache.append({"ids": ids, "states": states})
        del out
    return cache


def get_patch_params(recipient, rec_cache, donor_cache, hidden_indices, span):
    """Compute (common, k, rec_pos_np, per_layer_repl) for batch=1 hook."""
    prompt_text = exp04.render_prompt(tok, recipient["messages"])
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    plen = len(prompt_ids)
    common = exp04.longest_common_suffix(rec_cache["ids"], donor_cache["ids"])
    if common <= 0:
        raise RuntimeError("No aligned common suffix.")
    k = 1 if span == "last1" else common
    rec_pos_np = np.arange(plen - k, plen, dtype=np.int64)
    dlen = len(donor_cache["ids"])
    donor_pos_np = np.arange(dlen - k, dlen, dtype=np.int64)
    per_layer = {
        hidx: donor_cache["states"][hidx][donor_pos_np]
        for hidx in hidden_indices
    }
    return common, k, rec_pos_np, per_layer


def score_one_b1(model, tok, layers, recipient, rec_cache, donor_cache=None,
                 hidden_indices=None, span=None):
    """batch=1 per-candidate forward, no padding."""
    prompt_text = exp04.render_prompt(tok, recipient["messages"])
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    plen = len(prompt_ids)
    cand_ids = [tok(c, add_special_tokens=False)["input_ids"]
                for c in recipient["candidates"]]

    common, k, rec_pos_np, per_layer = None, 0, None, None
    if donor_cache is not None:
        common, k, rec_pos_np, per_layer = get_patch_params(
            recipient, rec_cache, donor_cache, hidden_indices, span)

    device = model.get_input_embeddings().weight.device
    scores = []
    for ci in cand_ids:
        row = prompt_ids + ci
        c = len(ci)
        input_ids = torch.tensor([row], dtype=torch.long, device=device)
        attention = torch.ones_like(input_ids)
        with ExitStack() as stack:
            if donor_cache is not None:
                positions = torch.tensor(
                    rec_pos_np.reshape(1, -1), dtype=torch.long, device=device)
                for hidx in hidden_indices:
                    repl = per_layer[hidx]
                    replacement = torch.tensor(
                        repl.reshape(1, k, -1), dtype=torch.float32, device=device)
                    stack.enter_context(
                        exp04.patch_hook(layers[hidx - 1], positions, replacement))
            out = model(input_ids=input_ids, attention_mask=attention,
                        use_cache=False, return_dict=True)
        lp = torch.log_softmax(out.logits.float(), dim=-1)[0]
        pred_pos = torch.arange(plen - 1, plen + c - 1, device=device)
        targ_pos = torch.arange(plen, plen + c, device=device)
        targets = input_ids[0, targ_pos]
        scores.append(float(lp[pred_pos, targets].mean().cpu()))
    return {
        "test_mean_logprob": scores[0],
        "impl_mean_logprob": scores[1],
        "margin": scores[1] - scores[0],
        "common_suffix_len": common,
        "patched_token_count": k,
    }


needed = list(range(19, 29))

# ---- Compare original vs batch1 on task 0 ----
ti = 0
entries = by_task[ti]
maps = exp04.donor_maps(entries)
cache = capture_f32(model, tok, entries, needed)

print(f"=== Task {ti} comparison (original batch2 vs new batch1) ===")
print(f"{'entry':<26} {'orig_base':>10} {'b1_base':>10} {'orig_self_patch':>16} {'b1_self_patch':>16}")
for i, e in enumerate(entries):
    base_b2 = exp04.score_one(model, tok, layers, e, cache[i])
    base_b1 = score_one_b1(model, tok, layers, e, cache[i])
    self_b2 = exp04.score_one(model, tok, layers, e, cache[i], cache[i], needed, "common")
    self_b1 = score_one_b1(model, tok, layers, e, cache[i], cache[i], needed, "common")
    d_b2 = self_b2["margin"] - base_b2["margin"]
    d_b1 = self_b1["margin"] - base_b1["margin"]
    print(f"{e['condition']:<26} {base_b2['margin']:>10.5f} {base_b1['margin']:>10.5f} "
          f"{d_b2:>16.6f} {d_b1:>16.6f}")

# ---- Real transfer effect in batch1 ----
print(f"\n=== Real opposite_same_wording transfer (batch1) ===")
for i, rec in enumerate(entries):
    di = maps[i]["opposite_same_wording"]
    donor = entries[di]
    base = score_one_b1(model, tok, layers, rec, cache[i])
    p = score_one_b1(model, tok, layers, rec, cache[i], cache[di], needed, "common")
    raw = p["margin"] - base["margin"]
    sign = 1.0 if donor["label"] == exp04.LABEL_IMPL else -1.0
    print(f"{rec['condition']:<26} donor={donor['condition']:<26} signed={sign * raw:>10.6f}")

# ---- Aggregate self-patch noise over several tasks ----
print(f"\n=== Self-patch noise across first 4 tasks (batch1) ===")
self_deltas = []
for t in sorted(by_task)[:4]:
    ents = by_task[t]
    ch = capture_f32(model, tok, ents, needed)
    bases = [score_one_b1(model, tok, layers, e, ch[i]) for i, e in enumerate(ents)]
    for i, e in enumerate(ents):
        sp = score_one_b1(model, tok, layers, e, ch[i], ch[i], needed, "common")
        self_deltas.append(sp["margin"] - bases[i]["margin"])
    del ch
self_deltas = np.array(self_deltas)
print(f"n = {len(self_deltas)}")
print(f"self-patch mean  = {self_deltas.mean():.6f}")
print(f"self-patch |mean|  = {np.abs(self_deltas).mean():.6f}")
print(f"self-patch max   = {self_deltas.max():.6f}")
print(f"all |delta| < 0.01: {bool(np.all(np.abs(self_deltas) < 0.01))}")