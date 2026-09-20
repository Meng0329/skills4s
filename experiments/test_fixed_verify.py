#!/usr/bin/env python3
"""Verify fixed exp04: self-patch should be ~0, real transfer should remain."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys, numpy as np, torch
from pathlib import Path

sys.path.insert(0, "/data/mzb/skills4s/experiments")
from exp04_distributed_restoration import run as exp04
from transformers import AutoModelForCausalLM, AutoTokenizer

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

needed = list(range(19, 29))  # primary config

print("=== TASK 0: self-patch & real transfer ===")
ti = 0
entries = by_task[ti]
maps = exp04.donor_maps(entries)
cache = exp04.capture_prompt_states(model, tok, entries, needed)

print(f"\n{'entry':<26} {'baseline':>10} {'self_patch_delta':>18}")
for i, e in enumerate(entries):
    base = exp04.score_one(model, tok, layers, e, cache[i])
    sp   = exp04.score_one(model, tok, layers, e, cache[i], cache[i], needed, "common")
    d = sp["margin"] - base["margin"]
    print(f"{e['condition']:<26} {base['margin']:>10.5f} {d:>18.8f}")

print(f"\n{'recipient':<26} {'donor':<26} {'signed_transfer':>16}")
for i, rec in enumerate(entries):
    di = maps[i]["opposite_same_wording"]
    donor = entries[di]
    base = exp04.score_one(model, tok, layers, rec, cache[i])
    p    = exp04.score_one(model, tok, layers, rec, cache[i], cache[di], needed, "common")
    raw = p["margin"] - base["margin"]
    sign = 1.0 if donor["label"] == exp04.LABEL_IMPL else -1.0
    print(f"{rec['condition']:<26} {donor['condition']:<26} {sign*raw:>16.6f}")

print(f"\n=== FIRST 4 TASKS: self-patch summary ===")
deltas = []
for t in sorted(by_task)[:4]:
    ents = by_task[t]
    ch = exp04.capture_prompt_states(model, tok, ents, needed)
    bases = [exp04.score_one(model, tok, layers, e, ch[i]) for i, e in enumerate(ents)]
    for i, e in enumerate(ents):
        sp = exp04.score_one(model, tok, layers, e, ch[i], ch[i], needed, "common")
        deltas.append(sp["margin"] - bases[i]["margin"])
    del ch
d = np.array(deltas)
print(f"n = {len(d)}")
print(f"self-patch mean  = {d.mean():.8f}")
print(f"self-patch |mean|= {np.abs(d).mean():.8f}")
print(f"self-patch max   = {d.max():.8f}")
print(f"all |delta| < 0.005: {bool(np.all(np.abs(d) < 0.005))}")
