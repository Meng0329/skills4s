#!/usr/bin/env python3
"""Quick sanity check: is the large self-patch effect a float16 precision bug?"""
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

needed = list(range(19, 29))
ti = 0
entries = by_task[ti]

# Float32 cache (no precision loss)
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

# Original float16 cache
cache_f16 = exp04.capture_prompt_states(model, tok, entries, needed)
cache_f32 = capture_f32(model, tok, entries, needed)

# Baselines from f32 cache
baselines = [exp04.score_one(model, tok, layers, e, cache_f32[i])
             for i, e in enumerate(entries)]

print("=== Task 0 self-patch comparison ===")
print(f"{'entry':<12} {'base_margin':>12} {'f16_delta':>12} {'f32_delta':>12}")
for i, e in enumerate(entries):
    p16 = exp04.score_one(model, tok, layers, e, cache_f16[i], cache_f16[i], needed, "common")
    p32 = exp04.score_one(model, tok, layers, e, cache_f32[i], cache_f32[i], needed, "common")
    d16 = p16["margin"] - baselines[i]["margin"]
    d32 = p32["margin"] - baselines[i]["margin"]
    print(f"{e['condition']:<12} {baselines[i]['margin']:>12.6f} {d16:>12.6f} {d32:>12.6f}")

# Also test real (opposite_same_wording) patch in f32
print("\n=== Opposite-same-wording patch (real) f32 ===")
maps = exp04.donor_maps(entries)
for i, e in enumerate(entries):
    di = maps[i]["opposite_same_wording"]
    donor = entries[di]
    p = exp04.score_one(model, tok, layers, e, cache_f32[i], cache_f32[di], needed, "common")
    raw = p["margin"] - baselines[i]["margin"]
    sign = 1.0 if donor["label"] == 1 else -1.0
    print(f"{e['condition']:<12} donor={donor['condition']:<24} signed={sign*raw:>12.6f}")
