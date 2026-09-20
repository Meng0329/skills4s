#!/usr/bin/env python3
"""Quick sanity check: float16 cache precision bug hypothesis."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp04_distributed_restoration"))

# Monkey-patch the cache to use float32 instead of float16
from exp04_distributed_restoration import run as exp04
original_capture = exp04.capture_prompt_states

def capture_float32(model, tok, entries, hidden_indices):
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

# Import and monkey-patch
sys.modules["exp04_distributed_restoration.run"] = type(sys)("temp")
sys.modules["exp04_distributed_restoration.run"] = exp04
exp04.capture_prompt_states = capture_float32

from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = Path(__file__).resolve().parents[1].parents[1] / "models" / "Qwen2.5-Coder-7B-Instruct"
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

# Load EXP01b data
exp01b = exp04.load_exp01b()
manifest = exp04.load_manifest()
by_task = exp04.recreate_entries(exp01b, manifest)

needed = [19,20,21,22,23,24,25,26,27,28]

# Test on task 0 only
ti = 0
entries = by_task[ti]
maps = exp04.donor_maps(entries)

print("=== Task 0: Self-patch comparison ===")
print(f"Entries: {len(entries)} conditions")

# Float32 cache
cache_f32 = capture_float32(model, tok, entries, needed)
# Original float16 cache
cache_f16 = original_capture(model, tok, entries, needed)

# Compute baselines
baselines = [
    exp04.score_one(model, tok, layers, e, cache_f32[i])
    for i, e in enumerate(entries)
]

print("\nBaseline margins:")
for i, b in enumerate(baselines):
    print(f"  {entries[i]['condition']}: margin = {b['margin']:.6f}")

# Self-patch with float32 cache
self_effects_f32 = []
for i, rec in enumerate(entries):
    p = exp04.score_one(model, tok, layers, rec, cache_f32[i], cache_f32[i],
                        needed, "common")
    raw = p["margin"] - baselines[i]["margin"]
    self_effects_f32.append(raw)
    print(f"\n  Self-patch f32 (entry {i}): baseline={baselines[i]['margin']:.6f}, "
          f"patched={p['margin']:.6f}, delta={raw:.6f}")

# Self-patch with float16 cache
self_effects_f16 = []
for i, rec in enumerate(entries):
    p = exp04.score_one(model, tok, layers, rec, cache_f16[i], cache_f16[i],
                        needed, "common")
    raw = p["margin"] - baselines[i]["margin"]
    self_effects_f16.append(raw)
    print(f"\n  Self-patch f16 (entry {i}): baseline={baselines[i]['margin']:.6f}, "
          f"patched={p['margin']:.6f}, delta={raw:.6f}")

print(f"\n=== SUMMARY ===")
print(f"Self-patch mean |float32|: {np.mean(np.abs(self_effects_f32)):.6f}")
print(f"Self-patch mean |float16|: {np.mean(np.abs(self_effects_f16)):.6f}")
print(f"Self-patch raw mean float32: {np.mean(self_effects_f32):.6f}")
print(f"Self-patch raw mean float16: {np.mean(self_effects_f16):.6f}")
