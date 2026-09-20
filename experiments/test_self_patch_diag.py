#!/usr/bin/env python3
"""Diagnose: why is self-patch delta ~0.2 when it should be ~0?"""
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
print(f"padding_side = {tok.padding_side}, pad_token = {tok.pad_token_id}")

model = AutoModelForCausalLM.from_pretrained(
    str(model_dir), local_files_only=True, trust_remote_code=False,
    torch_dtype="auto", device_map="auto", low_cpu_mem_usage=True,
)
model.eval()
layers = exp04.get_layers(model)

exp01b = exp04.load_exp01b()
manifest = exp04.load_manifest()
by_task = exp04.recreate_entries(exp01b, manifest)
needed = [19]
ti = 0
entries = by_task[ti]

# --- Part 1: does prompt-only forward == prompt+candidate forward (prompt part)? ---
e0 = entries[0]
text = exp04.render_prompt(tok, e0["messages"])
prompt_ids = tok(text, add_special_tokens=False)["input_ids"]
plen = len(prompt_ids)
print(f"\nprompt len = {plen}")

# Forward prompt only
enc = tok(text, add_special_tokens=False, return_tensors="pt")
enc = {k: v.to(model.device) for k, v in enc.items()}
out1 = model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
h_prompt = out1.hidden_states[19][0].float().cpu()  # (plen, H)

# Forward prompt + candidate (like score_one does)
c = e0["candidates"][0]
ci = tok(c, add_special_tokens=False)["input_ids"]
row = prompt_ids + ci
max_len = len(row)
input_ids = torch.tensor([row], dtype=torch.long, device=model.device)
attn = torch.ones_like(input_ids)
out2 = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True,
             use_cache=False, return_dict=True)
h_cand = out2.hidden_states[19][0].float().cpu()  # (plen+len(ci), H)

# Compare prompt part
diff = (h_prompt - h_cand[:plen]).abs().max().item()
print(f"prompt-only vs prompt+candidate hidden[19] max abs diff (prompt part): {diff:.6f}")

# --- Part 2: what positions does score_one patch for self-patch? ---
cache = exp04.capture_prompt_states(model, tok, entries, needed)
rec_cache = cache[0]
donor_cache = cache[0]  # self
common = exp04.longest_common_suffix(rec_cache["ids"], donor_cache["ids"])
print(f"\nself-patch: common suffix len = {common} (prompt len = {plen})")
print(f"-> self-patch patches {common} token positions (the ENTIRE prompt!)")

# --- Part 3: score_one baseline vs patched - inspect raw margins ---
base = exp04.score_one(model, tok, layers, e0, cache[0])
patched = exp04.score_one(model, tok, layers, e0, cache[0], cache[0], needed, "common")
print(f"\nbaseline margin  = {base['margin']:.6f}  (test={base['test_mean_logprob']:.4f}, impl={base['impl_mean_logprob']:.4f})")
print(f"self-patch margin= {patched['margin']:.6f}  (test={patched['test_mean_logprob']:.4f}, impl={patched['impl_mean_logprob']:.4f})")
print(f"delta = {patched['margin'] - base['margin']:.6f}")

# --- Part 4: check if hook actually changes anything numerically ---
# Recompute patched with hook but replace with EXACT same values captured during this forward?
# Instead: check hidden states at layer 19 pre-hook vs cache value
layers19 = layers[18]
captured = []

def spy_hook(module, inputs, output):
    if isinstance(output, tuple):
        h = output[0]
    else:
        h = output
    captured.append(h[0].float().cpu())

handle = layers19.register_forward_hook(spy_hook)
out = model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
handle.remove()

h_hook = captured[0]  # (plen, H) at layer 18 output == hidden_states[19]
h_cache = cache[0]["states"][19]  # float16 cached
print(f"\nlayer18 hook output max abs diff vs cache: {(h_hook - h_cache.astype(np.float32)).abs().max().item():.6f}")
print(f"hook output dtype: {h_hook.dtype}, cache dtype: {h_cache.dtype}")

# Is h_hook == h_prompt?
print(f"hook output vs out1.hidden_states[19]: {(h_hook - h_prompt).abs().max().item():.6f}")
