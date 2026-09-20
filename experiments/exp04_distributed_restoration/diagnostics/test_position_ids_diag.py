#!/usr/bin/env python3
"""Diagnose root cause: position_ids vs batch-size effect"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys, numpy as np, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
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

exp01b = exp04.load_exp01b()
manifest = exp04.load_manifest()
by_task = exp04.recreate_entries(exp01b, manifest)
ti = 0
entries = by_task[ti]
e0 = entries[0]

text = exp04.render_prompt(tok, e0["messages"])
prompt_ids = tok(text, add_special_tokens=False)["input_ids"]
plen = len(prompt_ids)
print(f"prompt len = {plen}")

# Candidate 0
c = e0["candidates"][0]
ci = tok(c, add_special_tokens=False)["input_ids"]
row = prompt_ids + ci
row_len = len(row)
print(f"candidate len = {len(ci)}, row len = {row_len}")

# === Test 1: batch=1 forward (no padding) ===
enc1 = tok(text, add_special_tokens=False, return_tensors="pt")
enc1 = {k: v.to(model.device) for k, v in enc1.items()}
with torch.inference_mode():
    out1 = model(**enc1, output_hidden_states=True, use_cache=False, return_dict=True)
h1 = out1.hidden_states[19][0].float().cpu()  # (plen, H)

# === Test 2: batch=2 left-padded, NO position_ids ===
pad = tok.pad_token_id
max_len2 = row_len + 5  # simulate 5 pad tokens for row0
ids_padded = [
    [pad]*5 + row,
    [pad]*(max_len2 - len(row)) + row,  # row1 (same as row if max_len2 == row_len)
]
attn_padded = [
    [0]*5 + [1]*row_len,
    [1]*max_len2,  # row1 full
]
input_ids2 = torch.tensor(ids_padded, dtype=torch.long, device=model.device)
attn2 = torch.tensor(attn_padded, dtype=torch.long, device=model.device)

# Ensure batch2 row1 matches row0 for consistent comparison
# Actually let's just use row0 padded and compare prompt part
print(f"\nbatch2: input_ids shape = {input_ids2.shape}")

with torch.inference_mode():
    out2 = model(input_ids=input_ids2, attention_mask=attn2,
                 output_hidden_states=True, use_cache=False, return_dict=True)
h2_nopos = out2.hidden_states[19][0].float().cpu()  # row0

# Prompt part starts at position 5 in batch2
h2_prompt_nopos = h2_nopos[5:5+plen]
diff_nopos = (h1 - h2_prompt_nopos).abs().max().item()
print(f"batch2 (no position_ids) vs batch1: max_diff = {diff_nopos:.6f}")

# === Test 3: batch=2 left-padded, WITH explicit position_ids ===
pos_ids = torch.tensor([
    list(range(row_len + 5)),  # row0: positions 0..272
    list(range(max_len2)),     # row1: positions 0..272
], dtype=torch.long, device=model.device)

# Wait, row0 has 5 pads + 273 tokens = 278 positions, row1 has 273 positions
# Let me recalculate: max_len2 = row_len + 5 = 273 + 5 = 278
max_len2 = max(row_len + 5, len(row))
print(f"max_len2 = {max_len2}")

ids_padded2 = [
    [pad]*5 + row + [pad]*(max_len2 - 5 - row_len),
    [pad]*(max_len2 - row_len) + row,
]
attn_padded2 = [
    [0]*5 + [1]*row_len + [0]*(max_len2 - 5 - row_len),
    [0]*(max_len2 - row_len) + [1]*row_len,
]
input_ids3 = torch.tensor(ids_padded2, dtype=torch.long, device=model.device)
attn3 = torch.tensor(attn_padded2, dtype=torch.long, device=model.device)

# Explicit position_ids: row0 real tokens start at position 0
# row0: pad positions don't have meaningful positions, real tokens are at 0..row_len-1
pos_ids3 = torch.tensor([
    [0]*5 + list(range(row_len)) + [0]*(max_len2 - 5 - row_len),  # row0
    list(range(max_len2)),  # row1
], dtype=torch.long, device=model.device)

print(f"input_ids3: {input_ids3.shape}, pos_ids3: {pos_ids3.shape}")
print(f"row0 mask sum: {attn3[0].sum().item()}, row1 mask sum: {attn3[1].sum().item()}")

with torch.inference_mode():
    out3 = model(input_ids=input_ids3, attention_mask=attn3, position_ids=pos_ids3,
                 output_hidden_states=True, use_cache=False, return_dict=True)
h3_pos = out3.hidden_states[19][0].float().cpu()
h3_prompt = h3_pos[5:5+plen]
diff_pos = (h1 - h3_prompt).abs().max().item()
print(f"batch2 (WITH position_ids) vs batch1: max_diff = {diff_pos:.6f}")

# === Test 4: batch=2 same sequence in both rows, no padding ===
# Put same sequence in both rows without padding
ids_nopad = [row, row]
attn_nopad = [[1]*row_len, [1]*row_len]
input_ids4 = torch.tensor(ids_nopad, dtype=torch.long, device=model.device)
attn4 = torch.tensor(attn_nopad, dtype=torch.long, device=model.device)

with torch.inference_mode():
    out4 = model(input_ids=input_ids4, attention_mask=attn4,
                 output_hidden_states=True, use_cache=False, return_dict=True)
h4 = out4.hidden_states[19][0].float().cpu()
diff_nopad_batch = (h1 - h4[:plen]).abs().max().item()
print(f"\nbatch=2 (no padding, same seq) vs batch=1: max_diff = {diff_nopad_batch:.6f}")

# === Test 5: compare all approaches ===
print(f"\n=== SUMMARY ===")
print(f"batch1 (baseline) vs:")
print(f"  batch2 no-pad same-seq:  {diff_nopad_batch:.6f}")
print(f"  batch2 left-pad no-pos:  {diff_nopos:.6f}")
print(f"  batch2 left-pad w-pos:   {diff_pos:.6f}")

# Check: what if we pad batch1 too but with a dummy sequence?
ids_dummy = [row, [pad]*(max_len2 - row_len) + row]
attn_dummy = [[1]*row_len, [0]*(max_len2 - row_len) + [1]*row_len]
input_ids5 = torch.tensor(ids_dummy, dtype=torch.long, device=model.device)
attn5 = torch.tensor(attn_dummy, dtype=torch.long, device=model.device)

with torch.inference_mode():
    out5 = model(input_ids=input_ids5, attention_mask=attn5,
                 output_hidden_states=True, use_cache=False, return_dict=True)
h5 = out5.hidden_states[19][0].float().cpu()
diff_dummy = (h1 - h5[:plen]).abs().max().item()
print(f"  batch2 dummy-pad row0 vs batch1: {diff_dummy:.6f}")
