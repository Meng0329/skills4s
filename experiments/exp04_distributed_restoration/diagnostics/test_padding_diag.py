#!/usr/bin/env python3
"""Root-cause: does batch=2 padded forward change hidden[19] vs cache (batch=1)?"""
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
print(f"padding_side={tok.padding_side}")

model = AutoModelForCausalLM.from_pretrained(
    str(model_dir), local_files_only=True, trust_remote_code=False,
    torch_dtype="auto", device_map="auto", low_cpu_mem_usage=True,
)
model.eval()
print(f"model dtype: {model.dtype}")

exp01b = exp04.load_exp01b()
manifest = exp04.load_manifest()
by_task = exp04.recreate_entries(exp01b, manifest)
needed = [19]
ti = 0
entries = by_task[ti]
e0 = entries[0]

text = exp04.render_prompt(tok, e0["messages"])
prompt_ids = tok(text, add_special_tokens=False)["input_ids"]
plen = len(prompt_ids)
print(f"prompt len = {plen}")

# Cache (batch=1, no padding) - emulate capture_prompt_states but keep full precision
enc = tok(text, add_special_tokens=False, return_tensors="pt")
enc = {k: v.to(model.device) for k, v in enc.items()}
with torch.inference_mode():
    out1 = model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
h_cache19 = out1.hidden_states[19][0].float().cpu().numpy()  # (plen, H)
print(f"cache hidden[19] {h_cache19.shape}, dtype float32")

# score_one forwards BOTH candidates, padded to max_len (batch=2)
cand_ids = [tok(c, add_special_tokens=False)["input_ids"] for c in e0["candidates"]]
rows = [prompt_ids + ci for ci in cand_ids]
max_len = max(map(len, rows))
pad = tok.pad_token_id
ids_rows = [r + [pad] * (max_len - len(r)) for r in rows]
attn_rows = [[1] * len(r) + [0] * (max_len - len(r)) for r in rows]
print(f"max_len = {max_len} (prompt {plen} + candidates {len(cand_ids[0])}/{len(cand_ids[1])})")
print(f"row lens: {[len(r) for r in rows]}")

input_ids = torch.tensor(ids_rows, dtype=torch.long, device=model.device)
attn = torch.tensor(attn_rows, dtype=torch.long, device=model.device)

# Capture hook on layer 18 output (== hidden_states[19])
cap = {}
def spy(mod, inputs, output):
    if isinstance(output, tuple):
        h = output[0]
    else:
        h = output
    cap["h"] = h.float().cpu()
handle = model.model.layers[18].register_forward_hook(spy)

with torch.inference_mode():
    out2 = model(input_ids=input_ids, attention_mask=attn,
                 output_hidden_states=True, use_cache=False, return_dict=True)
handle.remove()

h_hook = cap["h"].numpy()      # (2, max_len, H)
h_fwd19 = out2.hidden_states[19][0].float().cpu().numpy()  # batch 0

# Compare per-token: row0 prompt part
diff_hook_vs_fwd = np.abs(h_hook[0][:plen] - h_fwd19[:plen]).max()
diff_cache_vs_fwd = np.abs(h_cache19[:plen] - h_fwd19[:plen]).max()
diff_cache_vs_hook = np.abs(h_cache19[:plen] - h_hook[0][:plen]).max()
print(f"\nrow0: hook vs hidden[19] out2    diff = {diff_hook_vs_fwd:.6f}  (should be ~0)")
print(f"row0: cache(batch1) vs fwd(batch2) diff = {diff_cache_vs_fwd:.6f}")
print(f"row0: cache(batch1) vs hook       diff = {diff_cache_vs_hook:.6f}")

# Also row1 (second candidate)
diff_row1 = np.abs(h_cache19[:plen] - h_hook[1][:plen]).max()
print(f"row1: cache vs hook diff = {diff_row1:.6f}")

# Padding side check: where are pads in row 0?
print(f"\nrow0 mask head: {attn_rows[0][:8]}, tail: {attn_rows[0][-8:]}")
print(f"row1 mask head: {attn_rows[1][:8]}, tail: {attn_rows[1][-8:]}")