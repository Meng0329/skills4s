#!/usr/bin/env python3
"""Clean isolation: batch-size effect vs padding effect vs seq-len effect"""
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

exp01b = exp04.load_exp01b()
manifest = exp04.load_manifest()
by_task = exp04.recreate_entries(exp01b, manifest)
ti = 0
entries = by_task[ti]
e0 = entries[0]

text = exp04.render_prompt(tok, e0["messages"])
prompt_ids = tok(text, add_special_tokens=False)["input_ids"]
plen = len(prompt_ids)

ci0 = tok(e0["candidates"][0], add_special_tokens=False)["input_ids"]
ci1 = tok(e0["candidates"][1], add_special_tokens=False)["input_ids"]
row0 = prompt_ids + ci0
row1 = prompt_ids + ci1
print(f"plen={plen} row0={len(row0)} row1={len(row1)}")

def fwd(ids_rows, attn_rows, label):
    input_ids = torch.tensor(ids_rows, dtype=torch.long, device=model.device)
    attn = torch.tensor(attn_rows, dtype=torch.long, device=model.device)
    out = model(input_ids=input_ids, attention_mask=attn,
                output_hidden_states=True, use_cache=False, return_dict=True)
    h = out.hidden_states[19].float().cpu()
    print(f"  {label}: shape={tuple(h.shape)}")
    return h

# E1: batch=1, row0 alone, NO padding
h1 = fwd([row0], [[1]*len(row0)], "E1 batch1 row0 alone")
# E2: batch=2, [row0, row1], right-padded (exactly like score_one)
max_len = max(len(row0), len(row1))
ids2 = [row0 + [tok.pad_token_id]*(max_len-len(row0)),
        row1 + [tok.pad_token_id]*(max_len-len(row1))]
att2 = [[1]*len(row0)+[0]*(max_len-len(row0)),
        [1]*len(row1)+[0]*(max_len-len(row1))]
h2 = fwd(ids2, att2, "E2 batch2 right-pad")
# E3: batch=2, [row0, row0] identical rows, right-padded
ids3 = [ids2[0], ids2[0]]
att3 = [att2[0], att2[0]]
h3 = fwd(ids3, att3, "E3 batch2 [row0,row0] right-pad")
# E4: batch=2, [row0, row1] but NO padding (max_len == len(row1), both rows padded to max_len)
# same as E2 since row1 already max. Skip.
# E5: batch=1, row0 + trailing pads (row0 padded to max_len)
ids5 = [ids2[0]]
att5 = [att2[0]]
h5 = fwd(ids5, att5, "E5 batch1 row0 padded")
# E6: batch=2, [row0, row1] unpadded version: build rows so both have no pad
# row0' = row0 + [pad]*diff so both equal len -> that's E2 again
# Try: batch=2, both rows = row0 content but row1 is a DIFFERENT candidate at same len?
# Make row1 same length as row0 by truncating? No - instead use two DIFFERENT prompts of equal len.
# E6: batch=1 row1 alone (prompt+cand1)
h6 = fwd([row1], [[1]*len(row1)], "E6 batch1 row1 alone")
# E7: batch=2, [row1, row0] order swapped, right-padded
ids7 = [ids2[1], ids2[0]]
att7 = [att2[1], att2[0]]
h7 = fwd(ids7, att7, "E7 batch2 [row1,row0]")

print(f"\n{'comparison':<45}{'max_diff':>12}")
def cmp(label, a, b, slc):
    d = (a[slc] - b[slc]).abs().max().item()
    print(f"{label:<45}{d:>12.6f}")

pslc = (0, slice(0, plen))  # prompt part of row0
cmp("E1 vs E2  batch1 vs batch2 right-pad", h1[0], h2[0], slice(0, plen))
cmp("E1 vs E3  batch1 vs batch2 [row0,row0]", h1[0], h3[0], slice(0, plen))
cmp("E1 vs E5  batch1 vs batch1+padded", h1[0], h5[0], slice(0, plen))
cmp("E2 vs E3  batch2 [r0,r1] vs [r0,r0]", h2[0], h3[0], slice(0, plen))
cmp("E2 vs E7  batch2 order swap", h2[0], h7[0], slice(0, plen))
cmp("E6 vs E2  batch1 row1 vs batch2 row1", h6[0], h2[1], slice(0, plen))
cmp("E1 vs E6  row0 vs row1 prompt parts (both batch1)", h1[0], h6[0], slice(0, plen))
