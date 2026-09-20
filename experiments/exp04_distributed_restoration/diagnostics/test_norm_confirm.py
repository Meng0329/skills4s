#!/usr/bin/env python3
"""CONFIRM: hidden_states[28] is the POST-final-norm value (misaligned with layers[27] hook)."""
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
e0 = by_task[0][0]

text = exp04.render_prompt(tok, e0["messages"])
enc = tok(text, add_special_tokens=False, return_tensors="pt")
enc = {k: v.to(model.device) for k, v in enc.items()}

# Capture layer 27 (last decoder block) output via hook, plus final norm via hook
layer27 = model.model.layers[27]
captured = {}
def spy(mod, inputs, output):
    h = output[0] if isinstance(output, tuple) else output
    captured["layer27_out"] = h.detach().float().cpu()

def spy_norm(mod, inputs, output):
    captured["final_norm_out"] = output.detach().float().cpu().clone()

h1 = layer27.register_forward_hook(spy)
h2 = model.model.norm.register_forward_hook(spy_norm)
out = model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
h1.remove()
h2.remove()

h_l27 = captured["layer27_out"][0]            # pre-norm, layer 27 output
h_norm = captured["final_norm_out"][0]        # post final-norm
h_hs28 = out.hidden_states[28][0].float().cpu()     # tuple element 28

print(f"layer27 out shape: {tuple(h_l27.shape)}")
print(f"hidden_states[28] == final_norm(layer27): max abs diff = "
      f"{(h_hs28 - h_norm).abs().max().item():.6f}")
print(f"hidden_states[28] == layer27 (pre-norm):   max abs diff = "
      f"{(h_hs28 - h_l27).abs().max().item():.6f}")

# Confirm rms identity: h_norm == h_l27 / rms(h_l27) * weight
rms = (h_l27.pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt()
manual = h_l27 / rms * model.model.norm.weight.float()
print(f"manual RMSNorm(layer27) == hidden_states[28]: max abs diff = "
      f"{(manual - h_hs28).abs().max().item():.6f}")
