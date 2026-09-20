#!/usr/bin/env python3
"""DEFINITIVE: (1) live-capture patch must be EXACTLY 0. (2) prompt-only vs prompt+cand per-layer cache diff."""
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
ti = 0
entries = by_task[ti]
e0 = entries[0]

text = exp04.render_prompt(tok, e0["messages"])
prompt_ids = tok(text, add_special_tokens=False)["input_ids"]
plen = len(prompt_ids)
ci0 = tok(e0["candidates"][0], add_special_tokens=False)["input_ids"]
row = prompt_ids + ci0
cand_len = len(ci0)
device = model.get_input_embeddings().weight.device

print(f"plen={plen}, cand={cand_len}, row={len(row)}")

# ---------- (1) live-capture self-patch: capture during SAME forward, patch with exact values ----------
input_ids = torch.tensor([row], dtype=torch.long, device=device)
attn = torch.ones_like(input_ids)

def live_score(patch_layer_values=None):
    """patch_layer_values: dict hidx -> (numpy (plen, H)) applied at prompt positions; None = no patch."""
    handles = []
    patch_keys = []  # stack (hidx, replacement tensor)
    with ExitStack() as stack:
        if patch_layer_values:
            for hidx, vec in patch_layer_values.items():
                positions = torch.tensor(np.arange(0, plen, dtype=np.int64).reshape(1, -1),
                                         dtype=torch.long, device=device)
                replacement = torch.tensor(vec.reshape(1, plen, -1), dtype=torch.float32, device=device)
                # need stack-based hooks; use manual registration
        out = model(input_ids=input_ids, attention_mask=attn,
                    use_cache=False, return_dict=True)
    lp = torch.log_softmax(out.logits.float(), dim=-1)[0]
    pred_pos = torch.arange(plen - 1, plen + cand_len - 1, device=device)
    targ_pos = torch.arange(plen, plen + cand_len, device=device)
    sc = float(lp[pred_pos, input_ids[0, targ_pos]].mean().cpu())
    return sc

# Instead: capture via hooks DURING score forward, then patch a SECOND forward with those exact same
# layer outputs. Two separate forwards with identical input & shape -> deterministic? Test that too.
def fwd_with_hooks(capture_list):
    caps = {}
    handles = []
    def make_fn(hidx):
        def fn(mod, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            caps[hidx] = h.detach().float().cpu()  # (1, seq, H)
        return fn
    for hidx in range(19, 29):
        handles.append(layers[hidx - 1].register_forward_hook(make_fn(hidx)))
    out = model(input_ids=input_ids, attention_mask=attn, use_cache=False, return_dict=True)
    for h in handles:
        h.remove()
    return out, caps

# Forward 1: capture live layer outputs at 19..28
out1, caps1 = fwd_with_hooks(None)
lp1 = torch.log_softmax(out1.logits.float(), dim=-1)[0]
pred_pos = torch.arange(plen - 1, plen + cand_len - 1, device=device)
targ_pos = torch.arange(plen, plen + cand_len, device=device)
base = float(lp1[pred_pos, input_ids[0, targ_pos]].mean().cpu())
print(f"baseline score (forward 1) = {base:.6f}")

# Forward 2: same input, no hooks — check determinism
out2 = model(input_ids=input_ids, attention_mask=attn, use_cache=False, return_dict=True)
lp2 = torch.log_softmax(out2.logits.float(), dim=-1)[0]
base2 = float(lp2[pred_pos, input_ids[0, targ_pos]].mean().cpu())
print(f"baseline score (forward 2, retry) = {base2:.6f}  (determinism delta) = {base2 - base:.6e}")

# Forward 3: patch with values captured in forward 1 at ALL positions incl candidate part (full identity)
import contextlib
def patch_forward(full_values):
    """full_values: dict hidx -> tensor (1, seq, H). Patch full sequence = identity test."""
    handles = []
    def make_fn(hidx):
        def fn(mod, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            return full_values[hidx].to(h.device, dtype=h.dtype)
        return fn
    for hidx in range(19, 29):
        handles.append(layers[hidx - 1].register_forward_hook(make_fn(hidx)))
    out = model(input_ids=input_ids, attention_mask=attn, use_cache=False, return_dict=True)
    for hnd in handles:
        hnd.remove()
    return out

caps_t = {hidx: caps1[hidx] for hidx in range(19, 29)}  # (1, 280, H)
out3 = patch_forward(caps_t)
lp3 = torch.log_softmax(out3.logits.float(), dim=-1)[0]
id_score = float(lp3[pred_pos, input_ids[0, targ_pos]].mean().cpu())
print(f"identity-patch (full-value replace) score = {id_score:.6f}, delta vs base = {id_score - base:.6e}")

# ---------- (2) prompt-only vs prompt+candidate cache diff per layer ----------
# capture prompt-only
enc = tok(text, add_special_tokens=False, return_tensors="pt")
enc = {k: v.to(device) for k, v in enc.items()}
outp = model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
print(f"\n{'hidx':<6}{'prompt-only-vs-livediff max':>28}{'fwd1-vs-fwd2 determinism max':>30}")
for hidx in range(19, 29):
    h_p = outp.hidden_states[hidx][0].float().cpu()          # (plen, H)
    h_l = caps1[hidx][0, :plen].float().cpu()                # live, prompt part
    d_cache = (h_p - h_l).abs().max().item()
    # determinism: are the two hooks identical across fwd1 and... we only have fwd1 caps; use outp vs caps on prompt part of forward 2 via another capture?
    print(f"{hidx:<6}{d_cache:>28.6f}")

# rel norm per layer for context
print(f"\nrel diff = max_abs / rms(live):")
for hidx in range(19, 29):
    h_p = outp.hidden_states[hidx][0].float().cpu()
    h_l = caps1[hidx][0, :plen].float().cpu()
    d = (h_p - h_l).abs().max().item()
    print(f"{hidx:<6}{d:>28.6f}   rms={h_l.pow(2).mean().sqrt().item():.3f}")