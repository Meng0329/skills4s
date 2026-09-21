#!/usr/bin/env python3
from __future__ import annotations
import os
os.environ.setdefault("HF_HUB_OFFLINE","1")
os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
os.environ.setdefault("HF_DATASETS_OFFLINE","1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY","1")

import argparse, hashlib, importlib.util, json, platform, random, subprocess
from contextlib import ExitStack, contextmanager
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch, transformers
from sklearn.model_selection import GroupKFold
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT=Path(__file__).resolve().parents[2]
MODELS_ROOT=REPO_ROOT/"models"
EXP01B_CODE=REPO_ROOT/"experiments"/"exp01b_counterbalanced_next_state"/"run.py"
EXP01B_MANIFEST=REPO_ROOT/"outputs"/"exp01b_counterbalanced_next_state"/"run_manifest.json"
DEFAULT_OUT=REPO_ROOT/"outputs"/"exp07_attention_heads"

LABEL_TEST=0
LABEL_IMPL=1
EARLY_HIDDEN=list(range(15,21))
ATTN_HIDDEN=list(range(21,29))
PRIMARY_K=16
EXPLORATORY_K=[4,32,64]
RANDOM_CONTROLS=5

def sha256_file(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
    return h.hexdigest()

def git_commit():
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO_ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:return None

def resolve_model_dir(arg):
    if arg:
        p=Path(arg).expanduser()
        if not p.is_absolute():
            p1=(REPO_ROOT/p).resolve(); p2=(MODELS_ROOT/p).resolve()
            p=p1 if p1.exists() else p2
        if not (p/"config.json").is_file(): raise FileNotFoundError(p)
        return p.resolve()
    cs=sorted({p.parent.resolve() for p in MODELS_ROOT.rglob("config.json")})
    if len(cs)!=1: raise RuntimeError(f"Expected exactly one local model, found {len(cs)}")
    return cs[0]

def load_exp01b():
    spec=importlib.util.spec_from_file_location("exp01b",EXP01B_CODE)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def load_manifest(): return json.loads(EXP01B_MANIFEST.read_text(encoding="utf-8"))

def get_attr_any(obj,names):
    for n in names:
        if hasattr(obj,n): return getattr(obj,n)
    raise AttributeError(names)

def api(m):
    return dict(
        make_tasks=get_attr_any(m,["make_tasks"]),
        history=get_attr_any(m,["build_history","history"]),
        messages=get_attr_any(m,["build_messages","messages"]),
        conditions=get_attr_any(m,["CONDITIONS","CONDS"]),
        skills=get_attr_any(m,["SKILLS"]),
    )

def recreate_entries(m,manifest):
    a=api(m); tasks=a["make_tasks"](int(manifest["num_tasks"]),int(manifest["seed"])); out={}
    for task in tasks:
        ti=int(task["task_idx"]); hist=a["history"](task,ti); es=[]
        for cond,wording,label in a["conditions"]:
            es.append(dict(
                task_idx=ti,task_id=task["task_id"],condition=cond,wording=wording,label=int(label),
                messages=a["messages"](task,hist,a["skills"][cond]),
                candidates=[f"read_file('{task['test_path']}')",f"read_file('{task['src_path']}')"]
            ))
        out[ti]=es
    return out

def donor_maps(entries):
    lookup={(e["wording"],e["label"]):i for i,e in enumerate(entries)}; out={}
    for i,e in enumerate(entries):
        opp=LABEL_IMPL if e["label"]==LABEL_TEST else LABEL_TEST
        ow="paraphrase" if e["wording"]=="canonical" else "canonical"
        out[i]=dict(
            opposite_same_wording=lookup[(e["wording"],opp)],
            opposite_cross_wording=lookup[(ow,opp)]
        )
    return out

def lcs(a,b):
    k=0
    while k<min(len(a),len(b)) and a[-1-k]==b[-1-k]: k+=1
    return k

def render(tok,messages):
    return tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)

def get_layers(model):
    return model.model.layers

@contextmanager
def residual_patch_hook(layer,positions,replacements):
    def fn(module,inputs,output):
        if isinstance(output,tuple): h,rest=output[0],output[1:]
        else: h,rest=output,None
        h=h.clone(); b=torch.arange(h.shape[0],device=h.device).unsqueeze(1)
        h[b,positions.to(h.device),:]=replacements.to(h.device,dtype=h.dtype)
        return h if rest is None else (h,)+rest
    hd=layer.register_forward_hook(fn)
    try:yield
    finally:hd.remove()

@contextmanager
def head_patch_hook(o_proj,positions,head_idx,replacements,num_heads,head_dim):
    def fn(module,inputs):
        x=inputs[0].clone(); shape=x.shape; xh=x.view(*shape[:-1],num_heads,head_dim)
        pos=positions.to(x.device); rep=replacements.to(x.device,dtype=x.dtype)
        for row in range(xh.shape[0]):
            if head_idx is None:
                xh[row,pos[row],:,:]=rep[row]
            else:
                idx=head_idx.to(x.device)
                for pi,tp in enumerate(pos[row]):
                    xh[row,tp,idx,:]=rep[row,pi]
        return (xh.reshape(*shape),)+tuple(inputs[1:])
    hd=o_proj.register_forward_pre_hook(fn)
    try:yield
    finally:hd.remove()

def tok_entry(tok,e):
    t=render(tok,e["messages"]); ids=tok(t,add_special_tokens=False)["input_ids"]; return ids

@torch.inference_mode()
def capture(model,tok,layers,e,max_suffix,num_heads,head_dim,early_donor=None,early_k=None):
    device=model.get_input_embeddings().weight.device
    text=render(tok,e["messages"]); enc=tok(text,add_special_tokens=False,return_tensors="pt")
    ids=enc["input_ids"][0].tolist(); plen=len(ids); enc={k:v.to(device) for k,v in enc.items()}
    store={"early":{},"heads":{}}; handles=[]
    with ExitStack() as stack:
        if early_donor is not None:
            k=int(early_k); pos=np.arange(plen-k,plen,dtype=np.int64)
            positions=torch.tensor(pos[None,:],dtype=torch.long,device=device)
            for hidx in EARLY_HIDDEN:
                d=early_donor["early"][hidx][-k:].astype(np.float32)
                rep=torch.tensor(d[None],dtype=torch.float32,device=device)
                stack.enter_context(residual_patch_hook(layers[hidx-1],positions,rep))
        def eh(hidx):
            def f(module,inputs,output):
                t=output[0] if isinstance(output,tuple) else output; take=min(max_suffix,t.shape[1])
                store["early"][hidx]=t[0,-take:].detach().float().cpu().numpy().astype(np.float32)
            return f
        def hh(hidx):
            def f(module,inputs):
                x=inputs[0]; take=min(max_suffix,x.shape[1]); z=x[0,-take:].view(take,num_heads,head_dim)
                store["heads"][hidx]=z.detach().float().cpu().numpy().astype(np.float32)
            return f
        for hidx in EARLY_HIDDEN: handles.append(layers[hidx-1].register_forward_hook(eh(hidx)))
        for hidx in ATTN_HIDDEN: handles.append(layers[hidx-1].self_attn.o_proj.register_forward_pre_hook(hh(hidx)))
        try:model(**enc,use_cache=False,return_dict=True)
        finally:
            for h in handles:h.remove()
    return {"ids":ids,**store}

def build_task_stats(model,tok,layers,entries,num_heads,head_dim):
    maps=donor_maps(entries); ids=[tok_entry(tok,e) for e in entries]
    maxs=[lcs(ids[i],ids[maps[i]["opposite_same_wording"]]) for i in range(len(entries))]
    base=[capture(model,tok,layers,entries[i],maxs[i],num_heads,head_dim) for i in range(len(entries))]
    aligned=np.zeros((len(ATTN_HIDDEN),num_heads),np.float64); energy=np.zeros_like(aligned); counts=np.zeros(len(ATTN_HIDDEN),np.int64)
    for i in range(len(entries)):
        di=maps[i]["opposite_same_wording"]; k=lcs(base[i]["ids"],base[di]["ids"])
        p=capture(model,tok,layers,entries[i],k,num_heads,head_dim,early_donor=base[di],early_k=k)
        for li,hidx in enumerate(ATTN_HIDDEN):
            r=base[i]["heads"][hidx][-k:].astype(np.float64); d=base[di]["heads"][hidx][-k:].astype(np.float64); pp=p["heads"][hidx][-k:].astype(np.float64)
            dr=d-r; pr=pp-r
            aligned[li]+=np.sum(pr*dr,axis=(0,2)); energy[li]+=np.sum(dr*dr,axis=(0,2)); counts[li]+=k
    return (aligned/counts[:,None]).astype(np.float32),(energy/counts[:,None]).astype(np.float32)

def o_norms(layers,num_heads,head_dim):
    out=[]
    for hidx in ATTN_HIDDEN:
        w=layers[hidx-1].self_attn.o_proj.weight.detach().float().cpu().numpy()
        out.append([np.linalg.norm(w[:,h*head_dim:(h+1)*head_dim]) for h in range(num_heads)])
    return np.asarray(out,np.float32)

def select_heads(train_a,train_e,onorm,k):
    a=train_a.mean(0); e=train_e.mean(0); rec=a/(e+1e-12)
    impact=np.maximum(rec,0)*np.sqrt(np.maximum(e,0))*onorm
    flat=impact.ravel(); kk=min(k,len(flat)); inds=np.argpartition(flat,-kk)[-kk:]; inds=inds[np.argsort(flat[inds])[::-1]]
    rows=[]; nh=impact.shape[1]
    for rank,fi in enumerate(inds,1):
        li=int(fi//nh); head=int(fi%nh)
        rows.append(dict(rank=rank,hidden_state_index=int(ATTN_HIDDEN[li]),head_index=head,impact_score=float(impact[li,head]),recovery_ratio=float(rec[li,head]),aligned_mean=float(a[li,head]),donor_energy_mean=float(e[li,head]),o_proj_head_block_norm=float(onorm[li,head])))
    return rows

def by_layer(rows):
    out={}
    for r in rows: out.setdefault(r["hidden_state_index"],[]).append(r["head_index"])
    return {h:np.asarray(sorted(v),np.int64) for h,v in out.items()}

def random_matched(rows,num_heads,seed):
    rng=np.random.default_rng(seed); out={}
    for hidx,chosen in by_layer(rows).items():
        pool=np.asarray([x for x in range(num_heads) if x not in set(chosen)],np.int64)
        out[hidx]=rng.choice(pool,size=len(chosen),replace=False)
    return out

def prep_cache(model,tok,layers,entries,num_heads,head_dim):
    maps=donor_maps(entries); ids=[tok_entry(tok,e) for e in entries]
    maxs=[]
    for i in range(len(entries)):
        maxs.append(max(lcs(ids[i],ids[maps[i][r]]) for r in ["opposite_same_wording","opposite_cross_wording"]))
    return [capture(model,tok,layers,entries[i],maxs[i],num_heads,head_dim) for i in range(len(entries))],maps

@torch.inference_mode()
def score(model,tok,layers,rec,rec_cache,num_heads,head_dim,donor_cache=None,k=None,early=False,group=None,source=None,all_heads=False):
    device=model.get_input_embeddings().weight.device
    ptxt=render(tok,rec["messages"]); pids=tok(ptxt,add_special_tokens=False)["input_ids"]; plen=len(pids); vals=[]
    for cand in rec["candidates"]:
        cids=tok(cand,add_special_tokens=False)["input_ids"]; ids=pids+cids
        inp=torch.tensor([ids],dtype=torch.long,device=device)
        with ExitStack() as stack:
            if donor_cache is not None and k is not None:
                pos=np.arange(plen-k,plen,dtype=np.int64); positions=torch.tensor(pos[None],dtype=torch.long,device=device)
                if early:
                    for hidx in EARLY_HIDDEN:
                        d=donor_cache["early"][hidx][-k:].astype(np.float32)
                        stack.enter_context(residual_patch_hook(layers[hidx-1],positions,torch.tensor(d[None],dtype=torch.float32,device=device)))
                if all_heads or group:
                    src=donor_cache if source=="donor" else rec_cache
                    for hidx in ATTN_HIDDEN:
                        if all_heads:
                            idx=None; v=src["heads"][hidx][-k:].astype(np.float32)
                        else:
                            if hidx not in group: continue
                            ii=group[hidx]; idx=torch.tensor(ii,dtype=torch.long,device=device); v=src["heads"][hidx][-k:,ii,:].astype(np.float32)
                        stack.enter_context(head_patch_hook(layers[hidx-1].self_attn.o_proj,positions,idx,torch.tensor(v[None],dtype=torch.float32,device=device),num_heads,head_dim))
            out=model(input_ids=inp,attention_mask=torch.ones_like(inp),use_cache=False,return_dict=True)
        lp=torch.log_softmax(out.logits.float(),-1); c=len(cids); pp=torch.arange(plen-1,plen+c-1,device=device); tp=torch.arange(plen,plen+c,device=device); targets=inp[0,tp]
        vals.append(float(lp[0,pp,targets].mean().cpu()))
    return vals[1]-vals[0]

def boot(x,seed,n=5000):
    x=np.asarray(x,np.float64); rng=np.random.default_rng(seed)
    bs=np.asarray([rng.choice(x,size=len(x),replace=True).mean() for _ in range(n)])
    return float(x.mean()),float(np.quantile(bs,.025)),float(np.quantile(bs,.975))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",default=None); ap.add_argument("--seed",type=int,default=4707); ap.add_argument("--out",default=str(DEFAULT_OUT)); ap.add_argument("--random-controls",type=int,default=RANDOM_CONTROLS); args=ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    mod=load_exp01b(); mani=load_manifest(); bytask=recreate_entries(mod,mani)
    mdir=resolve_model_dir(args.model); out=Path(args.out); out=(REPO_ROOT/out).resolve() if not out.is_absolute() else out; out.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(str(mdir),local_files_only=True,trust_remote_code=False)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(str(mdir),local_files_only=True,trust_remote_code=False,torch_dtype="auto",device_map="auto",low_cpu_mem_usage=True); model.eval(); layers=get_layers(model)
    nh=int(model.config.num_attention_heads); hs=int(model.config.hidden_size); hd=int(getattr(model.config,"head_dim",hs//nh))
    if nh*hd!=hs: raise RuntimeError("Unexpected head geometry")
    on=o_norms(layers,nh,hd); tids=sorted(bytask); tpos={t:i for i,t in enumerate(tids)}
    A=np.zeros((len(tids),len(ATTN_HIDDEN),nh),np.float32); E=np.zeros_like(A)
    for n,ti in enumerate(tids,1):
        A[tpos[ti]],E[tpos[ti]]=build_task_stats(model,tok,layers,bytask[ti],nh,hd)
        print(f"[discovery {n}/{len(tids)}] task {ti}")
    np.savez_compressed(out/"head_task_stats.npz",task_ids=np.asarray(tids),aligned_mean=A,donor_energy_mean=E,o_proj_head_norm=on)
    selected_rows=[]; results=[]; ks=[PRIMARY_K]+EXPLORATORY_K
    uniq=np.asarray(tids); gkf=GroupKFold(4)
    for fold,(tr,te) in enumerate(gkf.split(np.zeros((len(uniq),1)),np.zeros(len(uniq)),uniq)):
        sels={}
        for ksel in ks:
            sels[ksel]=select_heads(A[[tpos[int(t)] for t in uniq[tr]]],E[[tpos[int(t)] for t in uniq[tr]]],on,ksel)
            for r in sels[ksel]: selected_rows.append({"fold":fold,"group":f"top{ksel}",**r})
        primary=by_layer(sels[PRIMARY_K]); randoms=[random_matched(sels[PRIMARY_K],nh,args.seed+fold*1000+r) for r in range(args.random_controls)]
        for ti in uniq[te]:
            ti=int(ti); entries=bytask[ti]; cache,maps=prep_cache(model,tok,layers,entries,nh,hd)
            for relation in ["opposite_same_wording","opposite_cross_wording"]:
                for i,rec in enumerate(entries):
                    di=maps[i][relation]; donor=entries[di]; k=lcs(cache[i]["ids"],cache[di]["ids"]); sign=1.0 if donor["label"]==LABEL_IMPL else -1.0
                    base=score(model,tok,layers,rec,cache[i],nh,hd)
                    early=score(model,tok,layers,rec,cache[i],nh,hd,cache[di],k,early=True); earlyeff=sign*(early-base)
                    if relation=="opposite_same_wording":
                        alls=score(model,tok,layers,rec,cache[i],nh,hd,cache[di],k,all_heads=True,source="donor")
                        alln=score(model,tok,layers,rec,cache[i],nh,hd,cache[di],k,early=True,all_heads=True,source="recipient")
                        for cfg,val,size in [
                            ("early_selector_effect",earlyeff,0),
                            ("all_attention_sufficiency",sign*(alls-base),len(ATTN_HIDDEN)*nh),
                            ("all_attention_necessity_loss",earlyeff-sign*(alln-base),len(ATTN_HIDDEN)*nh),
                        ]: results.append(dict(fold=fold,task_idx=ti,task_id=rec["task_id"],recipient_condition=rec["condition"],recipient_wording=rec["wording"],relation=relation,config=cfg,group_size=size,control_id=-1,value=val))
                        for ksel in ks:
                            group=by_layer(sels[ksel])
                            s=score(model,tok,layers,rec,cache[i],nh,hd,cache[di],k,group=group,source="donor")
                            n=score(model,tok,layers,rec,cache[i],nh,hd,cache[di],k,early=True,group=group,source="recipient")
                            results += [
                                dict(fold=fold,task_idx=ti,task_id=rec["task_id"],recipient_condition=rec["condition"],recipient_wording=rec["wording"],relation=relation,config=f"top{ksel}_sufficiency",group_size=ksel,control_id=-1,value=sign*(s-base)),
                                dict(fold=fold,task_idx=ti,task_id=rec["task_id"],recipient_condition=rec["condition"],recipient_wording=rec["wording"],relation=relation,config=f"top{ksel}_necessity_loss",group_size=ksel,control_id=-1,value=earlyeff-sign*(n-base)),
                            ]
                        for rid,group in enumerate(randoms):
                            s=score(model,tok,layers,rec,cache[i],nh,hd,cache[di],k,group=group,source="donor")
                            n=score(model,tok,layers,rec,cache[i],nh,hd,cache[di],k,early=True,group=group,source="recipient")
                            results += [
                                dict(fold=fold,task_idx=ti,task_id=rec["task_id"],recipient_condition=rec["condition"],recipient_wording=rec["wording"],relation=relation,config="random16_sufficiency",group_size=PRIMARY_K,control_id=rid,value=sign*(s-base)),
                                dict(fold=fold,task_idx=ti,task_id=rec["task_id"],recipient_condition=rec["condition"],recipient_wording=rec["wording"],relation=relation,config="random16_necessity_loss",group_size=PRIMARY_K,control_id=rid,value=earlyeff-sign*(n-base)),
                            ]
                    else:
                        s=score(model,tok,layers,rec,cache[i],nh,hd,cache[di],k,group=primary,source="donor")
                        n=score(model,tok,layers,rec,cache[i],nh,hd,cache[di],k,early=True,group=primary,source="recipient")
                        results += [
                            dict(fold=fold,task_idx=ti,task_id=rec["task_id"],recipient_condition=rec["condition"],recipient_wording=rec["wording"],relation=relation,config="top16_sufficiency_cross_wording",group_size=PRIMARY_K,control_id=-1,value=sign*(s-base)),
                            dict(fold=fold,task_idx=ti,task_id=rec["task_id"],recipient_condition=rec["condition"],recipient_wording=rec["wording"],relation=relation,config="top16_necessity_loss_cross_wording",group_size=PRIMARY_K,control_id=-1,value=earlyeff-sign*(n-base)),
                        ]
            print(f"[fold {fold+1}/4] task {ti}")
    sdf=pd.DataFrame(selected_rows); rdf=pd.DataFrame(results); sdf.to_csv(out/"selected_heads.csv",index=False); rdf.to_csv(out/"intervention_results.csv",index=False)
    task=(rdf.groupby(["config","relation","control_id","task_idx","task_id"],as_index=False)["value"].mean())
    rows=[]
    for (cfg,rel,cid),g in task.groupby(["config","relation","control_id"]):
        m,lo,hi=boot(g["value"],args.seed+sum(map(ord,cfg+rel))+int(cid+1)*31)
        rows.append(dict(config=cfg,relation=rel,control_id=int(cid),mean=m,ci_low=lo,ci_high=hi,num_tasks=int(g.task_idx.nunique())))
    sm=pd.DataFrame(rows); sm.to_csv(out/"summary_by_config.csv",index=False)
    def q(name,rel="opposite_same_wording",cid=-1): return sm[(sm.config==name)&(sm.relation==rel)&(sm.control_id==cid)].iloc[0]
    def diff(sel,ran,seed):
        a=task[(task.config==sel)&(task.relation=="opposite_same_wording")&(task.control_id==-1)][["task_idx","value"]].rename(columns={"value":"a"})
        b=task[(task.config==ran)&(task.relation=="opposite_same_wording")].groupby("task_idx",as_index=False).value.mean().rename(columns={"value":"b"})
        m=a.merge(b,on="task_idx"); return boot(m.a-m.b,seed)
    suffdiff=diff("top16_sufficiency","random16_sufficiency",args.seed+9001); necdiff=diff("top16_necessity_loss","random16_necessity_loss",args.seed+9002)
    dose=[]
    for ksel in sorted(set([PRIMARY_K]+EXPLORATORY_K)):
        s=q(f"top{ksel}_sufficiency"); n=q(f"top{ksel}_necessity_loss")
        dose.append(dict(k=ksel,sufficiency_mean=float(s["mean"]),sufficiency_ci_low=float(s["ci_low"]),sufficiency_ci_high=float(s["ci_high"]),necessity_loss_mean=float(n["mean"]),necessity_ci_low=float(n["ci_low"]),necessity_ci_high=float(n["ci_high"])))
    ddf=pd.DataFrame(dose); ddf.to_csv(out/"topk_dose_response.csv",index=False)
    counts=sdf[sdf.group=="top16"].groupby(["fold","hidden_state_index"]).size().reset_index(name="count")
    fig=plt.figure(figsize=(8,5)); plt.plot(ddf.k,ddf.sufficiency_mean,marker="o",label="sufficiency"); plt.plot(ddf.k,ddf.necessity_loss_mean,marker="o",label="necessity"); plt.axhline(0,ls="--"); plt.xscale("log"); plt.legend(); plt.tight_layout(); fig.savefig(out/"topk_dose_response.png",dpi=180); plt.close(fig)
    fig=plt.figure(figsize=(8,5)); counts.groupby("hidden_state_index")["count"].mean().plot(kind="bar"); plt.tight_layout(); fig.savefig(out/"selected_layer_counts.png",dpi=180); plt.close(fig)
    ps=q("top16_sufficiency"); pn=q("top16_necessity_loss"); alls=q("all_attention_sufficiency"); alln=q("all_attention_necessity_loss"); earlyq=q("early_selector_effect"); cs=q("top16_sufficiency_cross_wording","opposite_cross_wording"); cn=q("top16_necessity_loss_cross_wording","opposite_cross_wording")
    summary=dict(
        experiment="EXP07_attention_heads",primary_k=PRIMARY_K,early_selector_effect_mean=float(earlyq["mean"]),
        top16_sufficiency_mean=float(ps["mean"]),top16_sufficiency_95ci=[float(ps["ci_low"]),float(ps["ci_high"])],
        top16_necessity_loss_mean=float(pn["mean"]),top16_necessity_loss_95ci=[float(pn["ci_low"]),float(pn["ci_high"])],
        top16_sufficiency_minus_matched_random_mean=suffdiff[0],top16_sufficiency_minus_matched_random_95ci=[suffdiff[1],suffdiff[2]],
        top16_necessity_minus_matched_random_mean=necdiff[0],top16_necessity_minus_matched_random_95ci=[necdiff[1],necdiff[2]],
        all_attention_sufficiency_mean=float(alls["mean"]),all_attention_sufficiency_95ci=[float(alls["ci_low"]),float(alls["ci_high"])],
        all_attention_necessity_loss_mean=float(alln["mean"]),all_attention_necessity_loss_95ci=[float(alln["ci_low"]),float(alln["ci_high"])],
        cross_wording_top16_sufficiency_mean=float(cs["mean"]),cross_wording_top16_necessity_loss_mean=float(cn["mean"]),
        dose_response=ddf.to_dict(orient="records")
    )
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    manifest=dict(experiment="EXP07_attention_heads",git_commit=git_commit(),seed=args.seed,model_dir=str(mdir),model_name=mdir.name,num_attention_heads=nh,num_key_value_heads=int(model.config.num_key_value_heads),head_dim=hd,primary_k=PRIMARY_K,script_sha256=sha256_file(Path(__file__).resolve()),model_config_sha256=sha256_file(mdir/"config.json"),torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,offline_only=True)
    (out/"run_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
