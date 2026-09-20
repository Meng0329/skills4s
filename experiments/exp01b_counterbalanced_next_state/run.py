#!/usr/bin/env python3
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import argparse, hashlib, json, platform, random, subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = REPO_ROOT / "models"
DEFAULT_OUT = REPO_ROOT / "outputs" / "exp01b_counterbalanced_next_state"
CONFIRMATORY_LAYER = 19
TEST, IMPL = 0, 1
LABEL_NAMES = ["INSPECT_TEST", "INSPECT_IMPLEMENTATION"]

SKILLS = {
"canonical_test": """# Systematic Debugging Procedure
Follow these steps in exactly this order.
1. REPRODUCE — Run the failing test and establish the failure.
2. INSPECT_TEST — Read the failing test and identify the expected behavior.
3. INSPECT_IMPLEMENTATION — Read the relevant implementation and locate the discrepancy.
4. REPAIR — Make the smallest change that addresses the root cause.
5. TARGET_VERIFY — Run the directly affected test.
6. REGRESSION_VERIFY — Run the complete test suite.
""",
"canonical_impl": """# Systematic Debugging Procedure
Follow these steps in exactly this order.
1. REPRODUCE — Run the failing test and establish the failure.
2. INSPECT_IMPLEMENTATION — Read the relevant implementation and locate the discrepancy.
3. INSPECT_TEST — Read the failing test and identify the expected behavior.
4. REPAIR — Make the smallest change that addresses the root cause.
5. TARGET_VERIFY — Run the directly affected test.
6. REGRESSION_VERIFY — Run the complete test suite.
""",
"paraphrase_test": """# Debugging Workflow
Use the workflow in the stated sequence.
1. Confirm the reported failure by executing the relevant test.
2. Examine the failing assertion to determine the behavior the repository expects.
3. After that, examine the source code responsible for the behavior.
4. Apply only the smallest repair supported by the evidence.
5. Re-run the directly affected test.
6. Finish by checking the complete test suite.
""",
"paraphrase_impl": """# Debugging Workflow
Use the workflow in the stated sequence.
1. Confirm the reported failure by executing the relevant test.
2. Examine the source code responsible for the behavior.
3. After that, examine the failing assertion to determine the behavior the repository expects.
4. Apply only the smallest repair supported by the evidence.
5. Re-run the directly affected test.
6. Finish by checking the complete test suite.
""",
}
CONDS = [
    ("canonical_test", "canonical", TEST),
    ("canonical_impl", "canonical", IMPL),
    ("paraphrase_test", "paraphrase", TEST),
    ("paraphrase_impl", "paraphrase", IMPL),
]


def sha256_file(p: Path):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def resolve_model(arg):
    if arg:
        p = Path(arg).expanduser()
        if not p.is_absolute():
            a, b = (REPO_ROOT / p).resolve(), (MODELS_ROOT / p).resolve()
            p = a if a.exists() else b
        p = p.resolve()
        if not (p / "config.json").is_file():
            raise FileNotFoundError(f"Missing config.json: {p}")
        return p
    cands = sorted({p.parent.resolve() for p in MODELS_ROOT.rglob("config.json")})
    if len(cands) != 1:
        raise RuntimeError(f"Expected exactly one local model under {MODELS_ROOT}; found {len(cands)}. Use --model.")
    return cands[0]


def make_tasks(n, seed):
    rng = random.Random(seed)
    templates = [
        ("boundary", "A valid boundary input is rejected.", "the boundary value should be accepted"),
        ("default", "An empty input returns the wrong default.", "the documented default should be returned"),
        ("sign", "A numeric transformation returns the wrong sign.", "the sign should match the assertion"),
        ("off_by_one", "A sequence helper returns one element too few.", "the complete requested range should be returned"),
        ("branch", "A valid input incorrectly follows the fallback branch.", "the primary branch should be selected"),
        ("normalization", "A text helper normalizes a valid input incorrectly.", "the normalized text should match the assertion"),
        ("ordering", "A helper returns items in the wrong order.", "the result order should match the test"),
        ("boolean", "A predicate returns the opposite truth value for one case.", "the predicate should satisfy the test condition"),
    ]
    prefixes = [
        "CI reports one deterministic regression.",
        "A previously passing unit test now fails.",
        "A narrow edge case broke after a refactor.",
        "One reproducible unit-test failure remains.",
        "The current branch contains a localized regression.",
        "A small behavior change caused a stable test failure.",
    ]
    out = []
    for i in range(n):
        kind, failure, expected = templates[i % len(templates)]
        symbol, pkg = f"feature_{i:03d}", f"pkg_{i:03d}"
        test_path, src_path = f"tests/test_{symbol}.py", f"src/{pkg}/{symbol}.py"
        paths = [test_path, src_path]
        if i % 2: paths.reverse()
        out.append(dict(task_id=f"task_{i:03d}", task_idx=i, bug_kind=kind,
                        issue=f"{prefixes[(i*5+rng.randrange(len(prefixes)))%len(prefixes)]} {failure}",
                        expected=expected, test_path=test_path, src_path=src_path,
                        symbol=symbol, relevant_paths=paths))
    return out


def history(task, i):
    rows = []
    if i % 3 == 1:
        rows.append("Action: read_file('pyproject.toml')\nObservation: Test configuration appears normal.")
    elif i % 3 == 2:
        rows.append("Action: search('deprecated')\nObservation: No relevant deprecated API usage was found.")
    rows.append(f"Action: run_tests('{task['test_path']}')\nObservation: FAIL: {task['test_path']}::test_{task['symbol']}; observed behavior contradicts the expected behavior.")
    return "\n\n".join(rows)


def messages(task, hist, skill=None):
    system = "You are a coding agent. Follow any supplied workflow in its stated order. Decide the single next tool action. Do not skip ahead."
    sb = "" if skill is None else f"\n\nWorkflow guidance:\n----- SKILL START -----\n{skill.strip()}\n----- SKILL END -----\n"
    paths = "\n".join(f"- {p}" for p in task["relevant_paths"])
    user = f"""Repository issue:\n{task['issue']}\n\nExpected behavior:\n{task['expected']}\n\nPotentially relevant files:\n{paths}{sb}\nExecution history:\n{hist}\n\nChoose the single next action now.\n"""
    return [{"role":"system","content":system},{"role":"user","content":user}]


def render(tok, msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def extract_hidden(model, tok, batch_msgs):
    texts = [render(tok, m) for m in batch_msgs]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    lengths = enc["attention_mask"].sum(1)
    last = lengths - 1
    dev = model.get_input_embeddings().weight.device
    enc = {k:v.to(dev) for k,v in enc.items()}; last = last.to(dev)
    out = model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
    bi = torch.arange(len(batch_msgs), device=dev)
    hs = torch.stack([h[bi,last].detach().float().cpu() for h in out.hidden_states], 1)
    return hs.numpy().astype(np.float16), [int(x) for x in lengths.tolist()]


@torch.inference_mode()
def score_candidates(model, tok, prompt_msgs, candidate_pairs):
    items = []
    for pi,(m,cands) in enumerate(zip(prompt_msgs,candidate_pairs)):
        pids = tok(render(tok,m), add_special_tokens=False)["input_ids"]
        for ci,c in enumerate(cands):
            cids = tok(c, add_special_tokens=False)["input_ids"]
            items.append((pi,ci,c,pids,cids,pids+cids))
    mx = max(len(x[5]) for x in items); pad = tok.pad_token_id
    ids, mask = [], []
    for *_, seq in items:
        n = mx-len(seq); ids.append(seq+[pad]*n); mask.append([1]*len(seq)+[0]*n)
    dev = model.get_input_embeddings().weight.device
    ids = torch.tensor(ids,device=dev); mask=torch.tensor(mask,device=dev)
    lp = torch.log_softmax(model(input_ids=ids,attention_mask=mask,use_cache=False).logits.float(),-1)
    res = [[None,None] for _ in prompt_msgs]
    for r,(pi,ci,c,pids,cids,_) in enumerate(items):
        p,cnt=len(pids),len(cids)
        pos=torch.arange(p-1,p+cnt-1,device=dev); tgt=torch.tensor(cids,device=dev)
        vals=lp[r,pos,tgt]
        res[pi][ci]={"candidate":c,"token_count":cnt,"sum_logprob":float(vals.sum().cpu()),"mean_logprob":float(vals.mean().cpu())}
    return res


def probe():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000,class_weight="balanced",C=0.5,solver="liblinear"))


def within_oof(X,y,g,w,k):
    yt,yp=[],[]
    for fam in ["canonical","paraphrase"]:
        m=w==fam; Xf,yf,gf=X[m],y[m],g[m]
        pred=np.empty_like(yf)
        for tr,te in GroupKFold(k).split(Xf,yf,gf):
            c=probe(); c.fit(Xf[tr],yf[tr]); pred[te]=c.predict(Xf[te])
        yt.append(yf); yp.append(pred)
    yt,yp=np.concatenate(yt),np.concatenate(yp)
    return f1_score(yt,yp,average="macro"),accuracy_score(yt,yp)


def cross_oof(X,y,g,w,k):
    ug=np.unique(g); dummy=np.zeros((len(ug),1)); dy=np.zeros(len(ug))
    Y,P,S,I=[],[],[],[]
    for tri,tei in GroupKFold(k).split(dummy,dy,ug):
        trg,teg=set(ug[tri]),set(ug[tei])
        for a,b in [("canonical","paraphrase"),("paraphrase","canonical")]:
            tr=np.array([(gg in trg) and (ww==a) for gg,ww in zip(g,w)])
            te=np.array([(gg in teg) and (ww==b) for gg,ww in zip(g,w)])
            c=probe(); c.fit(X[tr],y[tr]);
            Y.append(y[te]); P.append(c.predict(X[te])); S.append(c.decision_function(X[te])); I.append(np.where(te)[0])
    return dict(y=np.concatenate(Y),pred=np.concatenate(P),score=np.concatenate(S),idx=np.concatenate(I))


def random_control(X,g,w,cond,k,seed,repeats=25):
    rng=np.random.default_rng(seed); ug=np.unique(g); vals=[]
    for _ in range(repeats):
        flip={int(x):int(rng.integers(0,2)) for x in ug}; yy=[]
        for gg,cc in zip(g,cond):
            sem=TEST if cc.endswith("_test") else IMPL
            yy.append(sem ^ flip[int(gg)])
        o=cross_oof(X,np.array(yy),g,w,k); vals.append(f1_score(o["y"],o["pred"],average="macro"))
    return vals


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model"); ap.add_argument("--num-tasks",type=int,default=48); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--confirmatory-layer",type=int,default=CONFIRMATORY_LAYER); ap.add_argument("--out",default=str(DEFAULT_OUT)); a=ap.parse_args()
    if a.num_tasks < 16: raise ValueError("Use at least 16 tasks; 48 recommended.")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    model_dir=resolve_model(a.model); out=Path(a.out); out = out if out.is_absolute() else (REPO_ROOT/out).resolve(); out.mkdir(parents=True,exist_ok=True)
    print(f"EXP01b | model={model_dir} | tasks={a.num_tasks} | confirmatory_layer={a.confirmatory_layer} | OFFLINE")
    tok=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True,trust_remote_code=False)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    tok.padding_side="right"
    model=AutoModelForCausalLM.from_pretrained(str(model_dir),local_files_only=True,trust_remote_code=False,torch_dtype="auto",device_map="auto",low_cpu_mem_usage=True); model.eval()

    H=[]; meta=[]; beh=[]; nsbeh=[]
    for i,t in enumerate(make_tasks(a.num_tasks,a.seed)):
        hist=history(t,i); msgs=[messages(t,hist,SKILLS[c]) for c,_,_ in CONDS]; ns=messages(t,hist,None)
        h,lens=extract_hidden(model,tok,msgs)
        if a.confirmatory_layer>=h.shape[1]: raise ValueError(f"Layer {a.confirmatory_layer} invalid for {h.shape[1]} hidden-state tensors")
        cand=[f"read_file('{t['test_path']}')",f"read_file('{t['src_path']}')"]
        scores=score_candidates(model,tok,msgs+[ns],[cand for _ in range(5)])
        for j,(c,w,y) in enumerate(CONDS):
            H.append(h[j]); meta.append(dict(task_id=t["task_id"],task_idx=i,bug_kind=t["bug_kind"],condition=c,wording=w,label=y,label_name=LABEL_NAMES[y],prompt_tokens=lens[j],test_path_first=int(t["relevant_paths"][0]==t["test_path"])))
            tm,im=scores[j][0]["mean_logprob"],scores[j][1]["mean_logprob"]; margin=im-tm; pred=int(margin>0)
            beh.append(dict(task_id=t["task_id"],task_idx=i,condition=c,wording=w,label=y,label_name=LABEL_NAMES[y],test_mean_logprob=tm,impl_mean_logprob=im,impl_minus_test_margin=margin,behavior_pred=pred,behavior_correct=int(pred==y),signed_margin_for_correct_action=margin if y==IMPL else -margin))
        tm,im=scores[-1][0]["mean_logprob"],scores[-1][1]["mean_logprob"]
        nsbeh.append(dict(task_id=t["task_id"],task_idx=i,test_mean_logprob=tm,impl_mean_logprob=im,impl_minus_test_margin=im-tm,prefers_impl=int(im>tm)))
        print(f"[{i+1:03d}/{a.num_tasks:03d}] {t['task_id']}")

    H=np.stack(H); meta=pd.DataFrame(meta); beh=pd.DataFrame(beh); nsbeh=pd.DataFrame(nsbeh)
    meta.to_csv(out/"metadata.csv",index=False); beh.to_csv(out/"behavior.csv",index=False); nsbeh.to_csv(out/"no_skill_behavior.csv",index=False); np.savez_compressed(out/"activations.npz",hidden=H)
    (out/"stimuli.json").write_text(json.dumps({"skills":SKILLS,"conditions":[{"condition":c,"wording":w,"label":y} for c,w,y in CONDS]},ensure_ascii=False,indent=2),encoding="utf-8")

    y=meta.label.to_numpy(np.int64); g=meta.task_idx.to_numpy(np.int64); w=meta.wording.to_numpy(); cond=meta.condition.to_numpy(); k=min(4,a.num_tasks//4)
    rows=[]; cache={}
    for layer in range(H.shape[1]):
        X=H[:,layer,:].astype(np.float32); wf,wa=within_oof(X,y,g,w,k); co=cross_oof(X,y,g,w,k); cf=f1_score(co["y"],co["pred"],average="macro"); ca=accuracy_score(co["y"],co["pred"])
        rows.append(dict(layer=layer,within_wording_macro_f1=wf,within_wording_accuracy=wa,cross_wording_macro_f1=cf,cross_wording_accuracy=ca)); cache[layer]=co
    ldf=pd.DataFrame(rows); ldf.to_csv(out/"layer_probe.csv",index=False)

    L=a.confirmatory_layer; crow=ldf[ldf.layer==L].iloc[0]; co=cache[L]
    cm=confusion_matrix(co["y"],co["pred"],labels=[TEST,IMPL]); pd.DataFrame(cm,index=LABEL_NAMES,columns=LABEL_NAMES).to_csv(out/"confirmatory_confusion_matrix.csv")
    ctrl=random_control(H[:,L,:].astype(np.float32),g,w,cond,k,a.seed+991); pd.DataFrame({"repeat":range(len(ctrl)),"random_mapping_macro_f1":ctrl}).to_csv(out/"control_probe.csv",index=False)

    margin=beh.iloc[co["idx"]].impl_minus_test_margin.to_numpy(float); pscore=co["score"].astype(float)
    corr=float(np.corrcoef(pscore,margin)[0,1]) if np.std(pscore)>0 and np.std(margin)>0 else float("nan")
    best=ldf.loc[ldf.cross_wording_macro_f1.idxmax()]

    plt.figure(figsize=(9,5.5)); plt.plot(ldf.layer,ldf.within_wording_macro_f1,label="Within-wording / held-out tasks"); plt.plot(ldf.layer,ldf.cross_wording_macro_f1,label="Cross-wording + held-out tasks",linewidth=2.2); plt.axhline(.5,linestyle="--",label="Chance"); plt.axvline(L,linestyle=":",label=f"Preregistered layer {L}"); plt.xlabel("Hidden-state index"); plt.ylabel("Macro-F1"); plt.title("Skill-prescribed next-state decodability"); plt.legend(); plt.tight_layout(); plt.savefig(out/"layer_probe.png",dpi=180); plt.close()

    summary=dict(experiment="EXP01b_counterbalanced_next_state",num_tasks=a.num_tasks,chance_binary_macro_f1=.5,confirmatory_layer=L,
        confirmatory_cross_wording_macro_f1=float(crow.cross_wording_macro_f1),confirmatory_cross_wording_accuracy=float(crow.cross_wording_accuracy),confirmatory_within_wording_macro_f1=float(crow.within_wording_macro_f1),
        exploratory_best_layer=int(best.layer),exploratory_best_cross_wording_macro_f1=float(best.cross_wording_macro_f1),random_mapping_control_macro_f1_mean=float(np.mean(ctrl)),random_mapping_control_macro_f1_std=float(np.std(ctrl)),
        behavior_action_accuracy_all=float(beh.behavior_correct.mean()),behavior_action_accuracy_canonical=float(beh[beh.wording=="canonical"].behavior_correct.mean()),behavior_action_accuracy_paraphrase=float(beh[beh.wording=="paraphrase"].behavior_correct.mean()),behavior_mean_signed_logprob_margin=float(beh.signed_margin_for_correct_action.mean()),
        no_skill_impl_choice_rate=float(nsbeh.prefers_impl.mean()),no_skill_mean_impl_minus_test_margin=float(nsbeh.impl_minus_test_margin.mean()),probe_behavior_margin_pearson_r=corr,primary_tool_control="read_file for both labels",primary_endpoint_preregistered_from_exp01=True)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")

    cfg=model_dir/"config.json"; script=Path(__file__).resolve(); manifest=dict(experiment="EXP01b",git_commit=git_commit(),seed=a.seed,num_tasks=a.num_tasks,confirmatory_layer=L,model_dir=str(model_dir),model_name=model_dir.name,model_config_sha256=sha256_file(cfg) if cfg.exists() else None,script_sha256=sha256_file(script),python=platform.python_version(),platform=platform.platform(),torch=torch.__version__,transformers=transformers.__version__,sklearn=sklearn.__version__,cuda_available=bool(torch.cuda.is_available()),cuda_version=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,offline_only=True)
    (out/"run_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print("\n=== EXP01b SUMMARY ==="); print(json.dumps(summary,ensure_ascii=False,indent=2)); print(f"\nSaved to {out}")

if __name__ == "__main__": main()
