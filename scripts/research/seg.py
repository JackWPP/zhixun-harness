import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import numpy as np
from rsi.data import load_jsonl_dataset
from rsi.metrics import classification_metrics
from rsi.model import ExtendedModel
from rsi.tasks import resolve_tasks
TASK=resolve_tasks(["A"])[0]; CL=list(TASK.classes); C=len(CL)
tr=load_jsonl_dataset(TASK.train_path,"classification",CL); va=load_jsonl_dataset(TASK.validation_path,"classification",CL)
X,Y,XV,YV=tr.x,tr.y,va.x,va.y
m=ExtendedModel.from_checkpoint(TASK.checkpoint_path)

def macro(p,y): 
    r=classification_metrics(p,y,CL); return r["macro_f1"],r

def rc_segments(x, bounds, mode="mean", w=1.0):
    # map each segment's stats back to model train stats
    out=np.empty_like(x)
    for a,b in bounds:
        seg=x[a:b]
        if mode=="mean": tm,ts=seg.mean(0),np.maximum(seg.std(0),1e-9)
        else:
            q1,q3=np.percentile(seg,[25,75],axis=0); tm,ts=np.median(seg,axis=0),np.maximum((q3-q1)/1.349,1e-9)
        out[a:b]=(seg-tm)/ts*m.scale+m.mean
    return out

def rc_windows(x, W, mode="mean", causal=False):
    out=np.empty_like(x); n=len(x)
    for i in range(n):
        a=max(0,i-W+1) if causal else max(0,i-W//2); b=min(n,a+W) if not causal else i+1
        if causal: a=max(0,i-W+1)
        seg=x[a:b]
        if mode=="mean": tm,ts=seg.mean(0),np.maximum(seg.std(0),1e-9)
        else: 
            q1,q3=np.percentile(seg,[25,75],axis=0); tm,ts=np.median(seg,axis=0),np.maximum((q3-q1)/1.349,1e-9)
        out[i]=(x[i]-tm)/ts*m.scale+m.mean
    return out

base=macro(m.predict(XV),YV); print("baseline",round(base[0],4),[round(v,3) for v in base[1]["per_class_f1"].values()])
# oracle segments from true labels
runs=[]; s=0
for i in range(1,len(YV)+1):
    if i==len(YV) or YV[i]!=YV[s]: runs.append((s,i)); s=i
print("oracle runs",len(runs))
for mode in ("mean","robust"):
    p=m.predict(rc_segments(XV,runs,mode)); sm,r=macro(p,YV)
    print(f"oracle-seg rc {mode}: {sm:.4f}", [round(v,3) for v in r["per_class_f1"].values()])
# sliding windows
for W in (15,25,50,100,200):
    for mode in ("mean","robust"):
        for causal in (False,True):
            p=m.predict(rc_windows(XV,W,mode,causal)); sm,r=macro(p,YV)
            print(f"win W={W:3d} {mode:6s} causal={int(causal)}: {sm:.4f}", [round(v,3) for v in r["per_class_f1"].values()])
