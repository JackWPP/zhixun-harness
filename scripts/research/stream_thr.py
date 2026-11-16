import sys; from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import numpy as np
from rsi.data import load_jsonl_dataset, read_uci_gas_dat
from rsi.metrics import classification_metrics
from rsi.model import ExtendedModel
from rsi.tasks import resolve_tasks, gas_probe_root
T=resolve_tasks(["A"])[0]; CL=list(T.classes); C=len(CL)
m=ExtendedModel.from_checkpoint(T.checkpoint_path)
def mf(pred,y): return classification_metrics(np.eye(C)[pred],y,CL)["macro_f1"]
def best_delta(s,y):
    # search threshold for AA-vs-Eth on labeled data; AA if s<delta
    base_pred=np.where(np.ones(len(y),bool),None,None)
    lg = None
    best=(-1,None)
    for d in np.linspace(np.percentile(s,1),np.percentile(s,99),97):
        pred=np.where(s<d,3,0)  # only 0/3 rows assumed
        v=mf(pred,y)
        if v>best[0]: best=(v,d)
    return best[1]
def run(name,batches):
    hist_x=[]; hist_y=[]
    out=[]
    for i,(x,y) in enumerate(batches):
        lg=m.predict_logits(x); pred=lg.argmax(1); s=lg[:,3]-lg[:,0]
        saw_aa = int(np.sum(np.concatenate(hist_y)==3)) if hist_y else 0
        if hist_x and saw_aa >= 10:
            hx=np.concatenate(hist_x); hy=np.concatenate(hist_y); hlg=m.predict_logits(hx); hs=hlg[:,3]-hlg[:,0]
            mask=np.isin(hlg.argmax(1),[0,3])
            d=best_delta(hs[mask],hy[mask])
            pm=pred.copy(); m2=np.isin(pred,[0,3]); pm[m2 & (s<d)]=3; pm[m2 & (s>=d)]=0
        else:
            pm=pred.copy(); d=None
        out.append((i,len(y),round(mf(pred,y),4),round(mf(pm,y),4),round(float(d) if d is not None else 0,2),saw_aa))
        hist_x.append(x); hist_y.append(y)
    print(name)
    for r in out: print("  batch %d n=%d base=%.4f stream=%.4f delta=%s sawAA=%d" % r)
    print("  mean base=%.4f stream=%.4f" % (np.mean([r[2] for r in out]), np.mean([r[3] for r in out])))
va=load_jsonl_dataset(T.validation_path,"classification",CL)
n=8; idx=np.array_split(np.arange(len(va.y)),n)
run("official val 8 chunks", [(va.x[i],va.y[i]) for i in idx])
root=gas_probe_root(); batches=[]
for b in range(3,11):
    f=root/f"batch{b}.dat"
    if f.exists():
        ds=read_uci_gas_dat(f); batches.append((ds.x,ds.y))
run("uci batches 3-10",batches)
run("uci batches 6-10",batches[3:])
