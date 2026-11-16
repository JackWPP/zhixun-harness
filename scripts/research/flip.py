import sys; from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import numpy as np
from rsi.data import load_jsonl_dataset
from rsi.metrics import classification_metrics
from rsi.model import ExtendedModel
from rsi.tasks import resolve_tasks
T=resolve_tasks(["A"])[0]; CL=list(T.classes); C=len(CL)
va=load_jsonl_dataset(T.validation_path,"classification",CL); XV,YV=va.x,va.y
m=ExtendedModel.from_checkpoint(T.checkpoint_path); lg=m.predict_logits(XV)
def softmax(z):
    z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
def mf(p): 
    r=classification_metrics(p,YV,CL); return r["macro_f1"],r
base,r=mf(softmax(lg)); print("base",round(base,4),[round(v,3) for v in r["per_class_f1"].values()])
s=lg[:,3]-lg[:,0]
# reversed: AA when s < delta, Eth when s >= delta
print("delta scan (AA if s<delta):")
best=None
for d in np.linspace(np.percentile(s,0.1),np.percentile(s,99.9),20):
    l=lg.copy(); l[:,3]-=d*(1 if False else 0)  # noop
    pred=lg.argmax(1).copy()
    mask=np.isin(pred,[0,3]); pred[mask & (s<d)]=3; pred[mask & (s>=d)]=0
    p=np.eye(C)[pred]; v,rr=mf(p)
    if best is None or v>best[0]: best=(v,d,pred,rr)
    print(f"  d={d:7.3f} macro={v:.4f} AA={rr['per_class_f1'].get('3',0):.3f} Eth={rr['per_class_f1'].get('0',0):.3f} nAA={(pred==3).sum()}")
v,d,pred,rr=best; print("BEST",round(v,4),"delta",round(float(d),3),"perclass",[round(x,3) for x in rr['per_class_f1'].values()])
# also try threshold on log p3-log p0 in [-6,6] fine grid
best2=None
for d in np.linspace(-6,6,97):
    pred=lg.argmax(1).copy(); mask=np.isin(pred,[0,3]); pred[mask & (s<d)]=3; pred[mask & (s>=d)]=0
    vv,_=mf(np.eye(C)[pred])
    if best2 is None or vv>best2[0]: best2=(vv,d)
print("fine best",round(best2[0],4),"delta",round(float(best2[1]),3))
