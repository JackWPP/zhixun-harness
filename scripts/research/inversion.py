import sys; from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import numpy as np
from rsi.data import load_jsonl_dataset, read_uci_gas_dat
from rsi.model import ExtendedModel
from rsi.tasks import resolve_tasks, gas_probe_root
T=resolve_tasks(["A"])[0]; CL=list(T.classes)
m=ExtendedModel.from_checkpoint(T.checkpoint_path)
tr=load_jsonl_dataset(T.train_path,"classification",CL)
def auc(score,y):
    o=np.argsort(np.argsort(score)); n1=(y==1).sum(); n0=(y==0).sum()
    return float((o[y==1].sum()-n1*(n1-1)/2)/(n1*n0)) if n1 and n0 else float('nan')
def pairs(x,y):
    lg=m.predict_logits(x); out={}
    for c in range(1,6):
        mask=np.isin(y,[0,c]); yy=(y[mask]==c); s=lg[mask,c]-lg[mask,0]
        out[c]=(round(auc(s,yy),3), int(mask.sum()))
    return out
va=load_jsonl_dataset(T.validation_path,"classification",CL)
print("official val pairs (auc vs class0):", pairs(va.x,va.y))
root=gas_probe_root(); print("gas root",root)
for b in range(1,11):
    f=root/f"batch{b}.dat"
    if not f.exists(): continue
    ds=read_uci_gas_dat(f)
    print(f"batch{b}:", pairs(ds.x,ds.y))
