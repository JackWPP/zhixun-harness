import sys; from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import numpy as np
from rsi.data import load_jsonl_dataset, read_uci_gas_dat
from rsi.model import ExtendedModel
from rsi.tasks import resolve_tasks, gas_probe_root
T=resolve_tasks(["A"])[0]; CL=list(T.classes)
m=ExtendedModel.from_checkpoint(T.checkpoint_path)
def show(name,x,y):
    lg=m.predict_logits(x); s=lg[:,3]-lg[:,0]
    print(name)
    for c in (0,3):
        v=s[y==c]
        if len(v)==0: continue
        qs=np.percentile(v,[0,5,25,50,75,95,100])
        print(f"  true {CL[c]:14s} n={len(v):5d} quantiles {np.round(qs,2).tolist()}")
    print("  all:", np.round(np.percentile(s,[0,1,5,25,50,75,95,99,100]),2).tolist())
va=load_jsonl_dataset(T.validation_path,"classification",CL); show("official val",va.x,va.y)
root=gas_probe_root()
for b in (1,3,4,5,6,7,8,9,10):
    f=root/f"batch{b}.dat"
    if f.exists():
        ds=read_uci_gas_dat(f); show(f"uci batch{b}",ds.x,ds.y)
