"""P1 delayed-label calibration simulation for A.

For each batch i: predict with the frozen base model + calibration learned ONLY from
labels of batches < i (prediction-time honest). Then reveal batch i labels and update.

Methods: base / adjacent threshold / cumulative threshold / adjacent logistic.
Also a gated variant that only enables calibration when history evidence is sufficient
and improves history macro by >= margin.
"""
import sys; from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import numpy as np
from rsi.data import load_jsonl_dataset, read_uci_gas_dat
from rsi.metrics import classification_metrics
from rsi.model import ExtendedModel
from rsi.tasks import resolve_tasks, gas_probe_root

T=resolve_tasks(["A"])[0]; CL=list(T.classes); C=len(CL)
M=ExtendedModel.from_checkpoint(T.checkpoint_path)

def macro_f1(pred,y):
    return float(classification_metrics(np.eye(C)[pred],y,CL)["macro_f1"])

def flip(pred, s, delta):
    out=pred.copy(); m=np.isin(pred,[0,3])
    out[m & (s<delta)]=3; out[m & (s>=delta)]=0
    return out

def best_delta(s,y,pred):
    m=np.isin(pred,[0,3])
    if m.sum()==0 or (y[m]==3).sum()==0 or (y[m]==0).sum()==0: return None
    best=(-1,None)
    lo,hi=np.percentile(s[m],[0.5,99.5])
    for d in np.linspace(lo,hi,161):
        v=macro_f1(flip(pred[m],s[m],d),y[m])
        if v>best[0]: best=(v,float(d))
    return best

def fit_logistic(s,y,pred):
    m=np.isin(pred,[0,3]); s=s[m]; t=(y[m]==3).astype(float)
    if len(np.unique(t))<2: return None
    mu,sd=s.mean(),s.std()+1e-9; z=(s-mu)/sd
    w,b=0.0,np.log((t.mean()+1e-6)/(1-t.mean()+1e-6))
    lr=0.2; n=len(z)
    for _ in range(400):
        p=1/(1+np.exp(-(w*z+b+1e-9)))
        gw=(z*(p-t)).mean()+1e-3*w; gb=(p-t).mean()
        w-=lr*gw; b-=lr*gb
    return (float(mu),float(sd),float(w),float(b))

def apply_logistic(pred,s,par):
    if par is None: return pred
    mu,sd,w,b=par; p=1/(1+np.exp(-(w*((s-mu)/sd)+b)))
    out=pred.copy(); m=np.isin(pred,[0,3]); out[m & (p>0.5)]=3; out[m & (p<=0.5)]=0
    return out

def run(name,batches):
    preds=[]; s_all=[]
    for x,y in batches:
        lg=M.predict_logits(x); preds.append(lg.argmax(1)); s_all.append(lg[:,3]-lg[:,0])
    keys=["base","adj_thr","cum_thr","adj_logit","pool3_thr","rvg"]
    res={k:[] for k in keys}; c3={k:[] for k in keys}
    hist_s=[]; hist_y=[]; hist_pred=[]
    for i in range(len(batches)):
        pred=preds[i]; s=s_all[i]; y=batches[i][1]
        # calibrations from history only
        adj_thr=cum_thr=adj_logit=None
        if hist_s:
            hs=np.concatenate(hist_s); hy=np.concatenate(hist_y); hp=np.concatenate(hist_pred)
            bd=best_delta(hs,hy,hp)
            if bd and bd[1] is not None: cum_thr=bd[1]; cum_score=bd[0]
            else: cum_score=None
            if len(hist_s[-1]):
                bd2=best_delta(hist_s[-1],hist_y[-1],hist_pred[-1])
                if bd2 and bd2[1] is not None: adj_thr=bd2[1]; adj_gain=bd2[0]-macro_f1(hist_pred[-1],hist_y[-1])
                else: adj_gain=0
            else: adj_gain=0
            adj_logit=fit_logistic(hs,hy,hp)
        out={"base":pred}
        out["adj_thr"]=flip(pred,s,adj_thr) if adj_thr is not None else pred
        out["cum_thr"]=flip(pred,s,cum_thr) if cum_thr is not None else pred
        out["adj_logit"]=apply_logistic(pred,s,adj_logit)
        # gated: enable adjacent threshold only if history has enough AA and prev-batch gain>=0.02
        gate = False
        if hist_s:
            hs=np.concatenate(hist_s); hy=np.concatenate(hist_y); hp=np.concatenate(hist_pred)
            gate = (adj_thr is not None and adj_gain>=0.02
                    and int((hy[hp==3]==3).sum())>=10 and int((hy[hp==0]==0).sum())>=20)
        # pooled last-3-batches threshold
        pool=None
        if len(hist_s)>=2:
            hs3=np.concatenate(hist_s[-3:]); hy3=np.concatenate(hist_y[-3:]); hp3=np.concatenate(hist_pred[-3:])
            bd3=best_delta(hs3,hy3,hp3)
            if bd3 and bd3[1] is not None: pool=bd3[1]
        out["pool3_thr"]=flip(pred,s,pool) if pool is not None else pred
        # rolling-validation gate: fit on history[:-1], test on last labeled batch
        rvg=None
        if len(hist_s)>=2:
            hs_prev=np.concatenate(hist_s[:-1]); hy_prev=np.concatenate(hist_y[:-1]); hp_prev=np.concatenate(hist_pred[:-1])
            bd4=best_delta(hs_prev,hy_prev,hp_prev)
            if bd4 and bd4[1] is not None:
                cand=flip(hist_pred[-1],hist_s[-1],bd4[1])
                gain=macro_f1(cand,hist_y[-1])-macro_f1(hist_pred[-1],hist_y[-1])
                if gain>=0.02 and int((hist_y[-1]==3).sum())>=5: rvg=bd4[1]
        out["rvg"]=flip(pred,s,rvg) if rvg is not None else pred
        for k in keys:
            res[k].append(macro_f1(out[k],y))
            f1=classification_metrics(np.eye(C)[out[k]],y,CL)["per_class_f1"]
            c3[k].append(float(f1.get("3", f1.get(3, 0.0))))
        hist_s.append(s); hist_y.append(y); hist_pred.append(pred)
    print(name)
    print("  mean macro: " + "  ".join(f"{k}={np.mean(v):.4f}" for k,v in res.items()))
    print("  mean AA F1: " + "  ".join(f"{k}={np.mean(c3[k]):.4f}" for k in keys))
    for i in range(len(batches)):
        print("  b%-2d " % i + " ".join(f"{k}:{res[k][i]:.3f}" for k in keys))
    return res

va=load_jsonl_dataset(T.validation_path,"classification",CL)
n=8; idx=np.array_split(np.arange(len(va.y)),n)
run("official val, 8 sequential chunks", [(va.x[i],va.y[i]) for i in idx])

root=gas_probe_root(); uci=[]
for b in range(3,11):
    f=root/f"batch{b}.dat"
    if f.exists():
        ds=read_uci_gas_dat(f); uci.append((ds.x,ds.y))
run("UCI batches 3-10", uci)
run("UCI batches 6-10", uci[3:])
