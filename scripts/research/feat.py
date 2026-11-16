import sys; from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import numpy as np
from rsi.data import load_jsonl_dataset
from rsi.tasks import resolve_tasks
T=resolve_tasks(["A"])[0]; CL=list(T.classes)
tr=load_jsonl_dataset(T.train_path,"classification",CL); va=load_jsonl_dataset(T.validation_path,"classification",CL)
X,Y,XV,YV=tr.x,tr.y,va.x,va.y
def cls(x,y,c): return x[y==c]
print("train class means spread:", [round(float(cls(X,Y,c).mean()),4) for c in range(6)])
print("val class means spread:", [round(float(cls(XV,YV,c).mean()),4) for c in range(6)])
# per-feature mean by class: correlation between train and val class-mean difference vectors
for a,b in [(3,0),(1,0),(2,0),(4,0),(5,0)]:
    dt=cls(X,Y,a).mean(0)-cls(X,Y,b).mean(0)
    dv=cls(XV,YV,a).mean(0)-cls(XV,YV,b).mean(0)
    r=float(np.corrcoef(dt,dv)[0,1]); 
    # correlation of abs
    ra=float(np.corrcoef(np.abs(dt),np.abs(dv))[0,1])
    print(f"class {a} vs {b}: corr(delta_train,delta_val)={r:.3f} corr(|.|)={ra:.3f} sign agree={np.mean(np.sign(dt)==np.sign(dv)):.3f}")
# global train/val feature mean & scale correlation
mt, st = X.mean(0), X.std(0); mv, sv = XV.mean(0), XV.std(0)
print("corr feature mean train-val:", round(float(np.corrcoef(mt,mv)[0,1]),3), "corr scale:", round(float(np.corrcoef(st,sv)[0,1]),3))
print("mean shift z-score per feature: median", round(float(np.median(np.abs(mt-mv)/st)),3), "max", round(float(np.max(np.abs(mt-mv)/st)),3))
# per-class within-train separability of AA vs Eth: simple 1D score = projection on (mean_AA-mean_Eth)
d=cls(X,Y,3).mean(0)-cls(X,Y,0).mean(0); d/=np.linalg.norm(d)
mask=np.isin(YV,[0,3]); s=XV[mask]@d; yy=(YV[mask]==3)
auc=float((np.argsort(np.argsort(s))[yy].sum()-yy.sum()*(yy.sum()-1)/2)/(yy.sum()*(~yy).sum()))
print("val projection-score AUC (train direction):",round(auc,4))
# also train direction from LDA with covariance whitening
Xs=np.vstack([cls(X,Y,3),cls(X,Y,0)]); ys=np.r_[np.ones(len(cls(X,Y,3))),np.zeros(len(cls(X,Y,0)))]
Sw=(np.cov(cls(X,Y,3).T)+np.cov(cls(X,Y,0).T))/2
Sw+=1e-3*np.trace(Sw)/len(Sw)*np.eye(len(Sw))
w=np.linalg.solve(Sw, cls(X,Y,3).mean(0)-cls(X,Y,0).mean(0))
s=XV[mask]@w
auc2=float((np.argsort(np.argsort(s))[yy].sum()-yy.sum()*(yy.sum()-1)/2)/(yy.sum()*(~yy).sum()))
print("val projection-score AUC (LDA direction):",round(auc2,4))
