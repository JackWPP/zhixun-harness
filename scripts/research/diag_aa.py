"""A: diagnose Acetaldehyde separability and calibration upper bounds."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from rsi.data import load_jsonl_dataset
from rsi.metrics import classification_metrics
from rsi.model import ExtendedModel
from rsi.tasks import resolve_tasks

OUT = ROOT / ".scratch" / "a_push"
TASK = resolve_tasks(["A"])[0]
CLASSES = list(TASK.classes)
C = len(CLASSES)
tr = load_jsonl_dataset(TASK.train_path, "classification", CLASSES)
va = load_jsonl_dataset(TASK.validation_path, "classification", CLASSES)
X, Y, XV, YV = tr.x, tr.y, va.x, va.y
m = ExtendedModel.from_checkpoint(TASK.checkpoint_path)
pT = m.predict(XV)
yv = YV.argmax(0) if YV.ndim > 1 else YV


def macro(p, y):
    return classification_metrics(p, y, CLASSES)["macro_f1"]


base = macro(pT, YV)
print("baseline val macro", round(base, 5))
cm = np.zeros((C, C), int)
for t, p in zip(YV, pT.argmax(1)):
    cm[t, p] += 1
print("confusion rows=true, cols=pred")
for i in range(C):
    print(f"  class{i} {CLASSES[i]:14s} {cm[i].tolist()}")
print("pred dist", np.bincount(pT.argmax(1), minlength=C).tolist(), "true dist", np.bincount(YV, minlength=C).tolist())

# binary separability AA(3) vs Ethanol(0), trained on official train only
r = np.random.default_rng(0)
sel = np.isin(Y, [0, 3])
Xb, yb = X[sel], (Y[sel] == 3).astype(float)
mu, sd = Xb.mean(0), np.maximum(Xb.std(0), 1e-9)
Z = (Xb - mu) / sd
w = np.zeros(X.shape[1]); b = 0.0
for _ in range(3000):
    z = Z @ w + b
    p = 1 / (1 + np.exp(-z))
    g = p - yb
    w -= 0.5 * (Z.T @ g / len(Z) + 1e-3 * w)
    b -= 0.5 * g.mean()
selv = np.isin(YV, [0, 3])
Zv = (XV[selv] - mu) / sd
pv = 1 / (1 + np.exp(-(Zv @ w + b)))
yv2 = (YV[selv] == 3)
order = np.argsort(pv)
ranks = np.empty_like(order, dtype=float)
ranks[order] = np.arange(len(order))
auc = (ranks[yv2].sum() - yv2.sum() * (yv2.sum() - 1) / 2) / (yv2.sum() * (~yv2).sum())
print("AA-vs-Ethanol logistic AUC on val:", round(float(auc), 4), "n:", len(yv2))
# base model p3/(p0+p3) AUC among true 0/3
score = pT[:, 3] / np.maximum(pT[:, 0] + pT[:, 3], 1e-9)
print("base p3/(p0+p3) AUC:", round(float((np.argsort(np.argsort(score[selv]))[yv2].sum() - yv2.sum()*(yv2.sum()-1)/2)/(yv2.sum()*(~yv2).sum())), 4))

# oracle upper bound: per-class logit bias search (analytic-ish coordinate ascent)
lg = m.predict_logits(XV)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


best = (base, np.zeros(C))
for it in range(3):
    for c in range(C):
        grid = np.linspace(-3, 3, 25)
        cur = best[0]
        bestv = cur
        bestc = best[1].copy()
        for g in grid:
            b = best[1].copy(); b[c] = g
            v = macro(softmax(lg + b[None, :]), YV)
            if v > bestv:
                bestv, bestc = v, b
        best = (bestv, bestc)
print("oracle per-class logit bias val:", round(best[0], 4), "biases", np.round(best[1], 2).tolist())

# rule: boost class 3 relative to 0 by delta; scan
p0p3 = np.stack([pT[:, 0], pT[:, 3]], 1)
ratio = np.log(np.maximum(pT[:, 3], 1e-12)) - np.log(np.maximum(pT[:, 0], 1e-12))
rows = []
for delta in [-2, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]:
    pp = pT.copy()
    # shift AA vs Ethanol logits
    l = np.log(np.maximum(pp, 1e-12)); l[:, 3] += delta
    pp = softmax(l)
    rows.append((delta, round(macro(pp, YV), 4), int((pp.argmax(1) == 3).sum()), int((pp.argmax(1) == 0).sum())))
print("AA-vs-Ethanol logit shift scan (delta, macro, n_AA, n_Eth):")
for r_ in rows: print("  ", r_)

# sequence structure of labels in official validation
runs=[]; start=0
for i in range(1,len(YV)+1):
    if i==len(YV) or YV[i]!=YV[start]:
        runs.append((int(YV[start]), start, i-1, i-start)); start=i
print("val contiguous runs:", len(runs))
for r_ in runs[:40]: print("   class%d [%d..%d] n=%d" % r_)
print("AA positions:", np.where(YV==3)[0].min(), np.where(YV==3)[0].max(), "n", int((YV==3).sum()))
print("score p3/(p0+p3) by 100-row blocks:", [round(float(np.mean(score[i:i+100])),3) for i in range(0,len(score),100)])
# rate-targeted rule: among rows whose argmax in {0,3}, flip top-r ratio to AA
mask = np.isin(pT.argmax(1), [0, 3])
idx = np.where(mask)[0]
order = idx[np.argsort(-ratio[mask])]
print("rate-targeted AA boost (rate, macro, AA_f1, Eth_f1, n_AA):")
for rate in [0.02, 0.04, 0.06, 0.067, 0.08, 0.10, 0.12, 0.15]:
    k = int(rate * len(YV))
    pp = pT.argmax(1).copy(); pp[order[:k]] = 3
    onehot = np.eye(C)[pp]
    mm = classification_metrics(onehot, YV, CLASSES)
    print("  ", rate, round(mm["macro_f1"], 4), round(mm["per_class_f1"].get("3", 0), 3),
          round(mm["per_class_f1"].get("0", 0), 3), int((pp == 3).sum()))
