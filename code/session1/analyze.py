"""Load-bearing analysis: geometric cosine vs functional efficiency, per distillation pair.

For each outer leave-one-group-out fold (layer chosen on base, inner folds only):
  AUC_base   base-trained probe on base
  AUC_own    distilled model's own probe on distilled            (ceiling for transfer)
  AUC_tr     base-trained probe on distilled (base scaler / distilled scaler)
  cos        cosine between z-scored diff-of-means directions, base vs distilled
  eta        d'(AUC_tr) / d'(AUC_own): share of the distilled model's own separability
             that the base probe recovers. Gaussian geometry predicts eta ~ cos.
Decision: eta clearly above cos  -> rotation lies mostly in directions the probe ignores.
          eta ~ cos, AUC_own > AUC_base -> distilled model is more separable and compensates.
"""
import argparse, json
import numpy as np, pandas as pd
from scipy.stats import norm, t as tdist
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

p = argparse.ArgumentParser()
p.add_argument("--csv", required=True)
p.add_argument("--base", required=True)
p.add_argument("--arm", required=True)
p.add_argument("--name", required=True)
p.add_argument("--C", type=float, default=1.0)
a = p.parse_args()

df = pd.read_csv(a.csv)
y = (df["label"].str.lower() == "unsafe").astype(int).values
# 8 groups: each safe type with its unsafe contrast type (XSTest v2 official CSV).
# The three privacy types form one group; the three discrimination types form one group.
g = df["type"].str.replace(r"^contrast_", "", regex=True)
g = g.where(~g.str.contains("privacy"), "privacy").where(~g.str.contains("discr"), "discr").values

B, A = np.load(a.base), np.load(a.arm)
n_mismatch = int((B["tokh"] != A["tokh"]).sum())
print(f"prompts tokenised differently by base and arm: {n_mismatch} of {len(B['tokh'])}")
XB, XA = B["acts"].astype(np.float32), A["acts"].astype(np.float32)   # [L, N, d]
L = XB.shape[0]

print(pd.crosstab(g, y).rename(columns={0: "safe", 1: "unsafe"}))

def zdir(X, yy):
    Z = (X - X.mean(0)) / (X.std(0) + 1e-6)
    d = Z[yy == 1].mean(0) - Z[yy == 0].mean(0)
    return d / np.linalg.norm(d)

def proj_auc(Xtr, ytr, Xte, yte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    d = ((Xtr - mu) / sd)[ytr == 1].mean(0) - ((Xtr - mu) / sd)[ytr == 0].mean(0)
    return roc_auc_score(yte, ((Xte - mu) / sd) @ d)

def two_class(idx):
    return len(np.unique(y[idx])) == 2

def pick_layer(tr):
    gs = np.unique(g[tr]); scores = np.zeros(L)
    for l in range(L):
        s = [proj_auc(XB[l][tr][g[tr] != h], y[tr][g[tr] != h], XB[l][tr][g[tr] == h], y[tr][g[tr] == h])
             for h in gs if two_class(tr[g[tr] == h])]
        scores[l] = np.mean(s)
    return int(scores.argmax())

def probe(Xtr, ytr):
    sc = StandardScaler().fit(Xtr)
    return sc, LogisticRegression(C=a.C, max_iter=5000).fit(sc.transform(Xtr), ytr)

dp = lambda u: np.sqrt(2) * norm.ppf(np.clip(u, 1e-4, 1 - 1e-4))
def dm_scores(Xtr, ytr, Xte, mu, sd):
    """Project standardised Xte onto the z-scored diff-of-means direction fitted on (Xtr, ytr) with stats (mu, sd)."""
    Z = (Xtr - mu) / sd
    d = Z[ytr == 1].mean(0) - Z[ytr == 0].mean(0)
    return ((Xte - mu) / sd) @ (d / np.linalg.norm(d))

rows = []
for h in np.unique(g):
    te, tr = np.where(g == h)[0], np.where(g != h)[0]
    if not two_class(te):
        continue                                   # single-class group: trains, never scored
    l = pick_layer(tr)
    scB, lrB = probe(XB[l][tr], y[tr])
    scA, lrA = probe(XA[l][tr], y[tr])
    sB = lambda X, sc: lrB.decision_function(sc.transform(X))
    r = dict(group=h, layer=l,
             auc_base=roc_auc_score(y[te], sB(XB[l][te], scB)),
             auc_own=roc_auc_score(y[te], lrA.decision_function(scA.transform(XA[l][te]))),
             auc_tr_basescaler=roc_auc_score(y[te], sB(XA[l][te], scB)),
             auc_tr_armscaler=roc_auc_score(y[te], sB(XA[l][te], scA)),
             cos=float(zdir(XB[l][tr], y[tr]) @ zdir(XA[l][tr], y[tr])))
    # same comparison with the difference-of-means estimator, so eta and cos use the same directions
    muB, sdB = XB[l][tr].mean(0), XB[l][tr].std(0) + 1e-6
    muA, sdA = XA[l][tr].mean(0), XA[l][tr].std(0) + 1e-6
    r["auc_dm_base"] = roc_auc_score(y[te], dm_scores(XB[l][tr], y[tr], XB[l][te], muB, sdB))
    r["auc_dm_own"] = roc_auc_score(y[te], dm_scores(XA[l][tr], y[tr], XA[l][te], muA, sdA))
    r["auc_dm_tr"] = roc_auc_score(y[te], dm_scores(XB[l][tr], y[tr], XA[l][te], muB, sdB))
    wB, wA = lrB.coef_.ravel(), lrA.coef_.ravel()
    r["cos_lr"] = float(wB @ wA / (np.linalg.norm(wB) * np.linalg.norm(wA)))
    r["eta_basescaler"] = dp(r["auc_tr_basescaler"]) / dp(r["auc_own"])
    r["eta_armscaler"] = dp(r["auc_tr_armscaler"]) / dp(r["auc_own"])
    r["eta_dm"] = dp(r["auc_dm_tr"]) / dp(r["auc_dm_own"])
    r["diff_tr"] = r["auc_tr_basescaler"] - r["auc_base"]
    r["diff_dm_tr"] = r["auc_dm_tr"] - r["auc_dm_base"]
    rows.append(r)

R = pd.DataFrame(rows)
def ci(v):
    v = np.asarray(v); n = len(v)
    return float(v.mean()), float(tdist.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n))
summary = {k: ci(R[k]) for k in ["auc_base", "auc_own", "auc_tr_basescaler", "auc_tr_armscaler",
                                 "diff_tr", "cos", "cos_lr", "eta_basescaler", "eta_armscaler",
                                 "auc_dm_base", "auc_dm_own", "auc_dm_tr", "diff_dm_tr", "eta_dm"]}
summary["eta_minus_cos"] = ci(R["eta_basescaler"] - R["cos"])
summary["eta_dm_minus_cos"] = ci(R["eta_dm"] - R["cos"])
summary["n_folds"] = len(R); summary["layers"] = R["layer"].tolist(); summary["n_token_mismatch"] = n_mismatch
print(R.round(3).to_string(index=False))
for k, (m, hw) in [(k, v) for k, v in summary.items() if isinstance(v, tuple)]:
    print(f"{a.name:10s} {k:18s} {m:+.3f} ± {hw:.3f}")
R.to_csv(f"{a.name}_folds.csv", index=False)
json.dump(summary, open(f"{a.name}_summary.json", "w"), indent=1)

# Per-layer sweep with the difference-of-means estimator: cosine and transfer at every layer (no layer selection).
sw = []
for l in range(L):
    cs, ab, at = [], [], []
    for h in np.unique(g):
        te, tr = np.where(g == h)[0], np.where(g != h)[0]
        if not two_class(te):
            continue
        muB, sdB = XB[l][tr].mean(0), XB[l][tr].std(0) + 1e-6
        cs.append(float(zdir(XB[l][tr], y[tr]) @ zdir(XA[l][tr], y[tr])))
        ab.append(roc_auc_score(y[te], dm_scores(XB[l][tr], y[tr], XB[l][te], muB, sdB)))
        at.append(roc_auc_score(y[te], dm_scores(XB[l][tr], y[tr], XA[l][te], muB, sdB)))
    sw.append(dict(layer=l, cos=np.mean(cs), auc_dm_base=np.mean(ab), auc_dm_tr=np.mean(at)))
pd.DataFrame(sw).round(4).to_csv(f"{a.name}_layers.csv", index=False)
print("per-layer sweep written to", f"{a.name}_layers.csv")
