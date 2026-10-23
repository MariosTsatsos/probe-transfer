"""Direction cosines in a common coordinate system, plus a fixed-layer probe check.

For each family and layer, the difference-of-means direction (unsafe minus safe) is compared between
base and distilled model in three coordinate systems:
  raw        raw residual-stream coordinates (shared by construction: same width, same architecture)
  basescale  both models' raw directions divided elementwise by the BASE model's per-coordinate std
  ownscale   each model's direction in its own standardised coordinates (the v1/v2 number, for comparison)
The logistic-probe weight vectors are compared after mapping each into raw coordinates (w / sd of the
model it was trained on), which is the coordinate system in which transfer actually happens.
Layers: the fixed layer used for controls (median of the fold layers), every fold's chosen layer, and
a per-layer sweep. Also: the probe at the fixed layer on the same 8 folds (fixed-layer robustness check).
Usage: python cosines.py --csv xstest_prompts.csv --base acts_X_base.npz --arm acts_X_arm.npz --name X
"""
import argparse, json
import numpy as np, pandas as pd
from scipy.stats import t as tdist
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

p = argparse.ArgumentParser()
p.add_argument("--csv", required=True); p.add_argument("--base", required=True)
p.add_argument("--arm", required=True); p.add_argument("--name", required=True)
p.add_argument("--C", type=float, default=1.0)
a = p.parse_args()

df = pd.read_csv(a.csv)
y = (df["label"].str.lower() == "unsafe").astype(int).values
g = df["type"].str.replace(r"^contrast_", "", regex=True)
g = g.where(~g.str.contains("privacy"), "privacy").where(~g.str.contains("discr"), "discr").values
XB = np.load(a.base)["acts"].astype(np.float32); XA = np.load(a.arm)["acts"].astype(np.float32)
L = XB.shape[0]
folds = pd.read_csv(f"{a.name}_folds.csv")
fixed = int(np.median(folds["layer"]))

def unit(v): return v / (np.linalg.norm(v) + 1e-12)
def cosines(Xb, Xa, yy):
    dB = Xb[yy == 1].mean(0) - Xb[yy == 0].mean(0)
    dA = Xa[yy == 1].mean(0) - Xa[yy == 0].mean(0)
    sB, sA = Xb.std(0) + 1e-6, Xa.std(0) + 1e-6
    return dict(raw=float(unit(dB) @ unit(dA)),
                basescale=float(unit(dB / sB) @ unit(dA / sB)),
                ownscale=float(unit(dB / sB) @ unit(dA / sA)))
def ci(v):
    v = np.asarray(v); return float(v.mean()), float(tdist.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))

# 1. fixed layer, all prompts
allp = np.arange(len(y))
out = {"fixed_layer": fixed, "fixed_layer_all_prompts": cosines(XB[fixed], XA[fixed], y)}

# 2. per fold at that fold's chosen layer: diff-of-means in three systems + probe weights in raw coordinates
rows = []
for _, r in folds.iterrows():
    h, l = r["group"], int(r["layer"]); tr = np.where(g != h)[0]
    c = cosines(XB[l][tr], XA[l][tr], y[tr])
    scB = StandardScaler().fit(XB[l][tr]); scA = StandardScaler().fit(XA[l][tr])
    wB = LogisticRegression(C=a.C, max_iter=5000).fit(scB.transform(XB[l][tr]), y[tr]).coef_.ravel()
    wA = LogisticRegression(C=a.C, max_iter=5000).fit(scA.transform(XA[l][tr]), y[tr]).coef_.ravel()
    c["probe_raw"] = float(unit(wB / scB.scale_) @ unit(wA / scA.scale_))   # raw-coordinate probe directions
    c["probe_ownscale"] = float(unit(wB) @ unit(wA))                        # the v2 number
    rows.append(dict(group=h, layer=l, **c))
R = pd.DataFrame(rows); R.to_csv(f"{a.name}_cosines_folds.csv", index=False)
out["fold_mean"] = {k: ci(R[k]) for k in ["raw", "basescale", "ownscale", "probe_raw", "probe_ownscale"]}

# 3. per-layer sweep, all prompts
sw = [dict(layer=l, **cosines(XB[l], XA[l], y)) for l in range(L)]
pd.DataFrame(sw).round(4).to_csv(f"{a.name}_cosines_layers.csv", index=False)

# 4. fixed-layer probe check on the same 8 folds (base, own, transfer with base scaler)
fx = []
for h in sorted(set(g)):
    te, tr = np.where(g == h)[0], np.where(g != h)[0]
    scB = StandardScaler().fit(XB[fixed][tr]); scA = StandardScaler().fit(XA[fixed][tr])
    lrB = LogisticRegression(C=a.C, max_iter=5000).fit(scB.transform(XB[fixed][tr]), y[tr])
    lrA = LogisticRegression(C=a.C, max_iter=5000).fit(scA.transform(XA[fixed][tr]), y[tr])
    fx.append(dict(group=h, auc_base=roc_auc_score(y[te], lrB.decision_function(scB.transform(XB[fixed][te]))),
                   auc_own=roc_auc_score(y[te], lrA.decision_function(scA.transform(XA[fixed][te]))),
                   auc_tr=roc_auc_score(y[te], lrB.decision_function(scB.transform(XA[fixed][te])))))
F = pd.DataFrame(fx); F["diff_tr"] = F.auc_tr - F.auc_base
F.to_csv(f"{a.name}_fixedlayer_folds.csv", index=False)
out["fixed_layer_probe"] = {k: ci(F[k]) for k in ["auc_base", "auc_own", "auc_tr", "diff_tr"]}

json.dump(out, open(f"{a.name}_cosines_summary.json", "w"), indent=1)
print(f"== {a.name}: fixed layer {fixed}")
print("diff-of-means cosine, fixed layer, all prompts:", {k: round(v, 3) for k, v in out["fixed_layer_all_prompts"].items()})
print("fold means (mean, 95% half-width):", {k: (round(m, 3), round(h, 3)) for k, (m, h) in out["fold_mean"].items()})
print("fixed-layer probe:", {k: (round(m, 3), round(h, 3)) for k, (m, h) in out["fixed_layer_probe"].items()})
