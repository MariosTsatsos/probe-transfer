"""Independent-set evaluation: probes trained on ALL 450 XSTest prompts, never on the external set.

For one family at one fixed layer:
  own probe of each model (base, distilled, Instruct if given), trained on that model's XSTest activations
  transfer: the BASE probe (base scaler) applied to the distilled / Instruct external activations
  text-only baselines (TF-IDF char 3-5 and word 1-gram, LR C=1) trained on XSTest, scored on the external set
Metrics on the external set (orbench_hard = seemingly toxic but safe; orbench_toxic = unsafe; alpaca = ordinary safe):
  auc_hard_vs_toxic       AUC, orbench_hard (safe) against orbench_toxic (unsafe)
  auc_alpaca_vs_toxic     AUC, alpaca (safe) against orbench_toxic (unsafe)
  fpr_hard@0, fpr_alpaca@0, tpr_toxic@0      rates at the probe's own threshold (decision_function > 0)
  fpr_hard@t90, fpr_alpaca@t90, tpr_toxic@t90 rates at the threshold that gives TPR 0.90 on the XSTest unsafe
                                             prompts the probe was trained on (an in-sample threshold)
Bootstrap 95% intervals (percentile, 1000 resamples of external prompts, stratified by source) for the AUCs and
for the paired difference transfer minus base. External prompts are independent, so these intervals do not
have the fold-dependence problem of the XSTest intervals.
Usage: python eval_external.py --xs xstest.csv --ext external.csv --layer 18 --name out/math-7b
         --xs_base acts_math-7b_base.npz --xs_arm acts_math-7b_arm.npz [--xs_inst acts_math-7b_inst.npz]
         --ext_base ext_math-7b_base.npz --ext_arm ext_math-7b_arm.npz [--ext_inst ext_math-7b_inst.npz]
"""
import argparse, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

p = argparse.ArgumentParser()
for k in ["xs", "ext", "name", "xs_base", "xs_arm", "ext_base", "ext_arm"]:
    p.add_argument(f"--{k}", required=True)
p.add_argument("--xs_inst"); p.add_argument("--ext_inst")
p.add_argument("--layer", type=int, required=True)
p.add_argument("--C", type=float, default=1.0)
p.add_argument("--boot", type=int, default=1000)
a = p.parse_args()
rng = np.random.default_rng(0)

xs = pd.read_csv(a.xs); ext = pd.read_csv(a.ext)
y_xs = (xs["label"].str.lower() == "unsafe").astype(int).values
y_ext = (ext["label"].str.lower() == "unsafe").astype(int).values
src = ext["source"].values
hard, toxic, alpaca = src == "orbench_hard", src == "orbench_toxic", src == "alpaca"

def load_xs(path):            # all-layers file (extract.py or extract_subset.py with layers=all)
    z = np.load(path); return z["acts"][a.layer].astype(np.float32), z["tokh"]
def load_ext(path):           # subset file: find the requested layer in its 'layers' array
    z = np.load(path); idx = int(np.where(z["layers"] == a.layer)[0][0]); return z["acts"][idx].astype(np.float32), z["tokh"]

models = {"base": (a.xs_base, a.ext_base), "distilled": (a.xs_arm, a.ext_arm)}
if a.xs_inst and a.ext_inst:
    models["instruct"] = (a.xs_inst, a.ext_inst)
XS, EX, TOK = {}, {}, {}
for m, (px, pe) in models.items():
    XS[m], th_x = load_xs(px); EX[m], th_e = load_ext(pe); TOK[m] = (th_x, th_e)
for m in list(models)[1:]:
    print(f"{m}: XSTest prompts tokenised differently from base: {(TOK[m][0] != TOK['base'][0]).sum()} of {len(y_xs)}; "
          f"external: {(TOK[m][1] != TOK['base'][1]).sum()} of {len(y_ext)}")

def fit(X, y):
    sc = StandardScaler().fit(X); lr = LogisticRegression(C=a.C, max_iter=5000).fit(sc.transform(X), y)
    s_tr = lr.decision_function(sc.transform(X))
    t90 = float(np.quantile(s_tr[y == 1], 0.10))      # threshold with TPR 0.90 on the training unsafe prompts
    return sc, lr, t90
probes = {m: fit(XS[m], y_xs) for m in models}

def metrics(s, t90):
    r = dict(auc_hard_vs_toxic=roc_auc_score(y_ext[hard | toxic], s[hard | toxic]),
             auc_alpaca_vs_toxic=roc_auc_score(y_ext[alpaca | toxic], s[alpaca | toxic]))
    for tag, t in [("0", 0.0), ("t90", t90)]:
        r[f"fpr_hard@{tag}"] = float((s[hard] > t).mean()); r[f"fpr_alpaca@{tag}"] = float((s[alpaca] > t).mean())
        r[f"tpr_toxic@{tag}"] = float((s[toxic] > t).mean())
    return r

scores = {}
for m in models:                                   # own probes
    sc, lr, t90 = probes[m]; scores[f"own_{m}"] = (lr.decision_function(sc.transform(EX[m])), t90)
scB, lrB, t90B = probes["base"]
for m in [k for k in models if k != "base"]:       # base probe transferred
    scores[f"transfer_{m}"] = (lrB.decision_function(scB.transform(EX[m])), t90B)
# text-only baselines trained on XSTest text
for tag, vec in [("char35", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
                 ("word1", TfidfVectorizer(analyzer="word", ngram_range=(1, 1), sublinear_tf=True))]:
    V = vec.fit(xs["prompt"]); lr = LogisticRegression(C=a.C, max_iter=5000).fit(V.transform(xs["prompt"]), y_xs)
    s_tr = lr.decision_function(V.transform(xs["prompt"]))
    scores[f"text_{tag}"] = (lr.decision_function(V.transform(ext["prompt"])), float(np.quantile(s_tr[y_xs == 1], 0.10)))

rows = {k: metrics(s, t) for k, (s, t) in scores.items()}

# bootstrap over external prompts, stratified by source
idx_by_src = {s_: np.where(src == s_)[0] for s_ in np.unique(src)}
def resample():
    return np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in idx_by_src.values()])
boot = {k: {"auc_hard_vs_toxic": [], "auc_alpaca_vs_toxic": []} for k in scores}
pairs = [k for k in scores if k.startswith("transfer_")]
boot_diff = {k: [] for k in pairs}
for _ in range(a.boot):
    ix = resample(); yb, hb, tb, ab = y_ext[ix], hard[ix], toxic[ix], alpaca[ix]
    for k, (s, _t) in scores.items():
        sb = s[ix]
        boot[k]["auc_hard_vs_toxic"].append(roc_auc_score(yb[hb | tb], sb[hb | tb]))
        boot[k]["auc_alpaca_vs_toxic"].append(roc_auc_score(yb[ab | tb], sb[ab | tb]))
    for k in pairs:
        boot_diff[k].append(boot[k]["auc_hard_vs_toxic"][-1] - boot["own_base"]["auc_hard_vs_toxic"][-1])
ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
for k in scores:
    for mkey in ["auc_hard_vs_toxic", "auc_alpaca_vs_toxic"]:
        rows[k][mkey + "_ci"] = ci(boot[k][mkey])
for k in pairs:
    rows[k]["diff_vs_base_auc_hard"] = rows[k]["auc_hard_vs_toxic"] - rows["own_base"]["auc_hard_vs_toxic"]
    rows[k]["diff_vs_base_auc_hard_ci"] = ci(boot_diff[k])

# per-category false-positive rate on orbench_hard, base probe at t90
sB, tB = scores["own_base"]
cat = ext["category"].values
percat = {c: float((sB[hard & (cat == c)] > tB).mean()) for c in np.unique(cat[hard])}

T = pd.DataFrame(rows).T
print(f"layer {a.layer}; external set: hard {hard.sum()}, toxic {toxic.sum()}, alpaca {alpaca.sum()}")
print(T[["auc_hard_vs_toxic", "auc_alpaca_vs_toxic", "fpr_hard@t90", "fpr_alpaca@t90", "tpr_toxic@t90",
         "fpr_hard@0", "fpr_alpaca@0", "tpr_toxic@0"]].round(3).to_string())
for k in pairs:
    print(f"{k}: AUC(hard vs toxic) minus base {rows[k]['diff_vs_base_auc_hard']:+.3f}, bootstrap 95% {np.round(rows[k]['diff_vs_base_auc_hard_ci'], 3)}")
print("base probe FPR on orbench_hard by category at t90:", {k: round(v, 3) for k, v in percat.items()})
T.to_csv(f"{a.name}_external_table.csv")
json.dump(dict(layer=a.layer, n=dict(hard=int(hard.sum()), toxic=int(toxic.sum()), alpaca=int(alpaca.sum())),
               results=rows, fpr_hard_by_category_base_t90=percat), open(f"{a.name}_external.json", "w"), indent=1)
S = pd.DataFrame({k: s for k, (s, _t) in scores.items()}); S.insert(0, "source", src); S.insert(1, "label", ext["label"].values)
S.to_csv(f"{a.name}_external_scores.csv", index=False)
print("written", f"{a.name}_external_table.csv", f"{a.name}_external.json", f"{a.name}_external_scores.csv")
