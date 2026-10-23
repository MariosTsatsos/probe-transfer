"""Rotation of many directions under distillation, to place the safety direction in context.

v2 (controls_bs.py): adds the common-coordinate ("base-scaled") cosines. In the own-scaled columns each model
is divided by its own per-coordinate standard deviation (different coordinate systems); in the base-scaled
columns BOTH models are divided by the base model's standard deviations (one coordinate system).
  cos_full_bs, cos_cross_bs, floor_bs, ratio_bs   base-scaled versions of the columns below
Outputs <name>_controls_bs.csv and <name>_controls_bs_summary.json (new names, nothing overwritten).

For each labelling of the prompts, build the z-scored difference-of-means direction in the base
model and in the distilled model (same fixed layer) and measure:
  cos_full   cosine between the two directions built on all prompts
  cos_cross  cosine between base direction (one random half) and distilled direction (other half), mean and 95% range
  floor      within-model split-half cosine (two random halves of the same model), the noise ceiling
  ratio      cos_cross / floor: agreement corrected for estimation noise (lower = more rotation).
             Empirically equal to cos_full within 0.01, so cos_full is already a clean rotation measure.
  rotation   floor - cos_cross (kept for reference; a difference, not the right correction)
  assoc      |phi| correlation between the labelling and the safety label (confound check)
Labellings: safety; all 35 balanced 4-vs-4 splits of the 8 prompt-type groups; question form;
named entity present; prompt length.
Usage: python controls.py --csv xstest_prompts.csv --base acts_X_base.npz --arm acts_X_arm.npz --name X [--layer L]
       (default layer = median of the layers chosen in X_folds.csv)
"""
import argparse, itertools, json, re
import numpy as np, pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--csv", required=True); p.add_argument("--base", required=True)
p.add_argument("--arm", required=True); p.add_argument("--name", required=True)
p.add_argument("--layer", type=int, default=None); p.add_argument("--boot", type=int, default=200)
a = p.parse_args()
rng = np.random.default_rng(0)

df = pd.read_csv(a.csv)
y_safe = (df["label"].str.lower() == "unsafe").astype(int).values
g = df["type"].str.replace(r"^contrast_", "", regex=True)
g = g.where(~g.str.contains("privacy"), "privacy").where(~g.str.contains("discr"), "discr").values
groups = sorted(set(g))

if a.layer is None:
    a.layer = int(np.median(pd.read_csv(f"{a.name}_folds.csv")["layer"]))
XB = np.load(a.base)["acts"][a.layer].astype(np.float32)
XA = np.load(a.arm)["acts"][a.layer].astype(np.float32)
N = len(df)

# ---- labellings -------------------------------------------------------------------------------
labels = {"safety": y_safe}
for k, half in enumerate(itertools.combinations(groups, 4)):
    if groups[0] not in half:           # each split once (its complement is the same split)
        continue
    labels[f"typesplit_{k:02d}"] = np.isin(g, half).astype(int)
first = df["prompt"].str.strip().str.split().str[0].str.lower()
labels["question_how"] = (first == "how").astype(int)
labels["named_entity"] = df["prompt"].apply(
    lambda s: int(bool(re.search(r"(?<!^)(?<![.?!]\s)\b[A-Z][a-z]+", s.strip())))).values
labels["length_long"] = (df["prompt"].str.len() > df["prompt"].str.len().median()).astype(int).values

# ---- helpers ----------------------------------------------------------------------------------
def zdir(X, yy, idx):
    Xi, yi = X[idx], yy[idx]
    Z = (Xi - Xi.mean(0)) / (Xi.std(0) + 1e-6)
    d = Z[yi == 1].mean(0) - Z[yi == 0].mean(0)
    return d / (np.linalg.norm(d) + 1e-12)

def sdir(X, yy, idx, sd):
    """difference of means with a GIVEN per-coordinate scale sd (base-scaled when sd = base std)"""
    Xi, yi = X[idx], yy[idx]
    d = (Xi[yi == 1].mean(0) - Xi[yi == 0].mean(0)) / sd
    return d / (np.linalg.norm(d) + 1e-12)

SD_B = XB.std(0) + 1e-6          # base model's scale, used for both models in the base-scaled columns

def cos(u, v): return float(u @ v)

def phi(u, v):
    u, v = u.astype(float), v.astype(float)
    if u.std() == 0 or v.std() == 0: return float("nan")
    return float(abs(np.corrcoef(u, v)[0, 1]))

allidx = np.arange(N)
rows = []
for name, yy in labels.items():
    n1 = int(yy.sum())
    if n1 < 20 or N - n1 < 20:
        continue
    c_full = cos(zdir(XB, yy, allidx), zdir(XA, yy, allidx))
    c_full_bs = cos(sdir(XB, yy, allidx, SD_B), sdir(XA, yy, allidx, SD_B))
    # Cross-model and within-model cosines on the same random half-splits, so they are comparable.
    cr, fl, cr_bs, fl_bs = [], [], [], []
    for _ in range(a.boot):
        perm = rng.permutation(N); h1, h2 = perm[: N // 2], perm[N // 2:]
        if len(np.unique(yy[h1])) < 2 or len(np.unique(yy[h2])) < 2: continue
        cr.append(cos(zdir(XB, yy, h1), zdir(XA, yy, h2)))          # base half vs distilled other half
        fl.append(cos(zdir(XB, yy, h1), zdir(XB, yy, h2)))          # base half vs base other half
        fl.append(cos(zdir(XA, yy, h1), zdir(XA, yy, h2)))          # distilled vs distilled
        cr_bs.append(cos(sdir(XB, yy, h1, SD_B), sdir(XA, yy, h2, SD_B)))
        fl_bs.append(cos(sdir(XB, yy, h1, SD_B), sdir(XB, yy, h2, SD_B)))
        fl_bs.append(cos(sdir(XA, yy, h1, SD_B), sdir(XA, yy, h2, SD_B)))
    rows.append(dict(direction=name, n_pos=n1, n_neg=N - n1, assoc_with_safety=round(phi(yy, y_safe), 3),
                     cos_full=round(c_full, 4), cos_cross=round(float(np.mean(cr)), 4),
                     ci_lo=round(np.percentile(cr, 2.5), 4), ci_hi=round(np.percentile(cr, 97.5), 4),
                     floor=round(float(np.mean(fl)), 4),
                     ratio=round(float(np.mean(cr) / np.mean(fl)), 4),          # noise-corrected agreement; equals cos_full to ~0.01
                     rotation=round(float(np.mean(fl) - np.mean(cr)), 4),
                     cos_full_bs=round(c_full_bs, 4), cos_cross_bs=round(float(np.mean(cr_bs)), 4),
                     floor_bs=round(float(np.mean(fl_bs)), 4),
                     ratio_bs=round(float(np.mean(cr_bs) / np.mean(fl_bs)), 4)))

R = pd.DataFrame(rows)
R.to_csv(f"{a.name}_controls_bs.csv", index=False)
splits = R[R.direction.str.startswith("typesplit")]
s = R[R.direction == "safety"].iloc[0]
rank = int((splits.ratio < s.ratio).sum())            # own-scaled: splits that rotate MORE than safety
rank_bs = int((splits.cos_full_bs < s.cos_full_bs).sum())   # base-scaled, on cos_full_bs
print(f"{a.name}: layer {a.layer}")
print(R[~R.direction.str.startswith("typesplit")].to_string(index=False))
print(f"type splits (n={len(splits)}), own-scaled ratio: median {splits.ratio.median():.3f}, "
      f"range {splits.ratio.min():.3f} to {splits.ratio.max():.3f}; {rank} of {len(splits)} rotate MORE than safety (safety {s.ratio:.3f})")
print(f"type splits (n={len(splits)}), base-scaled cos_full_bs: median {splits.cos_full_bs.median():.3f}, "
      f"range {splits.cos_full_bs.min():.3f} to {splits.cos_full_bs.max():.3f}; {rank_bs} of {len(splits)} rotate MORE than safety "
      f"(safety {s.cos_full_bs:.3f})")
json.dump(dict(layer=a.layer, safety_cos_full=float(s.cos_full), safety_ratio=float(s.ratio),
               split_ratio_median=float(splits.ratio.median()), n_splits_rotating_more=rank,
               safety_cos_full_bs=float(s.cos_full_bs), split_cos_bs_median=float(splits.cos_full_bs.median()),
               split_cos_bs_min=float(splits.cos_full_bs.min()), split_cos_bs_max=float(splits.cos_full_bs.max()),
               n_splits_rotating_more_bs=rank_bs, n_splits=len(splits)),
          open(f"{a.name}_controls_bs_summary.json", "w"), indent=1)
