"""Figures for the paper. Reads numbers.json (built by gather.py from the result files). Writes fig/*.pdf."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "numbers.json")))
FAM = ["math-7b", "llama-8b", "qwen-14b", "qwen-32b"]
BLUE, ORANGE, GREY, DARK = "#2F6FD6", "#D9711F", "#8A94A6", "#14213D"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "font.family": "serif",
                     "mathtext.fontset": "cm"})
OUT = os.path.join(HERE, "fig")
os.makedirs(OUT, exist_ok=True)


def ext_err(f, key):
    e = D[f]["ext"][key]
    m = e["diff_vs_base_auc_hard"]
    lo, hi = e["diff_vs_base_auc_hard_ci"]
    return m, m - lo, hi - m


# ---------------- Figure 1: transfer ----------------
fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.5), gridspec_kw={"width_ratios": [1.25, 1]})
x = np.arange(len(FAM))
off = [-0.27, -0.09, 0.09, 0.27]
for i, f in enumerate(FAM):
    r = D[f]
    pts = [(r["dist_change"][0], r["dist_change"][1], r["dist_change"][1], BLUE, "o", True),
           (r["inst_change"][0], r["inst_change"][1], r["inst_change"][1], ORANGE, "s", True)]
    m, lo, hi = ext_err(f, "transfer_distilled"); pts.append((m, lo, hi, BLUE, "o", False))
    m, lo, hi = ext_err(f, "transfer_instruct"); pts.append((m, lo, hi, ORANGE, "s", False))
    for k, (m, lo, hi, c, mk, filled) in enumerate(pts):
        a.errorbar(i + off[k], m, yerr=[[lo], [hi]], fmt=mk, color=c, ms=4.5, capsize=2, lw=1,
                   mfc=c if filled else "white", mec=c, mew=1.1)
a.axhline(0, color=DARK, lw=0.8)
a.set_xticks(x); a.set_xticklabels(FAM)
a.set_ylabel("AUC change\n(successor $-$ base, same probe)")
a.set_ylim(-0.075, 0.075)
a.set_title("(a) Base probe applied unchanged", loc="left")
h = [plt.Line2D([], [], color=BLUE, marker="o", ls="", ms=4.5, label="distilled, XSTest"),
     plt.Line2D([], [], color=ORANGE, marker="s", ls="", ms=4.5, label="Instruct, XSTest"),
     plt.Line2D([], [], color=BLUE, marker="o", ls="", ms=4.5, mfc="white", label="distilled, OR-Bench"),
     plt.Line2D([], [], color=ORANGE, marker="s", ls="", ms=4.5, mfc="white", label="Instruct, OR-Bench")]
a.legend(handles=h, ncol=2, loc="lower center", frameon=False, handletextpad=0.3, columnspacing=0.8)

LBL = {("qwen-32b", "d"): (26.0, 0.052), ("qwen-14b", "d"): (27.0, -0.050), ("math-7b", "d"): (36.3, 0.035),
       ("llama-8b", "d"): (36.0, -0.055), ("qwen-14b", "i"): (13.0, 0.055), ("qwen-32b", "i"): (13.0, -0.030),
       ("llama-8b", "i"): (21.5, 0.058), ("math-7b", "i"): (29.5, -0.068)}
for f in FAM:
    r = D[f]
    for ang, ch, c, mk, t in [(r["dist_angle"], r["dist_change"], BLUE, "o", "d"), (r["inst_angle"], r["inst_change"], ORANGE, "s", "i")]:
        b.errorbar(ang, ch[0], yerr=ch[1], fmt=mk, color=c, ms=4.5, capsize=2, lw=1)
        b.annotate(f, (ang, ch[0]), xytext=LBL[(f, t)], textcoords="data", fontsize=6.5, color=c,
                   arrowprops=dict(arrowstyle="-", color=c, lw=0.5, shrinkA=0, shrinkB=3))
b.axhline(0, color=DARK, lw=0.8)
b.set_xlim(12, 41); b.set_ylim(-0.075, 0.075)
b.set_xlabel("Rotation of the label direction (degrees)")
b.set_title("(b) Rotation versus change (XSTest)", loc="left")
b.legend(handles=[plt.Line2D([], [], color=BLUE, marker="o", ls="", ms=4.5, label="distilled"),
                  plt.Line2D([], [], color=ORANGE, marker="s", ls="", ms=4.5, label="Instruct")],
         loc="upper right", frameon=False)
fig.tight_layout(w_pad=1.5)
fig.savefig(os.path.join(OUT, "fig1_transfer.pdf")); fig.savefig(os.path.join(OUT, "fig1_transfer.png"), dpi=200)
plt.close(fig)

# ---------------- Figure 2: textual cues ----------------
fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.5))
w = 0.34
xs = [D[f]["base"][0] for f in FAM]
xe = [D[f]["base"][1] for f in FAM]
ob = [D[f]["ext"]["own_base"]["auc_hard_vs_toxic"] for f in FAM]
obe = [[o - D[f]["ext"]["own_base"]["auc_hard_vs_toxic_ci"][0] for o, f in zip(ob, FAM)],
       [D[f]["ext"]["own_base"]["auc_hard_vs_toxic_ci"][1] - o for o, f in zip(ob, FAM)]]
for i in range(len(FAM)):
    a.plot([i - 0.12, i + 0.12], [xs[i], ob[i]], color=GREY, lw=0.8, zorder=1)
a.errorbar(x - 0.12, xs, yerr=xe, fmt="o", color=BLUE, ms=5, capsize=2, lw=1, label="probe, XSTest")
a.errorbar(x + 0.12, ob, yerr=obe, fmt="o", color=ORANGE, ms=5, capsize=2, lw=1, mfc="white", mew=1.2, label="probe, OR-Bench")
tx = D["text"]["char35"]["mean_ci"][0]
to = D["math-7b"]["ext"]["text_char35"]["auc_hard_vs_toxic"]
a.axhline(tx, color=BLUE, ls="--", lw=1, label="text only, XSTest"); a.axhline(to, color=ORANGE, ls="--", lw=1, label="text only, OR-Bench")
a.set_ylim(0.5, 1.0); a.set_xticks(x); a.set_xticklabels(FAM)
a.set_ylabel("AUC (0.5 = chance)")
a.set_title("(a) Probe versus text-only classifier", loc="left")
a.legend(loc="lower center", frameon=False, ncol=2, handletextpad=0.4, columnspacing=1.0)

cats = [("orbench_toxic", "tpr_toxic@0", "toxic (correctly flagged)", DARK),
        ("orbench_hard", "fpr_hard@0", "benign but sounds harmful", ORANGE),
        ("alpaca", "fpr_alpaca@0", "ordinary (Alpaca)", GREY)]
w3 = 0.26
for k, (_, key, lab, c) in enumerate(cats):
    vals = [100 * D[f]["ext"]["own_base"][key] for f in FAM]
    b.bar(x + (k - 1) * w3, vals, w3, color=c, label=lab)
b.set_ylim(0, 100); b.set_xticks(x); b.set_xticklabels(FAM)
b.set_ylabel("Prompts flagged (%)")
b.set_title("(b) Flagged at the probe's own threshold", loc="left")
b.legend(loc="upper left", frameon=False, ncol=1, bbox_to_anchor=(0, 1.02))
fig.tight_layout(w_pad=1.5)
fig.savefig(os.path.join(OUT, "fig2_cues.pdf")); fig.savefig(os.path.join(OUT, "fig2_cues.png"), dpi=200)
plt.close(fig)

# ---------------- Figure A1: layer sweep ----------------
fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.4))
cols = {"math-7b": "#4C72B0", "llama-8b": "#DD8452", "qwen-14b": "#55A868", "qwen-32b": "#8172B3"}
for f in FAM:
    s = D[f]["sweep"]
    n = len(s["dist_cos"]); depth = np.arange(n) / (n - 1)
    a.plot(depth, s["dist_cos"], color=cols[f], lw=1.2, label=f)
    a.plot(depth, s["inst_cos"], color=cols[f], lw=1.2, ls="--")
    b.plot(depth, s["dist_change"], color=cols[f], lw=1.2)
    b.plot(depth, s["inst_change"], color=cols[f], lw=1.2, ls="--")
a.set_xlabel("Relative depth (layer / last layer)"); a.set_ylabel("Cosine, base vs successor")
a.set_title("(a) Label-direction cosine at every layer", loc="left"); a.set_ylim(0.6, 1.0)
h = [plt.Line2D([], [], color=cols[f], lw=1.2, label=f) for f in FAM] + \
    [plt.Line2D([], [], color=DARK, lw=1.2, label="distilled"), plt.Line2D([], [], color=DARK, lw=1.2, ls="--", label="Instruct")]
b.legend(handles=h, loc="lower left", frameon=False, ncol=2)
b.axhline(0, color=DARK, lw=0.8)
b.set_xlabel("Relative depth (layer / last layer)"); b.set_ylabel("AUC change, mean-difference reader")
b.set_title("(b) Transfer change at every layer", loc="left")
mx = max(max(abs(v) for v in D[f]["sweep"][k]) for f in FAM for k in ["dist_change", "inst_change"])
b.set_ylim(-1.1 * mx, 1.1 * mx)
fig.tight_layout(w_pad=1.5)
fig.savefig(os.path.join(OUT, "figA1_layers.pdf")); fig.savefig(os.path.join(OUT, "figA1_layers.png"), dpi=200)
plt.close(fig)

# ---------------- Figure A2: control directions ----------------
fig, ax = plt.subplots(figsize=(6.5, 2.2))
rng = np.random.default_rng(0)
for i, f in enumerate(FAM):
    c = D[f]["controls"]
    jit = rng.uniform(-0.12, 0.12, len(c["splits"]))
    ax.scatter(i + jit, c["splits"], s=9, color=GREY, alpha=0.8, label="35 prompt-type splits" if i == 0 else None, lw=0)
    ax.scatter(i + 0.3, c["label"], s=40, color=BLUE, marker="D", label="XSTest label direction" if i == 0 else None, zorder=3)
    ax.scatter(i + 0.42, c["how"], s=22, color=ORANGE, marker="^", label="first word \"How\"" if i == 0 else None)
    ax.scatter(i + 0.18, c["named"], s=22, color="#55A868", marker="v", label="capitalised word" if i == 0 else None)
    ax.scatter(i + 0.3, c["length"], s=22, color=DARK, marker="x", label="longer than median" if i == 0 else None)
ax.set_xticks(np.arange(4)); ax.set_xticklabels(FAM); ax.set_xlim(-0.5, 3.6)
ax.set_ylabel("Cosine, base vs distilled\n(lower = more rotation)")
ax.legend(loc="lower left", frameon=False, ncol=3, fontsize=6.5)
ax.set_ylim(0.5, 0.95)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figA2_controls.pdf")); fig.savefig(os.path.join(OUT, "figA2_controls.png"), dpi=200)
plt.close(fig)
print("figures written to", OUT)
