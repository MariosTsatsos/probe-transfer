"""Figure 1: sketch of the idea in 3D (a, b) plus the measured share kept (c)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "numbers.json")))
FAM = ["math-7b", "llama-8b", "qwen-14b", "qwen-32b"]
SAFE, UNSAFE, INK, BLUE, GREY = "#2A9D8F", "#E76F51", "#1F2937", "#2F6FD6", "#9AA3B2"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "cm", "pdf.fonttype": 42})

fig = plt.figure(figsize=(6.5, 2.9))
fig.patch.set_facecolor("white")
ax_a = fig.add_axes([0.03, 0.0, 0.33, 1.0], projection="3d")
ax_b = fig.add_axes([0.34, 0.0, 0.33, 1.0], projection="3d")
ax_c = fig.add_axes([0.745, 0.36, 0.225, 0.5])

rng = np.random.default_rng(5)
n = 140
sep = 3.2


def rot(theta_deg, phi_deg):
    """direction at angle theta from x; the rotation is split between the populated y axis and the sparse z axis"""
    t = np.radians(theta_deg); p = np.radians(phi_deg)
    return np.array([np.cos(t), np.sin(t) * np.cos(p), np.sin(t) * np.sin(p)])


def clouds(d):
    # anisotropic noise: wide along x and y (populated), thin along z (sparse)
    base = rng.normal(size=(2 * n, 3)) * np.array([1.0, 1.3, 0.35])
    y = np.r_[np.zeros(n), np.ones(n)]
    X = base + np.outer(y - 0.5, sep * d)
    return X[y == 0], X[y == 1]


def panel(ax, d, title, note, base_dir=None):
    safe, unsafe = clouds(d)
    ax.scatter(*safe.T, s=5, color=SAFE, alpha=0.55, lw=0, depthshade=False)
    ax.scatter(*unsafe.T, s=5, color=UNSAFE, alpha=0.55, lw=0, depthshade=False)
    # probe boundary: plane x = 0 (perpendicular to the base direction), drawn as a translucent sheet
    yy, zz = np.meshgrid([-4, 4], [-2.2, 2.2])
    ax.plot_surface(np.zeros_like(yy), yy, zz, color=INK, alpha=0.10, edgecolor=INK, linewidth=0.8, linestyle="--")
    # direction arrow through the origin
    if base_dir is not None:
        # the base model's direction, for reference, and the arc marking theta between the two
        ax.quiver(-2.4 * base_dir[0], -2.4 * base_dir[1], -2.4 * base_dir[2], 4.8 * base_dir[0], 4.8 * base_dir[1], 4.8 * base_dir[2],
                  color=GREY, arrow_length_ratio=0.12, linewidth=1.6)
        ts = np.linspace(0, 1, 40)
        arc = np.array([(1 - t) * base_dir + t * d for t in ts]); arc = 2.9 * arc / np.linalg.norm(arc, axis=1, keepdims=True)
        ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color=BLUE, lw=1.6, zorder=10)
        m = arc[len(arc) // 2] * 1.25
        ax.text(m[0], m[1], m[2], r"$\theta$", color=BLUE, fontsize=11, weight="bold", zorder=11)
    ax.quiver(-2.4 * d[0], -2.4 * d[1], -2.4 * d[2], 4.8 * d[0], 4.8 * d[1], 4.8 * d[2],
              color=INK, arrow_length_ratio=0.12, linewidth=2.6)
    # sparse axis marker
    ax.plot([0, 0], [0, 0], [-2.4, 2.4], color=GREY, lw=0.8, ls=":")
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_zlim(-2.4, 2.4)
    ax.set_axis_off()
    ax.view_init(elev=26, azim=-55)
    ax.set_title(title, loc="left", fontsize=8.5, color=INK, y=0.9, x=0.08)
    ax.text2D(0.08, 0.05, note, transform=ax.transAxes, fontsize=6.6, color=INK)
    ax.text2D(0.62, 0.86, "sparse\ndirection", transform=ax.transAxes, fontsize=6.5, color=GREY)


d0 = rot(0, 0)
panel(ax_a, d0, "(a) Base model", "dashed plane: probe boundary\narrow: safety direction")
ax_a.text2D(0.12, 0.62, "safe", transform=ax_a.transAxes, color=SAFE, fontsize=7.5, weight="bold")
ax_a.text2D(0.74, 0.62, "unsafe", transform=ax_a.transAxes, color=UNSAFE, fontsize=7.5, weight="bold")

d1 = rot(32, 72)   # 32 degrees away from the base direction, more than half of it into the sparse z axis
panel(ax_b, d1, "(b) Successor, same probe", "safety direction rotated by $\\theta\\approx30^\\circ$\nfrom the base direction (grey),\npartly into the sparse direction;\nsame boundary, classes still split", base_dir=d0)

# (c) measured
y = np.arange(len(FAM))[::-1]
for i, f in enumerate(FAM):
    r = D[f]
    pred = r["dist_cos_bs"][0]
    obs, err = r["dist_share"][0], r["dist_share"][1]
    ax_c.plot([pred, obs], [y[i], y[i]], color=GREY, lw=1.0, zorder=1)
    ax_c.scatter(pred, y[i], s=34, marker="|", color=INK, lw=1.6, zorder=3, label="predicted from the angle ($\\cos\\theta$)" if i == 0 else None)
    ax_c.errorbar(obs, y[i], xerr=err, fmt="o", color=BLUE, ms=5, capsize=2, lw=1, zorder=4, label="measured" if i == 0 else None)
ax_c.axvline(1.0, color=GREY, lw=0.8, ls=":")
ax_c.set_yticks(y); ax_c.set_yticklabels(FAM)
ax_c.set_xlim(0.64, 1.14); ax_c.set_xticks([0.7, 0.8, 0.9, 1.0, 1.1])
ax_c.set_xlabel("Separation kept by base probe", fontsize=7.5)
ax_c.set_title("(c) Measured (distilled)", loc="left", fontsize=8.5, color=INK)
ax_c.legend(loc="upper left", bbox_to_anchor=(-0.3, -0.36), frameon=False, fontsize=6.3, handletextpad=0.5, labelspacing=0.3)
for s in ["top", "right", "left"]:
    ax_c.spines[s].set_visible(False)
ax_c.tick_params(axis="y", length=0)
out = os.path.join(HERE, "fig")
fig.savefig(os.path.join(out, "fig0_schematic.pdf")); fig.savefig(os.path.join(out, "fig0_schematic.png"), dpi=220)
print("ok")
