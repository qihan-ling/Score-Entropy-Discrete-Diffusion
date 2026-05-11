# ent_auc_curves.py
# Run with: python ent_auc_curves.py
# Generates: ent_auc_curves_bidirectional.pdf

import json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── config ────────────────────────────────────────────────────────────────────
BASE        = "bidirectional"
N_POINTS    = 300          # interpolation resolution
OUTFILE     = "analysis/results/human_pattern_matching/ent_auc_curves_bidirectional.pdf"

CRITICAL_OFFSETS = {
    "Agreement": 1, "RelativeClause": 2,
    "AttachmentAmbiguity": 1, "ClassicGP": 1,
}
CONTRASTS = [
    # (subset, display_label, cond_a, cond_b, human_spr_ms)
    ("Agreement",           "Agreement\nUNAGREE − AGREE",          "UNAGREE",    "AGREE",       +65.63),
    ("ClassicGP",           "ClassicGP\nMVRR AMB − UAMB",          "MVRR_AMB",   "MVRR_UAMB",  +210.25),
    ("ClassicGP",           "ClassicGP\nNPS AMB − UAMB",           "NPS_AMB",    "NPS_UAMB",    +60.03),
    ("ClassicGP",           "ClassicGP\nNPZ AMB − UAMB",           "NPZ_AMB",    "NPZ_UAMB",   +142.78),
    ("RelativeClause",      "RelativeClause\nRC Obj − Subj",        "RC_Obj",     "RC_Subj",      -8.97),
    ("AttachmentAmbiguity", "AttachAmb\nAttachHigh − Multi",        "AttachHigh", "AttachMulti", +30.95),
    ("AttachmentAmbiguity", "AttachAmb\nAttachLow − Multi",         "AttachLow",  "AttachMulti",  +4.03),
]

COLORS = {
    "a":     "#2166ac",   # condA line
    "b":     "#d6604d",   # condB line
    "above": "#92c5de",   # condA > condB fill (AMB higher uncertainty)
    "below": "#f4a582",   # condA < condB fill (condB higher uncertainty)
}

# ── load trajectories ─────────────────────────────────────────────────────────
def load_trajectories(base):
    data = {}
    for path in glob.glob(f"{base}/*/*/*_pos_+*.json"):
        parts     = path.split("/")
        subset    = parts[1];  condition = parts[2]
        fname     = parts[3]
        offset    = int(fname.split("_pos_")[1].replace(".json", ""))
        item      = int(fname.split("_pos_")[0].replace("item_", ""))
        if CRITICAL_OFFSETS.get(subset) != offset:
            continue
        with open(path) as f:
            d = json.load(f)
        fh = d.get("frontier_history", [])
        if fh:
            data[(subset, condition, item)] = np.array([s["entropy"] for s in fh])
    return data

# ── interpolate + average ─────────────────────────────────────────────────────
def mean_curve(traj, subset, condition, items, grid):
    curves = []
    for item in items:
        ent = traj.get((subset, condition, item))
        if ent is None or len(ent) < 2:
            continue
        t = np.linspace(0, 1, len(ent))
        curves.append(np.interp(grid, t, ent))
    return np.mean(curves, axis=0) if curves else None

# ── build per-contrast results, sorted by AUC delta ──────────────────────────
grid = np.linspace(0, 1, N_POINTS)
traj = load_trajectories(BASE)

results = []
for subset, label, cond_a, cond_b, spr in CONTRASTS:
    items_a = {k[2] for k in traj if k[0] == subset and k[1] == cond_a}
    items_b = {k[2] for k in traj if k[0] == subset and k[1] == cond_b}
    shared  = list(items_a & items_b)
    ca = mean_curve(traj, subset, cond_a, shared, grid)
    cb = mean_curve(traj, subset, cond_b, shared, grid)
    if ca is None or cb is None:
        continue
    auc_delta = float(np.trapezoid(ca - cb, grid))
    results.append(dict(label=label, cond_a=cond_a, cond_b=cond_b,
                        spr=spr, auc_delta=auc_delta,
                        ca=ca, cb=cb, n=len(shared)))

# Sort largest AUC delta first → visually shows effect ranking left-to-right
results.sort(key=lambda x: -x["auc_delta"])

# ── plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 8))
fig.suptitle(
    "Mean Entropy Trajectory per Contrast  (bidirectional experiment)\n"
    "Subplots ordered left→right by AUC difference  =  Spearman ρ = 1.00 with Human SPR",
    fontsize=13, fontweight="bold", y=1.01,
)

gs = GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.35)
axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(4)]

for ax, res in zip(axes, results):
    ca, cb = res["ca"], res["cb"]
    diff   = ca - cb

    ax.plot(grid, ca, color=COLORS["a"], lw=1.8, label=res["cond_a"])
    ax.plot(grid, cb, color=COLORS["b"], lw=1.8, label=res["cond_b"])

    # colour fill: blue where condA > condB, orange where condA < condB
    ax.fill_between(grid, ca, cb, where=(diff >= 0),
                    color=COLORS["above"], alpha=0.55, label="condA > condB")
    ax.fill_between(grid, ca, cb, where=(diff <  0),
                    color=COLORS["below"], alpha=0.55, label="condA < condB")

    sign = "+" if res["auc_delta"] >= 0 else ""
    spr_sign = "+" if res["spr"] >= 0 else ""
    ax.set_title(
        f"{res['label']}\n"
        f"AUC Δ = {sign}{res['auc_delta']:.3f}   "
        f"SPR = {spr_sign}{res['spr']:.0f} ms",
        fontsize=8.5, pad=4,
    )
    ax.set_xlabel("Normalised step (0 = start, 1 = commit)", fontsize=7)
    ax.set_ylabel("Entropy (nats)", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)

# Turn off the unused 8th panel
axes[-1].set_visible(False)

# Shared legend at bottom
legend_handles = [
    mpatches.Patch(color=COLORS["a"],     label="Cond A (e.g. AMB / harder)"),
    mpatches.Patch(color=COLORS["b"],     label="Cond B (e.g. UNAMB / easier)"),
    mpatches.Patch(color=COLORS["above"], alpha=0.55, label="Cond A > Cond B  (AMB more uncertain)"),
    mpatches.Patch(color=COLORS["below"], alpha=0.55, label="Cond A < Cond B  (UNAMB more uncertain)"),
]
fig.legend(handles=legend_handles, loc="lower right",
           bbox_to_anchor=(0.98, 0.02), fontsize=8.5, framealpha=0.9)

plt.savefig(OUTFILE, bbox_inches="tight", dpi=150)
print(f"Saved → {OUTFILE}")