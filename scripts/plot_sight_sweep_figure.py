#!/usr/bin/env python3
"""Generate the main sight-sweep figure for the MQP report."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

# --- Data ---

# Q3_rw multi-seed (5 seeds x 500 eps each)
rw_sights = np.array([3, 4, 5, 6, 7])
rw_iqm = np.array([0.159575, 0.200017, 0.147089, 0.153996, 0.145011])
rw_ci_lo = np.array([0.150321, 0.182177, 0.137588, 0.144322, 0.121096])
rw_ci_hi = np.array([0.198716, 0.272840, 0.175688, 0.180089, 0.155423])

# Per-seed IQMs for jitter/swarm
rw_per_seed = {
    3: [0.2051, 0.2323, 0.2698, 0.2359, 0.2358],
    4: [0.2580, 0.3140, 0.3268, 0.3050, 0.2923],
    5: [0.1850, 0.2112, 0.2428, 0.1852, 0.2140],
    6: [0.2214, 0.2301, 0.2749, 0.1964, 0.2074],
    7: [0.1939, 0.1895, 0.1955, 0.2012, 0.1927],
}

# Q1 baseline (sight=8, levels ON, 5 seeds)
q1_iqm = 0.110
q1_ci_lo = 0.101
q1_ci_hi = 0.136
q1_per_seed = [0.1528, 0.1577, 0.1845, 0.1469, 0.1607]

# Q3-inf-aux single seed
inf_sights = np.array([3, 4, 5, 6, 7])
inf_iqm = np.array([0.161035, 0.192941, 0.117487, 0.064164, 0.158488])
inf_ci_lo = np.array([0.114648, 0.155620, 0.064000, 0.045672, 0.131465])
inf_ci_hi = np.array([0.189600, 0.281643, 0.136534, 0.099112, 0.240687])

# --- Plot ---

fig, ax = plt.subplots(figsize=(10, 6.5))

# Background shading for the "sweet spot" region
ax.axvspan(3.6, 4.4, alpha=0.08, color="#2ecc71", zorder=0)

# Q1 baseline band (sight=8, full obs)
ax.axhspan(q1_ci_lo, q1_ci_hi, alpha=0.12, color="#e74c3c", zorder=0)
ax.axhline(q1_iqm, color="#e74c3c", linewidth=1.2, linestyle="--", alpha=0.7, zorder=1)
ax.text(7.55, q1_iqm + 0.004, "Q1 baseline\n(sight=8, levels visible)",
        fontsize=8.5, color="#c0392b", ha="right", va="bottom",
        fontstyle="italic")

# Q3_rw main curve with CI band
ax.fill_between(rw_sights, rw_ci_lo, rw_ci_hi,
                alpha=0.18, color="#2980b9", zorder=2)
ax.plot(rw_sights, rw_iqm, "o-", color="#2980b9", linewidth=2.5,
        markersize=9, markeredgecolor="white", markeredgewidth=1.5,
        zorder=4, label="GPL-Q (levels hidden, 5 seeds)")

# Per-seed jitter dots
rng = np.random.default_rng(42)
for s in rw_sights:
    seeds = rw_per_seed[s]
    jitter = rng.uniform(-0.08, 0.08, size=len(seeds))
    ax.scatter(np.full(len(seeds), s) + jitter, seeds,
              color="#2980b9", alpha=0.35, s=22, zorder=3,
              edgecolors="none")

# Q3-inf-aux curve (thinner, dashed)
ax.fill_between(inf_sights, inf_ci_lo, inf_ci_hi,
                alpha=0.10, color="#8e44ad", zorder=2)
ax.plot(inf_sights, inf_iqm, "s--", color="#8e44ad", linewidth=1.8,
        markersize=7, markeredgecolor="white", markeredgewidth=1.2,
        zorder=4, label="GPL-Q + aux head (levels hidden, 1 seed)")

# Q1 per-seed dots at sight=8
q1_jitter = rng.uniform(-0.08, 0.08, size=len(q1_per_seed))
ax.scatter(np.full(len(q1_per_seed), 8) + q1_jitter, q1_per_seed,
          color="#e74c3c", alpha=0.45, s=22, zorder=3,
          edgecolors="none")
ax.plot(8, q1_iqm, "D", color="#e74c3c", markersize=9,
        markeredgecolor="white", markeredgewidth=1.5, zorder=4,
        label="Q1 baseline (full observability, 5 seeds)")

# Annotate the peak
peak_idx = np.argmax(rw_iqm)
ax.annotate(
    "Peak: sight = {}\nIQM = {:.3f}".format(rw_sights[peak_idx], rw_iqm[peak_idx]),
    xy=(rw_sights[peak_idx], rw_iqm[peak_idx]),
    xytext=(5.2, 0.24),
    fontsize=9.5, fontweight="bold", color="#1a5276",
    arrowprops=dict(arrowstyle="-|>", color="#1a5276", lw=1.5,
                    connectionstyle="arc3,rad=-0.2"),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
              edgecolor="#2980b9", alpha=0.9),
    zorder=5
)

# Annotate the drop
ax.annotate(
    "More sight\n= worse",
    xy=(6.5, 0.149),
    xytext=(6.5, 0.115),
    fontsize=8.5, color="#7f8c8d", ha="center",
    arrowprops=dict(arrowstyle="-|>", color="#7f8c8d", lw=1.2),
    zorder=5
)

# Formatting
ax.set_xlabel("Sight Radius", fontsize=13, labelpad=8)
ax.set_ylabel("IQM Return (Stationary Greedy)", fontsize=13, labelpad=8)
ax.set_title(
    "Observation Design as Information Bottleneck:\n"
    "More Teammate Visibility Is Not Monotonically Better",
    fontsize=14, fontweight="bold", pad=14
)

ax.set_xticks([3, 4, 5, 6, 7, 8])
ax.set_xticklabels(["3", "4", "5", "6", "7", "8\n(full grid)"],
                    fontsize=10.5)
ax.set_xlim(2.5, 8.5)
ax.set_ylim(0.02, 0.30)
ax.tick_params(axis="y", labelsize=10.5)

ax.legend(loc="lower left", fontsize=9, framealpha=0.9,
          edgecolor="#bdc3c7")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3, linewidth=0.8)

# Bottom annotation
fig.text(0.5, 0.01,
         "8\u00d78 LBF \u2022 3 agents \u2022 3 food \u2022 50 steps/ep \u2022 "
         "levels hidden \u2022 128k training episodes \u2022 "
         "greedy eval (500 ep/seed)",
         ha="center", fontsize=8, color="#95a5a6", style="italic")

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig("figures/sight_sweep_main.png", dpi=200, bbox_inches="tight",
            facecolor="white")
plt.savefig("figures/sight_sweep_main.pdf", bbox_inches="tight",
            facecolor="white")
print("Saved figures/sight_sweep_main.png and .pdf")
