#!/usr/bin/env python
"""Plot sight-sweep stationary greedy performance curves.

Supports two variants:
  - inf_aux : Q3-inf-aux (info-nerf + auxiliary level head)
  - rw      : Q3_rw      (info-nerf only, no auxiliary head)

For each variant this reads per-episode return CSVs at sight in {3,4,5,6,7}:
  Q3-inf-aux  sight=3    results/q3_inf_aux_rw_stationary_greedy_eval_500/
  Q3-inf-aux  sight=K    results/q3_inf_aux_sight{K}_rw_stationary_greedy_eval_500/
  Q3_rw       sight=3    results/q3_rw_stationary_greedy_eval_500/
  Q3_rw       sight=K    results/q3_rw_sight{K}_stationary_greedy_eval_500/

Per sight point it computes:
  - mean return (+/- 1 sd over episodes)
  - IQM return (25-75 trimmed mean) with 95% bootstrap CI
and renders a PNG with IQM on y, sight on x.

Default is the both-variant overlay (the clean visual for the
aux-head-vs-observability-saturation question).

Usage:
  python scripts/plot_sight_sweep.py                         # both curves
  python scripts/plot_sight_sweep.py --variant inf_aux       # single curve
  python scripts/plot_sight_sweep.py --variant rw
  python scripts/plot_sight_sweep.py --out figures/x.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


EVAL_FILENAME = "stationary_eval_seed42.csv"
SIGHTS = [3, 4, 5, 6, 7]

VARIANTS = {
    "inf_aux": {
        "label": "Q3-inf-aux (aux level head)",
        "color": "#1f77b4",
        "marker": "o",
        "baseline_dir": "results/q3_inf_aux_rw_stationary_greedy_eval_500",
        "sweep_fmt": "results/q3_inf_aux_sight{K}_rw_stationary_greedy_eval_500",
    },
    "rw": {
        "label": "Q3_rw (no auxiliary head)",
        "color": "#d62728",
        "marker": "s",
        "baseline_dir": "results/q3_rw_stationary_greedy_eval_500",
        "sweep_fmt": "results/q3_rw_sight{K}_stationary_greedy_eval_500",
    },
}


def eval_dir(variant: str, sight: int) -> Path:
    spec = VARIANTS[variant]
    if sight == 3:
        return Path(spec["baseline_dir"])
    return Path(spec["sweep_fmt"].format(K=sight))


def load_returns(csv_path: Path) -> np.ndarray:
    returns: list[float] = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            returns.append(float(row["return"]))
    return np.asarray(returns, dtype=np.float64)


def compute_iqm(returns: np.ndarray) -> float:
    """Interquartile mean: trim top and bottom 25%, average the middle 50%."""
    if returns.size == 0:
        return float("nan")
    lo, hi = np.percentile(returns, [25.0, 75.0])
    mask = (returns >= lo) & (returns <= hi)
    return float(returns[mask].mean()) if mask.any() else float("nan")


def bootstrap_iqm_ci(
    returns: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the IQM (episode-level resampling)."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = returns.size
    if n == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        boots[b] = compute_iqm(returns[idx[b]])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def summarise(
    variant: str,
    sight: int,
    csv_path: Path,
    rng: np.random.Generator,
) -> dict:
    returns = load_returns(csv_path)
    iqm = compute_iqm(returns)
    ci_lo, ci_hi = bootstrap_iqm_ci(returns, rng=rng)
    return {
        "variant": variant,
        "sight": sight,
        "n": int(returns.size),
        "mean": float(returns.mean()) if returns.size else float("nan"),
        "sd": float(returns.std(ddof=1)) if returns.size > 1 else 0.0,
        "iqm": iqm,
        "iqm_ci_lo": ci_lo,
        "iqm_ci_hi": ci_hi,
    }


def collect_rows(
    variants: list[str],
    sights: list[int],
    rng: np.random.Generator,
) -> list[dict]:
    rows: list[dict] = []
    for v in variants:
        for k in sights:
            cp = eval_dir(v, k) / EVAL_FILENAME
            if not cp.is_file():
                print(
                    f"[skip] variant={v} sight={k}: {cp} not found",
                    file=sys.stderr,
                )
                continue
            rows.append(summarise(v, k, cp, rng))
            print(f"[ok]   variant={v} sight={k}: {cp}")
    return rows


def print_summary_table(rows: list[dict]) -> None:
    print("\n=== Sight sweep summary ===")
    header = (
        f"{'variant':>9} {'sight':>5} {'n':>5} {'mean':>10} {'sd':>10} "
        f"{'IQM':>10} {'IQM_CI_lo':>10} {'IQM_CI_hi':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda r: (r["variant"], r["sight"])):
        print(
            f"{r['variant']:>9} {r['sight']:>5d} {r['n']:>5d} "
            f"{r['mean']:>10.4f} {r['sd']:>10.4f} {r['iqm']:>10.4f} "
            f"{r['iqm_ci_lo']:>10.4f} {r['iqm_ci_hi']:>10.4f}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variant",
        choices=["inf_aux", "rw", "both"],
        default="both",
        help="which curve(s) to plot (default: both)",
    )
    p.add_argument(
        "--out",
        default=None,
        help=(
            "output PNG path (default depends on --variant: "
            "figures/sight_sweep_{variant}_iqm.png, or "
            "figures/sight_sweep_overlay_iqm.png for both)"
        ),
    )
    p.add_argument(
        "--n-boot",
        type=int,
        default=1000,
        help="bootstrap resamples for IQM CI (default: 1000)",
    )
    p.add_argument(
        "--sights",
        type=int,
        nargs="+",
        default=SIGHTS,
        help="sight values to include (default: 3 4 5 6 7)",
    )
    return p.parse_args()


def default_out_path(variant_arg: str) -> Path:
    if variant_arg == "both":
        return Path("figures/sight_sweep_overlay_iqm.png")
    return Path(f"figures/sight_sweep_{variant_arg}_iqm.png")


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(0)

    variants = ["inf_aux", "rw"] if args.variant == "both" else [args.variant]
    rows = collect_rows(variants, args.sights, rng)
    if not rows:
        print(
            "No eval CSVs found. Submit and run the sweep(s) first:\n"
            "  bash scripts/slurm/submit_sight_sweep.sh     --with-eval  "
            "# Q3-inf-aux\n"
            "  bash scripts/slurm/submit_rw_sight_sweep.sh  --with-eval  "
            "# Q3_rw",
            file=sys.stderr,
        )
        return 1

    print_summary_table(rows)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "\nmatplotlib not available; summary printed above. "
            "Install it (`pip install matplotlib`) to render the PNG.",
            file=sys.stderr,
        )
        return 0

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for v in variants:
        spec = VARIANTS[v]
        vrows = sorted([r for r in rows if r["variant"] == v], key=lambda r: r["sight"])
        if not vrows:
            continue
        sights = np.array([r["sight"] for r in vrows])
        iqms = np.array([r["iqm"] for r in vrows])
        los = np.array([r["iqm_ci_lo"] for r in vrows])
        his = np.array([r["iqm_ci_hi"] for r in vrows])
        err = np.stack([iqms - los, his - iqms])
        ax.errorbar(
            sights,
            iqms,
            yerr=err,
            fmt=f"{spec['marker']}-",
            capsize=4,
            color=spec["color"],
            label=f"{spec['label']} (IQM, 95% bootstrap CI)",
        )

    ax.set_xlabel("Sight radius")
    ax.set_ylabel("Stationary IQM return (500 eps, greedy)")
    title_suffix = (
        "Q3-inf-aux vs Q3_rw"
        if args.variant == "both"
        else VARIANTS[args.variant]["label"]
    )
    ax.set_title(f"Sight sweep: {title_suffix}")
    ax.set_xticks(sorted(args.sights))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    out = Path(args.out) if args.out else default_out_path(args.variant)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out}")

    csv_out = out.with_suffix(".csv")
    with open(csv_out, "w") as f:
        f.write("variant,sight,n,mean,sd,iqm,iqm_ci_lo,iqm_ci_hi\n")
        for r in sorted(rows, key=lambda r: (r["variant"], r["sight"])):
            f.write(
                f"{r['variant']},{r['sight']},{r['n']},{r['mean']:.6f},"
                f"{r['sd']:.6f},{r['iqm']:.6f},"
                f"{r['iqm_ci_lo']:.6f},{r['iqm_ci_hi']:.6f}\n"
            )
    print(f"Summary CSV saved to {csv_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
