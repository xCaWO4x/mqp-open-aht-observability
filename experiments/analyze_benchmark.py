"""
Analyze benchmark results across LBF and Wolfpack observability sweeps.
Aggregates metrics across seeds and produces comparative summary tables.
"""

import os
import glob
import sys
import numpy as np
from collections import defaultdict


def parse_dir_name(dirname):
    # Format: {env}_sight{sight}_{teammate}_seed{seed}
    base = os.path.basename(dirname)
    parts = base.split("_")
    if len(parts) < 4:
        return None
    env = parts[0]
    sight = int(parts[1].replace("sight", ""))
    teammate = parts[2]
    seed = int(parts[3].replace("seed", ""))
    return env, sight, teammate, seed


def load_metrics_summary(csv_path):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) < 100:
        return None

    episodes = []
    returns = []
    lengths = []
    epsilons = []

    try:
        with open(csv_path, "r") as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    try:
                        ep = int(parts[0])
                        ret = float(parts[1])
                        ln = float(parts[2])
                        eps = float(parts[3]) if len(parts) > 3 else 0.0
                        episodes.append(ep)
                        returns.append(ret)
                        lengths.append(ln)
                        epsilons.append(eps)
                    except ValueError:
                        continue
    except Exception as e:
        return None

    if not episodes:
        return None

    episodes = np.array(episodes)
    returns = np.array(returns)
    lengths = np.array(lengths)

    total_episodes = episodes[-1]
    
    # Recent window: last 2000 episodes (or all if < 2000)
    w = min(len(returns), 2000)
    recent_return = float(np.mean(returns[-w:]))
    recent_std = float(np.std(returns[-w:]))
    recent_length = float(np.mean(lengths[-w:]))

    # Intermediate window (episodes 20,000 to 40,000 if available)
    mid_mask = (episodes >= 20000) & (episodes <= 40000)
    mid_return = float(np.mean(returns[mid_mask])) if np.any(mid_mask) else None

    # Early window (first 5,000 episodes)
    early_mask = episodes <= 5000
    early_return = float(np.mean(returns[early_mask])) if np.any(early_mask) else None

    return {
        "total_episodes": total_episodes,
        "recent_return": recent_return,
        "recent_std": recent_std,
        "recent_length": recent_length,
        "mid_return": mid_return,
        "early_return": early_return,
    }


def main():
    dirs = sorted(glob.glob("results/*_*_*_seed*"))
    print(f"Found {len(dirs)} candidate directories in results/.")
    sys.stdout.flush()

    # Group data by (env, teammate, sight) -> list of seed results
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    total_scanned = 0
    total_valid = 0

    for d in dirs:
        meta = parse_dir_name(d)
        if not meta:
            continue
        total_scanned += 1
        env, sight, teammate, seed = meta
        csv_path = os.path.join(d, "metrics.csv")
        stats = load_metrics_summary(csv_path)
        if stats and stats["total_episodes"] >= 1000:
            total_valid += 1
            stats["seed"] = seed
            stats["dir"] = d
            stats["is_final"] = os.path.exists(os.path.join(d, "checkpoints", "gpl_final.pt"))
            groups[env][teammate][sight].append(stats)

    print(f"Scanned {total_scanned} directories; {total_valid} runs have >= 1,000 recorded episodes.\n")
    sys.stdout.flush()

    # Generate Markdown Report
    lines = []
    lines.append("# Empirical Benchmark Analytics: Impact of Observability Radius\n")

    for env in sorted(groups.keys()):
        lines.append(f"## Environment: {env.upper()}\n")
        for teammate in sorted(groups[env].keys()):
            lines.append(f"### Teammate Policy: `{teammate}`\n")
            lines.append("| Sight Radius ($r$) | Seeds Evaluated | Max Ep Reached | Mean Return (Late Stage) | Std Return | Mean Ep Length | Early Return (<=5k) | Mid Return (20k-40k) |")
            lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

            sights = sorted(groups[env][teammate].keys())
            for s in sights:
                seed_list = groups[env][teammate][s]
                n_seeds = len(seed_list)
                if n_seeds == 0:
                    continue
                max_ep = max(x["total_episodes"] for x in seed_list)
                recent_rets = [x["recent_return"] for x in seed_list]
                mean_ret = float(np.mean(recent_rets))
                std_ret = float(np.std(recent_rets)) if len(recent_rets) > 1 else 0.0
                mean_len = float(np.mean([x["recent_length"] for x in seed_list]))

                early_rets = [x["early_return"] for x in seed_list if x["early_return"] is not None]
                mean_early = f"{np.mean(early_rets):.3f}" if early_rets else "N/A"

                mid_rets = [x["mid_return"] for x in seed_list if x["mid_return"] is not None]
                mean_mid = f"{np.mean(mid_rets):.3f}" if mid_rets else "N/A"

                lines.append(
                    f"| **r = {s}** | {n_seeds} seeds | {max_ep:,} | **{mean_ret:.3f}** | ±{std_ret:.3f} | {mean_len:.1f} | {mean_early} | {mean_mid} |"
                )
            lines.append("\n")

    report_text = "\n".join(lines)
    print(report_text)
    sys.stdout.flush()

    out_path = "results/benchmark_summary.md"
    with open(out_path, "w") as f:
        f.write(report_text)
    print(f"\nSummary successfully written to {out_path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
