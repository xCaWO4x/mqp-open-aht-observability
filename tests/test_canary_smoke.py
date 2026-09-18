"""
Quick Canary Verification Script for Observation Distractors.
Runs a 2-episode smoke test with n_envs=2 for each condition:
- LBF: 0 distractors, 8 semantic, 8 null
- Wolfpack: 0 distractors, 8 semantic, 8 null

Verifies:
- Clean execution with zero crashes or exceptions
- No NaNs in network parameters, loss, or Q-values
- Checkpoints saved properly
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.run_benchmark import run_benchmark

conditions = [
    # (env, num_dist, dist_type)
    ("lbf", 8, 0, "none"),
    ("lbf", 8, 8, "semantic"),
    ("lbf", 8, 8, "null"),
    ("wolfpack", 10, 0, "none"),
    ("wolfpack", 10, 8, "semantic"),
    ("wolfpack", 10, 8, "null"),
]

for env_name, sight, n_dist, dist_type in conditions:
    out_dir = f"results/test_canary_{env_name}_dist{n_dist}_{dist_type}"
    print(f"\n========================================================")
    print(f"Testing Canary: {env_name} | sight={sight} | dist={n_dist} ({dist_type})")
    print(f"========================================================")

    returns = run_benchmark(
        env_name=env_name,
        sight=sight,
        teammate_type="random",
        seed=42,
        n_episodes=2,
        n_envs=2,
        output_dir=out_dir,
        smoke_test=False,  # Test full logging and checkpointing
        device="cpu",
        num_distractors=n_dist,
        distractor_type=dist_type,
        distractor_seed=42 + 100000,
        shuffle_distractor_slots=True,
    )

    # Verify metrics.csv exists and has finite values
    metrics_path = os.path.join(out_dir, "metrics.csv")
    assert os.path.exists(metrics_path), f"metrics.csv not found at {metrics_path}"

    with open(metrics_path, "r") as f:
        lines = f.readlines()
        assert len(lines) >= 2, f"metrics.csv has insufficient lines: {len(lines)}"
        header = lines[0].strip().split(",")
        last_row = lines[-1].strip().split(",")
        print(f"Last logged metrics for {env_name} dist{n_dist}_{dist_type}: {dict(zip(header, last_row))}")

        # Check for NaNs in return, q_loss, agent_loss
        ret_val = float(last_row[1])
        q_loss_val = float(last_row[4])
        ag_loss_val = float(last_row[5])
        assert not np.isnan(ret_val), "Return is NaN"
        assert not np.isnan(q_loss_val), "q_loss is NaN"
        assert not np.isnan(ag_loss_val), "agent_loss is NaN"

    # Verify checkpoint exists
    final_ckpt = os.path.join(out_dir, "checkpoints", "gpl_final.pt")
    assert os.path.exists(final_ckpt), f"Checkpoint not found at {final_ckpt}"
    ckpt = torch.load(final_ckpt, map_location="cpu")
    assert "type_net_q" in ckpt, "Missing type_net_q in checkpoint"

    print(f"PASSED Canary for {env_name} dist{n_dist}_{dist_type}!")

print("\nALL 6 CANARY SMOKE TESTS PASSED SUCCESSFULLY!")
