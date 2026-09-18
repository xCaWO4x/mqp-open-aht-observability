"""
Sight-Radius Equivalence Test Across MARL Baselines.

Verifies that for identical environment states and sight radii:
visible entities under GPL == visible entities supplied to Transformer-Q == non-zero entities in IPPO flat_obs
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from envs.entity_adapter import extract_entities
from envs.env_utils import preprocess_lbf, preprocess_wolfpack
from experiments.run_benchmark import make_benchmark_env, unpack_reset


def test_lbf_sight_equivalence():
    """Verify sight equivalence on LBF across radii r in [1, 2, 3, 4, 8]."""
    radii = [1, 2, 3, 4, 8]
    n_agents = 3
    n_food = 3
    grid_size = 8
    num_entities = n_agents + n_food  # 6 entities

    print("=== Testing LBF Sight-Radius Equivalence ===")
    for r in radii:
        env = make_benchmark_env(
            "lbf",
            sight=r,
            seed=42,
            grid_size=grid_size,
            n_agents=n_agents,
            n_food=n_food,
            max_steps=50,
            observe_agent_levels=False,
        )
        raw_obs = unpack_reset(env.reset())

        # 1. GPL Visibility
        # Under ego-perspective, agent 0's observation has teammates and food.
        # Entities outside sight radius are masked to -1 in preprocess_lbf.
        # Format of obs_0: [food_0(3), food_1(3), food_2(3), self(2), tm_1(2), tm_2(2)]
        obs_0 = np.asarray(raw_obs[0], dtype=np.float32)
        food_feats = obs_0[:n_food * 3].reshape(n_food, 3)
        self_feat = obs_0[n_food * 3: n_food * 3 + 2]
        tm_feats = obs_0[n_food * 3 + 2:].reshape(n_agents - 1, 2)

        # Ordering in entity adapter: [self, tm_1, tm_2, food_0, food_1, food_2]
        gpl_vis = []
        # Self
        gpl_vis.append(self_feat[0] >= 0 and self_feat[1] >= 0)
        # Teammates
        for tm in tm_feats:
            gpl_vis.append(tm[0] >= 0 and tm[1] >= 0)
        # Foods
        for f in food_feats:
            gpl_vis.append(f[0] >= 0 and f[1] >= 0 and f[2] > 0)
        gpl_vis = np.array(gpl_vis, dtype=bool)

        # 2. Transformer-Q Visibility (from visible_mask)
        batch = extract_entities(raw_obs, "lbf", n_agents=n_agents, n_food=n_food, grid_size=float(grid_size))
        trans_vis = batch.visible_mask[0].cpu().numpy()

        # 3. IPPO Visibility (from non-zero entity entries in flat_obs)
        flat = batch.flat_obs[0].cpu().numpy().reshape(num_entities, 3)
        # An entity is visible in flat_obs iff its coordinates are non-zero (or active)
        ippo_vis = np.array([np.any(flat[k] != 0.0) for k in range(num_entities)], dtype=bool)

        # Compare equivalence
        print(f"Radius r={r:2d} | GPL visible: {gpl_vis.sum()}/{num_entities} | Transformer visible: {trans_vis.sum()}/{num_entities} | IPPO visible: {ippo_vis.sum()}/{num_entities}")

        assert np.array_equal(gpl_vis, trans_vis), f"GPL vs Transformer visibility mismatch at r={r}!\nGPL: {gpl_vis}\nTrans: {trans_vis}"
        assert np.array_equal(trans_vis, ippo_vis), f"Transformer vs IPPO visibility mismatch at r={r}!\nTrans: {trans_vis}\nIPPO: {ippo_vis}"

    print("LBF Sight-Radius Equivalence test PASSED 100% across all radii!\n")


def test_wolfpack_sight_equivalence():
    """Verify sight equivalence on Wolfpack across radii r in [2, 4, 6, 10]."""
    radii = [2, 4, 6, 10]
    n_wolves = 3
    n_prey = 2
    grid_size = 10
    num_entities = n_wolves + n_prey  # 5 entities

    print("=== Testing Wolfpack Sight-Radius Equivalence ===")
    for r in radii:
        env = make_benchmark_env(
            "wolfpack",
            sight=r,
            seed=42,
            grid_size=grid_size,
            n_wolves=n_wolves,
            n_prey=n_prey,
            max_steps=50,
        )
        raw_obs = unpack_reset(env.reset())

        # obs_0: [prey_0(3), prey_1(3), self(2), other_1(2), other_2(2)]
        obs_0 = np.asarray(raw_obs[0], dtype=np.float32)
        prey_feats = obs_0[:n_prey * 3].reshape(n_prey, 3)
        self_feat = obs_0[n_prey * 3: n_prey * 3 + 2]
        other_feats = obs_0[n_prey * 3 + 2:].reshape(n_wolves - 1, 2)

        # Ordering in entity adapter: [self, wolf_1, wolf_2, prey_0, prey_1]
        gpl_vis = []
        # Self
        gpl_vis.append(self_feat[0] >= 0 and self_feat[1] >= 0)
        # Teammate wolves
        for w in other_feats:
            gpl_vis.append(w[0] >= 0 and w[1] >= 0)
        # Prey
        for p in prey_feats:
            gpl_vis.append(p[0] >= 0 and p[1] >= 0 and p[2] > 0)
        gpl_vis = np.array(gpl_vis, dtype=bool)

        # Transformer-Q Visibility
        batch = extract_entities(raw_obs, "wolfpack", n_wolves=n_wolves, n_prey=n_prey, grid_size=float(grid_size))
        trans_vis = batch.visible_mask[0].cpu().numpy()

        # IPPO Visibility
        flat = batch.flat_obs[0].cpu().numpy().reshape(num_entities, 3)
        ippo_vis = np.array([np.any(flat[k] != 0.0) for k in range(num_entities)], dtype=bool)

        print(f"Radius r={r:2d} | GPL visible: {gpl_vis.sum()}/{num_entities} | Transformer visible: {trans_vis.sum()}/{num_entities} | IPPO visible: {ippo_vis.sum()}/{num_entities}")

        assert np.array_equal(gpl_vis, trans_vis), f"Wolfpack GPL vs Transformer visibility mismatch at r={r}!"
        assert np.array_equal(trans_vis, ippo_vis), f"Wolfpack Transformer vs IPPO visibility mismatch at r={r}!"

    print("Wolfpack Sight-Radius Equivalence test PASSED 100% across all radii!\n")


if __name__ == "__main__":
    test_lbf_sight_equivalence()
    test_wolfpack_sight_equivalence()
    print("ALL SIGHT-RADIUS EQUIVALENCE TESTS PASSED!")
