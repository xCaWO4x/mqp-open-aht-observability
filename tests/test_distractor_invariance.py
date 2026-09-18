"""
Automated Validation Suite for Observation-Only Distractor Controls.

Validates:
1. Environmental Invariance: Underlying environment dynamics, real entity positions,
   rewards, terminations, teammate actions, real prey behavior, and real food spawning
   are 100% bitwise/numerically identical across:
     - 0 distractors (baseline)
     - 8 semantic distractors
     - 8 null distractors
   under identical random seeds and forced actions.
2. Schema Indistinguishability: Semantic distractors have identical coordinate
   domains, feature ranges, and schema as real entities.
3. Null Distractor Marker: Null distractors have the distinct explicit marker (0.0).
4. Slot Interleaving: Distractors are distributed across slots, not pinned to the tail.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.distractor_wrapper import DistractorEnvWrapper
from experiments.run_benchmark import make_benchmark_env, unpack_reset, unpack_step


def test_lbf_environmental_invariance():
    """Verify that adding distractors has ZERO effect on real LBF dynamics."""
    seed = 42
    n_steps = 50

    # 1. Baseline environment (0 distractors)
    base_env = make_benchmark_env("lbf", sight=8, seed=seed, grid_size=8, n_agents=3, n_food=3, observe_agent_levels=False)
    # 2. Semantic distractors (8 distractors)
    sem_env = DistractorEnvWrapper(
        make_benchmark_env("lbf", sight=8, seed=seed, grid_size=8, n_agents=3, n_food=3, observe_agent_levels=False),
        num_distractors=8,
        distractor_type="semantic",
        distractor_seed=seed + 100000,
        shuffle_slots=True,
    )
    # 3. Null distractors (8 distractors)
    null_env = DistractorEnvWrapper(
        make_benchmark_env("lbf", sight=8, seed=seed, grid_size=8, n_agents=3, n_food=3, observe_agent_levels=False),
        num_distractors=8,
        distractor_type="null",
        distractor_seed=seed + 200000,
        shuffle_slots=True,
    )

    # Reset all
    base_obs = unpack_reset(base_env.reset(seed=seed))
    sem_obs = unpack_reset(sem_env.reset(seed=seed))
    null_obs = unpack_reset(null_env.reset(seed=seed))

    # Real food positions and player positions must be identical at start
    base_u = base_env.unwrapped
    sem_u = sem_env.unwrapped
    null_u = null_env.unwrapped

    np.testing.assert_array_equal(base_u.field, sem_u.field, err_msg="LBF initial field mismatch with semantic distractors")
    np.testing.assert_array_equal(base_u.field, null_u.field, err_msg="LBF initial field mismatch with null distractors")

    for i in range(3):
        np.testing.assert_array_equal(base_u.players[i].position, sem_u.players[i].position)
        np.testing.assert_array_equal(base_u.players[i].position, null_u.players[i].position)
        assert base_u.players[i].level == sem_u.players[i].level == null_u.players[i].level

    # Step through fixed action sequence
    rng_actions = np.random.default_rng(999)
    for step_i in range(n_steps):
        actions = rng_actions.integers(0, 6, size=3).tolist()

        base_out = base_env.step(actions)
        sem_out = sem_env.step(actions)
        null_out = null_env.step(actions)

        _, base_r, base_done, _ = unpack_step(base_out)
        _, sem_r, sem_done, _ = unpack_step(sem_out)
        _, null_r, null_done, _ = unpack_step(null_out)

        # Rewards and dones must match bitwise
        assert base_r == sem_r == null_r, f"Reward mismatch at step {step_i}: base={base_r}, sem={sem_r}, null={null_r}"
        assert base_done == sem_done == null_done, f"Done mismatch at step {step_i}"

        # Real field and player positions must match bitwise
        np.testing.assert_array_equal(base_u.field, sem_u.field, err_msg=f"Field diverged at step {step_i}")
        np.testing.assert_array_equal(base_u.field, null_u.field, err_msg=f"Field diverged at step {step_i}")

        for p_idx in range(3):
            np.testing.assert_array_equal(base_u.players[p_idx].position, sem_u.players[p_idx].position)
            np.testing.assert_array_equal(base_u.players[p_idx].position, null_u.players[p_idx].position)

        if base_done:
            break


def test_wolfpack_environmental_invariance():
    """Verify that adding distractors has ZERO effect on real Wolfpack dynamics."""
    seed = 42
    n_steps = 50

    base_env = make_benchmark_env("wolfpack", sight=10, seed=seed, grid_size=10, n_wolves=3, n_prey=2)
    sem_env = DistractorEnvWrapper(
        make_benchmark_env("wolfpack", sight=10, seed=seed, grid_size=10, n_wolves=3, n_prey=2),
        num_distractors=8,
        distractor_type="semantic",
        distractor_seed=seed + 100000,
        shuffle_slots=True,
    )
    null_env = DistractorEnvWrapper(
        make_benchmark_env("wolfpack", sight=10, seed=seed, grid_size=10, n_wolves=3, n_prey=2),
        num_distractors=8,
        distractor_type="null",
        distractor_seed=seed + 200000,
        shuffle_slots=True,
    )

    unpack_reset(base_env.reset(seed=seed))
    unpack_reset(sem_env.reset(seed=seed))
    unpack_reset(null_env.reset(seed=seed))

    base_u = base_env.unwrapped
    sem_u = sem_env.unwrapped
    null_u = null_env.unwrapped

    np.testing.assert_array_equal(base_u.wolf_positions, sem_u.wolf_positions)
    np.testing.assert_array_equal(base_u.wolf_positions, null_u.wolf_positions)
    np.testing.assert_array_equal(base_u.prey_positions, sem_u.prey_positions)
    np.testing.assert_array_equal(base_u.prey_positions, null_u.prey_positions)
    np.testing.assert_array_equal(base_u.prey_active, sem_u.prey_active)
    np.testing.assert_array_equal(base_u.prey_active, null_u.prey_active)

    rng_actions = np.random.default_rng(888)
    for step_i in range(n_steps):
        actions = rng_actions.integers(0, 5, size=3).tolist()

        base_out = base_env.step(actions)
        sem_out = sem_env.step(actions)
        null_out = null_env.step(actions)

        _, base_r, base_done, _ = unpack_step(base_out)
        _, sem_r, sem_done, _ = unpack_step(sem_out)
        _, null_r, null_done, _ = unpack_step(null_out)

        assert base_r == sem_r == null_r, f"Wolfpack reward mismatch at step {step_i}"
        assert base_done == sem_done == null_done, f"Wolfpack done mismatch at step {step_i}"

        np.testing.assert_array_equal(base_u.wolf_positions, sem_u.wolf_positions)
        np.testing.assert_array_equal(base_u.wolf_positions, null_u.wolf_positions)
        np.testing.assert_array_equal(base_u.prey_positions, sem_u.prey_positions)
        np.testing.assert_array_equal(base_u.prey_positions, null_u.prey_positions)
        np.testing.assert_array_equal(base_u.prey_active, sem_u.prey_active)
        np.testing.assert_array_equal(base_u.prey_active, null_u.prey_active)

        if base_done:
            break


def test_schema_and_slot_shuffling():
    """Verify entity encoding schema, null markers, and slot interleaving."""
    seed = 123
    # LBF check
    wrapper_lbf = DistractorEnvWrapper(
        make_benchmark_env("lbf", sight=8, seed=seed, grid_size=8, n_agents=3, n_food=3, observe_agent_levels=False),
        num_distractors=8,
        distractor_type="semantic",
        distractor_seed=seed,
        shuffle_slots=True,
    )
    obs_lbf, _ = wrapper_lbf.reset(seed=seed)
    # Total food = 3 + 8 = 11. Food features = 11 * 3 = 33. Agent features = 3 * 2 = 6. Total len = 39.
    assert len(obs_lbf[0]) == 39
    food_feats = obs_lbf[0][:33].reshape(11, 3)

    # Check that all coordinates are inside [0, 7] and levels are valid (in {2, 3})
    for f in food_feats:
        y, x, lvl = f
        assert 0 <= y <= 7, f"Illegal y coord {y}"
        assert 0 <= x <= 7, f"Illegal x coord {x}"
        assert lvl in (1.0, 2.0, 3.0), f"Semantic food level not in valid range: {lvl}"

    # LBF null check
    null_lbf = DistractorEnvWrapper(
        make_benchmark_env("lbf", sight=8, seed=seed, grid_size=8, n_agents=3, n_food=3, observe_agent_levels=False),
        num_distractors=8,
        distractor_type="null",
        distractor_seed=seed,
        shuffle_slots=True,
    )
    null_obs, _ = null_lbf.reset(seed=seed)
    null_food_feats = null_obs[0][:33].reshape(11, 3)

    null_levels = [f[2] for f in null_food_feats]
    assert null_levels.count(0.0) == 8, f"Expected 8 null distractors with level=0.0, got {null_levels.count(0.0)}"

    # Slot permutation check: over multiple resets, distractors should not be pinned to the tail
    positions_seen = []
    for ep in range(10):
        o, _ = wrapper_lbf.reset()
        perm = wrapper_lbf.slot_permutation
        # Find which slot indices hold ghost entities (ghost indices are 3..10)
        ghost_slots = [slot_idx for slot_idx, orig_idx in enumerate(perm) if orig_idx >= 3]
        positions_seen.extend(ghost_slots)

    # Verify ghost entities appear across all slot indices (0 to 10)
    unique_slots = set(positions_seen)
    assert len(unique_slots) == 11, f"Ghost entities not uniformly distributed across all slots: seen {unique_slots}"


def test_attribute_forwarding():
    """Verify that setting env attributes on the wrapper forwards them to unwrapped."""
    seed = 42
    env = DistractorEnvWrapper(
        make_benchmark_env("lbf", sight=8, seed=seed, grid_size=8, n_agents=3, n_food=3, observe_agent_levels=False),
        num_distractors=4,
        distractor_type="semantic",
    )
    env.min_player_level = np.array([2, 3, 2])
    env.min_food_level = np.array([3, 3, 3])

    np.testing.assert_array_equal(env.unwrapped.min_player_level, np.array([2, 3, 2]))
    np.testing.assert_array_equal(env.unwrapped.min_food_level, np.array([3, 3, 3]))


if __name__ == "__main__":
    test_lbf_environmental_invariance()
    test_wolfpack_environmental_invariance()
    test_schema_and_slot_shuffling()
    test_attribute_forwarding()
    print("All distractor invariance and schema tests passed successfully!")
