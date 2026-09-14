"""
Teammate policies for multi-agent benchmarks (Level-Based Foraging and Wolfpack).

Supports two policy modes:
1. 'random': Uniform random legal action selection.
2. 'greedy':
    - LBF: Pathfind toward the closest compatible food item and execute LOAD (5) when adjacent.
    - Wolfpack: Move toward the nearest active prey.
"""

from typing import List, Optional, Tuple, Union
import numpy as np


class BaseTeammatePolicy:
    """Base interface for teammate policy."""

    def select_action(self, agent_idx: int, env, obs=None) -> int:
        raise NotImplementedError


# ======================================================================
# Level-Based Foraging (LBF) Policies
# ======================================================================

class RandomLBFPolicy(BaseTeammatePolicy):
    """Uniform random policy for LBF."""

    def __init__(self, action_dim: int = 6, rng: Optional[np.random.Generator] = None):
        self.action_dim = action_dim
        self.rng = rng if rng is not None else np.random.default_rng()

    def select_action(self, agent_idx: int, env=None, obs=None) -> int:
        return int(self.rng.integers(0, self.action_dim))


class GreedyLBFPolicy(BaseTeammatePolicy):
    """Heuristic greedy policy for LBF teammates.

    1. Identifies active food items.
    2. Filters for compatible level (food_level <= agent_level). If none, considers all active food.
    3. Finds the closest food by Manhattan distance.
    4. If adjacent (Manhattan distance == 1), executes LOAD (5).
    5. Otherwise, steps toward the food along the axis with largest difference.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng if rng is not None else np.random.default_rng()

    def select_action(self, agent_idx: int, env=None, obs=None) -> int:
        # Extract agent position and level from env or ego obs
        if env is not None and hasattr(env, "players"):
            player = env.players[agent_idx]
            pos_y, pos_x = player.position
            level = player.level

            # Extract food from env.field
            food_items = []  # list of (y, x, food_level)
            for r in range(env.field.shape[0]):
                for c in range(env.field.shape[1]):
                    val = env.field[r, c]
                    if val > 0:
                        food_items.append((r, c, val))
        elif obs is not None:
            # Fallback to parsing ego observation
            # Obs layout: [food_0(3), ..., food_{M-1}(3), self(feat_dim), ...]
            obs_arr = np.asarray(obs[agent_idx] if isinstance(obs, (list, tuple)) else obs, dtype=float)
            # Find food entries (> 0 at index 2)
            food_items = []
            # Assuming up to 3 foods
            for k in range(0, 9, 3):
                fy, fx, fl = obs_arr[k], obs_arr[k + 1], obs_arr[k + 2]
                if fl > 0 and fy >= 0 and fx >= 0:
                    food_items.append((int(fy), int(fx), int(fl)))
            pos_y, pos_x = int(obs_arr[9]), int(obs_arr[10])
            level = int(obs_arr[11]) if len(obs_arr) > 11 and obs_arr[11] > 0 else 1
        else:
            return int(self.rng.integers(0, 6))

        if not food_items:
            return int(self.rng.integers(0, 6))

        # Filter compatible foods (food_level <= player_level)
        compatible = [f for f in food_items if f[2] <= level]
        targets = compatible if compatible else food_items

        # Find closest target
        dists = [abs(f[0] - pos_y) + abs(f[1] - pos_x) for f in targets]
        best_idx = int(np.argmin(dists))
        best_y, best_x, _ = targets[best_idx]
        dist = dists[best_idx]

        # Adjacent -> LOAD (action 5)
        if dist == 1:
            return 5  # LOAD

        # Move towards target (Actions: 0=None, 1=North, 2=South, 3=West, 4=East)
        dy = best_y - pos_y
        dx = best_x - pos_x

        if abs(dy) >= abs(dx):
            if dy < 0:
                return 1  # NORTH
            elif dy > 0:
                return 2  # SOUTH
        if dx < 0:
            return 3  # WEST
        elif dx > 0:
            return 4  # EAST
        elif dy < 0:
            return 1  # NORTH
        elif dy > 0:
            return 2  # SOUTH

        return 0  # NONE


# ======================================================================
# Wolfpack Policies
# ======================================================================

class RandomWolfpackPolicy(BaseTeammatePolicy):
    """Uniform random policy for Wolfpack."""

    def __init__(self, action_dim: int = 5, rng: Optional[np.random.Generator] = None):
        self.action_dim = action_dim
        self.rng = rng if rng is not None else np.random.default_rng()

    def select_action(self, agent_idx: int, env=None, obs=None) -> int:
        return int(self.rng.integers(0, self.action_dim))


class GreedyWolfpackPolicy(BaseTeammatePolicy):
    """Heuristic greedy policy for Wolfpack teammates.

    Moves along the shortest Manhattan path toward the nearest active prey.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng if rng is not None else np.random.default_rng()

    def select_action(self, agent_idx: int, env=None, obs=None) -> int:
        if env is not None and hasattr(env, "wolf_positions"):
            pos_y, pos_x = env.wolf_positions[agent_idx]
            prey_active = env.prey_active
            prey_positions = env.prey_positions

            active_prey = [prey_positions[k] for k in range(len(prey_positions)) if prey_active[k]]
        elif obs is not None:
            obs_arr = np.asarray(obs[agent_idx] if isinstance(obs, (list, tuple)) else obs, dtype=float)
            # Layout: [prey_0(3), prey_1(3), self(2), other_wolves...]
            active_prey = []
            for k in range(0, 6, 3):
                py, px, act = obs_arr[k], obs_arr[k + 1], obs_arr[k + 2]
                if act > 0 and py >= 0 and px >= 0:
                    active_prey.append((int(py), int(px)))
            pos_y, pos_x = int(obs_arr[6]), int(obs_arr[7])
        else:
            return int(self.rng.integers(0, 5))

        if not active_prey:
            return int(self.rng.integers(0, 5))

        # Find closest prey by Manhattan distance
        dists = [abs(p[0] - pos_y) + abs(p[1] - pos_x) for p in active_prey]
        best_idx = int(np.argmin(dists))
        best_y, best_x = active_prey[best_idx]

        dy = best_y - pos_y
        dx = best_x - pos_x

        # Action: 0=STAY, 1=NORTH, 2=SOUTH, 3=WEST, 4=EAST
        if abs(dy) >= abs(dx):
            if dy < 0:
                return 1  # NORTH
            elif dy > 0:
                return 2  # SOUTH
        if dx < 0:
            return 3  # WEST
        elif dx > 0:
            return 4  # EAST
        elif dy < 0:
            return 1  # NORTH
        elif dy > 0:
            return 2  # SOUTH

        return 0  # STAY


def get_teammate_policy(env_name: str, policy_type: str, rng: Optional[np.random.Generator] = None) -> BaseTeammatePolicy:
    """Factory helper to get the corresponding teammate policy."""
    env_lower = env_name.lower()
    pol_lower = policy_type.lower()

    if "lbf" in env_lower or "foraging" in env_lower:
        if pol_lower == "greedy":
            return GreedyLBFPolicy(rng=rng)
        return RandomLBFPolicy(rng=rng)
    elif "wolfpack" in env_lower:
        if pol_lower == "greedy":
            return GreedyWolfpackPolicy(rng=rng)
        return RandomWolfpackPolicy(rng=rng)
    else:
        raise ValueError(f"Unknown environment name: {env_name}")
