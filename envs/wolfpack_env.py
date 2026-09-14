"""
Wolfpack (Predator-Prey) Multi-Agent Environment.

Grid: 10x10
Wolves: 3 agents
Prey (stags): 2 items
Max horizon: 50 steps
Capture condition: >= 2 wolves adjacent (<= 1 cell Chebyshev distance) to the same prey simultaneously.
Observation radius: r in {3, 4, 6, 10} masks entities outside field of view to -1.
"""

from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


class Action(IntEnum):
    STAY = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4


DELTAS = {
    Action.STAY: (0, 0),
    Action.NORTH: (-1, 0),
    Action.SOUTH: (1, 0),
    Action.WEST: (0, -1),
    Action.EAST: (0, 1),
}


class WolfpackEnv(gym.Env):
    """Wolfpack Multi-Agent Environment with entity observations and sight radius masking.

    Parameters
    ----------
    grid_size : int, default 10
        Size of the square grid (10x10).
    n_wolves : int, default 3
        Number of predator agents.
    n_prey : int, default 2
        Number of prey items.
    max_steps : int, default 50
        Maximum episode steps.
    sight : int, default 10
        Observation radius. Chebyshev distance > sight masks coordinates to -1.
    coop_radius : int, default 1
        Radius defining adjacency for capture (<= 1 cell).
    min_coop_wolves : int, default 2
        Minimum adjacent wolves required to capture a prey.
    capture_reward : float, default 5.0
        Reward given to the team upon successful capture.
    respawn_prey : bool, default True
        Whether captured prey respawns at a random unoccupied cell.
    seed : Optional[int]
        Random seed.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        grid_size: int = 10,
        n_wolves: int = 3,
        n_prey: int = 2,
        max_steps: int = 50,
        sight: int = 10,
        coop_radius: int = 1,
        min_coop_wolves: int = 2,
        capture_reward: float = 5.0,
        respawn_prey: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.n_wolves = n_wolves
        self.n_prey = n_prey
        self.max_steps = max_steps
        self.sight = sight
        self.coop_radius = coop_radius
        self.min_coop_wolves = min_coop_wolves
        self.capture_reward = capture_reward
        self.respawn_prey = respawn_prey

        self.rng = np.random.default_rng(seed)
        self.current_step = 0

        # Positions: (y, x)
        self.wolf_positions = np.zeros((self.n_wolves, 2), dtype=int)
        self.prey_positions = np.zeros((self.n_prey, 2), dtype=int)
        self.prey_active = np.ones(self.n_prey, dtype=bool)

        # Action and observation spaces
        self.action_space = spaces.Tuple([spaces.Discrete(len(Action)) for _ in range(self.n_wolves)])

        # Single agent flat obs length:
        # 3 * n_prey (y, x, active) + 2 (self y, x) + 2 * (n_wolves - 1) (other wolves y, x)
        self.obs_dim_per_agent = 3 * self.n_prey + 2 + 2 * (self.n_wolves - 1)
        self.observation_space = spaces.Tuple(
            [spaces.Box(low=-1.0, high=float(self.grid_size), shape=(self.obs_dim_per_agent,), dtype=np.float32)
             for _ in range(self.n_wolves)]
        )

    def seed(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def _random_unoccupied_cell(self, occupied: List[Tuple[int, int]]) -> Tuple[int, int]:
        all_cells = [(y, x) for y in range(self.grid_size) for x in range(self.grid_size) if (y, x) not in occupied]
        idx = self.rng.integers(0, len(all_cells))
        return all_cells[idx]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[List[np.ndarray], dict]:
        if seed is not None:
            self.seed(seed)

        self.current_step = 0
        occupied: List[Tuple[int, int]] = []

        # Spawn wolves
        for i in range(self.n_wolves):
            pos = self._random_unoccupied_cell(occupied)
            self.wolf_positions[i] = pos
            occupied.append(pos)

        # Spawn prey
        for j in range(self.n_prey):
            pos = self._random_unoccupied_cell(occupied)
            self.prey_positions[j] = pos
            occupied.append(pos)

        self.prey_active.fill(True)

        observations = self._get_obs()
        return observations, {}

    def _is_in_sight(self, center: Tuple[int, int], target: Tuple[int, int]) -> bool:
        """Chebyshev field-of-view check."""
        if self.sight >= self.grid_size:
            return True
        dy = abs(center[0] - target[0])
        dx = abs(center[1] - target[1])
        return max(dy, dx) <= self.sight

    def _get_obs(self) -> List[np.ndarray]:
        """Build ego-centric entity observations for each wolf with sight masking.

        Format per agent i:
            [prey_0(y, x, active), ..., prey_{M-1}(y, x, active),
             self(y, x),
             other_1(y, x), ..., other_{N-1}(y, x)]
        If an entity is out of sight, its position is masked to -1.0.
        """
        obs_list = []
        for i in range(self.n_wolves):
            self_y, self_x = self.wolf_positions[i]

            # Prey features
            prey_feats = []
            for j in range(self.n_prey):
                py, px = self.prey_positions[j]
                active = float(self.prey_active[j])
                if active > 0 and self._is_in_sight((self_y, self_x), (py, px)):
                    prey_feats.extend([float(py), float(px), 1.0])
                else:
                    # Masked out of sight or inactive
                    prey_feats.extend([-1.0, -1.0, 0.0])

            # Self position (always known to self)
            self_feat = [float(self_y), float(self_x)]

            # Other wolves
            other_feats = []
            for other_idx in range(self.n_wolves):
                if other_idx == i:
                    continue
                oy, ox = self.wolf_positions[other_idx]
                if self._is_in_sight((self_y, self_x), (oy, ox)):
                    other_feats.extend([float(oy), float(ox)])
                else:
                    other_feats.extend([-1.0, -1.0])

            agent_obs = np.array(prey_feats + self_feat + other_feats, dtype=np.float32)
            obs_list.append(agent_obs)

        return obs_list

    def _move_prey(self):
        """Simple prey movement (random walk avoiding grid boundaries and occupied wolf cells)."""
        for j in range(self.n_prey):
            if not self.prey_active[j]:
                continue
            py, px = self.prey_positions[j]
            # Try a random move
            possible_actions = list(Action)
            self.rng.shuffle(possible_actions)
            for act in possible_actions:
                dy, dx = DELTAS[act]
                ny, nx = py + dy, px + dx
                if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                    # Avoid stepping directly into a wolf
                    if not any(np.array_equal([ny, nx], self.wolf_positions[w]) for w in range(self.n_wolves)):
                        self.prey_positions[j] = [ny, nx]
                        break

    def step(self, actions: Union[List[int], Tuple[int, ...], np.ndarray]):
        self.current_step += 1

        # 1. Move wolves
        for i in range(self.n_wolves):
            act = Action(actions[i])
            dy, dx = DELTAS[act]
            wy, wx = self.wolf_positions[i]
            ny = np.clip(wy + dy, 0, self.grid_size - 1)
            nx = np.clip(wx + dx, 0, self.grid_size - 1)
            self.wolf_positions[i] = [ny, nx]

        # 2. Check capture conditions
        team_reward = 0.0
        captured_prey_indices = []

        for j in range(self.n_prey):
            if not self.prey_active[j]:
                continue
            py, px = self.prey_positions[j]

            # Count adjacent wolves within coop_radius (<= 1 cell Chebyshev)
            adj_wolves = 0
            for i in range(self.n_wolves):
                wy, wx = self.wolf_positions[i]
                if max(abs(wy - py), abs(wx - px)) <= self.coop_radius:
                    adj_wolves += 1

            if adj_wolves >= self.min_coop_wolves:
                captured_prey_indices.append(j)
                team_reward += self.capture_reward

        # Process captures
        for j in captured_prey_indices:
            if self.respawn_prey:
                occupied = [tuple(self.wolf_positions[w]) for w in range(self.n_wolves)] + [
                    tuple(self.prey_positions[k]) for k in range(self.n_prey) if k != j and self.prey_active[k]
                ]
                self.prey_positions[j] = self._random_unoccupied_cell(occupied)
            else:
                self.prey_active[j] = False

        # 3. Move surviving prey
        self._move_prey()

        # 4. Check termination
        terminated = False
        if not self.respawn_prey and not self.prey_active.any():
            terminated = True
        truncated = self.current_step >= self.max_steps

        obs = self._get_obs()
        rewards = [team_reward] * self.n_wolves
        dones = [terminated or truncated] * self.n_wolves
        info = {
            "captures": len(captured_prey_indices),
            "team_reward": team_reward,
            "prey_active": self.prey_active.copy(),
        }

        return obs, rewards, dones, info
