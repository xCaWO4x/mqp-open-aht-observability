"""
Observation-Only Distractor Wrapper for Level-Based Foraging and Wolfpack.

CRITICAL INVARIANCE GUARANTEES:
1. Distractors exist strictly in the observation returned to the agent.
2. Distractors never affect:
   - rewards
   - collisions
   - food collection / prey capture
   - teammate or prey behavior
   - environment transitions
   - episode termination
   - real entity spawning or level generation
   - underlying environment RNG
3. A completely separate RNG stream is used for ghost entity placement,
   trajectories, and slot permutation.
4. Slot ordering is randomly permuted so distractors do not predictably
   occupy tail indices.
"""

from typing import List, Optional, Tuple, Union
import gymnasium as gym
import numpy as np


class DistractorEnvWrapper(gym.Wrapper):
    """Observation wrapper adding causally irrelevant ghost entities."""

    def __init__(
        self,
        env: gym.Env,
        num_distractors: int = 0,
        distractor_type: str = "none",  # "none", "semantic", "null"
        distractor_seed: Optional[int] = None,
        shuffle_slots: bool = True,
    ):
        super().__init__(env)
        self.num_distractors = int(num_distractors)
        self.distractor_type = str(distractor_type).lower().strip()
        self.distractor_seed = distractor_seed
        self.shuffle_slots = shuffle_slots

        if self.distractor_type not in ("none", "semantic", "null"):
            raise ValueError(f"Unknown distractor_type: {distractor_type}")
        if self.num_distractors == 0:
            self.distractor_type = "none"

        # Determine environment type
        env_unwrapped = env.unwrapped
        env_class_name = env_unwrapped.__class__.__name__.lower()
        self.is_lbf = "foraging" in env_class_name or hasattr(env_unwrapped, "field_size")
        self.is_wolfpack = "wolfpack" in env_class_name or hasattr(env_unwrapped, "n_wolves")

        if not (self.is_lbf or self.is_wolfpack):
            raise ValueError(f"Unsupported environment for distractor wrapper: {env_class_name}")

        if self.is_lbf:
            self.grid_size = int(env_unwrapped.field_size[0])
            self.n_agents = int(env_unwrapped.players) if hasattr(env_unwrapped, "players") and isinstance(env_unwrapped.players, int) else len(env_unwrapped.players)
            self.n_real_objects = int(env_unwrapped.max_num_food)
            self.object_feat_dim = 3  # (y, x, level)
            self.agent_feat_dim = 3 if getattr(env_unwrapped, "observe_agent_levels", False) else 2
        else:
            self.grid_size = int(env_unwrapped.grid_size)
            self.n_agents = int(env_unwrapped.n_wolves)
            self.n_real_objects = int(env_unwrapped.n_prey)
            self.object_feat_dim = 3  # (y, x, active)
            self.agent_feat_dim = 2   # (y, x)

        self.total_objects = self.n_real_objects + self.num_distractors

        # Internal distractor state
        self._rng = np.random.default_rng(distractor_seed)
        self.ghost_positions = np.zeros((self.num_distractors, 2), dtype=int)
        self.ghost_features = np.zeros((self.num_distractors,), dtype=float)
        self.slot_permutation = np.arange(self.total_objects)

    def __setattr__(self, name, value):
        if name in (
            "env", "num_distractors", "distractor_type", "distractor_seed", "shuffle_slots",
            "is_lbf", "is_wolfpack", "grid_size", "n_agents", "n_real_objects",
            "object_feat_dim", "agent_feat_dim", "total_objects", "_rng",
            "ghost_positions", "ghost_features", "slot_permutation", "_action_space", "_observation_space"
        ):
            super().__setattr__(name, value)
        else:
            if hasattr(self, "env"):
                setattr(self.env, name, value)
            else:
                super().__setattr__(name, value)

    def seed_distractor(self, seed: Optional[int] = None):
        """Seed the dedicated distractor RNG without touching the environment RNG."""
        self._rng = np.random.default_rng(seed)

    def _sample_unoccupied_cell(self, occupied: List[Tuple[int, int]]) -> Tuple[int, int]:
        all_cells = [
            (y, x)
            for y in range(self.grid_size)
            for x in range(self.grid_size)
            if (y, x) not in occupied
        ]
        if not all_cells:
            # Fallback to any valid cell if completely occupied
            y = int(self._rng.integers(0, self.grid_size))
            x = int(self._rng.integers(0, self.grid_size))
            return (y, x)
        idx = int(self._rng.integers(0, len(all_cells)))
        return all_cells[idx]

    def _spawn_ghost_entities(self):
        """Spawn ghost entities at unoccupied coordinates with proper features."""
        if self.num_distractors == 0:
            return

        env_u = self.env.unwrapped
        occupied: List[Tuple[int, int]] = []

        if self.is_lbf:
            # Collect real player and food positions
            for player in env_u.players:
                if player.position is not None:
                    occupied.append(tuple(player.position))
            if hasattr(env_u, "food_positions") and env_u.food_positions is not None:
                for pos in env_u.food_positions:
                    occupied.append(tuple(pos))
            elif hasattr(env_u, "field"):
                foods = np.argwhere(env_u.field > 0)
                for f in foods:
                    occupied.append(tuple(f))

            # Sample ghost food
            food_levels_dist = [2, 3]
            food_probs = [0.6, 0.4]
            for i in range(self.num_distractors):
                pos = self._sample_unoccupied_cell(occupied)
                self.ghost_positions[i] = pos
                occupied.append(pos)

                if self.distractor_type == "semantic":
                    level = float(self._rng.choice(food_levels_dist, p=food_probs))
                    self.ghost_features[i] = level
                elif self.distractor_type == "null":
                    # Explicit null marker: level 0.0 (real food is strictly >= 1.0)
                    self.ghost_features[i] = 0.0

        elif self.is_wolfpack:
            # Collect wolf and prey positions
            for wy, wx in env_u.wolf_positions:
                occupied.append((int(wy), int(wx)))
            for py, px in env_u.prey_positions:
                occupied.append((int(py), int(px)))

            # Sample ghost prey
            for i in range(self.num_distractors):
                pos = self._sample_unoccupied_cell(occupied)
                self.ghost_positions[i] = pos
                occupied.append(pos)

                if self.distractor_type == "semantic":
                    # Matches real prey: active flag = 1.0
                    self.ghost_features[i] = 1.0
                elif self.distractor_type == "null":
                    # Explicit null marker: active flag = 0.0
                    self.ghost_features[i] = 0.0

        # Slot permutation: shuffle slots across real + distractor entities
        if self.shuffle_slots:
            self.slot_permutation = self._rng.permutation(self.total_objects)
        else:
            self.slot_permutation = np.arange(self.total_objects)

    def _move_ghost_prey(self):
        """Advance ghost prey along persistent random-walk trajectories matching real prey."""
        if not self.is_wolfpack or self.num_distractors == 0:
            return

        env_u = self.env.unwrapped
        wolf_positions = env_u.wolf_positions

        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # N, S, W, E, Stay

        for i in range(self.num_distractors):
            py, px = self.ghost_positions[i]
            # Try a random move
            perm = self._rng.permutation(len(deltas))
            for act_idx in perm:
                dy, dx = deltas[act_idx]
                ny, nx = py + dy, px + dx
                if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                    # Avoid stepping directly into a wolf position (matching real prey)
                    if not any(np.array_equal([ny, nx], wolf_positions[w]) for w in range(self.n_agents)):
                        self.ghost_positions[i] = [ny, nx]
                        break

    def _augment_obs(self, raw_obs) -> List[np.ndarray]:
        """Insert ghost entities into per-agent observations with slot interleaving."""
        if self.num_distractors == 0 or self.distractor_type == "none":
            return raw_obs

        real_obj_end = self.n_real_objects * self.object_feat_dim
        augmented_obs = []

        for i in range(self.n_agents):
            agent_obs = np.asarray(raw_obs[i], dtype=np.float32)

            # 1. Extract real object features
            real_obj_feats = agent_obs[:real_obj_end].reshape(self.n_real_objects, self.object_feat_dim)

            # 2. Extract agent features (self + teammates)
            agent_feats = agent_obs[real_obj_end:]

            # 3. Build ghost object features for agent i
            ghost_obj_feats = np.zeros((self.num_distractors, self.object_feat_dim), dtype=np.float32)
            for d in range(self.num_distractors):
                gy, gx = self.ghost_positions[d]
                gval = self.ghost_features[d]
                ghost_obj_feats[d] = [float(gy), float(gx), float(gval)]

            # 4. Concatenate real + ghost objects: shape (total_objects, object_feat_dim)
            combined_objects = np.concatenate([real_obj_feats, ghost_obj_feats], axis=0)

            # 5. Apply slot permutation
            interleaved_objects = combined_objects[self.slot_permutation].flatten()

            # 6. Reconstruct agent observation: [objects, agents]
            new_agent_obs = np.concatenate([interleaved_objects, agent_feats]).astype(np.float32)
            augmented_obs.append(new_agent_obs)

        return augmented_obs

    def reset(self, **kwargs):
        # 1. Reset real environment (real seed and transitions untouched)
        obs, info = self.env.reset(**kwargs)

        # 2. Spawn ghost entities with independent distractor RNG
        self._spawn_ghost_entities()

        # 3. Return augmented observations
        aug_obs = self._augment_obs(obs)
        return aug_obs, info

    def step(self, actions):
        # 1. Advance real environment (dynamics completely unchanged)
        step_out = self.env.step(actions)

        # 2. Handle step return arity (4-tuple or 5-tuple)
        if len(step_out) == 5:
            obs, rewards, terminated, truncated, info = step_out
            ret_5 = True
        else:
            obs, rewards, dones, info = step_out
            ret_5 = False

        # 3. Advance ghost prey trajectories if applicable
        if self.is_wolfpack:
            self._move_ghost_prey()

        # 4. Augment observation
        aug_obs = self._augment_obs(obs)

        if ret_5:
            return aug_obs, rewards, terminated, truncated, info
        else:
            return aug_obs, rewards, dones, info
