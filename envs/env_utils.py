"""
Environment utilities: PREPROCESS and helpers for GPL input formatting.

Implements the PREPROCESS function from Appendix C.1 (Rahman et al. 2023):

    1. Split raw state s_t into per-agent features x_j and shared features u.
       - x_j: features whose values differ per agent (position, orientation, etc.)
       - u:   features shared across all agents (food/ball location, etc.)
    2. Concatenate: B_j = [x_j ; u] for each agent j.
    3. Return input batch B = {B_1, ..., B_N}.

This ensures each agent's type vector (computed by the LSTM) depends only
on its own trajectory + global context, not on other agents' features.

Also handles LSTM hidden state management for open agent sets:
    - New agents: initialise hidden state rows to zero.
    - Departed agents: remove corresponding hidden state rows.
    - Reordering: tracked by agent ID → index mapping.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


# ======================================================================
# PREPROCESS — Appendix C.1
# ======================================================================

def preprocess(
    raw_obs,
    agent_feature_slices: Dict[int, slice],
    shared_feature_slice: slice,
    prev_agent_ids: Optional[List[int]] = None,
    curr_agent_ids: Optional[List[int]] = None,
    prev_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    hidden_dim: int = 128,
    device: str = "cpu",
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[int]]:
    """PREPROCESS function from Appendix C.1.

    Splits raw observation into per-agent input batch B and manages LSTM
    hidden states when the agent set changes (openness).

    Parameters
    ----------
    raw_obs : array-like
        Flat state vector s_t from the environment.
    agent_feature_slices : dict of {agent_id: slice}
        Maps each agent's ID to the slice of raw_obs containing its features.
    shared_feature_slice : slice
        Slice of raw_obs containing shared (global) features u.
    prev_agent_ids : list of int or None
        Agent IDs present at the previous timestep (for hidden state tracking).
    curr_agent_ids : list of int or None
        Agent IDs present at the current timestep.  If None, inferred from
        agent_feature_slices keys.
    prev_hidden : tuple of (h, c) each shape (N_prev, hidden_dim) or None
        LSTM hidden states from the previous timestep.
    hidden_dim : int
        LSTM hidden state dimension (for zero-initialising new agents).
    device : str
        Torch device.

    Returns
    -------
    B : Tensor, shape (N, obs_dim)
        Per-agent input batch where B[j] = [x_j ; u].
        obs_dim = agent_feature_dim + shared_feature_dim.
    hidden : tuple of (h, c) each shape (N, hidden_dim)
        LSTM hidden states aligned to the current agent ordering.
        New agents get zeros; departed agents are removed.
    curr_agent_ids : list of int
        Ordered agent IDs matching B's row ordering.
    """
    obs = np.asarray(raw_obs, dtype=np.float32)

    # Extract shared features u
    u = obs[shared_feature_slice]

    # Current agent IDs
    if curr_agent_ids is None:
        curr_agent_ids = sorted(agent_feature_slices.keys())

    # Build per-agent input batch: B_j = [x_j ; u]
    B_rows = []
    for agent_id in curr_agent_ids:
        x_j = obs[agent_feature_slices[agent_id]]
        B_rows.append(np.concatenate([x_j, u]))

    B = torch.tensor(np.stack(B_rows), dtype=torch.float32, device=device)
    N = len(curr_agent_ids)

    # --- LSTM hidden state management for openness ---
    if prev_hidden is None or prev_agent_ids is None:
        # No prior state: zero-initialise
        h = torch.zeros(N, hidden_dim, device=device)
        c = torch.zeros(N, hidden_dim, device=device)
    else:
        h_prev, c_prev = prev_hidden
        # Build a mapping from previous agent ID → row index
        prev_id_to_idx = {aid: i for i, aid in enumerate(prev_agent_ids)}

        h = torch.zeros(N, hidden_dim, device=device)
        c = torch.zeros(N, hidden_dim, device=device)

        for new_idx, agent_id in enumerate(curr_agent_ids):
            if agent_id in prev_id_to_idx:
                # Carry forward hidden state
                old_idx = prev_id_to_idx[agent_id]
                h[new_idx] = h_prev[old_idx]
                c[new_idx] = c_prev[old_idx]
            # else: new agent → stays zero-initialised

    return B, (h, c), curr_agent_ids


# ======================================================================
# LBF PREPROCESS
# ======================================================================

# LBF obs layout (from _make_gym_obs in lbforaging/foraging/environment.py):
#   Agent i's obs = [food_0_y, food_0_x, food_0_level,
#                    food_1_y, food_1_x, food_1_level,
#                    ...,
#                    self_y, self_x, self_level,
#                    other1_y, other1_x, other1_level, ...]
#
# **FOOD features come FIRST, then AGENT features.**
#
# Layout sizes:
#   food features:  3 per food  (y, x, level)   — FIRST
#   agent features: 3 per agent (y, x, level)   — SECOND (self first among agents)
#   total per-agent obs = 3 * n_food + 3 * n_agents
#
# For PREPROCESS, we reconstruct global state and produce:
#   x_j = agent j's (y, x, level) — 3 features
#   u   = food features — 3 * n_food features
#   B_j = [x_j ; u]

LBF_AGENT_FEAT_DIM = 3   # (y, x, level) per agent
LBF_FOOD_FEAT_DIM = 3    # (y, x, level) per food


def preprocess_lbf(
    raw_obs,
    n_agents: int,
    n_food: int = 1,
    prev_agent_ids: Optional[List[int]] = None,
    prev_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    hidden_dim: int = 128,
    device: str = "cpu",
    observe_agent_levels: bool = True,
    from_ego_perspective: bool = True,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[int]]:
    """PREPROCESS specialised for Level-Based Foraging.

    LBF returns a tuple of per-agent ego-centric observations.
    Each agent's obs is: [food(3 each), self(feat_dim), other_agents(feat_dim each)].
    Food features come FIRST, then agent features (self first among agents).

    When observe_agent_levels=True:
        agent features = (y, x, level) -> 3 per agent
    When observe_agent_levels=False:
        agent features = (y, x) -> 2 per agent

    When from_ego_perspective=True:
        Teammate features are extracted from agent 0's field of view. Entities outside
        the spatial sight radius are masked to -1.
    When from_ego_perspective=False:
        Each agent's features are extracted from its own ego-perspective (always visible to itself).

    Returns
    -------
    B : Tensor, shape (n_agents, agent_feat_dim + 3*n_food)
    hidden : tuple of (h, c) each shape (n_agents, hidden_dim)
    agent_ids : list of int
    """
    agent_feat_dim = LBF_AGENT_FEAT_DIM if observe_agent_levels else 2
    food_end = n_food * LBF_FOOD_FEAT_DIM
    agent_start = food_end

    if isinstance(raw_obs, (list, tuple)):
        obs_0 = np.asarray(raw_obs[0], dtype=np.float32)
        food_features = obs_0[:food_end]

        agent_features = []
        if from_ego_perspective:
            # Agent 0 self
            agent_features.append(obs_0[agent_start:agent_start + agent_feat_dim])
            # Teammates from agent 0's observation (masked to -1 if outside sight)
            for j in range(1, n_agents):
                tm_start = agent_start + j * agent_feat_dim
                agent_features.append(obs_0[tm_start:tm_start + agent_feat_dim])
        else:
            for i in range(n_agents):
                obs_i = np.asarray(raw_obs[i], dtype=np.float32)
                agent_features.append(obs_i[agent_start:agent_start + agent_feat_dim])

        global_state = np.concatenate(agent_features + [food_features])
    else:
        global_state = np.asarray(raw_obs, dtype=np.float32)

    agent_feature_slices = {}
    curr_agent_ids = list(range(n_agents))
    for i in range(n_agents):
        start = i * agent_feat_dim
        agent_feature_slices[i] = slice(start, start + agent_feat_dim)

    shared_feature_slice = slice(n_agents * agent_feat_dim, len(global_state))

    return preprocess(
        global_state, agent_feature_slices, shared_feature_slice,
        prev_agent_ids, curr_agent_ids, prev_hidden,
        hidden_dim, device,
    )


# ======================================================================
# Wolfpack PREPROCESS
# ======================================================================

WOLFPACK_WOLF_FEAT_DIM = 2    # (y, x) per wolf
WOLFPACK_PREY_FEAT_DIM = 3    # (y, x, active) per prey


def preprocess_wolfpack(
    raw_obs,
    n_wolves: int = 3,
    n_prey: int = 2,
    grid_size: float = 10.0,
    normalize_coords: bool = True,
    prev_agent_ids: Optional[List[int]] = None,
    prev_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    hidden_dim: int = 128,
    device: str = "cpu",
    from_ego_perspective: bool = True,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[int]]:
    """PREPROCESS specialised for Wolfpack.

    Obs layout per agent i:
        [prey_0(3), ..., prey_{M-1}(3), self(2), other_1(2), ..., other_{N-1}(2)]
    Entities out of sight are masked to -1.0.

    When normalize_coords=True:
        Valid coordinates [0, grid_size) are divided by grid_size (10.0) into [0, 1].
        Masked values remain -1.0.

    Constructs:
        x_j = wolf j's (y, x) -> 2 features
        u   = prey features = 3 * n_prey features
        B_j = [x_j ; u] -> obs_dim = 2 + 3 * n_prey (e.g. 2 + 6 = 8)

    Returns
    -------
    B : Tensor, shape (n_wolves, 2 + 3*n_prey)
    hidden : tuple of (h, c) each shape (n_wolves, hidden_dim)
    agent_ids : list of int
    """
    prey_end = n_prey * WOLFPACK_PREY_FEAT_DIM
    wolf_start = prey_end

    def _norm_coords(arr):
        if not normalize_coords:
            return arr
        res = arr.copy()
        mask = res >= 0
        res[mask] = res[mask] / grid_size
        return res

    if isinstance(raw_obs, (list, tuple)):
        obs_0 = np.asarray(raw_obs[0], dtype=np.float32)
        raw_prey = obs_0[:prey_end].copy()
        
        # Normalize prey coordinates (y, x) while preserving active flag
        prey_features = raw_prey.copy()
        if normalize_coords:
            for p_idx in range(n_prey):
                base = p_idx * WOLFPACK_PREY_FEAT_DIM
                if prey_features[base] >= 0:
                    prey_features[base] /= grid_size
                if prey_features[base + 1] >= 0:
                    prey_features[base + 1] /= grid_size

        wolf_features = []
        if from_ego_perspective:
            # Wolf 0 (self)
            w0 = _norm_coords(obs_0[wolf_start:wolf_start + WOLFPACK_WOLF_FEAT_DIM])
            wolf_features.append(w0)
            # Teammate wolves from wolf 0's observation
            for j in range(1, n_wolves):
                w_start = wolf_start + j * WOLFPACK_WOLF_FEAT_DIM
                wj = _norm_coords(obs_0[w_start:w_start + WOLFPACK_WOLF_FEAT_DIM])
                wolf_features.append(wj)
        else:
            for i in range(n_wolves):
                obs_i = np.asarray(raw_obs[i], dtype=np.float32)
                wi = _norm_coords(obs_i[wolf_start:wolf_start + WOLFPACK_WOLF_FEAT_DIM])
                wolf_features.append(wi)

        global_state = np.concatenate(wolf_features + [prey_features])
    else:
        global_state = np.asarray(raw_obs, dtype=np.float32)

    agent_feature_slices = {}
    curr_agent_ids = list(range(n_wolves))
    for i in range(n_wolves):
        start = i * WOLFPACK_WOLF_FEAT_DIM
        agent_feature_slices[i] = slice(start, start + WOLFPACK_WOLF_FEAT_DIM)

    shared_feature_slice = slice(n_wolves * WOLFPACK_WOLF_FEAT_DIM, len(global_state))

    return preprocess(
        global_state, agent_feature_slices, shared_feature_slice,
        prev_agent_ids, curr_agent_ids, prev_hidden,
        hidden_dim, device,
    )


# ======================================================================
# Generic helpers
# ======================================================================

def make_env(env_id: str, seed: int = 0, **kwargs):
    """Create and seed an environment by id or type."""
    if env_id.lower().startswith("wolfpack"):
        from envs.wolfpack_env import WolfpackEnv
        env = WolfpackEnv(seed=seed, **kwargs)
        return env

    try:
        import gymnasium as gym
    except ImportError:
        import gym
    env = gym.make(env_id, **kwargs)
    if hasattr(env, "seed"):
        env.seed(seed)
    return env
