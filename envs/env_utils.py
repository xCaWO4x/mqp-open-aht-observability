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
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[int]]:
    """PREPROCESS specialised for Level-Based Foraging.

    LBF returns a tuple of per-agent ego-centric observations.
    Each agent's obs is: [food(3 each), self(feat_dim), other_agents(feat_dim each)].
    Food features come FIRST, then agent features (self first among agents).

    When observe_agent_levels=True (default):
        agent features = (y, x, level) → 3 per agent
    When observe_agent_levels=False:
        agent features = (y, x) → 2 per agent (level is dropped by LBF)

    We reconstruct the global state and produce:
        x_j = agent j's features (2 or 3 dims)
        u   = food features = 3 * n_food features
        B_j = [x_j ; u]    (obs_dim = agent_feat_dim + 3 * n_food)

    Parameters
    ----------
    raw_obs : tuple/list of arrays
        Per-agent observations from LBF.
        Format: [food_0(3), ..., food_m(3), self(feat), other_0(feat), ...]
        OR a single flat array (already global state).
    n_agents : int
        Number of agents in the environment.
    n_food : int
        Number of food items.
    prev_agent_ids : list of int or None
        Agent IDs from previous timestep (for hidden state tracking).
    prev_hidden : tuple of (h, c) or None
        Previous LSTM hidden states.
    hidden_dim : int
    device : str
    observe_agent_levels : bool
        Whether LBF was configured with observe_agent_levels=True.
        Determines per-agent feature dimension (3 vs 2).

    Returns
    -------
    B : Tensor, shape (n_agents, agent_feat_dim + 3*n_food)
    hidden : tuple of (h, c) each shape (n_agents, hidden_dim)
    agent_ids : list of int
    """
    agent_feat_dim = LBF_AGENT_FEAT_DIM if observe_agent_levels else 2

    if isinstance(raw_obs, (list, tuple)):
        # Multi-agent obs: reconstruct global state from ego-centric views.
        # LBF obs format: [food(3*n_food), self(agent_feat_dim), others(agent_feat_dim*(n_agents-1))]
        # Food comes FIRST, then agents (self first among agents).
        #
        # Strategy: extract each agent's own features from their ego obs,
        # and shared food features from any agent's obs.

        food_end = n_food * LBF_FOOD_FEAT_DIM
        agent_start = food_end  # agents start right after food

        agent_features = []   # list of (y, x[, level]) per agent
        for i in range(n_agents):
            obs_i = np.asarray(raw_obs[i], dtype=np.float32)
            # Self agent features start at food_end (first among agents)
            agent_features.append(obs_i[agent_start:agent_start + agent_feat_dim])

        # Food features from agent 0 (same positions for all, may differ in order)
        obs_0 = np.asarray(raw_obs[0], dtype=np.float32)
        food_features = obs_0[:food_end]

        # Build a flat global state: [agent_0(feat), agent_1(feat), ..., food(3*n_food)]
        global_state = np.concatenate(agent_features + [food_features])
    else:
        # Already a flat global state
        global_state = np.asarray(raw_obs, dtype=np.float32)

    # Build slices for the generic preprocess
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
# Generic helpers
# ======================================================================

def make_env(env_id: str, seed: int = 0, **kwargs):
    """Create and seed a gym environment by id."""
    try:
        import gymnasium as gym
    except ImportError:
        import gym
    env = gym.make(env_id, **kwargs)
    if hasattr(env, "seed"):
        env.seed(seed)
    return env
