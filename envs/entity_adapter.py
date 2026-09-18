"""
Entity Adapter for Level-Based Foraging (LBF) and Wolfpack.

Extracts structured entity tokens, entity types, and visibility masks from raw
observations without altering environment dynamics or sight-radius definitions.
Ensures identical observation masking across GPL, Transformer-Q, and IPPO-GRU.

CRITICAL INVARIANCE GUARANTEE:
Entities outside the sight radius are strictly zeroed in all feature dimensions
[0.0, 0.0, 0.0]. No sentinel coordinates, levels, or flags remain visible in
flat_obs or entity_features.
"""

from typing import List, NamedTuple, Optional, Tuple, Union
import numpy as np
import torch


# Entity type identifiers
ENTITY_TYPE_SELF = 0
ENTITY_TYPE_TEAMMATE = 1
ENTITY_TYPE_OBJECT = 2  # Food in LBF, Prey in Wolfpack


class EntityBatch(NamedTuple):
    """Container for structured entity representations."""
    entity_features: torch.Tensor  # Shape: (B, num_entities, feat_dim)
    entity_types: torch.Tensor     # Shape: (B, num_entities), dtype=torch.long
    visible_mask: torch.Tensor     # Shape: (B, num_entities), dtype=torch.bool (True = visible)
    flat_obs: torch.Tensor         # Shape: (B, flat_dim), flattened vector for MLP/PPO


def extract_entities_lbf(
    raw_obs,
    n_agents: int = 3,
    n_food: int = 3,
    grid_size: float = 8.0,
    observe_agent_levels: bool = False,
    sight_radius: Optional[int] = None,
    invisible_indices: Optional[List[int]] = None,
    device: str = "cpu",
) -> EntityBatch:
    """Extract entity tokens from LBF observation.

    LBF observation layout for agent 0:
    [food_0(3), ..., food_{M-1}(3), self(feat_dim), tm_1(feat_dim), ..., tm_{N-1}(feat_dim)]
    Invisible entities have coordinates -1.0.

    Tokens extracted (ordered: Self, Teammates, Foods):
    - Slot 0: Self token (type 0)
    - Slot 1..N-1: Teammate tokens (type 1)
    - Slot N..N+M-1: Food tokens (type 2)
    Total entities = n_agents + n_food (e.g. 3 + 3 = 6).

    INVARIANCE: Any invisible entity has ALL features strictly set to 0.0: [0.0, 0.0, 0.0].
    """
    agent_feat_dim = 3 if observe_agent_levels else 2
    food_feat_dim = 3
    food_end = n_food * food_feat_dim
    agent_start = food_end

    # Handle batch vs single obs
    if isinstance(raw_obs, (list, tuple)) and len(raw_obs) > 0 and isinstance(raw_obs[0], (list, tuple, np.ndarray)):
        if isinstance(raw_obs[0], (list, tuple, np.ndarray)) and len(raw_obs[0]) > 0 and not isinstance(raw_obs[0][0], (int, float, np.floating, np.integer)):
            obs_list = [np.asarray(o[0], dtype=np.float32) for o in raw_obs]
        else:
            obs_list = [np.asarray(raw_obs[0], dtype=np.float32)]
    else:
        obs_arr = np.asarray(raw_obs, dtype=np.float32)
        if obs_arr.ndim == 1:
            obs_list = [obs_arr]
        else:
            obs_list = [obs_arr[b] for b in range(len(obs_arr))]

    batch_size = len(obs_list)
    num_entities = n_agents + n_food
    feat_dim = 3  # (y, x, val)

    feats = np.zeros((batch_size, num_entities, feat_dim), dtype=np.float32)
    types = np.zeros((batch_size, num_entities), dtype=np.int64)
    vis_mask = np.zeros((batch_size, num_entities), dtype=bool)

    for b, obs_0 in enumerate(obs_list):
        # 1. Self entity (index 0)
        self_raw = obs_0[agent_start:agent_start + agent_feat_dim]
        sy, sx = self_raw[0], self_raw[1]
        s_val = self_raw[2] if observe_agent_levels else 1.0
        s_vis = (sy >= 0 and sx >= 0)
        if invisible_indices is not None and 0 in invisible_indices:
            s_vis = False

        if s_vis:
            feats[b, 0] = [sy / grid_size, sx / grid_size, s_val]
        else:
            feats[b, 0] = [0.0, 0.0, 0.0]
        types[b, 0] = ENTITY_TYPE_SELF
        vis_mask[b, 0] = s_vis

        # 2. Teammates (indices 1 to n_agents - 1)
        for j in range(1, n_agents):
            tm_start = agent_start + j * agent_feat_dim
            tm_raw = obs_0[tm_start:tm_start + agent_feat_dim]
            ty, tx = tm_raw[0], tm_raw[1]
            t_val = tm_raw[2] if observe_agent_levels else 1.0
            t_vis = (ty >= 0 and tx >= 0)

            if s_vis and sight_radius is not None:
                if max(abs(ty - sy), abs(tx - sx)) > sight_radius:
                    t_vis = False
            if invisible_indices is not None and j in invisible_indices:
                t_vis = False

            if t_vis:
                feats[b, j] = [ty / grid_size, tx / grid_size, t_val]
            else:
                feats[b, j] = [0.0, 0.0, 0.0]
            types[b, j] = ENTITY_TYPE_TEAMMATE
            vis_mask[b, j] = t_vis

        # 3. Food items (indices n_agents to n_agents + n_food - 1)
        for k in range(n_food):
            idx = n_agents + k
            f_start = k * food_feat_dim
            fy, fx, fl = obs_0[f_start], obs_0[f_start + 1], obs_0[f_start + 2]
            f_vis = (fy >= 0 and fx >= 0 and fl > 0)

            if s_vis and sight_radius is not None:
                if max(abs(fy - sy), abs(fx - sx)) > sight_radius:
                    f_vis = False
            if invisible_indices is not None and idx in invisible_indices:
                f_vis = False

            if f_vis:
                feats[b, idx] = [fy / grid_size, fx / grid_size, fl]
            else:
                feats[b, idx] = [0.0, 0.0, 0.0]
            types[b, idx] = ENTITY_TYPE_OBJECT
            vis_mask[b, idx] = f_vis

    flat_arr = feats.reshape(batch_size, -1)

    return EntityBatch(
        entity_features=torch.tensor(feats, dtype=torch.float32, device=device),
        entity_types=torch.tensor(types, dtype=torch.long, device=device),
        visible_mask=torch.tensor(vis_mask, dtype=torch.bool, device=device),
        flat_obs=torch.tensor(flat_arr, dtype=torch.float32, device=device),
    )


def extract_entities_wolfpack(
    raw_obs,
    n_wolves: int = 3,
    n_prey: int = 2,
    grid_size: float = 10.0,
    sight_radius: Optional[int] = None,
    invisible_indices: Optional[List[int]] = None,
    device: str = "cpu",
) -> EntityBatch:
    """Extract entity tokens from Wolfpack observation.

    Wolfpack observation layout for wolf 0:
    [prey_0(3), ..., prey_{M-1}(3), self(2), other_1(2), ..., other_{N-1}(2)]
    Prey layout: (y, x, active).
    Wolf layout: (y, x).
    Invisible entities have coordinates -1.0.

    Tokens extracted (ordered: Self, Teammates, Prey):
    - Slot 0: Self wolf (type 0)
    - Slot 1..N-1: Teammate wolves (type 1)
    - Slot N..N+M-1: Prey items (type 2)
    Total entities = n_wolves + n_prey (e.g. 3 + 2 = 5).

    INVARIANCE: Any invisible entity has ALL features strictly set to 0.0: [0.0, 0.0, 0.0].
    """
    prey_feat_dim = 3
    wolf_feat_dim = 2
    prey_end = n_prey * prey_feat_dim
    wolf_start = prey_end

    # Handle batch vs single obs
    if isinstance(raw_obs, (list, tuple)) and len(raw_obs) > 0 and isinstance(raw_obs[0], (list, tuple, np.ndarray)):
        if isinstance(raw_obs[0], (list, tuple, np.ndarray)) and len(raw_obs[0]) > 0 and not isinstance(raw_obs[0][0], (int, float, np.floating, np.integer)):
            obs_list = [np.asarray(o[0], dtype=np.float32) for o in raw_obs]
        else:
            obs_list = [np.asarray(raw_obs[0], dtype=np.float32)]
    else:
        obs_arr = np.asarray(raw_obs, dtype=np.float32)
        if obs_arr.ndim == 1:
            obs_list = [obs_arr]
        else:
            obs_list = [obs_arr[b] for b in range(len(obs_arr))]

    batch_size = len(obs_list)
    num_entities = n_wolves + n_prey
    feat_dim = 3  # (y, x, active)

    feats = np.zeros((batch_size, num_entities, feat_dim), dtype=np.float32)
    types = np.zeros((batch_size, num_entities), dtype=np.int64)
    vis_mask = np.zeros((batch_size, num_entities), dtype=bool)

    for b, obs_0 in enumerate(obs_list):
        # 1. Self wolf (index 0)
        sw_raw = obs_0[wolf_start:wolf_start + wolf_feat_dim]
        wy, wx = sw_raw[0], sw_raw[1]
        w_vis = (wy >= 0 and wx >= 0)
        if invisible_indices is not None and 0 in invisible_indices:
            w_vis = False

        if w_vis:
            norm_y = (wy / grid_size) if wy > 1.0 else wy
            norm_x = (wx / grid_size) if wx > 1.0 else wx
            feats[b, 0] = [norm_y, norm_x, 1.0]
        else:
            feats[b, 0] = [0.0, 0.0, 0.0]
        types[b, 0] = ENTITY_TYPE_SELF
        vis_mask[b, 0] = w_vis

        # 2. Teammate wolves (indices 1 to n_wolves - 1)
        for j in range(1, n_wolves):
            tw_start = wolf_start + j * wolf_feat_dim
            tw_raw = obs_0[tw_start:tw_start + wolf_feat_dim]
            ty, tx = tw_raw[0], tw_raw[1]
            t_vis = (ty >= 0 and tx >= 0)

            if w_vis and sight_radius is not None:
                if max(abs(ty - wy), abs(tx - wx)) > sight_radius:
                    t_vis = False
            if invisible_indices is not None and j in invisible_indices:
                t_vis = False

            if t_vis:
                norm_ty = (ty / grid_size) if ty > 1.0 else ty
                norm_tx = (tx / grid_size) if tx > 1.0 else tx
                feats[b, j] = [norm_ty, norm_tx, 1.0]
            else:
                feats[b, j] = [0.0, 0.0, 0.0]
            types[b, j] = ENTITY_TYPE_TEAMMATE
            vis_mask[b, j] = t_vis

        # 3. Prey items (indices n_wolves to n_wolves + n_prey - 1)
        for k in range(n_prey):
            idx = n_wolves + k
            p_start = k * prey_feat_dim
            py, px, p_act = obs_0[p_start], obs_0[p_start + 1], obs_0[p_start + 2]
            p_vis = (py >= 0 and px >= 0 and p_act > 0)

            if w_vis and sight_radius is not None:
                if max(abs(py - wy), abs(px - wx)) > sight_radius:
                    p_vis = False
            if invisible_indices is not None and idx in invisible_indices:
                p_vis = False

            if p_vis:
                norm_py = (py / grid_size) if py > 1.0 else py
                norm_px = (px / grid_size) if px > 1.0 else px
                feats[b, idx] = [norm_py, norm_px, p_act]
            else:
                feats[b, idx] = [0.0, 0.0, 0.0]
            types[b, idx] = ENTITY_TYPE_OBJECT
            vis_mask[b, idx] = p_vis

    flat_arr = feats.reshape(batch_size, -1)

    return EntityBatch(
        entity_features=torch.tensor(feats, dtype=torch.float32, device=device),
        entity_types=torch.tensor(types, dtype=torch.long, device=device),
        visible_mask=torch.tensor(vis_mask, dtype=torch.bool, device=device),
        flat_obs=torch.tensor(flat_arr, dtype=torch.float32, device=device),
    )


def extract_entities(
    raw_obs,
    env_name: str,
    device: str = "cpu",
    **kwargs,
) -> EntityBatch:
    """Unified factory for entity extraction."""
    env_lower = env_name.lower()
    if "lbf" in env_lower or "foraging" in env_lower:
        return extract_entities_lbf(
            raw_obs,
            n_agents=kwargs.get("n_agents", 3),
            n_food=kwargs.get("n_food", 3),
            grid_size=kwargs.get("grid_size", 8.0),
            observe_agent_levels=kwargs.get("observe_agent_levels", False),
            sight_radius=kwargs.get("sight_radius", None),
            invisible_indices=kwargs.get("invisible_indices", None),
            device=device,
        )
    elif "wolfpack" in env_lower:
        return extract_entities_wolfpack(
            raw_obs,
            n_wolves=kwargs.get("n_wolves", 3),
            n_prey=kwargs.get("n_prey", 2),
            grid_size=kwargs.get("grid_size", 10.0),
            sight_radius=kwargs.get("sight_radius", None),
            invisible_indices=kwargs.get("invisible_indices", None),
            device=device,
        )
    else:
        raise ValueError(f"Unknown environment for entity extraction: {env_name}")
