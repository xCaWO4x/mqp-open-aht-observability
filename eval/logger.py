"""
Lightweight logging wrapper for TensorBoard and optional wandb.

Usage:
    logger = Logger(log_dir="runs/experiment", use_wandb=False)
    logger.log_scalar("train/return", 42.0, step=100)
    logger.log_scalars("train", {"q_loss": 0.1, "agent_loss": 0.2}, step=100)
    logger.close()
"""

import os
from typing import Dict, Optional


class Logger:
    """Unified logger writing to TensorBoard and optionally wandb.

    Parameters
    ----------
    log_dir : str
        Directory for TensorBoard event files.
    use_wandb : bool
        Whether to also log to wandb (must be initialized externally).
    use_tensorboard : bool
        Whether to log to TensorBoard.
    """

    def __init__(
        self,
        log_dir: str = "runs/default",
        use_wandb: bool = False,
        use_tensorboard: bool = True,
    ):
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self._tb_writer = None

        if use_tensorboard:
            os.makedirs(log_dir, exist_ok=True)
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(log_dir)
            except ImportError:
                print("Warning: tensorboard not installed, disabling TB logging")
                self.use_tensorboard = False

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a single scalar value."""
        if self.use_tensorboard and self._tb_writer is not None:
            self._tb_writer.add_scalar(tag, value, step)
        if self.use_wandb:
            try:
                import wandb
                wandb.log({tag: value}, step=step)
            except ImportError:
                pass

    def log_scalars(self, prefix: str, metrics: Dict[str, float], step: int):
        """Log multiple scalars under a common prefix."""
        for key, value in metrics.items():
            self.log_scalar(f"{prefix}/{key}", value, step)

    def log_episode(
        self,
        episode: int,
        episode_return: float,
        episode_length: int,
        extra: Optional[Dict[str, float]] = None,
    ):
        """Convenience method for logging episode-level stats."""
        self.log_scalar("episode/return", episode_return, episode)
        self.log_scalar("episode/length", episode_length, episode)
        if extra:
            self.log_scalars("episode", extra, episode)

    def flush(self):
        if self._tb_writer is not None:
            self._tb_writer.flush()

    def close(self):
        if self._tb_writer is not None:
            self._tb_writer.close()
