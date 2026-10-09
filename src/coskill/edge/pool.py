"""Trajectory buffering and asynchronous evolution-trigger policies."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

from ..utils.models import Trajectory


@dataclass(slots=True)
class TriggerConfig:
    """Thresholds controlling when buffered trajectories are uploaded."""

    recent_window: int = 16
    failure_rate_threshold: float = 0.60
    fixed_batch_size: int | None = None


@dataclass(slots=True)
class TriggerDecision:
    """Decision and evidence returned after adding a trajectory to the pool."""

    triggered: bool
    reasons: list[str] = field(default_factory=list)


class TrajectoryPool:
    """Edge-side buffer and paper-aligned asynchronous upload trigger."""

    def __init__(self, config: TriggerConfig | None = None) -> None:
        self.config = config or TriggerConfig()
        if self.config.recent_window <= 0:
            raise ValueError("recent_window must be positive")
        if self.config.fixed_batch_size is not None and self.config.fixed_batch_size <= 0:
            raise ValueError("fixed_batch_size must be positive")
        self._pending: dict[str, list[Trajectory]] = defaultdict(list)
        self._recent: dict[str, deque[bool]] = defaultdict(
            lambda: deque(maxlen=self.config.recent_window)
        )

    def add(self, trajectory: Trajectory) -> TriggerDecision:
        """Buffer a trajectory and report whether cloud evolution should run."""
        task_type = trajectory.task_type
        self._pending[task_type].append(trajectory)
        recent = self._recent[task_type]
        recent.append(trajectory.success)
        reasons: list[str] = []
        if (
            self.config.fixed_batch_size is not None
            and len(self._pending[task_type]) >= self.config.fixed_batch_size
        ):
            reasons.append(
                "fixed update batch reached: "
                f"{len(self._pending[task_type])} trajectories"
            )
            return TriggerDecision(True, reasons)
        if self.config.fixed_batch_size is not None:
            return TriggerDecision(False, reasons)
        # The pending buffer is reset after an evolution. This means the gate
        # requires N newly accumulated trajectories instead of firing on every
        # episode after the first recent window.
        if len(self._pending[task_type]) >= self.config.recent_window:
            success_rate = sum(recent) / len(recent)
            failure_rate = 1.0 - success_rate
            if failure_rate >= self.config.failure_rate_threshold:
                reasons.append(
                    f"recent failure rate {failure_rate:.2f} >= "
                    f"{self.config.failure_rate_threshold:.2f}"
                )

        return TriggerDecision(bool(reasons), reasons)

    def drain(self, task_type: str) -> list[Trajectory]:
        """Remove and return all currently buffered trajectories for a task type."""
        trajectories = self._pending.pop(task_type, [])
        return trajectories

    def pending_count(self, task_type: str) -> int:
        return len(self._pending.get(task_type, []))
