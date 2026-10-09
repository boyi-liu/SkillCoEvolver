"""Co-Skill edge specialization: robust actions and adaptive upload triggers."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Sequence

from ..llm import LLMClient, LLMError, Message
from ..utils.models import TokenUsage, Trajectory
from .base import BaseEdge, TriggerConfig, TriggerDecision


class EdgeAgent(BaseEdge):
    """Add corrective retries to the baseline edge executor."""

    async def _complete_action(
        self,
        messages: list[Message],
        *,
        admissible_actions: Sequence[str] | None = None,
    ) -> tuple[dict[str, Any], Any]:
        total_usage = TokenUsage()
        last_payload: dict[str, Any] = {}
        for attempt in range(self.action_retries + 1):
            attempt_messages = messages
            if attempt:
                if admissible_actions is not None:
                    valid_labels = ", ".join(
                        str(index) for index in range(len(admissible_actions))
                    )
                    correction = (
                        f"Your previous response {last_payload!r} did not copy a valid "
                        f"menu label. The only valid labels are [{valid_labels}]. "
                        "Copy one integer printed before an action; do not count the "
                        "actions yourself. Return only "
                        '{"action_index": <one listed integer>}.'
                    )
                else:
                    correction = (
                        f"Your previous response {last_payload!r} did not contain exactly "
                        "one usable action. Return only "
                        '{"action": "one non-empty environment action"}.'
                    )
                attempt_messages = [*messages, Message("user", correction)]
            payload, response = await self.client.complete_json(attempt_messages)
            last_payload = payload
            total_usage = total_usage + response.usage
            if self._resolved_action(payload, admissible_actions):
                response.usage = total_usage
                return payload, response
        raise LLMError(
            f"Device model returned an invalid action after "
            f"{self.action_retries} retries; payload={last_payload!r}; "
            f"admissible_count={len(admissible_actions) if admissible_actions is not None else 'n/a'}"
        )


class TrajectoryPool:
    """Co-Skill buffer with fixed-batch and failure-rate upload triggers."""

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
        if len(self._pending[task_type]) >= self.config.recent_window:
            failure_rate = 1.0 - sum(recent) / len(recent)
            if failure_rate >= self.config.failure_rate_threshold:
                reasons.append(
                    f"recent failure rate {failure_rate:.2f} >= "
                    f"{self.config.failure_rate_threshold:.2f}"
                )
        return TriggerDecision(bool(reasons), reasons)

    def drain(self, task_type: str) -> list[Trajectory]:
        return self._pending.pop(task_type, [])

    def pending_count(self, task_type: str) -> int:
        return len(self._pending.get(task_type, []))


class CoSkillEdge(BaseEdge):
    """Co-Skill edge runtime composed from its agent and trajectory pool."""

    def __init__(
        self,
        client: LLMClient,
        *,
        max_steps: int = 40,
        action_retries: int = 2,
        trigger: TriggerConfig | None = None,
    ) -> None:
        super().__init__(
            client,
            max_steps=max_steps,
            action_retries=action_retries,
            trigger=trigger,
        )
        self.agent = EdgeAgent(
            client,
            max_steps=max_steps,
            action_retries=action_retries,
        )
        self.pool = TrajectoryPool(trigger)


__all__ = ["CoSkillEdge", "EdgeAgent", "TrajectoryPool"]
