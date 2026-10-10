"""Default cloud runtime and reusable flat-skill data structures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..llm import LLMClient, Message
from ..utils import Task, Trajectory
from .cloud_prompts import BASE_EVOLUTION_SYSTEM_PROMPT, base_cloud_prompt
from .contracts import EvolutionDraft, SkillArtifact, ValidationResult


@dataclass(slots=True)
class FlatSkill:
    """Single-instruction skill used by the default framework runtime."""

    task_type: str
    goal: str
    instruction: str
    revision: int = 0

    def guidance(self, *, max_depth: int | None = None) -> str:
        del max_depth
        return self.instruction

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "flat",
            "task_type": self.task_type,
            "goal": self.goal,
            "instruction": self.instruction,
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FlatSkill":
        return cls(
            task_type=str(value["task_type"]),
            goal=str(value["goal"]),
            instruction=str(value["instruction"]),
            revision=int(value.get("revision", 0)),
        )


@dataclass(slots=True)
class RawTrajectoryEvidence:
    """Uncompressed trajectory evidence used by the default cloud runtime."""

    task_type: str
    trajectories: list[Trajectory]

    def to_prompt(self) -> str:
        return json.dumps(
            [trajectory.to_dict() for trajectory in self.trajectories],
            ensure_ascii=False,
            indent=2,
            default=str,
        )


class BaseCloudRuntime:
    """Default registry, evidence, evolution, and validation implementation."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self._registry: dict[str, FlatSkill] = {}

    def retrieve_skill(self, task: Task) -> SkillArtifact:
        existing = self.lookup_skill(task.task_type)
        if existing is not None:
            return existing
        goal = task.skill_template or task.goal
        skill = FlatSkill(task.task_type, goal, goal)
        self.commit_skill(skill)
        return skill

    def lookup_skill(self, task_type: str) -> SkillArtifact | None:
        return self._registry.get(task_type)

    def commit_skill(self, skill: SkillArtifact) -> None:
        if not isinstance(skill, FlatSkill):
            raise TypeError("default cloud registry expects FlatSkill")
        self._registry[skill.task_type] = skill

    def compile_evidence(
        self,
        task_type: str,
        trajectories: list[Trajectory],
    ) -> RawTrajectoryEvidence:
        return RawTrajectoryEvidence(task_type, trajectories)

    async def evolve_skill(
        self,
        skill: SkillArtifact,
        evidence: Any,
        trigger_reasons: list[str],
        *,
        framework_name: str,
    ) -> EvolutionDraft:
        if not isinstance(evidence, RawTrajectoryEvidence):
            raise TypeError("default cloud evolution expects raw evidence")
        if not isinstance(skill, FlatSkill):
            raise TypeError("default cloud evolution expects FlatSkill")
        payload, response = await self.client.complete_json(
            [
                Message("system", BASE_EVOLUTION_SYSTEM_PROMPT),
                Message(
                    "user",
                    base_cloud_prompt(
                        framework_name,
                        skill.guidance(),
                        evidence.to_prompt(),
                        trigger_reasons,
                    ),
                ),
            ]
        )
        candidate = FlatSkill.from_dict(skill.to_dict())
        instruction = str(payload.get("instruction", "")).strip()
        if instruction and instruction != candidate.instruction:
            candidate.instruction = instruction
            candidate.revision += 1
        return EvolutionDraft(
            candidate=candidate,
            summary=str(payload.get("summary", "Flat instruction reviewed")),
            usage=response.usage,
            details={"mode": "raw-trajectory-flat-rewrite"},
        )

    def validate_evolution(
        self,
        current: SkillArtifact,
        draft: EvolutionDraft,
    ) -> ValidationResult:
        candidate = draft.candidate
        if candidate.task_type != current.task_type:
            return ValidationResult(False, "candidate changed task identity")
        if candidate.revision < current.revision:
            return ValidationResult(False, "candidate revision moved backwards")
        if not candidate.goal.strip() or not candidate.guidance().strip():
            return ValidationResult(False, "candidate contains an empty goal or skill")
        return ValidationResult(True, "basic structural checks passed")


__all__ = ["BaseCloudRuntime", "FlatSkill", "RawTrajectoryEvidence"]
