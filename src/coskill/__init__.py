"""Co-Skill: structured collaborative skill evolution."""

from .core import (
    CoSkill,
    CoSkillConfig,
    EdgeAgent,
    Environment,
    EpisodeResult,
    RunArtifactRecorder,
    StepResult,
    Task,
    TokenUsage,
    Trajectory,
    TrajectoryStep,
)
from .evolution import TrajectoryPool, TrajectoryTrie, TriggerConfig
from .llm import LLMClient, OpenAICompatibleClient
from .skills import ProgressiveSkillTree, SkillLibrary, SkillNode

__all__ = [
    "CoSkill",
    "CoSkillConfig",
    "EdgeAgent",
    "Environment",
    "EpisodeResult",
    "RunArtifactRecorder",
    "LLMClient",
    "OpenAICompatibleClient",
    "ProgressiveSkillTree",
    "SkillLibrary",
    "SkillNode",
    "StepResult",
    "Task",
    "TokenUsage",
    "Trajectory",
    "TrajectoryPool",
    "TrajectoryStep",
    "TrajectoryTrie",
    "TriggerConfig",
]
