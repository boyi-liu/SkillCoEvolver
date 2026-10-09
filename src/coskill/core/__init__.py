"""End-to-end orchestration for the edge/cloud Co-Skill pipeline."""

from .orchestrator import CoSkill, CoSkillConfig

__all__ = [
    "CoSkill",
    "CoSkillConfig",
]
