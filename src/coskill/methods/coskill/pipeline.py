"""Compose Co-Skill's edge/cloud specializations into a concrete pipeline."""

from __future__ import annotations

from pathlib import Path

from ...framework import BaseSkillEvolutionFramework
from ...llm import LLMClient
from ...utils import Task
from .cloud import CoSkillCloud
from .config import CoSkillConfig
from .edge import CoSkillEdge
from .library import SkillLibrary
from .tree import ProgressiveSkillTree


class CoSkill(BaseSkillEvolutionFramework):
    """Trie-compressed, progressive-tree instantiation of the framework."""

    def __init__(
        self,
        *,
        device_client: LLMClient,
        cloud_client: LLMClient,
        library: SkillLibrary,
        config: CoSkillConfig | None = None,
        max_steps: int | None = None,
        enable_evolution: bool | None = None,
        artifact_dir: str | Path | None = None,
    ) -> None:
        effective_config = config or CoSkillConfig()
        resolved_max_steps = (
            max_steps
            if max_steps is not None
            else effective_config.max_steps or 40
        )
        resolved_enable_evolution = (
            enable_evolution
            if enable_evolution is not None
            else (
                effective_config.enable_evolution
                if effective_config.enable_evolution is not None
                else True
            )
        )
        resolved_artifact_dir = (
            artifact_dir
            if artifact_dir is not None
            else effective_config.artifact_dir
        )
        cloud_runtime = CoSkillCloud(
            cloud_client,
            library,
            observation_limit=effective_config.trie_observation_limit,
            max_operations=effective_config.max_evolution_operations,
        )
        edge_runtime = CoSkillEdge(
            device_client,
            max_steps=resolved_max_steps,
            action_retries=effective_config.action_retries,
            trigger=effective_config.trigger,
        )
        super().__init__(
            config=effective_config,
            edge_runtime=edge_runtime,
            cloud_runtime=cloud_runtime,
            max_steps=resolved_max_steps,
            enable_evolution=resolved_enable_evolution,
            artifact_dir=resolved_artifact_dir,
        )
        self.library = library
        self.cloud = cloud_runtime.evolver

    @property
    def framework_name(self) -> str:
        return "coskill-trie-progressive-tree"

    # Preserve the old public helper used by downstream callers.
    def skill_tree_for(self, task: Task) -> ProgressiveSkillTree:
        skill = self.retrieve_skill(task)
        if not isinstance(skill, ProgressiveSkillTree):
            raise TypeError("CoSkill retrieval must return ProgressiveSkillTree")
        return skill
