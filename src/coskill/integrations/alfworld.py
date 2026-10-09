"""Optional adapter for the official text-only ALFWorld environment."""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..pipeline import CoSkill
from ..edge import Environment, StepResult
from ..utils import EpisodeResult, Task


ALFWORLD_TASK_LABELS = {
    "pick_and_place_simple": "Pick",
    "look_at_obj_in_light": "Look",
    "pick_clean_then_place_in_recep": "Clean",
    "pick_heat_then_place_in_recep": "Heat",
    "pick_cool_then_place_in_recep": "Cool",
    "pick_two_obj_and_place": "Pick-2",
}

ALFWORLD_SKILL_TEMPLATES = {
    "pick_and_place_simple": "Place {target} in or on {destination}.",
    "look_at_obj_in_light": "Look at {target} under {light_source}.",
    "pick_clean_then_place_in_recep": (
        "Clean {target}, then place it in or on {destination}."
    ),
    "pick_heat_then_place_in_recep": (
        "Heat {target}, then place it in or on {destination}."
    ),
    "pick_cool_then_place_in_recep": (
        "Cool {target}, then place it in or on {destination}."
    ),
    "pick_two_obj_and_place": (
        "Place two instances of {target} in or on {destination}."
    ),
}

SPLITS = {"train", "eval_in_distribution", "eval_out_of_distribution"}


class ALFWorldSetupError(RuntimeError):
    pass


def default_config() -> dict[str, Any]:
    """Minimal official-compatible config for text-only ALFWorld."""
    return {
        "dataset": {
            "data_path": "$ALFWORLD_DATA/json_2.1.1/train",
            "eval_id_data_path": "$ALFWORLD_DATA/json_2.1.1/valid_seen",
            "eval_ood_data_path": "$ALFWORLD_DATA/json_2.1.1/valid_unseen",
            "num_train_games": -1,
            "num_eval_games": -1,
        },
        "logic": {
            "domain": "$ALFWORLD_DATA/logic/alfred.pddl",
            "grammar": "$ALFWORLD_DATA/logic/alfred.twl2",
        },
        "env": {
            "type": "AlfredTWEnv",
            "domain_randomization": False,
            "task_types": [1, 2, 3, 4, 5, 6],
            "expert_timeout_steps": 150,
            "expert_type": "handcoded",
            "goal_desc_human_anns_prob": 0.0,
        },
        "general": {
            "random_seed": 42,
            "use_cuda": False,
            "training_method": "dagger",
        },
        "dagger": {"training": {"max_nb_steps_per_episode": 40}},
        "rl": {"training": {"max_nb_steps_per_episode": 40}},
    }


def load_config(path: str | Path | None, *, max_steps: int) -> dict[str, Any]:
    """Load an ALFWorld config and apply the requested episode step limit."""
    if path is None:
        config = default_config()
    else:
        try:
            import yaml
        except ImportError as exc:
            raise ALFWorldSetupError(
                "PyYAML is required for a custom ALFWorld config; install "
                "with `pip install -e '.[alfworld]'`."
            ) from exc
        with Path(path).expanduser().open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, dict):
            raise ALFWorldSetupError(f"Invalid ALFWorld config: {path}")

    config.setdefault("general", {})["training_method"] = "dagger"
    config.setdefault("dagger", {}).setdefault("training", {})[
        "max_nb_steps_per_episode"
    ] = max_steps
    config.setdefault("rl", {}).setdefault("training", {})[
        "max_nb_steps_per_episode"
    ] = max_steps
    return config


def _first(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (str, bytes, dict)):
        return value
    try:
        return value[0]
    except (IndexError, KeyError, TypeError):
        return default


def _scalar_bool(value: Any, default: bool = False) -> bool:
    item = _first(value, value)
    try:
        return bool(item)
    except (TypeError, ValueError):
        return default


def _scalar_float(value: Any, default: float = 0.0) -> float:
    item = _first(value, value)
    try:
        return float(item)
    except (TypeError, ValueError):
        return default


def _goal_from_observation(observation: str) -> str | None:
    match = re.search(
        r"(?:your\s+task\s+is\s+to|task)\s*:\s*([^\n]+)",
        observation,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def _trajectory_metadata(gamefile: str) -> dict[str, Any]:
    if not gamefile:
        return {}
    path = Path(gamefile).expanduser()
    metadata_path = path.parent / "traj_data.json"
    if not metadata_path.exists():
        return {}
    try:
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _task_type(gamefile: str, metadata: dict[str, Any]) -> str:
    explicit = str(metadata.get("task_type", ""))
    if explicit in ALFWORLD_TASK_LABELS:
        return explicit
    normalized_path = gamefile.replace("\\", "/").casefold()
    for task_type in ALFWORLD_TASK_LABELS:
        if task_type.casefold() in normalized_path:
            return task_type
    return "alfworld_unknown"


def _goal_from_metadata(metadata: dict[str, Any]) -> str | None:
    annotations = metadata.get("turk_annotations", {}).get("anns", [])
    if annotations and isinstance(annotations[0], dict):
        goal = annotations[0].get("task_desc")
        if goal:
            return str(goal).strip()
    return None


class ALFWorldEnvironment(Environment):
    """Batch-size-one adapter for the official AlfredTWEnv API."""

    def __init__(
        self,
        raw_env: Any,
        *,
        split: str = "train",
        max_steps: int = 40,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"Unsupported ALFWorld split: {split}")
        self.raw_env = raw_env
        self.split = split
        self.max_steps = max_steps
        self._prepared_task: Task | None = None
        self._raw_observation = ""
        self._admissible_actions: list[str] = []
        self._steps = 0
        self._done = False

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
        *,
        split: str = "train",
        max_steps: int = 40,
        seed: int = 42,
        data_dir: str | Path | None = None,
    ) -> "ALFWorldEnvironment":
        """Build a batch-size-one adapter from ALFWorld configuration."""
        if split not in SPLITS:
            raise ValueError(f"Unsupported ALFWorld split: {split}")
        if data_dir is not None:
            os.environ["ALFWORLD_DATA"] = str(Path(data_dir).expanduser().resolve())
        else:
            os.environ.setdefault(
                "ALFWORLD_DATA", str(Path("~/.cache/alfworld").expanduser())
            )
        config = load_config(config_path, max_steps=max_steps)
        config.setdefault("general", {})["random_seed"] = seed
        random.seed(seed)
        try:
            from alfworld.agents.environment import get_environment
        except ImportError as exc:
            raise ALFWorldSetupError(
                "ALFWorld is not installed. Run `pip install -e '.[alfworld]'`."
            ) from exc

        env_type = config.get("env", {}).get("type", "AlfredTWEnv")
        if env_type != "AlfredTWEnv":
            raise ALFWorldSetupError(
                "This initial integration supports text-only AlfredTWEnv; "
                f"got {env_type!r}."
            )
        data_path_key = {
            "train": "data_path",
            "eval_in_distribution": "eval_id_data_path",
            "eval_out_of_distribution": "eval_ood_data_path",
        }[split]
        data_path = Path(
            os.path.expandvars(str(config["dataset"][data_path_key]))
        ).expanduser()
        if not data_path.exists():
            raise ALFWorldSetupError(
                f"ALFWorld data not found at {data_path}. Run `alfworld-download` "
                "or pass --data-dir."
            )
        factory = get_environment(env_type)
        raw_env = factory(config, train_eval=split).init_env(batch_size=1)
        return cls(
            raw_env,
            split=split,
            max_steps=max_steps,
        )

    def prepare_episode(self) -> Task:
        """Reset ALFWorld and convert its metadata into a Co-Skill task."""
        observations, infos = self.raw_env.reset()
        self._raw_observation = str(_first(observations, ""))
        self._admissible_actions = self._extract_actions(infos)
        self._steps = 0
        self._done = False
        gamefile = str(
            _first(infos.get("extra.gamefile", infos.get("gamefile", "")), "")
        )
        metadata = _trajectory_metadata(gamefile)
        task_type = _task_type(gamefile, metadata)
        goal = (
            _goal_from_observation(self._raw_observation)
            or _goal_from_metadata(metadata)
            or "Complete the current ALFWorld task."
        )
        self._prepared_task = Task(
            task_type=task_type,
            goal=goal,
            task_id=gamefile or uuid4().hex,
            metadata={
                "benchmark": "ALFWorld",
                "split": self.split,
                "gamefile": gamefile,
                "task_label": ALFWORLD_TASK_LABELS.get(task_type, "Unknown"),
            },
            skill_template=ALFWORLD_SKILL_TEMPLATES.get(task_type),
        )
        return self._prepared_task

    async def reset(self, task: Task) -> str:
        if self._prepared_task is None:
            raise RuntimeError("Call prepare_episode() before CoSkill.run_episode().")
        if task.task_id != self._prepared_task.task_id:
            raise ValueError("Task does not match the prepared ALFWorld episode")
        return self._render_observation()

    async def step(self, action: str) -> StepResult:
        """Execute one admissible ALFWorld action and normalize its result."""
        if self._prepared_task is None or self._done:
            raise RuntimeError("No active ALFWorld episode")
        action = action.strip()
        if action not in self._admissible_actions:
            return StepResult(
                observation=self._render_observation(
                    prefix=f"Invalid action {action!r}; choose an admissible action."
                ),
                info={"invalid_action": True, "admissible_actions": self._admissible_actions},
            )
        observations, scores, dones, infos = self.raw_env.step([action])
        self._steps += 1
        self._raw_observation = str(_first(observations, ""))
        self._admissible_actions = self._extract_actions(infos)
        raw_done = _scalar_bool(dones)
        won_value = infos.get("won")
        success = (
            _scalar_bool(won_value)
            if won_value is not None
            else raw_done and _scalar_float(scores) > 0
        )
        self._done = raw_done or self._steps >= self.max_steps
        return StepResult(
            observation=self._render_observation(),
            reward=_scalar_float(scores),
            done=self._done,
            success=success,
            info={
                "won": success,
                "steps": self._steps,
                "admissible_actions": list(self._admissible_actions),
            },
        )

    def state_key(self, observation: str) -> str:
        # The action menu is prompt context, not part of the environment state.
        return self._raw_observation

    def admissible_actions(self) -> list[str]:
        return list(self._admissible_actions)

    def close(self) -> None:
        close = getattr(self.raw_env, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _extract_actions(infos: dict[str, Any]) -> list[str]:
        actions = _first(infos.get("admissible_commands"), [])
        if actions is None:
            return []
        return [str(action) for action in actions]

    def _render_observation(self, prefix: str = "") -> str:
        return "\n\n".join(part for part in (prefix, self._raw_observation) if part)


@dataclass(slots=True)
class ALFWorldEpisodeRecord:
    episode: int
    task_id: str
    task_type: str
    task_label: str
    goal: str
    success: bool
    steps: int
    tokens: int
    evolution_triggered: bool
    evolution_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ALFWorldRunSummary:
    records: list[ALFWorldEpisodeRecord] = field(default_factory=list)

    @property
    def successes(self) -> int:
        return sum(record.success for record in self.records)

    @property
    def success_rate(self) -> float:
        return self.successes / len(self.records) if self.records else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(record.tokens for record in self.records)

    def success_rate_by_task(self) -> dict[str, float]:
        grouped: dict[str, list[bool]] = {}
        for record in self.records:
            grouped.setdefault(record.task_label, []).append(record.success)
        return {
            label: sum(results) / len(results) for label, results in sorted(grouped.items())
        }


async def run_alfworld(
    system: CoSkill,
    environment: ALFWorldEnvironment,
    *,
    episodes: int,
    output_jsonl: str | Path | None = None,
) -> ALFWorldRunSummary:
    """Run a batch of ALFWorld episodes and optionally append JSONL records."""
    summary = ALFWorldRunSummary()
    output_path = Path(output_jsonl).expanduser() if output_jsonl else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    for index in range(1, episodes + 1):
        task = environment.prepare_episode()
        result: EpisodeResult = await system.run_episode(
            task, environment, episode_index=index
        )
        record = ALFWorldEpisodeRecord(
            episode=index,
            task_id=task.task_id,
            task_type=task.task_type,
            task_label=str(task.metadata.get("task_label", "Unknown")),
            goal=task.goal,
            success=result.trajectory.success,
            steps=len(result.trajectory.steps),
            tokens=result.token_usage.total_tokens,
            evolution_triggered=result.evolution_triggered,
            evolution_summary=result.evolution_summary,
        )
        summary.records.append(record)
        if output_path:
            with output_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return summary
