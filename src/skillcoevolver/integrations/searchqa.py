"""SearchQA dataset adapter and benchmark runner.

The adapter deliberately keeps dataset loading local and dependency-free.  It
accepts the original SearchQA shape (``question``, ``answer``, and
``search_results``), Hugging Face JSON exports, and MRQA-style examples with a
single ``context`` and an ``answers`` list.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..framework import Environment, SkillEvolutionFramework, StepResult
from ..utils import EpisodeResult, Task


SEARCHQA_TASK_TYPE = "searchqa"
SEARCHQA_SKILL_TEMPLATE = (
    "Answer the question from the retrieved search snippets. Resolve conflicting "
    "or noisy snippets using repeated evidence, and return only the shortest answer "
    "that directly answers the question."
)


class SearchQADataError(ValueError):
    """Raised when a local SearchQA file does not have a supported shape."""


@dataclass(frozen=True, slots=True)
class SearchQAExample:
    """One question together with private gold answers and retrieved snippets."""

    example_id: str
    question: str
    answers: tuple[str, ...]
    snippets: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        default_id: str,
    ) -> "SearchQAExample":
        question = str(value.get("question", "")).strip()
        answers = _extract_answers(value)
        snippets = _extract_snippets(value)
        if not question:
            raise SearchQADataError(f"SearchQA example {default_id!r} has no question")
        if not answers:
            raise SearchQADataError(f"SearchQA example {default_id!r} has no answer")

        raw_id = value.get("id", value.get("key", default_id))
        example_id = str(raw_id).strip() or default_id
        reserved = {
            "id",
            "key",
            "question",
            "answer",
            "answers",
            "context",
            "snippets",
            "search_results",
        }
        metadata = {key: item for key, item in value.items() if key not in reserved}
        return cls(
            example_id=example_id,
            question=question,
            answers=tuple(answers),
            snippets=tuple(snippets),
            metadata=metadata,
        )


@dataclass(frozen=True, slots=True)
class SearchQAEvaluation:
    prediction: str
    exact_match: float
    f1: float


def normalize_answer(text: str) -> str:
    """Apply the conventional SQuAD/SearchQA answer normalization."""

    lowered = text.casefold()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def exact_match_score(prediction: str, answers: Sequence[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    return float(
        any(normalized_prediction == normalize_answer(answer) for answer in answers)
    )


def token_f1_score(prediction: str, answers: Sequence[str]) -> float:
    """Return the maximum token F1 over all accepted gold answers."""

    prediction_tokens = normalize_answer(prediction).split()
    scores: list[float] = []
    for answer in answers:
        answer_tokens = normalize_answer(answer).split()
        if not prediction_tokens or not answer_tokens:
            scores.append(float(prediction_tokens == answer_tokens))
            continue
        common = Counter(prediction_tokens) & Counter(answer_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            scores.append(0.0)
            continue
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(answer_tokens)
        scores.append(2 * precision * recall / (precision + recall))
    return max(scores, default=0.0)


def load_searchqa(
    path: str | Path,
    *,
    split: str | None = None,
) -> list[SearchQAExample]:
    """Load SearchQA examples from a JSON or JSONL file.

    A JSON file may contain a list, a single example, a ``data`` list, or a
    mapping of split names to lists.  Pass ``split`` for the latter form.
    """

    source = Path(path).expanduser()
    if not source.is_file():
        raise SearchQADataError(f"SearchQA data file not found: {source}")
    try:
        if source.suffix.casefold() in {".jsonl", ".ndjson"}:
            rows = [
                json.loads(line)
                for line in source.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            payload = json.loads(source.read_text(encoding="utf-8"))
            rows = _select_rows(payload, split=split)
    except (OSError, json.JSONDecodeError) as exc:
        raise SearchQADataError(
            f"Could not read SearchQA data from {source}: {exc}"
        ) from exc

    if not isinstance(rows, list):
        raise SearchQADataError("Selected SearchQA data must be a list of examples")
    examples: list[SearchQAExample] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise SearchQADataError(f"SearchQA row {index} is not an object")
        examples.append(
            SearchQAExample.from_mapping(row, default_id=f"{split or 'data'}-{index}")
        )
    if not examples:
        raise SearchQADataError("SearchQA data contains no examples")
    return examples


class SearchQAEnvironment(Environment):
    """One-step extractive-QA environment over SearchQA search snippets."""

    def __init__(
        self,
        examples: Iterable[SearchQAExample],
        *,
        split: str = "train",
        max_snippets: int = 50,
        max_context_chars: int = 16_000,
    ) -> None:
        if max_snippets <= 0:
            raise ValueError("max_snippets must be positive")
        if max_context_chars <= 0:
            raise ValueError("max_context_chars must be positive")
        self.examples = list(examples)
        if not self.examples:
            raise ValueError("SearchQAEnvironment requires at least one example")
        self.split = split
        self.max_snippets = max_snippets
        self.max_context_chars = max_context_chars
        self._next_index = 0
        self._current: SearchQAExample | None = None
        self._prepared_task: Task | None = None
        self._observation = ""
        self._evaluation: SearchQAEvaluation | None = None

    @property
    def remaining(self) -> int:
        return len(self.examples) - self._next_index

    @property
    def evaluation(self) -> SearchQAEvaluation | None:
        return self._evaluation

    @property
    def current_example(self) -> SearchQAExample | None:
        return self._current

    def prepare_episode(self) -> Task:
        if self._next_index >= len(self.examples):
            raise StopIteration("No SearchQA examples remain")
        example = self.examples[self._next_index]
        self._next_index += 1
        self._current = example
        self._evaluation = None
        self._observation = self._render_observation(example)
        public_metadata = {
            key: value
            for key, value in example.metadata.items()
            if key not in {"answer", "answers"}
        }
        self._prepared_task = Task(
            task_type=SEARCHQA_TASK_TYPE,
            goal=example.question,
            task_id=example.example_id,
            metadata={
                "benchmark": "SearchQA",
                "split": self.split,
                **public_metadata,
            },
            skill_template=SEARCHQA_SKILL_TEMPLATE,
        )
        return self._prepared_task

    async def reset(self, task: Task) -> str:
        if self._prepared_task is None:
            raise RuntimeError("Call prepare_episode() before run_episode().")
        if task.task_id != self._prepared_task.task_id:
            raise ValueError("Task does not match the prepared SearchQA example")
        return self._observation

    async def step(self, action: str) -> StepResult:
        if self._current is None or self._evaluation is not None:
            raise RuntimeError("No active SearchQA example")
        prediction = action.strip()
        evaluation = SearchQAEvaluation(
            prediction=prediction,
            exact_match=exact_match_score(prediction, self._current.answers),
            f1=token_f1_score(prediction, self._current.answers),
        )
        self._evaluation = evaluation
        observation = (
            "Answer accepted for evaluation."
            if evaluation.exact_match
            else "Answer did not exactly match a reference answer."
        )
        return StepResult(
            observation=observation,
            reward=evaluation.f1,
            done=True,
            success=bool(evaluation.exact_match),
            info={
                "exact_match": evaluation.exact_match,
                "f1": evaluation.f1,
            },
        )

    def _render_observation(self, example: SearchQAExample) -> str:
        snippets: list[str] = []
        used_chars = 0
        for snippet in example.snippets[: self.max_snippets]:
            remaining = self.max_context_chars - used_chars
            if remaining <= 0:
                break
            rendered = snippet[:remaining].strip()
            if rendered:
                snippets.append(rendered)
                used_chars += len(rendered)
        context = "\n".join(
            f"[{index}] {snippet}" for index, snippet in enumerate(snippets, start=1)
        )
        return f"Question:\n{example.question}\n\nRetrieved snippets:\n{context or '(none)'}"


@dataclass(slots=True)
class SearchQAEpisodeRecord:
    episode: int
    task_id: str
    question: str
    prediction: str
    answers: list[str]
    exact_match: float
    f1: float
    tokens: int
    evolution_triggered: bool
    evolution_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SearchQARunSummary:
    records: list[SearchQAEpisodeRecord] = field(default_factory=list)

    @property
    def exact_match(self) -> float:
        return (
            sum(record.exact_match for record in self.records) / len(self.records)
            if self.records
            else 0.0
        )

    @property
    def f1(self) -> float:
        return (
            sum(record.f1 for record in self.records) / len(self.records)
            if self.records
            else 0.0
        )

    @property
    def total_tokens(self) -> int:
        return sum(record.tokens for record in self.records)


async def run_searchqa(
    system: SkillEvolutionFramework,
    environment: SearchQAEnvironment,
    *,
    episodes: int,
    output_jsonl: str | Path | None = None,
) -> SearchQARunSummary:
    """Run SearchQA examples and optionally append per-example JSONL records."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if episodes > environment.remaining:
        raise ValueError(
            f"Requested {episodes} SearchQA episodes but only "
            f"{environment.remaining} remain"
        )
    summary = SearchQARunSummary()
    output_path = Path(output_jsonl).expanduser() if output_jsonl else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    for index in range(1, episodes + 1):
        task = environment.prepare_episode()
        example = environment.current_example
        if example is None:  # pragma: no cover - guarded by prepare_episode
            raise RuntimeError("SearchQA example preparation failed")
        result: EpisodeResult = await system.run_episode(
            task, environment, episode_index=index
        )
        evaluation = environment.evaluation
        if evaluation is None:  # pragma: no cover
            raise RuntimeError("SearchQA episode ended without an evaluation")
        record = SearchQAEpisodeRecord(
            episode=index,
            task_id=task.task_id,
            question=example.question,
            prediction=evaluation.prediction,
            answers=list(example.answers),
            exact_match=evaluation.exact_match,
            f1=evaluation.f1,
            tokens=result.token_usage.total_tokens,
            evolution_triggered=result.evolution_triggered,
            evolution_summary=result.evolution_summary,
        )
        summary.records.append(record)
        if output_path:
            with output_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return summary


def _select_rows(payload: Any, *, split: str | None) -> Any:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, Mapping):
        raise SearchQADataError("SearchQA JSON must contain an object or list")
    if split is not None and split in payload:
        return payload[split]
    if "data" in payload:
        return payload["data"]
    if "question" in payload:
        return [payload]
    available = ", ".join(sorted(str(key) for key in payload))
    if split is None:
        raise SearchQADataError(
            f"SearchQA JSON contains named splits ({available}); specify split"
        )
    raise SearchQADataError(
        f"SearchQA split {split!r} not found; available keys: {available}"
    )


def _extract_answers(value: Mapping[str, Any]) -> list[str]:
    raw = value.get("answers", value.get("answer"))
    if isinstance(raw, Mapping):
        raw = raw.get("text", raw.get("answers", []))
    if isinstance(raw, str):
        candidates = [raw]
    elif isinstance(raw, Sequence):
        candidates = list(raw)
    else:
        candidates = []
    return _unique_strings(candidates)


def _extract_snippets(value: Mapping[str, Any]) -> list[str]:
    if isinstance(value.get("context"), str):
        return _unique_strings([value["context"]])
    raw: Any = value.get("snippets", value.get("search_results", []))
    candidates: list[Any] = []
    if isinstance(raw, Mapping):
        candidates.extend(_as_items(raw.get("snippets", raw.get("snippet", []))))
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for result in raw:
            if isinstance(result, Mapping):
                candidates.extend(
                    _as_items(result.get("snippets", result.get("snippet", [])))
                )
            else:
                candidates.append(result)
    elif isinstance(raw, str):
        candidates.append(raw)
    return _unique_strings(candidates)


def _as_items(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _unique_strings(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip() if value is not None else ""
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


__all__ = [
    "SEARCHQA_SKILL_TEMPLATE",
    "SEARCHQA_TASK_TYPE",
    "SearchQADataError",
    "SearchQAEnvironment",
    "SearchQAEpisodeRecord",
    "SearchQAEvaluation",
    "SearchQAExample",
    "SearchQARunSummary",
    "exact_match_score",
    "load_searchqa",
    "normalize_answer",
    "run_searchqa",
    "token_f1_score",
]
