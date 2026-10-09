"""Command-line interface for launching Co-Skill benchmark runs."""

from __future__ import annotations

import argparse
import asyncio
import random
from dataclasses import dataclass

from .framework import TriggerConfig
from .llm import OpenAICompatibleClient
from .methods.coskill import CoSkill, CoSkillConfig, SkillLibrary


@dataclass(frozen=True, slots=True)
class GlobalRunConfig:
    """Method-agnostic controls shared by benchmark pipelines."""

    max_steps: int
    enable_evolution: bool
    artifact_dir: str | None


def _coskill_config(args: argparse.Namespace) -> CoSkillConfig:
    """Build method-specific configuration shared by benchmark commands."""

    return CoSkillConfig(
        action_retries=args.action_retries,
        trigger=TriggerConfig(
            recent_window=args.trigger_window,
            failure_rate_threshold=args.failure_rate_threshold,
            fixed_batch_size=args.update_every,
        ),
        trie_observation_limit=args.trie_observation_limit,
        max_evolution_operations=args.max_evolution_operations,
    )


async def _alfworld(args: argparse.Namespace) -> None:
    """Configure the ALFWorld adapter and run the requested episodes."""
    from .integrations.alfworld import ALFWorldEnvironment, run_alfworld

    device = OpenAICompatibleClient.from_env("COSKILL_DEVICE")
    cloud = OpenAICompatibleClient.from_env("COSKILL_CLOUD")
    global_config = GlobalRunConfig(
        max_steps=args.max_steps,
        enable_evolution=not args.no_evolution,
        artifact_dir=args.artifacts_dir,
    )
    environment = ALFWorldEnvironment.from_config(
        args.config,
        split=args.split,
        max_steps=global_config.max_steps,
        seed=args.seed,
        data_dir=args.data_dir,
    )
    try:
        with SkillLibrary(args.db) as library:
            system = CoSkill(
                device_client=device,
                cloud_client=cloud,
                library=library,
                config=_coskill_config(args),
                max_steps=global_config.max_steps,
                enable_evolution=global_config.enable_evolution,
                artifact_dir=global_config.artifact_dir,
            )
            summary = await run_alfworld(
                system,
                environment,
                episodes=args.episodes,
                output_jsonl=args.output,
            )
            if args.flush_at_end and not args.no_evolution:
                for task_type in sorted({record.task_type for record in summary.records}):
                    message = await system.flush(
                        task_type, "end-of-run ALFWorld trajectory flush"
                    )
                    print(f"flush {task_type}: {message}")
            for record in summary.records:
                marker = "success" if record.success else "failure"
                evolved = " +evolved" if record.evolution_triggered else ""
                print(
                    f"episode {record.episode}: {record.task_label} {marker}, "
                    f"steps={record.steps}, tokens={record.tokens}{evolved}"
                )
            print(
                f"\nALFWorld: {summary.successes}/{len(summary.records)} success "
                f"({summary.success_rate:.1%}), tokens={summary.total_tokens}"
            )
            for label, rate in summary.success_rate_by_task().items():
                print(f"  {label}: {rate:.1%}")
    finally:
        environment.close()


async def _searchqa(args: argparse.Namespace) -> None:
    """Load local SearchQA data and run one-step question-answer episodes."""

    from .integrations.searchqa import (
        SEARCHQA_TASK_TYPE,
        SearchQAEnvironment,
        load_searchqa,
        run_searchqa,
    )

    device = OpenAICompatibleClient.from_env("COSKILL_DEVICE")
    cloud = OpenAICompatibleClient.from_env("COSKILL_CLOUD")
    examples = load_searchqa(args.data, split=args.split)
    if args.shuffle:
        random.Random(args.seed).shuffle(examples)
    global_config = GlobalRunConfig(
        max_steps=1,
        enable_evolution=not args.no_evolution,
        artifact_dir=args.artifacts_dir,
    )
    environment = SearchQAEnvironment(
        examples,
        split=args.split,
        max_snippets=args.max_snippets,
        max_context_chars=args.max_context_chars,
    )
    with SkillLibrary(args.db) as library:
        system = CoSkill(
            device_client=device,
            cloud_client=cloud,
            library=library,
            config=_coskill_config(args),
            max_steps=global_config.max_steps,
            enable_evolution=global_config.enable_evolution,
            artifact_dir=global_config.artifact_dir,
        )
        summary = await run_searchqa(
            system,
            environment,
            episodes=args.episodes,
            output_jsonl=args.output,
        )
        if args.flush_at_end and not args.no_evolution:
            message = await system.flush(
                SEARCHQA_TASK_TYPE, "end-of-run SearchQA trajectory flush"
            )
            print(f"flush {SEARCHQA_TASK_TYPE}: {message}")
        for record in summary.records:
            evolved = " +evolved" if record.evolution_triggered else ""
            print(
                f"episode {record.episode}: EM={record.exact_match:.0%}, "
                f"F1={record.f1:.1%}, tokens={record.tokens}{evolved}"
            )
        print(
            f"\nSearchQA: EM={summary.exact_match:.1%}, "
            f"F1={summary.f1:.1%}, tokens={summary.total_tokens}"
        )


def _add_coskill_arguments(parser: argparse.ArgumentParser) -> None:
    """Add Co-Skill-specific hyperparameters to a benchmark command."""

    parser.add_argument(
        "--action-retries",
        type=int,
        default=2,
        help="Co-Skill retries after an invalid device action",
    )
    parser.add_argument(
        "--trigger-window",
        type=int,
        default=16,
        help="Co-Skill recent window used by the failure-rate trigger",
    )
    parser.add_argument(
        "--failure-rate-threshold",
        type=float,
        default=0.60,
        help="Co-Skill failure rate that triggers cloud evolution",
    )
    parser.add_argument(
        "--update-every",
        type=int,
        default=None,
        help="trigger a skill update after this many new trajectories per task type",
    )
    parser.add_argument(
        "--trie-observation-limit",
        type=int,
        default=300,
        help="maximum observation characters rendered per trie node",
    )
    parser.add_argument(
        "--max-evolution-operations",
        type=int,
        choices=range(1, 4),
        default=3,
        help="maximum evidence-grounded tree patches per cloud update",
    )


def _add_evolution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-evolution",
        action="store_true",
        help="evaluate using stored skills without cloud updates",
    )
    parser.add_argument(
        "--flush-at-end",
        action="store_true",
        help="evolve from remaining trajectories even when no automatic trigger fired",
    )


def main() -> None:
    """Parse command-line arguments and dispatch to the selected command."""
    parser = argparse.ArgumentParser(description="Run the Co-Skill reference prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)
    alfworld = subparsers.add_parser(
        "alfworld", help="run Co-Skill on the official text-only ALFWorld environment"
    )
    alfworld.add_argument("--episodes", type=int, default=100)
    alfworld.add_argument(
        "--split",
        choices=["train", "eval_in_distribution", "eval_out_of_distribution"],
        default="train",
    )
    alfworld.add_argument("--config", help="optional ALFWorld YAML config")
    alfworld.add_argument("--data-dir", help="ALFWorld data directory")
    alfworld.add_argument("--db", default="coskill.db")
    alfworld.add_argument(
        "--artifacts-dir",
        help=(
            "save trajectories.jsonl and skill-tree snapshots under this directory"
        ),
    )
    alfworld.add_argument("--output", help="write per-episode JSONL records")
    alfworld.add_argument("--max-steps", type=int, default=40)
    alfworld.add_argument("--seed", type=int, default=42)
    _add_coskill_arguments(alfworld)
    _add_evolution_arguments(alfworld)

    searchqa = subparsers.add_parser(
        "searchqa", help="run Co-Skill on a local SearchQA JSON or JSONL dataset"
    )
    searchqa.add_argument("--data", required=True, help="SearchQA JSON or JSONL file")
    searchqa.add_argument(
        "--split",
        default="train",
        help="split key for JSON files containing train/validation/test mappings",
    )
    searchqa.add_argument("--episodes", type=int, default=100)
    searchqa.add_argument("--db", default="coskill.db")
    searchqa.add_argument("--artifacts-dir", help="save trajectory and skill artifacts")
    searchqa.add_argument("--output", help="write per-example JSONL records")
    searchqa.add_argument("--seed", type=int, default=42)
    searchqa.add_argument("--shuffle", action="store_true")
    searchqa.add_argument(
        "--max-snippets",
        type=int,
        default=50,
        help="maximum retrieved snippets included in each episode",
    )
    searchqa.add_argument(
        "--max-context-chars",
        type=int,
        default=16000,
        help="maximum total snippet characters included in each episode",
    )
    _add_coskill_arguments(searchqa)
    _add_evolution_arguments(searchqa)
    args = parser.parse_args()
    if args.command == "alfworld":
        asyncio.run(_alfworld(args))
    elif args.command == "searchqa":
        asyncio.run(_searchqa(args))


if __name__ == "__main__":
    main()
