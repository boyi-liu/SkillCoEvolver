"""Command-line interface for launching Co-Skill benchmark runs."""

from __future__ import annotations

import argparse
import asyncio

from .core import CoSkill, CoSkillConfig
from .evolution import TriggerConfig
from .llm import OpenAICompatibleClient
from .skills import SkillLibrary


async def _alfworld(args: argparse.Namespace) -> None:
    """Configure the ALFWorld adapter and run the requested episodes."""
    from .integrations.alfworld import ALFWorldEnvironment, run_alfworld

    device = OpenAICompatibleClient.from_env("COSKILL_DEVICE")
    cloud = OpenAICompatibleClient.from_env("COSKILL_CLOUD")
    environment = ALFWorldEnvironment.from_config(
        args.config,
        split=args.split,
        max_steps=args.max_steps,
        seed=args.seed,
        data_dir=args.data_dir,
    )
    try:
        with SkillLibrary(args.db) as library:
            system = CoSkill(
                device_client=device,
                cloud_client=cloud,
                library=library,
                config=CoSkillConfig(
                    max_steps=args.max_steps,
                    enable_evolution=not args.no_evolution,
                    artifact_dir=args.artifacts_dir,
                    trigger=(
                        TriggerConfig(fixed_batch_size=args.update_every)
                        if args.update_every is not None
                        else TriggerConfig()
                    ),
                ),
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
    alfworld.add_argument(
        "--update-every",
        type=int,
        default=None,
        help="trigger a skill update after this many new trajectories per task type",
    )
    alfworld.add_argument(
        "--no-evolution",
        action="store_true",
        help="evaluate using stored skills without cloud updates",
    )
    alfworld.add_argument(
        "--flush-at-end",
        action="store_true",
        help="evolve from remaining trajectories even when no automatic trigger fired",
    )
    args = parser.parse_args()
    if args.command == "alfworld":
        asyncio.run(_alfworld(args))


if __name__ == "__main__":
    main()
