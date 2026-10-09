# Co-Skill Evolution Pipeline

Reference implementation of the collaborative skill-evolution pipeline from
**Co-Skill: A Collaborative Communication Framework for Skill Evolution**.

## Pipeline

```text
Task + current skill
        |
        v
Device LLM selects an environment action
        |
        v
Environment returns the next observation
        |
        v
Trajectory is stored and merged into a prefix trie
        |
        v
Cloud LLM diagnoses failures and updates the skill tree
        |
        v
Updated skill is persisted for later tasks
```

## Package layout

The runtime is organized around the edge/cloud boundary:

```text
src/coskill/
├── edge/          # BaseEdge and CoSkill agent/trigger specialization
├── cloud/         # BaseCloud and CoSkill trie/tree specialization
├── pipeline/      # base loop and concrete CoSkill pipeline assembly
├── utils/         # shared data models and run-artifact recording
├── skills/        # progressive skill tree and persistent skill library
├── llm/           # provider-independent LLM client interfaces
└── integrations/  # optional environment adapters such as ALFWorld
```

## General framework and Co-Skill instantiation

`SkillEvolutionFramework` defines the shared control loop through six
replaceable operators: skill registry, edge execution, collaboration
scheduling, evidence compilation, cloud evolution, and evolution validation.
`NaiveSkillEvolutionFramework` is a runnable baseline that uploads raw
trajectories and rewrites a single flat instruction.

The elementary edge operators live in `edge/base.py`, while the elementary
cloud operators and baseline data contracts live in `cloud/base.py`.
`pipeline/base.py` only coordinates these two runtimes and handles concurrent
evolution rounds and artifact recording.

Side-specific helpers live under `edge/utils/` and `cloud/utils/`. Prompt
constants and builders are defined in each side's `utils/prompt.py`. Co-Skill's
trajectory trie and progressive evolver are consolidated in `cloud/coskill.py`.
Its robust action executor and adaptive trajectory trigger are consolidated in
`edge/coskill.py`.

`CoSkill` composes that framework with `CoSkillCloud`, which improves selected
operators through cross-task skill retrieval, prefix-trie trajectory
compression, and evidence-grounded progressive-tree patches. New algorithms
can inject another edge/cloud runtime without changing the control loop.

### Parameter ownership

Global run controls are assembled in `cli.py` and passed directly to a
pipeline: maximum episode steps, whether evolution is enabled, and artifact
output location. Method-specific hyperparameters live beside their concrete
pipeline. `pipeline/coskill.py` owns action retries, upload-trigger thresholds,
trie rendering limits, and the maximum number of cloud evolution operations.
This keeps benchmark/runtime settings separate from algorithm settings.

Use the layer-specific imports when extending the implementation, for example
`from coskill.edge import EdgeAgent` and
`from coskill.cloud import CloudSkillEvolver`. The package-level public API
(`from coskill import ...`) remains available for applications.

## Run on ALFWorld

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[alfworld]'
alfworld-download
```

Configure OpenAI-compatible device and cloud endpoints:

```bash
export COSKILL_DEVICE_BASE_URL=http://localhost:8123/v1
export COSKILL_DEVICE_API_KEY=EMPTY
export COSKILL_DEVICE_MODEL=Qwen/Qwen2.5-3B-Instruct

export COSKILL_CLOUD_BASE_URL=https://api.openai.com/v1
export COSKILL_CLOUD_API_KEY=...
export COSKILL_CLOUD_MODEL=...
```

Evolve skills on the training split:

```bash
coskill alfworld --split train --episodes 100 \
  --db runs/alfworld-skills.db \
  --output runs/train.jsonl \
  --artifacts-dir runs/train-artifacts
```

When `--artifacts-dir` is set, artifacts are flushed after every episode:

- `trajectories.jsonl`: complete trajectories, including states, actions, and observations.
- `skill_trees.jsonl`: chronological skill-tree snapshots.
- `skills/<task_type>.json`: the latest readable tree for each task type.

Evaluate the persisted skills without further evolution:

```bash
coskill alfworld --split eval_out_of_distribution --episodes 100 \
  --db runs/alfworld-skills.db \
  --output runs/eval.jsonl \
  --no-evolution
```

For the repeated Look-task experiment, use 20 training trajectories per round
and update the skill after every 20 trajectories. The default is 5 rounds; the
same skill database is reused across rounds:

```bash
scripts/remote_look_sweep.sh PROJECT_DIR ALFWORLD_DATA 5 look_train_sweep 20 18 CONFIG_PATH
```

Each round therefore performs one fixed-size skill update, then evaluates the
updated skill on the 18-task `valid_unseen` Look test split. Test results are
written to `runs/<run_name>_eval_round<N>.jsonl` and summarized as
`TEST_SUMMARY` in the main log. The evaluation uses a copied database so test
usage statistics do not affect the next training round.

## Test

```bash
python -m unittest discover -s tests -v
```
