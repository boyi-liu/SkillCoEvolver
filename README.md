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
├── edge/          # device agent, action prompts, trajectory buffering/triggers
├── cloud/         # trajectory trie, cloud prompts, and skill evolution
├── core/          # end-to-end orchestration across edge and cloud
├── utils/         # shared data models and run-artifact recording
├── skills/        # progressive skill tree and persistent skill library
├── llm/           # provider-independent LLM client interfaces
└── integrations/  # optional environment adapters such as ALFWorld
```

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
scripts/remote_look_sweep.sh PROJECT_DIR ALFWORLD_DATA 5 look_train_sweep 20
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
