# SkillCoEvolver

<p align="center">
  <a href="README.md">🇬🇧 English</a>
  &nbsp;•&nbsp;
  <a href="README_zh.md">🇨🇳 中文</a>
</p>

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

The runtime separates reusable base behavior from concrete method
specializations:

```text
src/coskill/
├── framework/     # contracts, default runtimes, and the shared control loop
├── methods/
│   └── coskill/   # CoSkill edge, cloud, skills, prompts, and pipeline
├── utils/         # shared data models and run-artifact recording
├── llm/           # provider-independent LLM client interfaces
└── integrations/  # optional environment adapters such as ALFWorld
```

## General framework and Co-Skill instantiation

`framework/` defines the runtime contracts, default edge agent/runtime,
failure-aware trajectory scheduler, flat-skill cloud runtime, structural
validation, and shared control loop. Its default pipeline uploads raw
trajectories and rewrites a single flat instruction.

`methods/coskill/` specializes those framework bases with corrective action
retries, cross-task retrieval, trajectory tries, progressive skill trees,
persistent storage, and evidence-grounded tree patches. Method code depends
only on `framework/`, never on another method package.

`CoSkill` composes that framework with `CoSkillCloud`, which improves selected
operators through cross-task skill retrieval, prefix-trie trajectory
compression, and evidence-grounded progressive-tree patches. A new algorithm
can be added as another `methods/<name>/` package without changing the shared
control loop or scattering its implementation across layer directories.

### Parameter ownership

Global run controls are assembled in `cli.py` and passed directly to a
pipeline: maximum episode steps, whether evolution is enabled, and artifact
output location. Method-specific hyperparameters live beside their concrete
pipeline. `methods/coskill/config.py` owns action retries, upload-trigger
thresholds, trie rendering limits, and the maximum number of cloud evolution
operations. This keeps benchmark/runtime settings separate from algorithm
settings.

Use method-scoped imports when extending the implementation, for example
`from coskill.methods.coskill import EdgeAgent, CloudSkillEvolver`. The
package-level public API (`from coskill import ...`) remains available for
applications.

### Algorithm selection

Both benchmark commands accept `--alg`. Omitting it (or passing an empty
string) runs `BaseSkillEvolutionFramework`; passing `--alg coskill` runs the
CoSkill specialization. The `--skill-library` option is used only by CoSkill.

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
coskill alfworld --alg coskill --split train --episodes 100 \
  --skill-library runs/alfworld-skills.json \
  --output runs/train.jsonl \
  --artifacts-dir runs/train-artifacts
```

When `--artifacts-dir` is set, artifacts are flushed after every episode:

- `trajectories.jsonl`: complete trajectories, including states, actions, and observations.
- `skill_trees.jsonl`: chronological skill-tree snapshots.
- `skills/<task_type>.json`: the latest readable tree for each task type.
- `skills/<task_type>.md`: the latest generated skill rendered as Markdown.

Evaluate the persisted skills without further evolution:

```bash
coskill alfworld --alg coskill --split eval_out_of_distribution --episodes 100 \
  --skill-library runs/alfworld-skills.json \
  --output runs/eval.jsonl \
  --no-evolution
```

## Run on SearchQA

The SearchQA integration reads local JSON or JSONL without adding a dataset
runtime dependency. It accepts the original fields (`question`, `answer`, and
`search_results.snippets`), Hugging Face JSON exports, and MRQA-style
`context`/`answers` examples. A JSON file may be a list of examples or a mapping
of split names to example lists.

```bash
coskill searchqa --alg coskill --data data/searchqa.json \
  --split train \
  --episodes 100 \
  --skill-library runs/searchqa-skills.json \
  --output runs/searchqa-train.jsonl \
  --artifacts-dir runs/searchqa-artifacts
```

Each question is a one-step episode: the edge model receives the question and
up to 50 retrieved snippets, then returns its shortest final answer as the open
environment action. Runs report normalized exact match and token F1. Gold
answers are retained by the evaluator and are not included in model prompts or
skill-evolution trajectories. Use `--max-snippets` and `--max-context-chars` to
bound prompt context, and `--shuffle --seed 42` for deterministic sampling.

For the repeated Look-task experiment, use 20 training trajectories per round
and update the skill after every 20 trajectories. The default is 5 rounds; the
the same JSON skill library is reused across rounds:

```bash
scripts/remote_look_sweep.sh PROJECT_DIR ALFWORLD_DATA 5 look_train_sweep 20 18 CONFIG_PATH
```

Each round therefore performs one fixed-size skill update, then evaluates the
updated skill on the 18-task `valid_unseen` Look test split. Test results are
written to `runs/<run_name>_eval_round<N>.jsonl` and summarized as
`TEST_SUMMARY` in the main log. The evaluation uses a copied skill library so test
usage statistics do not affect the next training round.

To run all six ALFWorld task families sequentially in the background, use:

```bash
nohup bash scripts/remote_all_task_sweeps.sh \
  /home/liuboyi/CoSkillEvo-Code \
  /home/liuboyi/CoSkillEvo-Code/data/alfworld \
  5 20 alfworld_all_tasks \
  > /home/liuboyi/CoSkillEvo-Code/runs/alfworld_all_tasks_launcher.log 2>&1 &
```

The batch script creates independent DBs and logs for Pick, Look, Clean, Heat,
Cool, and Pick-2, and stops if any task family fails.
