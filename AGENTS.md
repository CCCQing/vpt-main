# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains core code: `configs/` (CfgNode defaults), `data/` (loaders/transforms/datasets), `models/`, `engine/` (trainer/evaluator), `solver/`, and `utils/`.
- `configs/` stores experiment YAMLs (`prompt/`, `finetune/`, `linear/`, `TEST/`).
- Root entry points: `train.py` (main train/eval) and `launch.py`.
- Diagnostics/utilities live in `src/tools/`.
- Documentation is in `docs/`; generated runs and logs are typically under `output/`.

## Build, Test, and Development Commands
- Set up environment (Conda-based): `bash env_setup.sh` (or mirror the package list manually on Windows).
- Train with a config: `python train.py --config-file configs/prompt/cub.yaml`.
- Override config keys from CLI: `python train.py --config-file configs/prompt/cub.yaml OUTPUT_DIR output/dev RUN_N_TIMES 1`.

## Coding Style & Naming Conventions
- Python style: 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes, constants in `UPPER_SNAKE_CASE`.
- Keep modules focused; place new training logic in `src/engine/` or `src/solver/`, not in root scripts.
- Config keys follow existing uppercase dotted style (example: `SOLVER.BASE_LR`, `MODEL.PROMPT.NUM_TOKENS`).
- Prefer small, explicit helpers over large monolithic functions.
- Code reading default: prioritize executable logic over comments. Unless the user explicitly asks for comments/docstrings, skip `# ...` comments and triple-quoted explanatory text during code inspection, and only consult them when runtime behavior or protocol intent is otherwise unclear.

## Testing Guidelines
- No formal `pytest` suite is configured; use script-based smoke/regression checks.
- Minimum before PR: run one short training job (`RUN_N_TIMES 1`, reduced epochs) and one diagnostic script when touching training/loss code.
- Keep reproducibility in mind: set `SEED` and record config overrides in logs.

## Commit & Pull Request Guidelines
- Recent history favors short, focused commit subjects (often concise Chinese phrases). Keep subject lines imperative and specific to one change.
- Suggested commit format: `<area>: <what changed>` (example: `solver: fix affinity monitor scaling`).
- PRs should include: purpose, key config/file changes, exact run command(s), and before/after metrics or log snippets.
- Link related issues/tasks and call out dataset/checkpoint assumptions explicitly.
