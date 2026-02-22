# Agent Guidelines

## General
- Plan thoroughly before implementing; ask for clarification on vague or ill-advised prompts
- Prefer functional, declarative style over imperative OOP
- Add comments only where intent or implementation is not immediately clear
- All behavioral directives are defaults that can be overridden by request

## Project Layout
- `local/` — untracked scratch space for plans, ad-hoc scripts, logs, and results
  - `local/scratch/` — ad-hoc scripts and their outputs (plots, CSVs, etc.)
  - `local/logs/exec/` — execution logs
- `scripts/` — production CLIs for pipeline operations
- `scripts/misc/` — analysis CLIs saved in source control

## Environment
- Run `source ~/.projectrc` before anything else; it sets the working directory, PYTHONPATH, and env vars
- Never `cd` to the repo manually; always use `source ~/.projectrc`
- Never add `sys.path.append(...)` to scripts
- When running Python locally: `source .venv/bin/activate && python ...`
- Pre-commit: install `pyrefly` with `uv pip install pyrefly` if missing

## Execution
- This project is developed on a local CPU-only workstation unless instructed otherwise
- All GPU or CPU-intensive work must be offloaded to the TACC HPC cluster (see `.claude/skills/tacc/SKILL.md`)

## CLI
- Use `argparse` for CLIs unless instructed otherwise
- Initialize logging in `main` with `logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')`
- See `scripts/extract.py` for an example CLI program
- Always tee CLI output: `python ... 2>&1 | tee local/logs/exec/<name>.log`

## Coding
- NumPy-style docstrings
- `numpy.typing` for array annotations
- Liberal type annotations using Python 3.10+ syntax (`list`, `dict`, `int | None`)
- Early returns over deep nesting
- Loggers over print statements

## Testing
- `pytest` with `pytest.mark.parametrize` over boilerplate
- Start small, expand on request
- Absolute imports: `from src.module import ...`

## Naming
- No type suffixes on variables (use `attributes` not `attr_df`)
