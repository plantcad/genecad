# Agent Guidelines

## General
- IMPORTANT: Ask clarifying questions liberally
- Prefer functional, declarative style over imperative OOP
- Add comments only where intent or implementation is not immediately clear

## Project Layout
- `local/` — untracked scratch space for plans, ad-hoc scripts, logs, and results
  - `local/scratch/` — ad-hoc scripts and their outputs (plots, CSVs, etc.)
  - `local/logs/` — execution logs
- `scripts/` — production CLIs for pipeline operations
- `scripts/misc/` — ad-hoc CLIs saved in source control

## Environment
- IMPORTANT: All local libraries and execution needs to use `uv`
- Assume local dev (typically MacOS) and execution is mostly remote in HPC or Neocloud
- Assume remote environment from context or ask for clarification based on skills (`.claude/skills`)

## CLI
- Use `argparse` for CLIs unless instructed otherwise
- Initialize logging in `main` with `logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')`
- See `scripts/extract.py` for an example CLI program
- Always tee CLI output: `python ... 2>&1 | tee local/logs/exec/<name>.log`

## Coding
- IMPORTANT: Highly prefer errors over fallbacks
- IMPORTANT: No local imports
- NumPy-style docstrings
- `numpy.typing` for array annotations
- Liberal type annotations using Python 3.10+ syntax (`list`, `dict`, `int | None`)
- Early returns over deep nesting
- Loggers over print statements

## Testing
- `pytest` with `pytest.mark.parametrize` over boilerplate
- Start small (one or two tests max), expand on request
- Absolute imports: `from src.module import ...`


## Naming
- No type suffixes on variables (use `attributes` not `attr_df`)
