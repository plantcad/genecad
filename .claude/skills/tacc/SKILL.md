---
name: tacc
description: Manages experiment execution on the TACC Vista HPC cluster. Use when the user says "run on TACC", "run the pipeline", "allocate a GPU node", "check SLURM queue", "sync code to TACC", "download results from TACC", "cancel that job", or when any task requires GPU/CPU compute that cannot run locally. Do NOT use for local-only tasks like editing code, running tests, or plotting.
allowed-tools: Bash
---

# TACC HPC Cluster (Vista)

This skill targets the **Vista** supercomputer within TACC (Texas Advanced Computing Center). Queue names, module versions, and filesystem layout are Vista-specific and may differ on other TACC systems.

## First-Time Setup

Before any TACC operations, verify the user's environment is ready. Run these checks and guide the user through fixing any failures:

### Step 1: Verify SSH access
```bash
ssh tacc "echo 'SSH OK'"
```
If this fails, the user needs to configure `tacc` as an SSH host in `~/.ssh/config`.

### Step 2: Install command wrapper
```bash
ssh tacc "test -x \$HOME/local/bin/genecad/cmd && echo 'cmd OK' || echo 'MISSING'"
```
If missing, install it from the local skill scripts:
```bash
ssh tacc "mkdir -p ~/local/bin/genecad"
rsync -Pz .claude/skills/tacc/scripts/cmd tacc:~/local/bin/genecad/cmd
ssh tacc "chmod +x ~/local/bin/genecad/cmd"
```
The `cmd` script sources `.bashrc`, sets `PYTHONPATH`, cds to the repo, and exports `RANK`/`WORLD_SIZE` from SLURM variables, then execs the given command. It is the single entry point for running commands on compute nodes.

### Step 3: Verify Python environment
```bash
ssh tacc "bash -l -c 'which python && python --version'"
```
Expected: Python 3.11+ from a venv at `$WORK/envs/ml-rel/bin/activate`, sourced by `.bashrc`. Do NOT use conda, mamba, or micromamba — they are not installed on Vista.

### Step 4: Verify repository exists on TACC
```bash
ssh tacc "bash -l -c 'cd \$WORK/repos/genecad && git status --short'"
```
If the repo doesn't exist, clone it: `ssh tacc "bash -l -c 'git clone <REPO_URL> \$WORK/repos/genecad'"`.

## Filesystem Model

- **`$WORK`** — Small, persistent filesystem. All code and repositories live here. Treat stored data as **read-only** unless explicitly instructed otherwise.
- **`$SCRATCH`** — Effectively unlimited capacity but subject to automatic garbage collection (data expires if not accessed frequently enough). **`$SCRATCH/tmp` is the default output location for ALL experiment results** unless instructed otherwise. Any other directories under `$SCRATCH` should be treated as read-only data sources, not output destinations.

## Login Node vs Compute Node

The `tacc` SSH alias connects to a **login node**. Use it freely for:
- File exploration (`ls`, `find`, `du`)
- Environment checks (`which python`, `env`)
- Data inspection and analysis (Python + pandas, matplotlib, parsing results, generating tables/plots)
- Running analysis scripts that don't need GPUs or heavy compute
- Git operations (`git status`, `git pull`)
- Job management (`squeue`, `scancel`, `idev`)

**Compute nodes cost SU credits.** Only allocate a compute node when:
- The task requires a GPU (model inference, predictions)
- The task requires significant CPU or memory (large-scale post-processing)

Do NOT allocate a compute node just to run `ls`, check paths, inspect files, or run quick scripts. Use `ssh tacc "bash -l -c 'COMMAND'"` for that.

## Experiment Workflow

A typical experiment follows this sequence. Each step depends on the prior step completing successfully.

### Step 1: Sync code to TACC
```bash
bash .claude/skills/tacc/scripts/sync          # rsync (default, fast, no commit needed)
bash .claude/skills/tacc/scripts/sync --git    # git push + pull (requires clean commit)
```
**rsync (default):** Directly pushes local files without requiring a git commit. Use for in-progress work.
**git (`--git`):** Use only when changes are committed and you want the remote to match a specific branch/commit.

### Step 2: Check for existing compute nodes
```bash
ssh tacc "squeue -u \$USER -o '%.18i %.9P %.30j %.2t %.10M %.6D %.20R'"
```
Reuse a running node if one exists — each session has a minimum 15-minute charge.

### Step 3: Allocate a compute node

**Queues:**

| Queue | Type | Time Limit | Use Case |
|-------|------|------------|----------|
| `gg` | CPU-only | — | CPU-only jobs (post-processing, evaluation) |
| `gh-dev` | GPU dev | 2 hours | Try first for GPU work |
| `gh` | GPU prod | — | Fallback if `gh-dev` has no nodes |

Do not allocate a CPU-only node (`gg`) for GPU work. The prediction step requires a GPU.

**Single-node (default):** Use `idev` to allocate an interactive node:
```bash
idev -p gh-dev -N 1 -n 1 -t 2:00:00
```
Then find the allocated node:
```bash
ssh tacc "squeue -u \$USER -h -t R -o '%N'"
```

**Multi-node:** Use `srun` directly — no `idev` needed (see Step 4).

### Step 4: Run the experiment

Write all output to `$SCRATCH/tmp` unless instructed otherwise.

#### Single-node execution

After allocating a node with `idev` (Step 3), find the node name and run commands via `cmd`. Since `cmd` cds to the repo automatically, commands can use repo-relative paths directly:
```bash
NODE=$(ssh tacc "squeue -u \$USER -h -t R -o '%N'" | head -1)
ssh tacc "ssh $NODE ~/local/bin/genecad/cmd python scripts/predict.py ..."
```
For long-running single-node jobs, prefer `sbatch` over `idev` to survive SSH disconnects.

#### Multi-node execution with `srun`

Use `srun` to launch the same command across multiple nodes simultaneously. The `cmd` wrapper sets `RANK=$PMIX_RANK` and `WORLD_SIZE=$SLURM_NNODES`, so each node knows its rank and the total node count.

```bash
ssh tacc "bash -l -c '\
  srun -p gh-dev -N 8 -n 8 --tasks-per-node 1 -t 2:00:00 \
    --output \$SCRATCH/tmp/logs/<name>.log \
    --error  \$SCRATCH/tmp/logs/<name>.log \
    ~/local/bin/genecad/cmd python scripts/predict.py create_predictions \
      --input ... --output-dir ...'" 2>&1 &
```

**Important notes for multi-node `srun`:**
- Always use `--tasks-per-node 1` — each node runs one instance of the command.
- Environment variables set *before* the `srun` call propagate to all nodes.
- All nodes write to the same `--output`/`--error` log file (interleaved). Use `[rank=N]` prefixes in log messages to distinguish nodes.
- `srun` blocks until all nodes finish. Run it in background (`&`) and monitor via `tail` on the log file.

#### Monitoring progress
```bash
ssh tacc "tail -20 \$SCRATCH/tmp/logs/<name>.log"
```

### Step 5: Download results
```bash
rsync -Pz tacc:/remote/results/path local/results/path
```
Always use `rsync` instead of `scp`.

### Step 6: Cancel the job
```bash
ssh tacc "scancel <jobid>"
```
**Always cancel jobs when done.** Idle jobs consume SU credits.

## Environment

- Python env: standard venv at `$WORK/envs/ml-rel/bin/activate`, sourced automatically by `.bashrc`
- Do NOT use conda, mamba, or micromamba — they are not installed
- Modules loaded by `.bashrc`: `gcc/13.2.0`, `cuda/12.4`, `python3/3.11.8`
- `.bashrc` must be sourced for all remote commands — `cmd` handles this automatically

## Troubleshooting

### "No running compute node found"
Cause: No `idev` session is active, or the job expired.
Fix: Allocate a new node per Step 3 of the Experiment Workflow.

### Node allocated on wrong queue (e.g., `gg` for GPU work)
Cause: `gg` is CPU-only. GPU predictions will fail or silently use CPU (extremely slow).
Fix: Cancel the job (`scancel <jobid>`) and allocate on `gh-dev` or `gh`.

### File transfer fails with `scp`
Fix: Use `rsync -Pz` instead of `scp`.

### Python environment not found / `conda` not found
Cause: `.bashrc` was not sourced, or agent tried to use conda.
Fix: Ensure all compute node commands go through `cmd`. Never use conda/mamba/micromamba.

### Pre-commit `pyrefly` hook fails locally
Cause: `pyrefly` is not installed in the local dev environment.
Fix: `uv pip install pyrefly`

### SSH timeout during long-running command
Cause: Command was run in foreground and SSH connection dropped.
Fix: Always background long-running commands and redirect to a log file. Check progress by tailing the remote log.

### Hallucinated or incorrect numbers in result summaries
Cause: Model generated numbers from memory instead of reading source data.
Fix: ALWAYS read raw result files (`.stats`, `.tsv`) before quoting any numbers. Never guess or recall numbers from earlier in the conversation.
