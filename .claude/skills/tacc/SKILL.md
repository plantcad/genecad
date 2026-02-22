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

### Step 2: Verify remote environment wrapper
```bash
ssh tacc "test -x \$HOME/local/bin/tacc_env.sh && echo 'tacc_env.sh OK' || echo 'MISSING'"
```
If missing, create it — this script sources `.bashrc` before executing commands on compute nodes:
```bash
ssh tacc "mkdir -p \$HOME/local/bin && cat > \$HOME/local/bin/tacc_env.sh << 'SCRIPT'
#!/bin/bash
source ~/.bashrc
exec \"\$@\"
SCRIPT
chmod +x \$HOME/local/bin/tacc_env.sh"
```

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
bash .claude/skills/tacc/scripts/sync_tacc.sh          # rsync (default, fast, no commit needed)
bash .claude/skills/tacc/scripts/sync_tacc.sh --git    # git push + pull (requires clean commit)
```
**rsync (default):** Directly pushes local files without requiring a git commit. Use for in-progress work.
**git (`--git`):** Use only when changes are committed and you want the remote to match a specific branch/commit.

### Step 2: Check for existing compute nodes
```bash
ssh tacc "squeue -u \$USER -o '%.18i %.9P %.30j %.2t %.10M %.6D %.20R'"
```
Reuse a running node if one exists — each session has a minimum 15-minute charge.

### Step 3: Allocate a compute node
Choose the right queue based on whether the task needs a GPU:

| Queue | Type | Time Limit | Use Case |
|-------|------|------------|----------|
| `gg` | CPU-only | — | CPU-only jobs (post-processing, evaluation) |
| `gh-dev` | GPU dev | 2 hours | Try first for GPU work |
| `gh` | GPU prod | — | Fallback if `gh-dev` has no nodes |

Do not allocate a CPU-only node (`gg`) for GPU work. The prediction step requires a GPU.

For GPU jobs, try `gh-dev` first. If no node is provisioned within ~2 minutes, cancel and switch to `gh`:
```bash
idev -p gh-dev -N 1 -n 1 -t 2:00:00
# Check: ssh tacc "squeue -u \$USER -h -t R -p gh-dev -o '%N'"
# If empty after ~2 min: ssh tacc "scancel <jobid>"
idev -p gh -N 1 -n 1 -t 2:00:00
# Wait indefinitely on gh
```
`idev` exits immediately after job submission — do not sleep before checking for the node.

### Step 4: Run the experiment
Write all output to `$SCRATCH/tmp` unless instructed otherwise.
```bash
bash .claude/skills/tacc/scripts/run_tacc.sh "COMMAND 2>&1 | tee \$SCRATCH/tmp/logs/<name>.log"
```
Also tee locally for monitoring after SSH disconnects:
```bash
bash .claude/skills/tacc/scripts/run_tacc.sh "COMMAND 2>&1 | tee \$SCRATCH/tmp/logs/<name>.log" 2>&1 | tee local/logs/exec/<name>.log
```
For long-running GPU jobs (e.g., `make -f pipelines/prediction predictions`), background the remote command:
```bash
bash .claude/skills/tacc/scripts/run_tacc.sh "COMMAND > \$SCRATCH/tmp/logs/<name>.log 2>&1 &"
```
Then check progress periodically:
```bash
bash .claude/skills/tacc/scripts/run_tacc.sh "tail -20 \$SCRATCH/tmp/logs/<name>.log"
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

## User Quick Commands

When invoked directly with `/tacc`, interpret the argument:

- **`status`** or **`jobs`**: Show running SLURM jobs.
  ```bash
  ssh tacc "squeue -u \$USER -o '%.18i %.9P %.30j %.2t %.10M %.6D %.20R'"
  ```

- **`paths`**: Display key TACC filesystem paths:
  ```bash
  ssh tacc "bash -l -c 'echo WORK=\$WORK && echo SCRATCH=\$SCRATCH'"
  ```
  | Path | Location | Notes |
  |------|----------|-------|
  | Repository | `$WORK/repos/genecad` | Code lives here |
  | Experiment output | `$SCRATCH/tmp` | Default output for all experiments |
  | gffcompare | `$WORK/repos/misc/gffcompare/gffcompare` | Built from source |
  | `$WORK` | `/work/10459/$USER/vista` | Persistent, small, read-only for data |
  | `$SCRATCH` | `/scratch/10459/$USER` | Unlimited, GC'd, ephemeral outputs only |

- **No argument or free-text**: Interpret intent from context and follow the Experiment Workflow.

## Environment

- Python env: standard venv at `$WORK/envs/ml-rel/bin/activate`, sourced automatically by `.bashrc`
- Do NOT use conda, mamba, or micromamba — they are not installed
- Modules loaded by `.bashrc`: `gcc/13.2.0`, `cuda/12.4`, `python3/3.11.8`
- `.bashrc` must be sourced for all remote commands (loads modules, activates venv, exports keys)

## Command Execution

Run commands on TACC compute nodes:
```bash
bash .claude/skills/tacc/scripts/run_tacc.sh "COMMAND"
bash .claude/skills/tacc/scripts/run_tacc.sh --node NODE "COMMAND"
```

This auto-detects the first running compute node from `squeue`, or uses the explicit `--node`. It invokes `$HOME/local/bin/tacc_env.sh` on the remote side, which sources `.bashrc` before executing the command.

## File Transfer

Always use `rsync` instead of `scp` to copy files to/from TACC:
```bash
rsync -Pz /local/path tacc:/remote/path
rsync -Pz tacc:/remote/path /local/path
```

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
Fix: The venv is activated by `.bashrc`. Ensure all remote commands go through `run_tacc.sh` or `tacc_env.sh`. Never use conda/mamba/micromamba.

### Pre-commit `pyrefly` hook fails locally
Cause: `pyrefly` is not installed in the local dev environment.
Fix: `uv pip install pyrefly`

### SSH timeout during long-running command
Cause: Command was run in foreground and SSH connection dropped.
Fix: Always background long-running commands and redirect to a log file. Check progress by tailing the remote log.

### Hallucinated or incorrect numbers in result summaries
Cause: Model generated numbers from memory instead of reading source data.
Fix: ALWAYS read raw result files (`.stats`, `.tsv`) before quoting any numbers. Never guess or recall numbers from earlier in the conversation.
