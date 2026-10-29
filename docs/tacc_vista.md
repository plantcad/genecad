# GH200 / TACC Vista

GH200 combines an ARM64 Grace CPU with a Hopper GPU. Use native ARM64
packages or an ARM64 container; x86-64 CUDA wheels and SIF files cannot be reused.
Single-GPU plant inference and native CUDA numerical/backward checks have been
validated on Vista with Python 3.12, PyTorch 2.7.1+cu128, Triton 3.3.1,
FlashAttention 2.8.3, Mamba 2.2.4 and causal-conv1d 1.5.0.post8.

## Native installation

Install `uv` and Git, then run on an allocated GH200 node:

```bash
module load gcc/14.2.0 cuda/12.8  # Vista
git clone https://github.com/plantcad/genecad.git
cd genecad
bash scripts/install_release.sh
source .venv/bin/activate
genecad predict -i /absolute/path/genome.fa -o result -s MySpecies -m plant
```

The installer uses official ARM64 PyTorch/Triton wheels and builds the three
CUDA extensions from fixed sources. Allow at least 30 GB free space and a
two-hour allocation for the first build. Completed wheels are cached under
`$WORK/.cache/genecad/arm64-cu128-cp312`, or `~/.cache/...` when `WORK` is unset.
Use `--wheel-dir PATH`, `--venv PATH`, or `--jobs N` to override defaults.
Keep the checkout in place: GeneCAD is installed in editable mode.

For CPU build nodes with CUDA 12.8 loaded, add `--skip-gpu-check`. It checks
GeneCAD imports and installed versions; native CUDA imports and execution are
deferred because Triton/Mamba require an active GPU driver even during import.
Before inference, validate on a GH200 node:

```bash
source .venv/bin/activate
python scripts/check_install.py
```

The environment records versions and wheel hashes in `genecad-requirements.txt`
and `genecad-wheels.sha256`. GPU validation writes `genecad-install-check.json`.
Do not run `uv sync` afterward: its Triton lock entry is x86-only.

## Containers

Docker, Singularity and Apptainer use the same multi-platform image. Pull it
on the target architecture; a converted SIF is specific to that architecture.
See [container commands](alternative_installation_methods.md#containers-singularityapptainer-or-docker).
Tags published before ARM64 support remain x86-only. The new publishing workflow
creates a shared tag only after both native builds and CPU smoke checks pass.
Run `python /build/scripts/check_install.py` inside the container with GPU access
to validate native CUDA kernels before inference.

On Vista, load the available `tacc-apptainer` module and use an allocated GH node:

```bash
module load tacc-apptainer
apptainer pull genecad-arm64.sif docker://ghcr.io/plantcad/genecad_v1:latest
apptainer exec --nv --bind "$PWD:/workspace" --pwd /workspace \
  genecad-arm64.sif /usr/local/bin/genecad bash predict.sh \
  -i /workspace/genome.fa -o /workspace/result -s MySpecies -m plant
```

Consult the [Vista guide](https://docs.tacc.utexas.edu/hpc/vista/) for allocation,
module and queue policies. CPU-only GG nodes cannot run inference.
