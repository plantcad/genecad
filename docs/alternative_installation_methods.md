# Installing GeneCAD

## Installing from a pre-built wheel (recommended)

The recommended method uses a pre-built wheel to reduce installation time.

```bash
# Download the GeneCAD repository
git clone https://github.com/plantcad/genecad.git
cd genecad

# Create a virtual environment and install GeneCAD
uv venv
source .venv/bin/activate
bash scripts/install_release.sh
```

## Install from Source (Using uv)

If you are a developer and want to install GeneCAD from the source code instead of using the pre-built wheel, use [uv](https://docs.astral.sh/uv/):

```bash
# Clone the repository
git clone https://github.com/plantcad/genecad && cd genecad

# Install all dependencies
# mamba and causal-conv1d build from source — this can take 3–30 minutes
uv sync --extra torch --extra mamba

# Activate the virtual environment
source .venv/bin/activate
```

PyTorch is pinned to 2.7.1 (CUDA 12.8) to ensure compatibility with `mamba` (2.2.4) and `causal-conv1d` (1.5.0.post8). Newer combinations may work but are not officially tested.

`mamba` and `causal-conv1d` build from source without build isolation for reliability. To cache the built wheels and speed up future installs, add these entries to `pyproject.toml`:

```toml
[tool.uv.sources]
mamba-ssm = { path = "path/to/mamba_ssm-2.2.4-*.whl" }
causal-conv1d = { path = "path/to/causal_conv1d-1.5.0.post8-*.whl" }
# Or use a remote URL:
# mamba-ssm = { url = "https://.../mamba_ssm-2.2.4-*.whl" }
```

## Containers (Singularity/Apptainer or Docker)

The Docker image at `ghcr.io/plantcad/genecad_v1` bundles the complete runtime.
Source code is **mounted at run time**, so changes you make to the cloned repository
take effect immediately — no rebuild needed.

> [!TIP]
> **Understanding File Paths in Containers**
> The argument `-v $(pwd):/workspace` connects your current folder on the host machine
> to the `/workspace` folder inside the container. Always place your input FASTA
> files inside your current directory so the container can see them!

> [!IMPORTANT]
> If your environment requires a specific Docker alias (e.g., `docker1`), replace
> `docker` with your local command in the examples below.

```bash
# Clone the repository
git clone https://github.com/plantcad/genecad && cd genecad

# Pull the image
docker pull ghcr.io/plantcad/genecad_v1:latest

# Run GeneCAD in your Docker container
docker run --rm --gpus all \
  -v $(pwd):/workspace -w /workspace \
  ghcr.io/plantcad/genecad_v1:latest \
  bash predict.sh \
    -i /workspace/data/my_plant.fa \
    -o /workspace/output \
    -s Zmays \
    -m plant
```

For HPC environments where Docker is not available, you can use Singularity
(or Apptainer) to pull and run the Docker image directly. The `--nv` flag is
required to enable GPU support.

```bash
# Clone the repository
git clone https://github.com/plantcad/genecad && cd genecad

# Pull the Docker image and convert it to a Singularity image
singularity pull genecad.sif docker://ghcr.io/plantcad/genecad_v1:latest

# Run on the bundled Arabidopsis example (auto-downloads FASTA)
singularity exec --nv \
  --bind $(pwd):/workspace \
  --pwd /workspace \
  genecad.sif \
  /usr/local/bin/genecad bash predict.sh

# Run GeneCAD in your Singularity container
singularity exec --nv \
  --bind $(pwd):/workspace \
  --pwd /workspace \
  genecad.sif \
  /usr/local/bin/genecad bash predict.sh \
    -i /workspace/data/my_plant.fa \
    -o /workspace/output \
    -s Zmays \
    -m plant
```

## Working with SLURM on HPC Clusters

When installing GeneCAD [from source](#install-from-source-using-uv) or using
the [quick-start installation script](#download-and-install) on an HPC, make sure
you are performing installation on a node that has access to GPU resources. This is
essential for the package manager to identify the correct version of `pytorch` with
`CUDA` for your system. Alternatively, use the [container-based installation](#containers-singularityapptainer-or-docker).

You may need to load specific modules, such as your system's `cuda-toolkit`, to
successfully install the GeneCAD environment. Remember to activate your environment
with `source .venv/bin/activate` in your SLURM batch script before calling GeneCAD.

> [!TIP]
> **Multi-GPU on a Single Node:** To use all GPUs on a node, request them with `--gres=gpu:4` and pass `--gpus all` to
> `genecad predict`. If there are many chromosomes, they will be distributed across GPUs.
> If there are fewer chromosomes than GPUs, GeneCAD will automatically split the longest
> sequences into parallel windows across the GPUs.

> [!TIP]
> **Multi-Node Distributed Inference (e.g., TACC):** GeneCAD natively detects multi-node
> SLURM topologies. If you allocate multiple nodes (e.g., `#SBATCH --nodes=4`), you can
> use `srun` to seamlessly process enormous single contigs across the entire
> cluster. `genecad predict` will automatically bypass local launchers and
> allow PyTorch Lightning to route the distributed process windows.

> [!TIP]
> **Custom launcher:** If your cluster requires a non-standard Python entrypoint
> (e.g. `srun python` instead of `torchrun`), use `--launcher` or the `LAUNCHER`
> environment variable to override automatic detection.

## Cloud Provisioning (SkyPilot)

[SkyPilot](https://docs.skypilot.co/en/latest/docs/index.html) lets you provision on-demand cloud GPUs across many providers (AWS, GCP, Lambda, RunPod, etc.) without managing infrastructure manually. This is useful if you do not have access to a local GPU or HPC cluster.

> [!IMPORTANT]
> Run these commands from your **local machine or a login node with internet access** — not from an HPC compute node, which typically has no outbound internet and cannot reach cloud provider APIs.

**Step 1 — Install SkyPilot** (in a dedicated environment, separate from the GeneCAD venv to avoid dependency conflicts):

```bash
python3 -m venv ~/skypilot-env
source ~/skypilot-env/bin/activate
pip install "skypilot[lambda]>=0.10.3"
```

**Step 2 — Start the API server and configure credentials:**

```bash
# Start the SkyPilot API server (runs as a background process)
sky api start

# Verify credentials for your cloud provider
sky check lambda
```

If `sky check lambda` shows `Lambda: disabled`, configure your API key:
1. Generate a key at [https://cloud.lambdalabs.com/api-keys](https://cloud.lambdalabs.com/api-keys)
2. Add it to `~/.lambda_cloud/lambda_keys`:

```bash
mkdir -p ~/.lambda_cloud
echo "api_key = YOUR_API_KEY_HERE" > ~/.lambda_cloud/lambda_keys
sky check lambda   # should now show: Lambda: enabled ✓
```

**Step 3 — Launch a GPU node and run:**

*(Make sure your terminal is currently in the root of the cloned `genecad` repository)*

> [!NOTE]
> **Annotating your own genome:** When you are ready to annotate your own data, place your FASTA file inside the `genecad` directory (e.g., `genecad/my_genome.fa`) before running `sky launch`. This ensures SkyPilot automatically uploads it to the cloud. You would then append `-i my_genome.fa -s MySpecies` to the `bash predict.sh` command below.

```bash
# 1. Deploy the GPU node (--no-setup: Docker replaces uv sync)
sky launch --num-nodes 1 --yes --no-setup \
  --cluster genecad examples/configs/cluster.sky.yaml

# 2. Run the workload (this example runs the built-in Arabidopsis test)
sky exec genecad 'docker pull ghcr.io/plantcad/genecad_v1:latest && \
                  docker run --rm --gpus all \
                  -v $(pwd):/workspace -w /workspace \
                  ghcr.io/plantcad/genecad_v1:latest \
                  bash predict.sh'

# 3. Download the results back to your local machine
rsync -avz genecad:~/sky_workdir/genecad_result/ ./genecad_result/

# 4. Terminate the node to stop billing
sky down genecad --yes
```

> [!IMPORTANT]
> **Why Docker instead of `uv sync`?**
> `flash-attn`, `mamba-ssm`, and `causal-conv1d` must all compile from source, which can
> take **30–60 minutes** on a fresh cloud node and may fail if the CUDA/GCC versions mismatch.
> The pre-built Docker image includes all compiled packages, making cluster setup take seconds
> instead of hours.

See the [Throughput](../README.md#prerequisites) section for GPU cost comparisons across providers.

<details><summary>SkyPilot GPU rates</summary>

```
> sky show-gpus L4:1   # low-end development GPU
GPU  CLOUD   INSTANCE_TYPE  DEVICE_MEM  HOURLY_PRICE  REGION
L4   RunPod  1x_L4_SECURE   24GB        $ 0.440       CA
L4   GCP     g2-standard-4  24GB        $ 0.705       us-east4
L4   AWS     g6.xlarge      22GB        $ 0.805       us-east-2

> sky show-gpus H100:1  # high-end production GPU
GPU   CLOUD       INSTANCE_TYPE          DEVICE_MEM  HOURLY_PRICE  REGION
H100  Hyperbolic  1x-H100-75-722         80GB        $ 1.290       default
H100  Lambda      gpu_1x_h100_pcie       80GB        $ 2.490       europe-central-1
H100  GCP         a3-highgpu-1g          80GB        $ 5.383       us-central1
```

</details>
