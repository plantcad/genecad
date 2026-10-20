ARG TARGETARCH
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel AS base-amd64
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS base-arm64
FROM base-${TARGETARCH} AS runtime

ENV UV_INSTALL_DIR=/opt/uv/bin \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_CACHE_DIR=/tmp/uv-cache \
    PATH="/opt/uv/bin:$PATH" \
    MAX_JOBS=2 \
    NVCC_THREADS=1

# Install curl, git, and uv package manager
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl git wget && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy files required for dependency resolution and local package build metadata
COPY pyproject.toml LICENSE README.md predict.sh train.sh ./
COPY scripts ./scripts
COPY src ./src
RUN chmod -R a+rX /build

# Install dependencies
RUN if [ "$(uname -m)" = aarch64 ]; then \
      GENECAD_WHEEL_DIR=/tmp/genecad-wheels bash scripts/install_release.sh --skip-gpu-check; \
    else \
      uv sync --extra torch --extra mamba; \
    fi && rm -rf /tmp/uv-cache /tmp/genecad-wheels /root/.cache

# Create entrypoint script to source environment and add
# source dynamically from working directory rather than
# embedding it in the container itself
RUN echo '#!/bin/bash\n\
source /build/.venv/bin/activate\n\
export PYTHONPATH="$(pwd):$PYTHONPATH"\n\
exec "$@"' > /usr/local/bin/genecad && \
    chmod +x /usr/local/bin/genecad
ENTRYPOINT ["/usr/local/bin/genecad"]
