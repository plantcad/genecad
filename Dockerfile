FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

# Install curl, git, and uv package manager
RUN apt-get update && apt-get install -y curl git && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /build

# Copy project files and directories
COPY pyproject.toml LICENSE ./

# Build gffcompare from source
RUN mkdir -p /tmp/gffcompare && cd /tmp/gffcompare && \
    git clone https://github.com/gpertea/gffcompare && \
    cd gffcompare && \
    make release && \
    cp gffcompare /usr/local/bin/ && \
    cd / && rm -rf /tmp/gffcompare

# Install dependencies
RUN uv sync --extra torch --extra mamba

# Create entrypoint script to source environment and add
# source dynamically from working directory rather than
# embedding it in the container itself
RUN echo '#!/bin/bash\n\
source /build/.venv/bin/activate\n\
export PYTHONPATH="$(pwd):$PYTHONPATH"\n\
exec "$@"' > /usr/local/bin/genecad && \
    chmod +x /usr/local/bin/genecad
ENTRYPOINT ["/usr/local/bin/genecad"]
