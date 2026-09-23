FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    ca-certificates \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Install NeurInSpectre. The extra index enables CUDA wheels for torch when
# available (falls back to CPU wheels if not).
#
# To include dev tooling (pytest/black/etc), build with:
#   docker build --build-arg INSTALL_DEV=1 ...
ARG INSTALL_DEV=0
ARG INSTALL_MALWARE=0
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    if [ "${INSTALL_DEV}" = "1" ]; then \
      python3 -m pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu121 ; \
    else \
      python3 -m pip install -e "." --extra-index-url https://download.pytorch.org/whl/cu121 ; \
    fi && \
    if [ "${INSTALL_MALWARE}" = "1" ]; then \
      python3 -m pip install -e ".[malware]" && \
      python3 -m pip install "git+https://github.com/elastic/ember.git" --no-deps ; \
    fi && \
    rm -rf /root/.cache/pip

# Verify installation (no network; best-effort inventory).
RUN neurinspectre doctor

# NOTE: The .dockerignore excludes data/ and models/ for image size.
# Mount them at runtime:
#   docker run --gpus all --rm -it \
#     -v /path/to/data:/app/data \
#     -v /path/to/models:/app/models \
#     neurinspectre:ae table2 --config table2_config.yaml --strict-real-data
ENTRYPOINT ["neurinspectre"]

