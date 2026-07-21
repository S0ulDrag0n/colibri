# colibrì — CPU runtime image.
#
# Build context: the repo root (so `COPY c/ /opt/colibri/c/` resolves).
# The .dockerignore keeps the context small.
#
# Build:
#   docker build -f docker/Dockerfile -t colibri:1.0 .
# or via the Makefile in docker/:
#   make -C docker build
#
# Override the target CPU with --build-arg ARCH=x86-64-v3 for a portable
# binary, or ARCH=native (the default) for the fastest build on the host.
# The engine's Linux Makefile autodetects via `gcc -dumpmachine`, so the
# same Dockerfile works on x86-64, aarch64 (Graviton), and ppc64.

# ---------- builder stage: produce a clean, modern gcc for the build ----------
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Build-time only: gcc, make, libc headers. We discard this stage so the
# runtime image is not bloated with the toolchain.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        make \
        libc6-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/colibri-build

# Copy only the engine source first — this layer is cached and only re-runs
# when c/ changes.
COPY c/ /opt/colibri-build/c/

# ARCH arg:
#   - native  : fastest on the build host (default, Linux Makefile path
#               autodetects via `gcc -dumpmachine` and emits the right -march)
#   - x86-64-v3 : portable binary with AVX2 (e.g. for distributing the image)
#   - armv8-a  : portable aarch64 (Apple-shipping Graviton, Ampere)
#   - power8   : portable ppc64
# The `Makefile` (c/Makefile) maps these to the right compiler flag.
ARG ARCH=native
RUN make -C c glm ARCH=${ARCH}

# ---------- runtime stage: slim base, libgomp + python3 only ----------
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Runtime only: no gcc/make/libc-dev. libgomp1 is what libgomp-linked
# binaries need at runtime; python3 is the interpreter for `coli` and
# `openai_server.py` (both stdlib-only).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        python3 \
        python3-pip \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# huggingface_hub lets the entrypoint pull the model on first start
# (only used when COLI_HF_REPO is set; see docker/entrypoint.sh). We
# pin a wide range because the library is stable but new features ship
# often. No torch here — that would add ~2 GB and is not needed for
# a snapshot_download.
RUN pip install --no-cache-dir --break-system-packages \
        "huggingface_hub>=0.24,<1.0"

# Copy the engine, the CLI, and the small set of Python modules the CLI
# imports at runtime (resource_plan, doctor, openai_server). Tools/ is
# also copied because the CLI invokes subprocesses from it (eval_glm.py
# for bench, clean.py for clean, etc.).
COPY --from=builder /opt/colibri-build/c/glm          /opt/colibri/c/glm
COPY --from=builder /opt/colibri-build/c/coli         /opt/colibri/c/coli
COPY --from=builder /opt/colibri-build/c/openai_server.py /opt/colibri/c/openai_server.py
COPY --from=builder /opt/colibri-build/c/resource_plan.py /opt/colibri/c/resource_plan.py
COPY --from=builder /opt/colibri-build/c/doctor.py    /opt/colibri/c/doctor.py
COPY --from=builder /opt/colibri-build/c/tools        /opt/colibri/c/tools

# The healthcheck script lives in the docker/ directory at build time;
# install it at a stable path the compose healthcheck can call.
COPY docker/healthcheck.py /opt/colibri/docker/healthcheck.py
# Entrypoint wrapper + HF downloader. These run BEFORE the engine and
# are no-ops when COLI_HF_REPO is unset (the common case for users
# who bind-mount a pre-populated model directory).
COPY docker/entrypoint.sh /opt/colibri/docker/entrypoint.sh
COPY docker/hf_fetch.py  /opt/colibri/docker/hf_fetch.py
RUN chmod +x /opt/colibri/c/coli /opt/colibri/docker/entrypoint.sh /opt/colibri/docker/hf_fetch.py /opt/colibri/docker/healthcheck.py

# Default model location. The compose file bind-mounts the host's int4
# model directory at this path; users who want a different layout can
# override COLI_MODEL at run time.
ENV COLI_MODEL=/models/glm52_i4
ENV PATH=/opt/colibri/c:${PATH}
ENV PYTHONUNBUFFERED=1

# Expose the OpenAI-compatible HTTP API port.
EXPOSE 8000

# Default: run the OpenAI-compatible server, listening on all interfaces
# so docker port-forwarding (and the host's localhost) both work. Override
# at run time: `docker run colibri:1.0 chat`, `docker run colibri:1.0 info`,
# `docker run colibri:1.0 plan`, etc. (the CLI subcommands are passed as
# the CMD).
#
# We use the JSON-array form so the entrypoint/cmd are not shell-interpreted;
# this also lets `docker stop` deliver SIGINT to the engine's SIGINT
# handler (which gracefully stops the current turn — see c/coli:613)
# rather than killing the process mid-token.
# Entrypoint wrapper. When COLI_HF_REPO is set, it pulls the model
# from Hugging Face before launching the engine. When unset, it is
# a passthrough (zero cost).
ENTRYPOINT ["/opt/colibri/docker/entrypoint.sh"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
