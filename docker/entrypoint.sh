#!/usr/bin/env bash
# colibrì entrypoint wrapper.
#
# If COLI_HF_REPO is set AND ${COLI_MODEL}/config.json is missing, run the
# HF downloader first. Then exec the engine. The exec is critical: it
# replaces this shell so SIGINT/SIGTERM propagate to the engine's handler
# (c/coli:613 in chat mode), which `docker stop` delivers when `init: true`
# is set in compose.
#
# Behaviour matrix:
#   COLI_HF_REPO unset                                -> passthrough, zero cost
#   COLI_HF_REPO set, config.json present             -> passthrough, logs "skipping"
#   COLI_HF_REPO set, config.json missing, no mount   -> error: bind-mount missing
#   COLI_HF_REPO set, config.json missing, mounted    -> run hf_fetch.py, then engine
#
# Why a shell wrapper, not a Python wrapper:
#   - `exec` is one line, not a child process. SIGINT hits the engine
#     directly. A Python wrapper would need signal-handler plumbing.
#   - the wrapper itself is <30 lines, no logic worth a language upgrade.
set -euo pipefail

# 1. Optional model pre-fetch from Hugging Face.
if [ -n "${COLI_HF_REPO:-}" ]; then
    MODEL_DIR="${COLI_MODEL:-/models/glm52_i4}"
    if [ ! -f "${MODEL_DIR}/config.json" ]; then
        # Pre-flight: the destination must exist (it's the bind-mount). If
        # the user forgot to set up the bind-mount, fail with a clear
        # error BEFORE the "downloading" message, so they don't think the
        # container is wasting time on a download that can never succeed.
        if [ ! -d "${MODEL_DIR}" ]; then
            echo "[entrypoint] error: COLI_MODEL=${MODEL_DIR} is not a directory" >&2
            echo "                bind-mount your model directory there, e.g.:" >&2
            echo "                  -v /path/on/host:${MODEL_DIR}:rw" >&2
            echo "                (or set COLI_MODEL to a directory that exists in the image)" >&2
            exit 4
        fi
        echo "[entrypoint] COLI_HF_REPO=${COLI_HF_REPO} set; ${MODEL_DIR} not populated -> downloading"
        exec /usr/bin/env python3 /opt/colibri/docker/hf_fetch.py
    else
        echo "[entrypoint] ${MODEL_DIR}/config.json already exists; skipping download"
    fi
fi

# 2. Hand off to the engine. `exec` replaces this shell so signals go
#    directly to the engine (c/coli:613 handles SIGINT gracefully).
exec /opt/colibri/c/coli "$@"
