# Plan — Auto-download GLM-5.2 from Hugging Face on container startup

## Goal

When the user runs `docker run colibri:1.0` with `COLI_HF_REPO` set, the
container should populate [`$COLI_MODEL`](docker/README.md:1) with the
contents of that HF repo *before* launching the engine. If the directory
already has a `config.json`, skip the download. If `COLI_HF_REPO` is
unset, the entrypoint does nothing extra (zero-cost for users who bind-
mount a pre-populated model dir).

The two env vars the user wants:

| Var | Required? | Default | What it does |
|---|---|---|---|
| `COLI_HF_REPO` | yes, to trigger download | unset → no-op | the HF repo id, e.g. `zai-org/GLM-5.2-FP8` or `zai-org/GLM-5.2-FP8` |
| `HF_TOKEN` | only for private/gated repos | unset | forwarded to `huggingface_hub` as `token=...` |

## Design constraints (driven by the engine, not by opinion)

| Constraint | Source | Effect |
|---|---|---|
| Cold decode reads ~11 GB/token randomly | README | The download itself is the only one-time cost; the engine handles streaming from then on |
| Engine expects `config.json` + safetensors shards in `COLI_MODEL` | [`c/coli:136`](c/coli:136) | download must write a complete, runnable model directory |
| Container image is ~250 MB CPU / ~5 GB CUDA | existing Dockerfiles | adding `huggingface_hub` (+~30 MB) is fine; adding `torch` would blow the budget and is *not* needed (we only download; we don't convert at startup) |
| `init: true` in compose for SIGINT propagation | existing compose file | the entrypoint wrapper script must `exec` the engine so signals still reach it |
| Engine wants `~370 GB` of free disk on the bind-mount | [`c/setup.sh:60`](c/setup.sh:60) | the entrypoint must check free space before the download, not after |
| FP8+convert path is a separate `coli convert` step that needs torch | [`c/tools/convert_fp8_to_int4.py:1`](c/tools/convert_fp8_to_int4.py:1) | this design does NOT auto-run convert; that stays a manual step. We just download the repo contents verbatim and let the user run convert if they pulled FP8 |

## Decisions (locked in from Q&A)

| # | Decision | Rationale |
|---|---|---|
| 1 | `COLI_HF_REPO` is the single trigger; unset = no-op | the existing image must keep working for users who bind-mount a pre-populated dir |
| 2 | Default repo is `zai-org/GLM-5.2-FP8` (~756 GB) — the only public GLM-5.2 artifact on HF | the `convert` step is manual (see Docker README); we don't ship torch in the runtime image |
| 3 | Use `huggingface_hub.snapshot_download` (not `huggingface-cli`) | one call, one process, better error reporting, supports `token=...` natively |
| 4 | Wrap the entrypoint with a tiny shell script (not a Python wrapper) | shell `exec` is the cleanest way to preserve SIGINT/SIGTERM to the engine, and the wrapper is ~15 lines |
| 5 | Install `huggingface_hub` at build time via `pip install --no-cache-dir` | cheaper than a per-startup pip install; HF's releases are stable |
| 6 | Idempotency: skip if `${COLI_MODEL}/config.json` already exists | lets users kill+restart the container mid-download without re-pulling everything |
| 7 | Free-space check before download with a clear error message | `OSError: No space left on device` deep in hf_hub is awful to debug |
| 8 | Allow `COLI_HF_REVISION` (advanced) for pinning to a specific commit | not asked for, but trivially cheap to wire through and users will eventually want it |
| 9 | Don't auto-run `coli convert` after the download | keeps the image torch-free; user can run it as a one-shot if they pulled FP8 |

> **NOTE (2026-07-21):** the only public GLM-5.2 repo on Hugging Face is `zai-org/GLM-5.2-FP8`. There is no pre-quantised int4 repo to `COLI_HF_REPO=` yet. The HF fetcher will download the FP8 source verbatim; the user must then run `coli convert` (one-shot, with a torch-enabled image) to produce the int4 model the engine serves. If/when an int4 repo is published, set `COLI_HF_REPO=<that-repo>` and the same wrapper works without changes — no `convert` step.

## File layout (all under `docker/`)

```
docker/
├── entrypoint.sh        # NEW — wraps /opt/colibri/c/coli
├── hf_fetch.py          # NEW — stdlib-style downloader (uses huggingface_hub)
├── healthcheck.py       # unchanged
├── Dockerfile           # MOD — add huggingface_hub pip install; new ENTRYPOINT
├── Dockerfile.cuda      # MOD — same
├── docker-compose.yml   # MOD — wire COLI_HF_REPO, HF_TOKEN, COLI_HF_REVISION
├── Makefile             # MOD — add `make fetch` and `make build-fresh`
├── README.md            # MOD — new "Auto-download from Hugging Face" section
└── .dockerignore        # unchanged
```

## The wrapper: [`docker/entrypoint.sh`](docker/entrypoint.sh)

```bash
#!/usr/bin/env bash
# Wrap the engine. If COLI_HF_REPO is set, populate COLI_MODEL from
# HuggingFace before exec'ing the engine. Unset = no-op (zero cost).
set -euo pipefail

# 1. If COLI_HF_REPO is unset OR the model already looks populated, skip.
if [ -n "${COLI_HF_REPO:-}" ]; then
    MODEL_DIR="${COLI_MODEL:-/models/glm52_i4}"
    if [ ! -f "${MODEL_DIR}/config.json" ]; then
        echo "[entrypoint] COLI_HF_REPO=${COLI_HF_REPO} set, model not found at ${MODEL_DIR}"
        echo "[entrypoint] downloading via huggingface_hub..."
        exec /usr/bin/env python3 /opt/colibri/docker/hf_fetch.py
    else
        echo "[entrypoint] ${MODEL_DIR}/config.json exists, skipping download"
    fi
fi

# 2. Hand off to the engine. `exec` replaces this shell so SIGINT/SIGTERM
#    go straight to the engine's handler (c/coli:613 in chat mode).
exec /opt/colibri/c/coli "$@"
```

## The downloader: [`docker/hf_fetch.py`](docker/hf_fetch.py)

```python
#!/usr/bin/env python3
"""Idempotent HF repo -> COLI_MODEL snapshot download.

Called by entrypoint.sh when COLI_HF_REPO is set and COLI_MODEL has no
config.json. Uses huggingface_hub.snapshot_download with:
  - token = $HF_TOKEN (if set)
  - revision = $COLI_HF_REVISION (if set; default 'main')
  - allow_patterns = safetensors + json + tokenizer

Exits 0 on success, non-zero on failure. The entrypoint does NOT catch —
we want a hard failure if the download breaks (better than a silent
half-model).
"""
import os
import sys
import shutil

REPO  = os.environ.get("COLI_HF_REPO")
DEST  = os.environ.get("COLI_MODEL", "/models/glm52_i4")
REV   = os.environ.get("COLI_HF_REVISION", "main")
TOKEN = os.environ.get("HF_TOKEN") or None

if not REPO:
    print("error: COLI_HF_REPO is not set", file=sys.stderr)
    sys.exit(2)

# Free-space check (refuse to start if we don't have ~1.2x the expected
# model size, just like the existing convert workflow in
# c/tools/convert_fp8_to_int4.py:88).
EXPECTED_GB = int(os.environ.get("COLI_HF_EXPECT_GB", "450"))  # int4 ~=372, FP8 ~=756
free_gb = shutil.disk_usage(DEST).free / 1e9
if free_gb < EXPECTED_GB * 1.1:
    print(f"error: only {free_gb:.0f} GB free at {DEST}; need ~{EXPECTED_GB} GB + 10% margin",
          file=sys.stderr)
    sys.exit(3)

# Hand off to the library. The download itself is in hf_hub; we just
# pass our env vars through.
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=REPO,
    repo_type="model",
    revision=REV,
    local_dir=DEST,
    token=TOKEN,
    allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "*.tiktoken"],
    max_workers=int(os.environ.get("COLI_HF_WORKERS", "8")),
    tqdm_class=None,           # we get noisy container logs otherwise
)
print(f"[hf_fetch] downloaded {REPO}@{REV} -> {DEST}")
```

## Dockerfile delta

Both `Dockerfile` and `Dockerfile.cuda` need three small changes:

```dockerfile
# 1. install huggingface_hub (no torch, no transformers, just the lib)
RUN pip install --no-cache-dir --break-system-packages \
        "huggingface_hub>=0.24,<1.0"

# 2. copy the wrapper scripts
COPY docker/entrypoint.sh /opt/colibri/docker/entrypoint.sh
COPY docker/hf_fetch.py  /opt/colibri/docker/hf_fetch.py
RUN chmod +x /opt/colibri/docker/entrypoint.sh /opt/colibri/docker/hf_fetch.py

# 3. swap ENTRYPOINT
ENTRYPOINT ["/opt/colibri/docker/entrypoint.sh"]
# CMD is unchanged: ["serve", "--host", "0.0.0.0", "--port", "8000"]
```

Expected image-size impact: **+~30 MB** for `huggingface_hub` (it ships with `filelock`, `hf-xet`, `tqdm`, `typing-extensions`, `packaging`, `requests`, `pyyaml`).

## docker-compose delta

```yaml
# Add to BOTH services (coli and coli-gpu) under `environment:`:
environment:
  COLI_HF_REPO: "${COLI_HF_REPO:-}"        # e.g. zai-org/GLM-5.2-FP8
  HF_TOKEN:     "${HF_TOKEN:-}"            # forwarded as-is
  COLI_HF_REVISION: "${COLI_HF_REVISION:-}"  # optional: pin to a commit
  COLI_HF_WORKERS: "${COLI_HF_WORKERS:-8}"   # parallel shard downloads
  COLI_HF_EXPECT_GB: "${COLI_HF_EXPECT_GB:-450}"
```

The `.env` file is intentionally NOT created (the README's existing
"no .env file" decision is preserved). Users `export` the vars in
their shell before `docker compose up`.

## Makefile delta

```makefile
# Add a one-shot "fetch" target that runs the download in a one-off
# container, then exits. Useful when the user wants to pre-populate
# the bind-mount on a slow link without holding a long-running container.
.PHONY: fetch
fetch:
    @if [ -z "$$COLI_HF_REPO" ]; then \
        echo "set COLI_HF_REPO first, e.g.:"; \
        echo "  COLI_HF_REPO=zai-org/GLM-5.2-FP8 make fetch"; \
        exit 1; \
    fi
    $(COMPOSE) run --rm \
        -e COLI_HF_REPO -e HF_TOKEN -e COLI_HF_REVISION -e COLI_HF_WORKERS \
        --entrypoint '/usr/bin/env python3 /opt/colibri/docker/hf_fetch.py' \
        coli
```

## README delta

New section "Auto-download from Hugging Face" with:

1. **The two env vars** (table)
2. **Quick example**: `COLI_HF_REPO=zai-org/GLM-5.2-FP8 HF_TOKEN=hf_xxx docker compose up -d coli-gpu`
3. **Resume behavior**: kill the container mid-download, restart with the same env, it picks up from `.incomplete` files
4. **Disk budget**: ~450 GB free required for the default int4; the entrypoint refuses to start if there's less
5. **The 1.x-multiplier is a safety margin**, not because hf_hub is sloppy

## Validation plan

1. **No-op case**: `docker run --rm colibri:1.0 info` with `COLI_HF_REPO` unset → identical to today (entrypoint is a no-op passthrough)
2. **Empty-model case**: bind-mount an empty dir, set `COLI_HF_REPO=.../nonexistent-repo` → entrypoint fails fast with the HF error, exit code != 0
3. **Idempotency**: after a successful download, `docker run` again → entrypoint logs "skipping download" and launches the engine
4. **Image size**: `docker images colibri:1.0` shows ~+30 MB vs the previous build
5. **Smoke test**: `make up` with `COLI_HF_REPO` set and a fake token → the `doctor` subcommand should still report a real, populated model dir (this is the full-stack check, requires the real 372 GB model)

## Out of scope (intentional)

- **No auto-convert**: if the user pulls FP8, they have to run `docker compose run --rm coli convert --model /models/glm52_i4` themselves (this needs `torch` and is a separate container for that reason)
- **No S3/MinIO fallback**: HF only, per the user's request
- **No background pre-fetch**: the download is part of the startup path; users on slow links should pre-populate with `make fetch`
- **No resumable progress UI**: hf_hub's own tqdm goes to stderr; the entrypoint suppresses it for cleaner container logs
- **No mirror selection**: we go straight to `huggingface.co`. HF_ENDPOINT env var (already respected by hf_hub) covers the rare user behind a corporate mirror.

## Total scope

- **Files**: 2 new (`entrypoint.sh`, `hf_fetch.py`) + 4 modified (`Dockerfile`, `Dockerfile.cuda`, `docker-compose.yml`, `Makefile`, `README.md`)
- **Image size impact**: +~30 MB
- **No new external dependencies at runtime** other than `huggingface_hub` (no torch, no transformers)
- **No behaviour change** for users who don't set `COLI_HF_REPO`
