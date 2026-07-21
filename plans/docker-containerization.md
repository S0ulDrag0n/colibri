# Plan — Docker containerization for colibrì

## Goal

Run colibrì (the pure-C GLM-5.2 744B MoE inference engine) in Docker on a **Linux host**, with an opt-in GPU tier. The runtime container ships only what's needed to *run* the engine — the offline FP8→int4 converter is intentionally kept out, per the user's choice (Python 3.12 + venv, no torch in the image, conversion happens on a separate host).

## Constraints (driven by the engine, not by opinion)

| Constraint | Source | Effect on the container |
|---|---|---|
| Engine mmaps 144+ shards at startup | [`c/coli:28`](c/coli:28) bumps `RLIMIT_NOFILE` to 65536 | `--ulimit nofile=65536:65536` is non-negotiable; cgroups clip `RLIMIT_NOFILE` even after the Python bump |
| Cold decode reads ~11 GB/token randomly across shards | README "How to test it" section | The int4 model directory must be a **bind-mount from a local ext4/xfs filesystem** — never 9p / NFS / virtiofs |
| OpenMP kernels are memory-bound; SMT siblings hurt | README benchmarks table | `cpuset-cpus` should pin to **physical cores only**, never logical |
| Engine auto-caps expert cache at 88% of `MemAvailable` | `c/resource_plan.py` | `mem_limit` (cgroup) becomes the de-facto RAM budget — the engine reads `MemAvailable` through cgroup-aware APIs |
| Pure C, libgomp, no Python deps at runtime | [`c/Makefile`](c/Makefile) and `c/coli` is stdlib-only | Slim image: `gcc` (for the build), `libgomp1` (for runtime), `python3` (for `coli`); no `pip install` needed |
| The CLI's byte protocol relies on SIGINT being delivered cleanly | [`c/coli:613`](c/coli:613) `Ctrl-C` handling | `init: true` in compose so `docker stop` propagates SIGINT to the engine's handler (which gracefully stops the current turn) |
| Engine has no GPU requirement | `make` default is pure C | Two-image strategy: lean CPU default, optional CUDA image |
| Engine binary is built once per image | 30s build | No multi-stage / ccache — not worth the complexity |

## Decisions (locked in from Q&A)

| # | Decision | Rationale |
|---|---|---|
| 1 | Two Dockerfiles, not conditional `FROM` | Easier to reason about, lean default, GPU opt-in is a profile toggle |
| 2 | Layout under `docker/`, not repo root | No pollution of the existing clean root |
| 3 | Bind-mount default, named-volume documented | README: "never a network mount" — bind-mount is the only correct path |
| 4 | `ARCH=native` at build time | Linux Makefile autodetects via `gcc -dumpmachine`; same Dockerfile works on x86-64, aarch64 (Graviton), ppc64 |
| 5 | Single `docker-compose.yml` with a `gpu` profile | One file, one service, opt-in with `--profile gpu` |
| 6 | Wire `COLI_CUDA=1` and `CUDA_EXPERT_GB` into the GPU profile's env block | "Best for most users" per Q&A |
| 7 | Add a thin `Makefile` for friendly aliases | "Easy command to make the build" per Q&A — `make build`, `make up`, `make chat` |
| 8 | `coli serve` is the default `CMD` | Persistent engine, OpenAI-compatible API; chat and other subcommands come via `docker compose run` |
| 9 | Healthcheck is model-free (pings `/v1/models` only after the engine is up) | Model load is 30s+; a load-triggering healthcheck would fail every CI smoke test |

## Files to create (7 total, all under `docker/`)

### 1. [`docker/Dockerfile`](docker/Dockerfile) — CPU runtime

- **Base:** `ubuntu:24.04`
- **Apt packages:** `gcc`, `make`, `libc6-dev`, `libgomp1`, `python3`, `python3-pip` (unused but standard), `ca-certificates`
- **Build args:** `ARCH` (default `native`)
- **Steps:**
  1. `apt-get install` runtime + build deps in one layer
  2. `COPY c/ /opt/colibri/c/`
  3. `WORKDIR /opt/colibri/c`
  4. `make glm ARCH=$ARCH`
  5. `ENV COLI_MODEL=/models/glm52_i4 PATH=/opt/colibri/c:$PATH`
  6. `EXPOSE 8000`
  7. `ENTRYPOINT ["./coli"]`
  8. `CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]`
- **Expected size:** ~250 MB compressed
- **Key refs:** [`c/Makefile:102`](c/Makefile:102) for the Linux build path, [`c/coli:484`](c/coli:484) for the chat prompt format, [`c/coli:622`](c/coli:622) for `serve`

### 2. [`docker/Dockerfile.cuda`](docker/Dockerfile.cuda) — GPU runtime

- **Base:** `FROM Dockerfile as cpu-runtime` (only the build artifacts) + a `cuda-runtime` stage with `nvidia/cuda:12.8.0-devel-ubuntu24.04`
- **Why not multi-stage from scratch:** the CUDA image is large and the `gcc` it ships is older; we want both: the build tools from the slim image, the CUDA toolkit from the NVIDIA image. Two-stage with `COPY --from=cpu-runtime` is cleanest.
- **Apt packages:** `cuda-toolkit-12-8` (or use the preinstalled toolkit from `nvidia/cuda:*-devel`), `libcudart12`
- **Steps:**
  1. `FROM cpu-runtime AS cpu-build`
  2. `FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS cuda-runtime`
  3. Install engine's runtime deps (gcc, libgomp, python3)
  4. `COPY --from=cpu-build /opt/colibri/c/glm /opt/colibri/c/coli /opt/colibri/c/openai_server.py /opt/colibri/c/resource_plan.py /opt/colibri/c/doctor.py /opt/colibri/c/tools /opt/colibri/c/`
  5. Rebuild `glm` with `make CUDA=1` against the toolkit's headers (because we didn't copy the `.o` and the CUDA path needs the host's CUDA libs at link time)
  6. `ENV CUDA_HOME=/usr/local/cuda COLI_CUDA=1`
  7. Same `ENTRYPOINT` / `CMD`
- **Expected size:** ~5 GB compressed (heavy; the price of the toolkit)

### 3. [`docker/docker-compose.yml`](docker/docker-compose.yml) — service definition

```yaml
name: colibri
services:
  coli:
    profiles: ["default"]
    build:
      context: ..
      dockerfile: docker/Dockerfile
      args: { ARCH: ${ARCH:-native} }
    image: colibri:1.0
    restart: unless-stopped
    init: true
    ulimits:
      nofile: { soft: 65536, hard: 65536 }
    # cpuset: "0-11"     # EDIT THIS — physical cores only, no hyperthreads
    mem_limit: 30g         # EDIT THIS — engine auto-caps at 88% of MemAvailable
    ports: ["8000:8000"]
    volumes:
      - ./models:/models/glm52_i4:rw   # EDIT THIS — point at your int4 model dir
    healthcheck:
      test: ["CMD", "python3", "/opt/colibri/docker/healthcheck.py"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 120s   # engine load is 30s+ on slow disks

  coli-gpu:
    profiles: ["gpu"]
    build:
      context: ..
      dockerfile: docker/Dockerfile.cuda
      args: { ARCH: ${ARCH:-native} }
    image: colibri:1.0-cuda
    restart: unless-stopped
    init: true
    ulimits: { nofile: { soft: 65536, hard: 65536 } }
    mem_limit: 30g
    ports: ["8000:8000"]
    volumes:
      - ./models:/models/glm52_i4:rw
    environment:
      COLI_CUDA: "1"
      CUDA_EXPERT_GB: "auto"     # fill each device up to free VRAM minus dense + 2GB headroom
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "python3", "/opt/colibri/docker/healthcheck.py"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 120s
```

Commented-out named-volume variant included for users who want `coli_model` named volume seeded once.

### 4. [`docker/Makefile`](docker/Makefile) — friendly aliases

| Target | Underlying command | Purpose |
|---|---|---|
| `build` | `docker build -t colibri:1.0 -f Dockerfile ..` | Build the CPU image |
| `build-gpu` | `docker build -t colibri:1.0-cuda -f Dockerfile.cuda ..` | Build the CUDA image |
| `up` | `docker compose up -d coli` | Start CPU service in the background |
| `up-gpu` | `docker compose --profile gpu up -d coli-gpu` | Start GPU service |
| `down` | `docker compose down` | Stop & remove containers |
| `logs` | `docker compose logs -f coli` | Tail the engine's status |
| `chat` | `docker compose run --rm coli chat` | Interactive chat (TTY passthrough) |
| `info` | `docker compose run --rm coli info` | One-shot info dump, no model load |
| `plan` | `docker compose run --rm coli plan` | RAM/VRAM plan, no model load |
| `doctor` | `docker compose run --rm coli doctor` | Read-only validation, no model load |
| `shell` | `docker compose exec coli bash` | Drop into a running container |
| `rebuild` | `down` + `build --no-cache` | Force a clean rebuild |
| `clean` | `docker rmi colibri:1.0` | Remove the CPU image |

Variables: `ARCH` (default `native`), `IMAGE` (default `colibri:1.0`).

### 5. [`docker/.dockerignore`](docker/.dockerignore) — slim build context

```
# Build context = the repo root; the Dockerfile is at docker/Dockerfile.
# Allow only what the Dockerfile COPYs.

# Whitelist: nothing else is needed.
*
!c/
!c/**
```

Plus a comment block explaining why.

### 6. [`docker/healthcheck.py`](docker/healthcheck.py) — stdlib container-alive check

```python
#!/usr/bin/env python3
"""stdlib-only healthcheck for the colibrì container.

Pings the OpenAI-compatible API. Container-alive check, NOT a model-readiness
check — the engine takes 30+ seconds to load the 370 GB model and any real
readiness probe would have to wait for that, which would falsely fail CI
smoke tests. We exit 0 if /v1/models answers, 1 otherwise.
"""
import sys, urllib.request, urllib.error, os

url = os.environ.get("COLI_HEALTHCHECK_URL", "http://localhost:8000/v1/models")
try:
    with urllib.request.urlopen(url, timeout=3) as r:
        sys.exit(0 if 200 <= r.status < 300 else 1)
except (urllib.error.URLError, urllib.error.HTTPError, OSError):
    sys.exit(1)
```

### 7. [`docker/README.md`](docker/README.md) — quick start + gotchas

**Sections:**
1. **Quick start** — `cd docker && make build && make up && make logs`
2. **Prerequisites** — Linux host, Docker 24+, NVIDIA Container Toolkit (GPU only), `make`, ~400 GB free on the host for the int4 model
3. **The four gotchas** (one paragraph each):
   - `ulimit nofile=65536` — engine mmaps 144+ shards
   - `cpuset-cpus` — pin to physical cores, not logical
   - Bind-mount from a **local** filesystem (not NFS/9p)
   - `mem_limit` is the engine's view of available RAM
4. **GPU profile** — `make build-gpu && make up-gpu`, env vars to tune
5. **Custom commands** — `make chat`, `make plan`, `make doctor`, `make shell`
6. **Troubleshooting**:
   - Container exits immediately → `make logs`
   - `EAGAIN` on mmap → `ulimit` still too low
   - Cold decode is slower than the host's `iobench` → bind-mount is a 9p export (Docker Desktop on macOS/Windows)
   - OOM-killer → lower `mem_limit` or `--ram` via env
7. **Architecture diagram** — host FS → bind-mount → container, with the three tiers (VRAM / RAM / disk)

## Topology diagram

```mermaid
flowchart LR
  subgraph Host["Linux host"]
    Nvme["ext4 NVMe\n~370 GB int4 model\n.coli_kv / .coli_usage"]
    Cores["physical cores\ncpuset-cpus=0-N"]
    OptionalGPU["NVIDIA driver\nContainer Toolkit"]
  end

  Nvme -- bind mount rw --> Container
  Cores -- cpuset --> Container
  OptionalGPU -- --gpus all --> ContainerGpu["coli-gpu\nprofile: gpu"]

  subgraph Container["coli container (CPU)"]
    Engine["glm\n(libgomp, AVX2)"]
    Cli["coli\n(python3 stdlib)"]
    Engine --- Cli
  end

  subgraph ContainerGpu["coli-gpu container (CUDA)"]
    EngineCuda["glm CUDA=1\n(libcudart)"]
    CliCuda["coli\npython3 stdlib"]
    EngineCuda --- CliCuda
  end

  Container -- :8000 --> Client["OpenAI client\n(curl, Open WebUI,\ncolibri web)"]
  ContainerGpu -- :8000 --> Client
```

## Out of scope (intentional)

- **No converter service** — user chose: convert on a separate host, no torch in the runtime image
- **No web UI bundling** — `coli web` is a separate decision; the OpenAI API on `:8000` is the integration point
- **No multi-stage build** — engine builds in ~30s, not worth the complexity
- **No ccache** — same reason
- **No `.env` file** — everything lives in `docker-compose.yml` for one-command onboarding
- **No cgroup memory swap limit** — cgroup v1 vs v2 differ; document the gotcha instead of guessing

## Validation plan (for the implementation phase)

1. **Static check** — `docker compose config` exits 0; YAML parses
2. **Build check** — `make build` succeeds, image is ~250 MB
3. **Engine self-test inside the image** — `docker run --rm colibri:1.0 info` (no model needed, validates engine binary)
4. **No-mount smoke** — `docker compose run --rm coli info` exits 0 with "model not found" (the expected error when the bind mount is empty)
5. **With-mount smoke** — `docker compose run --rm coli plan` produces a real plan; `docker compose run --rm coli doctor` exits 0
6. **Manual cold-decode sanity** — `make chat`, ask a one-token question, check `make logs` for non-zero tok/s and reasonable expert hit rate

Steps 4–6 require the user's real 370 GB model and their physical-core layout; the README documents the commands, and the healthcheck is the only automated gate that exercises the full stack.

## Open question (defaults to a sensible answer)

**CPU vs CUDA selector:** the plan uses a single `docker-compose.yml` with a `gpu` profile. The user can switch with `make up` (CPU) or `make up-gpu` (CUDA). Alternative was two services in one file or two files; the profile is the least YAML and the cleanest mental model. If the user prefers something different, the compose file is the only place to change.

## Implementation order

1. [`docker/.dockerignore`](docker/.dockerignore) — smallest, no dependencies
2. [`docker/healthcheck.py`](docker/healthcheck.py) — independent
3. [`docker/Dockerfile`](docker/Dockerfile) — references only `c/`
4. [`docker/Dockerfile.cuda`](docker/Dockerfile.cuda) — references `Dockerfile`
5. [`docker/docker-compose.yml`](docker/docker-compose.yml) — references both Dockerfiles
6. [`docker/Makefile`](docker/Makefile) — references compose
7. [`docker/README.md`](docker/README.md) — references everything; written last so it documents the actual files

## Total scope

- **Files:** 7 new files, all under `docker/`
- **No existing files modified**
- **No root-level changes**
- **No new dependencies on the host** (Docker + Make + NVIDIA Container Toolkit are standard)
