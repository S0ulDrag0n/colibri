# colibrì in Docker

Run a 744B-parameter GLM-5.2 MoE model in a container, on a Linux host,
with an opt-in GPU tier. The container is small (no PyTorch at runtime),
the engine is built once at image-build time, and the entire workflow is
five commands.

```
cd docker
make build         # ~30 s, ~250 MB image
make up            # starts coli serve in the background
make logs          # watch the 744B wake up
curl http://localhost:8000/v1/models
```

That's the entire onboarding. Everything below is "what to change if
your machine isn't the default" and "what to do when something goes
wrong."

## Table of contents

- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [The four gotchas](#the-four-gotchas)
- [One-shot commands](#one-shot-commands)
- [GPU tier](#gpu-tier)
- [Storage tiers explained](#storage-tiers-explained)
- [Troubleshooting](#troubleshooting)
- [How the files fit together](#how-the-files-fit-together)

## Prerequisites

| What | Why | Notes |
|---|---|---|
| Linux host | The engine's I/O and OpenMP story | macOS / Windows Docker Desktop is not supported (the bind-mount goes through 9p/virtiofs, which makes cold decode 10–100× slower) |
| Docker 24+ | `init: true`, the new healthcheck syntax | |
| Docker Compose v2 | the `deploy.resources` GPU syntax | `docker compose version` should report v2.x |
| `make` | the friendlier command surface | the file is plain GNU make, no extensions |
| ~250 MB free | CPU image | CUDA image is ~5 GB |
| ~400 GB free **on the host**, not in the container | the int4 model directory | ext4/xfs/btrfs on a local NVMe; bind-mounted into the container at `/models/glm52_i4` |
| NVIDIA driver + Container Toolkit | **only for the GPU tier** | `nvidia-container-cli --version` should print something |

## Quick start

1. **Edit the bind-mount path** in [`docker-compose.yml`](docker-compose.yml). The line is marked with `EDIT THIS`. Point it at the directory containing your `glm52_i4` int4 model on the host.

   ```yaml
   volumes:
     - /nvme/glm52_i4:/models/glm52_i4:rw
   ```

2. **Build and run** with the Makefile:

   ```bash
   make build           # build colibri:1.0 (CPU, ~30 s)
   make up              # start coli serve in the background
   make logs            # tail the engine's status
   ```

3. **Probe**:

   ```bash
   curl http://localhost:8000/v1/models
   # {"data":[{"id":"glm-5.2-colibri", ...}]}

   curl http://localhost:8000/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"glm-5.2-colibri","messages":[{"role":"user","content":"Hello"}]}'
   ```

4. **Stop and remove**:

   ```bash
   make down
   ```

The Makefile is a thin wrapper. Everything it does is documented in [`Makefile`](Makefile), and the underlying compose commands are visible in the output. Override anything from the command line:

```bash
make build ARCH=x86-64-v3                       # portable binary
make up MODEL_DIR=/mnt/nvme/glm52_i4            # different host path
make up-gpu CUDA_EXPERT_GB=32                   # pin a VRAM budget
```

## The four gotchas

These are the four things that have to be right or the engine silently
performs badly. They're set correctly in [`docker-compose.yml`](docker-compose.yml) by
default, but you may need to uncomment / change them for your host.

### 1. `ulimit nofile` must be ≥ 65536

The engine mmaps 144+ safetensors shards at startup and bumps
`RLIMIT_NOFILE` to 65536 from the Python CLI
([`c/coli:28`](../c/coli:28)). cgroups can clip the inherited value, so we
set the ulimit explicitly in the compose file. **Don't lower it.** If
you see `EAGAIN` or `Too many open files` in `make logs`, the ulimit is
the cause.

### 2. `cpuset-cpus` must pin to physical cores only

The engine's quantized matmul kernels are memory-bound; SMT siblings
share execution units that these kernels already saturate, and OpenMP
threads scheduled on a hyperthread's sibling of a busy physical core
stall. Use physical cores only:

```bash
lscpu | grep '^Core(s) per socket'   # e.g. 12
# Uncomment and set the cpuset line in docker-compose.yml:
#   cpuset: "0-11"
```

### 3. The model directory must be a local filesystem bind-mount

Cold decode reads ~11 GB of expert bytes per token, randomly across 144+
shards. The README is explicit: **never a network/9p mount**. The
default `MODEL_DIR=./models` in [`docker-compose.yml`](docker-compose.yml) is a relative
bind-mount — point it at `/nvme/...` or `/mnt/nvme/...` (an ext4/xfs
NVMe partition) on the host.

Docker Desktop on macOS/Windows uses virtiofs for bind-mounts, which
makes this workload 10–100× slower. Run colibrì on Linux natively, or
on a remote Linux host via SSH + `docker compose up`.

### 4. `mem_limit` is the engine's view of available RAM

The engine sizes the expert cache to 88% of `MemAvailable`. Inside a
container, `MemAvailable` is the cgroup's view, which is the value of
`mem_limit`. **Set `mem_limit` to the amount of RAM you want the engine
to use, not your host's total RAM.** If your host has 128 GB but you
want the engine to stay at 30 GB (leaving room for the page cache, the
`.coli_kv` file, etc.), set `mem_limit: 30g`.

## One-shot commands

The Makefile has aliases for each CLI subcommand. They use `docker
compose run --rm`, so the container is created, runs the command, and
is removed on exit. Useful for diagnostics that don't need a
long-running service:

| Target | Subcommand | Loads the model? | Notes |
|---|---|---|---|
| `make info` | `coli info` | no | model directory, RAM, disk, engine status |
| `make plan` | `coli plan` | no (reads safetensors headers) | RAM/VRAM plan |
| `make doctor` | `coli doctor` | no | read-only validation, returns a versioned JSON report |
| `make chat` | `coli chat` | yes | interactive REPL, TTY passthrough |
| `make bench` | `coli bench` | yes | quality benchmarks (hellaswag, arc, mmlu) |

`info`, `plan`, and `doctor` are the right starting points when
debugging a setup issue — they run without loading the 370 GB model and
report exactly what's wrong (or right).

## GPU tier

To enable the CUDA expert tier:

```bash
make build-gpu        # build colibri:1.0-cuda (~5 GB image)
make up-gpu           # start with --gpus all via NVIDIA Container Toolkit
```

This is a separate service (`coli-gpu`) on the `gpu` profile. The
compose file wires in `COLI_CUDA=1` and `CUDA_EXPERT_GB=auto` by
default; the `auto` value tells the engine to fill each device up to
free VRAM minus the projected dense tensors and 2 GB of runtime
headroom ([`c/resource_plan.py`](../c/resource_plan.py)).

Useful overrides:

```bash
# Pin a specific VRAM budget across all selected devices
make up-gpu CUDA_EXPERT_GB=32

# Pin specific GPU indices (default: all visible)
NVIDIA_VISIBLE_DEVICES=0,2,4 make up-gpu

# Also distribute dense tensors round-robin across GPUs (advanced)
# Set in docker-compose.yml environment block: CUDA_DENSE: "1"
```

The community benchmark table in the main README shows that the CUDA
expert tier earns its VRAM only when the CPU is the weak link — on a
Ryzen 9 9950X3D with AVX-512 the AVX-512 CPU matmul already matches
the 5090, so `COLI_CUDA=1` buys ~0% there. Measure before assuming.

## Auto-download from Hugging Face

The container can populate `COLI_MODEL` from a Hugging Face repo on first
start, so you don't have to convert or rsync a 370 GB model to the host
manually. The behaviour is opt-in: when `COLI_HF_REPO` is unset, the
container does nothing extra and behaves exactly as before.

| Env var | Required? | Default | What it does |
|---|---|---|---|
| `COLI_HF_REPO` | yes, to trigger download | unset → no-op | HF repo id, e.g. `zai-org/GLM-5.2-FP8` |
| `HF_TOKEN` | only for private/gated repos | unset | forwarded to `huggingface_hub` |
| `COLI_HF_REVISION` | no | `main` | pin to a specific commit (e.g. `abc1234`) |
| `COLI_HF_WORKERS` | no | `8` | parallel shard downloads |
| `COLI_HF_EXPECT_GB` | no | `450` | free-space gate before the download starts |

### One-shot: pull a model into a bind-mount

The simplest way to get a model onto the host without a long-running
container is `make fetch`:

```bash
cd docker
export COLI_HF_REPO=zai-org/GLM-5.2-FP8
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx   # only if the repo is gated
make fetch
```

`make fetch` runs the downloader in a one-off container, populates the
`MODEL_DIR` bind-mount, and exits. The next `make up` finds the model
already there and skips the download.

### Run-and-fetch: download and serve in the same `docker run`

```bash
docker run --rm -it \
    -v /nvme/glm52_i4:/models/glm52_i4:rw \
    -p 8000:8000 \
    -e COLI_HF_REPO=zai-org/GLM-5.2-FP8 \
    -e HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx \
    s0uldrag0n/colibri:1.0-cuda
```

The first start downloads the model; subsequent starts detect
`config.json` and skip the download.

### How the download behaves

- **Resume**: hf_hub writes `.incomplete` sidecars. Killing the
  container mid-download and restarting with the same env resumes from
  the last completed shard. No need to re-pull what's already there.
- **Disk check**: the entrypoint refuses to start the download if
  `${COLI_MODEL}` has less than `COLI_HF_EXPECT_GB` (default 450 GB)
  free. Bump it to 820 when pulling the FP8 source.
- **Pre-quantised int4 default**: the example above pulls the
  pre-quantised int4 model (~372 GB). The engine serves it directly
  with no `coli convert` step.
- **FP8 source**: if you point `COLI_HF_REPO` at `zai-org/GLM-5.2-FP8`
  (~756 GB) you also need to convert it. The image does NOT auto-run
  `coli convert` (that needs `torch` and a multi-hour run). After the
  download finishes:
  ```bash
  docker compose run --rm coli convert --model /models/glm52_i4
  ```
  This step is intentionally manual — you don't want a freshly-deployed
  container to surprise you with a 4-hour quantisation job.

### Why the int4 default

The pre-quantised int4 model is what the engine consumes natively. The
FP8 source requires a torch-based conversion (`tools/convert_fp8_to_int4.py`,
~1.5-2 hours on a fast CPU). For most users the int4 model is the
right starting point; FP8 is there if you want a different quantisation
scheme (`int4-g128` etc., see [`tools/quant_ablation.py`](../c/tools/quant_ablation.py)).

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `error: COLI_MODEL=... is not a directory` | bind-mount not set up | add `-v /path/on/host:/models/glm52_i4:rw` (see [gotcha #3](#3-the-model-directory-must-be-a-local-filesystem-bind-mount)) |
| `error: only N GB free; need ~M GB` | disk budget not met | `COLI_HF_EXPECT_GB=N-1 make fetch` (or free space) |
| `error: COLI_HF_REPO is not set` | typo in env var name | check the case (`COLI_HF_REPO`, not `HF_REPO` or `HUGGINGFACE_REPO`) |
| Download hangs at 0% | network/firewall issue | try `HF_ENDPOINT=https://hf-mirror.com` (already respected by `huggingface_hub`) |
| `401 Unauthorized` | private repo, missing token | `export HF_TOKEN=...` and rerun |

## Storage tiers explained

The engine treats VRAM, RAM, and disk as one managed memory hierarchy.
The container exposes the three tiers as follows:

| Tier | Source | In the container |
|---|---|---|
| **VRAM** (hot) | optional, GPU profile | `CUDA_EXPERT_GB=auto` env var; the engine manages it directly |
| **RAM** (warm) | the cgroup's `mem_limit` | engine auto-caps the expert cache at 88% of `MemAvailable` |
| **Disk** (cold) | the bind-mounted model directory | `/models/glm52_i4`; ~370 GB of int4 expert weights |

The auto-tier pipeline (`make plan`) reports the projected placement
before the model is loaded. The `make doctor` command validates that
the chosen placement is runnable (it never starts the engine).

## Troubleshooting

### Container exits immediately

```bash
make logs              # see the engine's last words
make doctor            # run the read-only validation
```

`doctor` is the right tool here — it checks the model directory, the
config, the tokenizer, the engine binary, the cgroup's view of
available RAM, the requested GPU devices, and the placement budget
without starting inference. Failures have stable check IDs you can
automate against.

### "killed by SIGKILL" in the logs

The kernel's OOM-killer fired because the cgroup's RSS exceeded
`mem_limit`. Lower `mem_limit`, or shorten the context (`CTX=2048`),
or use the `--ram` flag to set a tighter expert-cache cap.

### Cold decode is much slower than the host's `iobench`

The bind-mount is going through a slow filesystem. Check:

```bash
# Inside the container, confirm the mount is local
make shell
> df -T /models/glm52_i4
> mount | grep glm52
```

You should see `ext4` / `xfs` / `btrfs` and a device path starting with
`/dev/nvme...` or `/dev/sd...`. If you see `fuse.virtiofs` or
`/Volumes/...` (macOS), the bind-mount is going through Docker
Desktop's translator, which is too slow for this workload.

### Engine reports "could not mmap" or "EAGAIN" on startup

The ulimit is too low. Confirm:

```bash
make shell
> ulimit -n
65536
```

If you see a smaller number, the ulimit in `docker-compose.yml` isn't
taking effect (often because of a Compose-version mismatch).

### "I forgot to set the model path" warnings on `make up`

The Makefile warns when `MODEL_DIR` is the placeholder `./models` and
that directory doesn't exist. Set it explicitly:

```bash
make up MODEL_DIR=/path/to/glm52_i4
```

Or edit [`docker-compose.yml`](docker-compose.yml) so the default is your real path.

### Container is up but `curl` times out

Wait 30–90 seconds. The first request to `/v1/models` only succeeds
*after* the engine has finished loading the 370 GB model. The
healthcheck's `start_period: 120s` accounts for this; `docker compose
ps` will show `(healthy)` once the engine is ready.

## How the files fit together

```
docker/
├── Dockerfile         # CPU runtime: ubuntu + gcc (build) + libgomp + python3 (runtime)
├── Dockerfile.cuda    # CUDA runtime: nvidia/cuda base, builds with make CUDA=1
├── docker-compose.yml # one coli service + a coli-gpu service on the gpu profile
├── Makefile           # build/up/chat/info/plan/doctor/down/shell aliases
├── healthcheck.py     # stdlib ping to /v1/models (container-alive, not model-ready)
├── .dockerignore      # whitelists c/ to keep the build context small
└── README.md          # this file
```

The build context for both Dockerfiles is the **repo root** (so
`COPY c/ /opt/colibri/c/` resolves). The `.dockerignore` keeps the
context to the engine directory so the `docker build` doesn't try to
tarball the 370 GB model.

The CLI subcommands (`info`, `plan`, `doctor`, `chat`, `serve`, `bench`,
`convert`, `web`) are passed as the `CMD` — see the [Makefile](Makefile)
for the one-shot wrappers, or run them directly:

```bash
docker compose run --rm coli info
docker compose run --rm coli plan
docker compose run --rm coli doctor
docker compose run --rm coli chat
```

For the underlying engine documentation, see the main
[`README.md`](../README.md). For the CLI's subcommands and their
environment variables, see [`c/coli`](../c/coli).
