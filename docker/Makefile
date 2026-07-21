# colibrì — friendly aliases for the Docker workflow.
#
# Run from this directory: `cd docker && make build` etc.
#
# Variables (override on the command line or via the environment):
#   ARCH     - target CPU arch passed to the engine's Makefile (default: native)
#   IMAGE    - CPU image tag (default: colibri:1.0)
#   CUDA_IMAGE - CUDA image tag (default: colibri:1.0-cuda)
#   COMPOSE  - docker compose binary (default: docker compose)
#
# Examples:
#   make build                       # build the CPU image
#   make build ARCH=x86-64-v3        # build a portable CPU binary
#   make build-gpu                   # build the CUDA image
#   make up                          # start the CPU service
#   make up-gpu                      # start the GPU service
#   make chat                        # one-off interactive chat
#   make info                        # one-off info dump (no model load)
#   make plan                        # one-off RAM/VRAM plan (no model load)
#   make doctor                      # one-off diagnostic (no model load)
#   make logs                        # tail the running service
#   make down                        # stop the service
#   make rebuild                     # clean rebuild from scratch
#
# Why a Makefile? The engine's CLI is the integration point — a
# container running `coli serve` and a container running `coli chat`
# differ only in the CMD. Make targets let the README stay short:
#   "make build && make up"   is the entire onboarding.

# ---------------------------------------------------------------- variables
ARCH        ?= native
IMAGE       ?= colibri:1.0
CUDA_IMAGE  ?= colibri:1.0-cuda
COMPOSE     ?= docker compose

# Per-run env so the compose file can read MODEL_DIR and friends
# without requiring a .env file in the repo. Override on the command
# line: `make up MODEL_DIR=/nvme/glm52_i4`.
MODEL_DIR   ?= ./models

# Pass the variables into compose via the shell. `env` is the
# POSIX-portable way to set a variable for a single command.
export ARCH
export MODEL_DIR

# ---------------------------------------------------------------- meta
.PHONY: help
help:
	@echo "colibrì — Docker Makefile"
	@echo ""
	@echo "Build:"
	@echo "  make build        Build the CPU image ($(IMAGE))"
	@echo "  make build-gpu    Build the CUDA image ($(CUDA_IMAGE))"
	@echo ""
	@echo "Run (background):"
	@echo "  make up           Start the CPU service"
	@echo "  make up-gpu       Start the GPU service"
	@echo "  make down         Stop the service"
	@echo "  make logs         Tail the engine's logs"
	@echo "  make shell        Drop into a running container"
	@echo ""
	@echo "One-shot (run a single CLI subcommand, then exit):"
	@echo "  make chat         Interactive chat (TTY passthrough)"
	@echo "  make info         Show model, RAM, disk status"
	@echo "  make plan         Show RAM/VRAM plan"
	@echo "  make doctor       Run read-only validation"
	@echo "  make bench        Run quality benchmarks"
	@echo ""
	@echo "Maintenance:"
	@echo "  make rebuild      Force a clean rebuild"
	@echo "  make clean        Remove the local images"
	@echo "  make prune        Remove all colibri* containers, images, and volumes"
	@echo ""
	@echo "Variables (override on the command line):"
	@echo "  ARCH=$(ARCH)  IMAGE=$(IMAGE)  MODEL_DIR=$(MODEL_DIR)"

# ---------------------------------------------------------------- build
.PHONY: build
build:
	@echo ">>> building $(IMAGE)  (ARCH=$(ARCH))"
	$(COMPOSE) build coli

.PHONY: build-gpu
build-gpu:
	@echo ">>> building $(CUDA_IMAGE)  (ARCH=$(ARCH))"
	$(COMPOSE) --profile gpu build coli-gpu

# ---------------------------------------------------------------- run
.PHONY: up
up:
	@echo ">>> starting CPU service (image: $(IMAGE))"
	@echo "    bind-mount: $(MODEL_DIR) -> /models/glm52_i4"
	@if [ "$(MODEL_DIR)" = "./models" ] && [ ! -d "./models" ]; then \
		echo ""; \
		echo "  WARNING: MODEL_DIR is the placeholder './models' which does not exist."; \
		echo "  Edit docker-compose.yml or run:  make up MODEL_DIR=/path/to/glm52_i4"; \
	fi
	$(COMPOSE) up -d coli

.PHONY: up-gpu
up-gpu:
	@echo ">>> starting GPU service (image: $(CUDA_IMAGE))"
	@echo "    bind-mount: $(MODEL_DIR) -> /models/glm52_i4"
	@if [ "$(MODEL_DIR)" = "./models" ] && [ ! -d "./models" ]; then \
		echo ""; \
		echo "  WARNING: MODEL_DIR is the placeholder './models' which does not exist."; \
		echo "  Edit docker-compose.yml or run:  make up-gpu MODEL_DIR=/path/to/glm52_i4"; \
	fi
	$(COMPOSE) --profile gpu up -d coli-gpu

.PHONY: down
down:
	$(COMPOSE) down

.PHONY: logs
logs:
	$(COMPOSE) logs -f coli

.PHONY: shell
shell:
	@if $(COMPOSE) ps --services --filter status=running | grep -q .; then \
		$(COMPOSE) exec coli bash; \
	else \
		echo "no coli service is running. start one with: make up"; \
		exit 1; \
	fi

# ---------------------------------------------------------------- one-shot
# These run a single CLI subcommand in a fresh container (--rm removes
# the container on exit). They use `docker compose run` (NOT `exec`)
# because the one-shot subcommands don't need a long-running service.
# The `info`, `plan`, and `doctor` subcommands do NOT load the model,
# so they're safe to run before the model dir is mounted correctly
# (they'll just report "model not found" — the same error you'd get
# running them on the host without a model).
.PHONY: chat
chat:
	$(COMPOSE) run --rm coli chat

.PHONY: info
info:
	$(COMPOSE) run --rm coli info

.PHONY: plan
plan:
	$(COMPOSE) run --rm coli plan

.PHONY: doctor
doctor:
	$(COMPOSE) run --rm coli doctor

.PHONY: bench
bench:
	$(COMPOSE) run --rm coli bench

# ---------------------------------------------------------------- hf auto-fetch
# Pre-populate the bind-mount with a model from Hugging Face WITHOUT
# starting a long-running service. Useful for slow links (the
# ~370 GB int4 model can take hours) and for offline CI that just
# needs the model on disk.
#
# The env vars are forwarded straight through; see plans/hf-auto-download.md.
.PHONY: fetch
fetch:
	@if [ -z "$$COLI_HF_REPO" ]; then \
		echo "set COLI_HF_REPO first. example:"; \
		echo "  export COLI_HF_REPO=cpatonn/GLM-5.2-Colibri-int4"; \
		echo "  export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx"; \
		echo "  make fetch"; \
		exit 1; \
	fi
	@echo ">>> fetching $$COLI_HF_REPO into $$MODEL_DIR"
	$(COMPOSE) run --rm \
		-e COLI_HF_REPO -e HF_TOKEN -e COLI_HF_REVISION -e COLI_HF_WORKERS -e COLI_HF_EXPECT_GB \
		--entrypoint '/usr/bin/env python3 /opt/colibri/docker/hf_fetch.py' \
		coli

# ---------------------------------------------------------------- maintenance
.PHONY: rebuild
rebuild: down
	@echo ">>> forcing clean rebuild (no cache)"
	$(COMPOSE) build --no-cache coli

.PHONY: clean
clean:
	-docker rmi $(IMAGE) 2>/dev/null || true
	-docker rmi $(CUDA_IMAGE) 2>/dev/null || true
	@echo "removed local colibri images (volumes preserved)"

.PHONY: prune
prune: down
	-docker rmi $(IMAGE) 2>/dev/null || true
	-docker rmi $(CUDA_IMAGE) 2>/dev/null || true
	-docker volume rm colibri_state 2>/dev/null || true
	@echo "removed images and the coli_state named volume"
	@echo "WARNING: this wipes the .coli_kv cache. conversations reopen cold."
