#!/usr/bin/env python3
"""Idempotent Hugging Face -> COLI_MODEL snapshot download.

Invoked by the entrypoint wrapper when COLI_HF_REPO is set and the
target directory has no config.json. Uses huggingface_hub.snapshot_download
with:

  - token = $HF_TOKEN (if set, otherwise the public anonymous quota)
  - revision = $COLI_HF_REVISION (default: 'main')
  - allow_patterns = the file types the engine can actually read
  - max_workers = $COLI_HF_WORKERS (default 8)

Exits non-zero on any failure. The entrypoint does NOT catch — a half-
downloaded model is worse than no model, so we hard-fail and let the
user investigate. hf_hub's own .incomplete files make this resumable on
the NEXT container start (just rerun with the same env).

Why a separate script and not inline in entrypoint.sh:
  - hf_hub needs Python; the engine is a separate C binary anyway
  - this script can be tested in isolation (run it on the host)
  - the env-var parsing + free-space check is easier in Python
"""
import os
import shutil
import sys

REPO  = os.environ.get("COLI_HF_REPO")
DEST  = os.environ.get("COLI_MODEL", "/models/glm52_i4")
REV   = os.environ.get("COLI_HF_REVISION", "main")
TOKEN = os.environ.get("HF_TOKEN") or None
WORKERS = int(os.environ.get("COLI_HF_WORKERS", "8"))
EXPECT_GB = int(os.environ.get("COLI_HF_EXPECT_GB", "450"))

if not REPO:
    print("error: COLI_HF_REPO is not set", file=sys.stderr)
    sys.exit(2)

# Pre-flight: the destination must exist (it's the bind-mount). If the
# user forgot to set up the bind-mount, fail loud.
if not os.path.isdir(DEST):
    print(f"error: COLI_MODEL={DEST} is not a directory; bind-mount it first",
          file=sys.stderr)
    sys.exit(4)

# Pre-flight: free space. The defaults target the int4 model (~372 GB)
# with 20% headroom for shard .incomplete sidecars. Users pulling FP8
# (~756 GB) should set COLI_HF_EXPECT_GB=820.
try:
    free_gb = shutil.disk_usage(DEST).free / 1e9
except FileNotFoundError:
    print(f"error: COLI_MODEL={DEST} does not exist", file=sys.stderr)
    sys.exit(4)
if free_gb < EXPECT_GB:
    print(f"error: {free_gb:.0f} GB free at {DEST}; need ~{EXPECT_GB} GB",
          file=sys.stderr)
    print(f"hint: free up space, mount a bigger disk, or set COLI_HF_EXPECT_GB={int(free_gb*0.9)}",
          file=sys.stderr)
    sys.exit(3)

# hf_hub prints a tqdm progress bar by default. In a container that
# flood of " 12%|█▍        | 3.2G/27G" lines drowns the engine's startup
# log. Suppress it; users can re-enable with COLI_HF_PROGRESS=1.
if not os.environ.get("COLI_HF_PROGRESS"):
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Use the classic HTTP path. hf_xet has known issues with container
# network namespaces (matches c/tools/convert_fp8_to_int4.py:413).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

print(f"[hf_fetch] repo={REPO}  rev={REV}  -> {DEST}  ({free_gb:.0f} GB free)")
if TOKEN:
    print(f"[hf_fetch] using HF_TOKEN from env (private/gated access enabled)")

# Import AFTER env setup so the library sees the right config.
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("error: huggingface_hub is not installed in this image", file=sys.stderr)
    print("       (the CPU/CUDA Dockerfile should pip-install it)", file=sys.stderr)
    sys.exit(5)

# allow_patterns: only the file types the engine actually reads. This
# avoids pulling huge .bin or .gguf sidecars that some repos have.
snapshot_download(
    repo_id=REPO,
    repo_type="model",
    revision=REV,
    local_dir=DEST,
    token=TOKEN,
    allow_patterns=[
        "*.safetensors",
        "*.json",
        "*.txt",
        "*.model",
        "*.tiktoken",
        "tokenizer*",
    ],
    max_workers=WORKERS,
)

# Sanity check: config.json must exist after the download. If it doesn't,
# the repo is missing the file the engine needs.
if not os.path.isfile(os.path.join(DEST, "config.json")):
    print(f"error: download finished but {DEST}/config.json is missing", file=sys.stderr)
    print(f"       (is {REPO} a model repo? snapshot_download expects 'model' type)",
          file=sys.stderr)
    sys.exit(6)

# Report what landed so the engine's startup log is easy to correlate.
total_bytes = 0
shard_count = 0
for root, _dirs, files in os.walk(DEST):
    for f in files:
        p = os.path.join(root, f)
        try:
            total_bytes += os.path.getsize(p)
            if f.endswith(".safetensors"):
                shard_count += 1
        except OSError:
            pass
print(f"[hf_fetch] done: {shard_count} shards, {total_bytes/1e9:.1f} GB total, in {DEST}")
