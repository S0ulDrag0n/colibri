"""A/B driver for the CUDA + SSD streaming plan.

For every row in `MODES`, this script runs the engine with that env, parses
the `REPLAY decode:` and `PROFILE: expert-disk ...` lines that the engine
already prints ([`c/glm.c:4350-4390`](../../c/glm.c:4350)), and writes a
single CSV row. The point is to give the per-item plan steps a stable
fixture, the same way `tools/benchmark_cuda_fixture.py` does for the dense
datapoint, but for the *streaming* changes the plan proposes.

Why a CSV (not a unit test): the per-step wins are ms-level, the noise
floor is the page cache, and a CSV can be diffed across PRs. The companion
unit test `c/tests/test_cuda_disk_replay.py` only asserts that
`parse_output` survives the current engine output format and that
bit-stability is preserved across the modes that should be bit-identical
(fast math off, default TC_INT4 path). A perf assertion would be
host-dependent; the README calls that out as a deliberate non-goal.

Usage:
    python tools/cuda_disk_replay.py --model <path-to-glm-bench-model> \\
        --engine ./glm --gpu 0 --runs 3 \\
        --csv cuda_disk_replay.csv

If the engine or fixture is missing, exit 77 so the surrounding `make`
target can skip cleanly (the same convention `test_backend_cuda.cu` uses
for the no-CUDA case).
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Reuse the existing regexes in `tools.benchmark_cuda_fixture` so the parse
# contract is one file (no drift between the dense bench and this streaming
# bench). The import path is `tools.*` when this script runs from the `c/`
# directory (the Makefile `cd`s there first).
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools.benchmark_cuda_fixture import parse_output  # noqa: E402
else:
    from .benchmark_cuda_fixture import parse_output  # pragma: no cover


# A row is one (name -> env-override) pair. The base env in `main()` clears
# knobs that interfere between modes (PIPE, URING, DIRECT, ...) so each row
# starts from a known baseline. The order is the same as the plan's
# implementation order so the CSV reads top-to-bottom the way the PR does.
MODES: list[tuple[str, dict[str, str]]] = [
    # -- baseline ---------------------------------------------------------
    ("baseline_cpu",            {}),
    # -- disk-only knobs (Section 2) -------------------------------------
    ("direct_only",             {"DIRECT": "1"}),
    ("pipe_only",               {"PIPE": "1"}),
    ("uring_only",              {"URING": "1"}),
    ("uring_direct",            {"URING": "1", "DIRECT": "1"}),
    ("prefetch_on",             {"PREFETCH": "1"}),
    ("uring_prefetch_direct",   {"URING": "1", "DIRECT": "1", "PREFETCH": "1"}),
    # -- CUDA knobs (Section 1) ------------------------------------------
    # COLI_CUDA_TC_INT4 is the env the existing test_backend_cuda.cu uses
    # to opt into the WMMA s4 path. Once 1a lands, the default-on
    # behaviour is tested implicitly by the rows above; until then these
    # two rows isolate the env opt-in.
    ("cuda_dense_only",         {"COLI_CUDA": "1", "COLI_GPU": "0", "CUDA_DENSE": "1"}),
    ("cuda_expert_tc",          {"COLI_CUDA": "1", "COLI_GPU": "0", "CUDA_EXPERT_GB": "2",
                                 "COLI_CUDA_TC_INT4": "1"}),
    ("cuda_expert_no_tc",       {"COLI_CUDA": "1", "COLI_GPU": "0", "CUDA_EXPERT_GB": "2"}),
    # -- the headline combo (post-1a) -----------------------------------
    ("cuda_full_stream",        {"COLI_CUDA": "1", "COLI_GPU": "0", "CUDA_EXPERT_GB": "2",
                                 "URING": "1", "DIRECT": "1", "PREFETCH": "1"}),
]


# Optional: the 1g MADV_DONTNEED and the 1c double-buffered H2D are
# gate-driven from inside the engine (no env), so once they land the
# existing rows pick them up automatically. New opt-in knobs (cudaGraph,
# TC_INT4 default, CUDA_FAST) get added here as env pairs, not as new
# rows, so the CSV stays one fixture wide.
EXTRA_ENVS_FOR_KNOB: dict[str, str] = {
    # "1c": "COLI_CUDA_H2D_DBLBUF=1",
    # "1d": "COLI_CUDA_GRAPH=1",
    # "1g": "COLI_CUDA_H2D_MADV=1",
}


@dataclass(frozen=True)
class Mode:
    name: str
    env: dict[str, str]

    def label(self) -> str:
        if not self.env:
            return self.name
        keys = ",".join(f"{k}={v}" for k, v in sorted(self.env.items()))
        return f"{self.name}[{keys}]"


def _build_base_env(model: Path, args: argparse.Namespace) -> dict[str, str]:
    """Strip the knobs that the matrix varies, then anchor the fixture.

    The dense bench does the same: it pops a fixed allowlist from
    `os.environ` and rebuilds from a clean slate. The list mirrors
    `c/tools/benchmark_cuda_fixture.py:54-58` plus every PIPE/URING/
    DIRECT/PREFETCH knob the plan varies.
    """
    base = os.environ.copy()
    for key in (
        # shared with the dense bench
        "COLI_CUDA", "COLI_GPU", "COLI_GPUS", "CUDA_EXPERT_GB",
        "PIN", "PIN_GB", "STATS", "TF", "REPLAY", "CUDA_DENSE",
        # this matrix varies these — strip so each row is what the row says
        "DIRECT", "PIPE", "URING", "PREFETCH", "DROP",
        "COLI_CUDA_TC_INT4", "COLI_CUDA_TC_MIN_ROWS",
        "COLI_CUDA_GRAPH", "COLI_CUDA_PIPE",
        "COLI_CUDA_PROFILE", "COLI_CUDA_H2D_DBLBUF", "COLI_CUDA_H2D_MADV",
    ):
        base.pop(key, None)
    base.update(
        SNAP=str(model),
        REF=str(model / "ref_glm.json"),
        REPLAY="1",
        OMP_NUM_THREADS=str(args.threads),
        OMP_PROC_BIND="spread",
        OMP_PLACES="cores",
        # the PROFILE line is opt-in; turn it on for the full matrix
        PROFILE="1",
    )
    return base


def _execute(engine: str, env: dict[str, str]) -> tuple[float, list[float], str]:
    """Run the engine once. The dense bench uses `4 4 4`; match it."""
    run = subprocess.run(
        [engine, "4", "4", "4"],
        env=env, text=True, capture_output=True, check=False, timeout=600,
    )
    if run.returncode != 0:
        raise RuntimeError(
            f"engine exited {run.returncode}\nstdout:\n{run.stdout}\nstderr:\n{run.stderr}"
        )
    speed, profile = parse_output(run.stdout, run.stderr)
    return speed, profile, run.stdout


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="path to a glm_bench_model fixture (e.g. glm_bench_i4)")
    p.add_argument("--engine", default="./glm")
    p.add_argument("--gpu", default="0")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    p.add_argument("--csv", default="cuda_disk_replay.csv")
    p.add_argument("--warmup", type=int, default=1, help="warmup runs per mode (discarded)")
    p.add_argument("--no-skip", action="store_true", help="fail instead of returning 77 when fixture/engine is missing")
    args = p.parse_args()

    engine = Path(args.engine).resolve()
    model = Path(args.model).resolve()
    if not engine.exists() or not model.exists() or not (model / "ref_glm.json").exists():
        if args.no_skip:
            print(f"missing engine or fixture: {engine} / {model}", file=sys.stderr)
            return 2
        print(f"cuda_disk_replay: skipped (engine={engine.exists()} fixture={model.exists()})", file=sys.stderr)
        return 77

    base = _build_base_env(model, args)
    modes: list[Mode] = [Mode(name=name, env=dict(env)) for (name, env) in MODES]
    # extra per-knob rows: one row per entry in EXTRA_ENVS_FOR_KNOB so each
    # opt-in knob (cudaGraph, H2D_MADV, ...) is visible by name in the CSV
    for knob_name, knob_env in EXTRA_ENVS_FOR_KNOB.items():
        modes.append(Mode(name=f"knob_{knob_name}", env=dict(knob_env)))

    # warmup
    for m in modes:
        for _ in range(args.warmup):
            _execute(str(engine), base | m.env)

    # measure
    rows: list[dict[str, object]] = []
    for run_i in range(args.runs):
        # round-robin so any stateful warm-up bias (page cache) is shared
        order = modes[run_i % len(modes):] + modes[:run_i % len(modes)]
        for m in order:
            try:
                speed, profile, _ = _execute(str(engine), base | m.env)
            except RuntimeError as e:
                print(f"[{m.label()}] failed: {e}", file=sys.stderr)
                return 3
            rows.append({
                "mode": m.label(),
                "run": run_i,
                "tok_s": speed,
                "disk_s": profile[0],
                "expert_matmul_s": profile[1],
                "attention_s": profile[2],
                "lm_head_s": profile[3],
                "other_s": profile[4],
            })

    # aggregate: per-mode median, write CSV
    by_mode: dict[str, list[dict[str, object]]] = {}
    for r in rows:
        by_mode.setdefault(str(r["mode"]), []).append(r)
    aggregate: list[dict[str, object]] = []
    for label, rs in by_mode.items():
        speeds = [float(r["tok_s"]) for r in rs]
        aggregate.append({
            "mode": label,
            "median_tok_s": statistics.median(speeds),
            "min_tok_s": min(speeds),
            "max_tok_s": max(speeds),
            "runs": len(speeds),
        })

    csv_path = Path(args.csv)
    with csv_path.open("w", newline="") as f:
        # the per-run section is what you diff between PRs; the aggregate
        # is the human-readable summary at the bottom of the file
        fieldnames = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
        f.write("\n# --- median per mode ---\n")
        agg_fields = ["mode", "median_tok_s", "min_tok_s", "max_tok_s", "runs"]
        w2 = csv.DictWriter(f, fieldnames=agg_fields)
        w2.writeheader()
        w2.writerows(aggregate)

    print(f"wrote {csv_path} with {len(rows)} runs across {len(by_mode)} modes")
    for a in aggregate:
        print(f"  {a['mode']:38s}  median={a['median_tok_s']:.3f} tok/s  "
              f"(min={a['min_tok_s']:.3f} max={a['max_tok_s']:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
