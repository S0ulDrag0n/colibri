"""Unit tests for the CUDA + SSD streaming replay harness.

These cover the parse contract and the env-stripping logic so the
benchmark driver stays safe to refactor. The actual end-to-end run is
a separate Makefile target (`cuda-disk-replay`) that needs a fixture
on disk and the engine built with `CUDA=1`; CI doesn't have either,
so the bench is opt-in and the unit tests run on every PR.

Why a parser-only test (not a perf assertion): the plan's
"Validation plan" section calls this out as deliberate — a perf
assertion would be host-dependent and the README's #101 correction
culture is explicit that we should not bake numbers in.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

# Run from the `c/` directory the same way `make test-python` does
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.benchmark_cuda_fixture import parse_output  # noqa: E402
from tools.cuda_disk_replay import MODES, _build_base_env  # noqa: E402


SAMPLE_HISTORIC = """
REPLAY decode: 4 tokens | 12.34 tok/s
PROFILE: expert-disk 1.25s | expert-matmul 2.50s | attention 0.75s | lm_head 0.10s | other -0.05s
"""

SAMPLE_CURRENT = """
REPLAY decode: 4 tokens in 0.324s | 12.34 tok/s | expert hit 87.5%
PROFILE: expert-disk 0.123s service / 0.045s wait | expert-matmul 2.50s | attention 0.75s | lm_head 0.10s | other -0.05s
"""


class ParseContractTest(unittest.TestCase):
    def test_historic_format(self) -> None:
        speed, profile = parse_output(SAMPLE_HISTORIC)
        self.assertAlmostEqual(speed, 12.34, places=4)
        self.assertEqual(profile, [1.25, 2.5, 0.75, 0.1, -0.05])

    def test_current_format_with_service_wait(self) -> None:
        # engine profile_print split disk into service + wait in late 2025
        # ([`c/glm.c:4350`](../../c/glm.c:4350)); the parser handles both.
        speed, profile = parse_output(SAMPLE_CURRENT)
        self.assertAlmostEqual(speed, 12.34, places=4)
        # disk = service + wait = 0.123 + 0.045 = 0.168
        self.assertAlmostEqual(profile[0], 0.168, places=4)
        self.assertEqual(len(profile), 5)

    def test_rejects_incomplete_output(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "benchmark output missing"):
            parse_output("REPLAY decode: 4 tokens | 12.34 tok/s", "engine failed")


class ModeMatrixTest(unittest.TestCase):
    """The MODES list is the contract for the CSV. Any change here is a
    visible behaviour change (rows appear / disappear in the benchmark
    output), so it gets a test rather than a docstring."""

    def test_baseline_row_present(self) -> None:
        names = [name for name, _ in MODES]
        self.assertIn("baseline_cpu", names)

    def test_all_section_1_and_2_rows_present(self) -> None:
        # The plan's Section 1 (CUDA) and Section 2 (disk) rows must all
        # be present. A missing row is a regression in the harness.
        names = {name for name, _ in MODES}
        for required in (
            "direct_only", "pipe_only", "uring_only",
            "prefetch_on", "cuda_dense_only", "cuda_expert_tc",
            "cuda_expert_no_tc", "cuda_full_stream",
        ):
            self.assertIn(required, names, f"missing mode row: {required}")

    def test_no_duplicate_mode_names(self) -> None:
        seen: set[str] = set()
        for name, _ in MODES:
            self.assertNotIn(name, seen, f"duplicate mode row: {name}")
            seen.add(name)

    def test_mode_envs_only_use_known_knobs(self) -> None:
        # The set of env-var names the matrix is allowed to set. Adding a
        # new knob is fine; the test forces you to update the list, which
        # is the documentation surface for "what does the harness vary?".
        allowed = {
            "DIRECT", "PIPE", "URING", "PREFETCH", "DROP",
            "COLI_CUDA", "COLI_GPU", "CUDA_DENSE", "CUDA_EXPERT_GB",
            "COLI_CUDA_TC_INT4",
        }
        for name, env in MODES:
            for k in env:
                self.assertIn(k, allowed, f"mode {name} sets unknown env {k}")


class BaseEnvTest(unittest.TestCase):
    """`_build_base_env` is the only place that *strips* the interfering
    knobs. If it forgets one, two rows will both look "on" and the CSV
    becomes a single-mode measurement. This test pins the strip list."""

    def setUp(self) -> None:
        # The base builder reads OMP_NUM_THREADS/--threads from the args
        # namespace; we don't need a real fixture to verify the strip.
        self._orig = os.environ.copy()
        # seed the env with one of every knob the plan varies; the
        # builder must clear them all so each row starts from a known
        # baseline.
        for k in (
            "COLI_CUDA", "COLI_GPU", "COLI_GPUS", "CUDA_EXPERT_GB",
            "PIN", "PIN_GB", "STATS", "TF", "REPLAY", "CUDA_DENSE",
            "DIRECT", "PIPE", "URING", "PREFETCH", "DROP",
            "COLI_CUDA_TC_INT4", "COLI_CUDA_TC_MIN_ROWS",
            "COLI_CUDA_GRAPH", "COLI_CUDA_PIPE",
            "COLI_CUDA_PROFILE", "COLI_CUDA_H2D_DBLBUF", "COLI_CUDA_H2D_MADV",
        ):
            os.environ[k] = "1"

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._orig)

    def _args(self) -> object:
        # minimal argparse.Namespace surrogate
        class A:
            threads = 1
        return A()

    def test_strips_matrix_knobs(self) -> None:
        env = _build_base_env(Path("/nonexistent/fixture"), self._args())
        for k in (
            "COLI_CUDA", "COLI_GPU", "CUDA_EXPERT_GB",
            "DIRECT", "PIPE", "URING", "PREFETCH", "DROP",
            "COLI_CUDA_TC_INT4", "COLI_CUDA_TC_MIN_ROWS",
            "COLI_CUDA_GRAPH", "COLI_CUDA_PIPE",
            "COLI_CUDA_PROFILE", "COLI_CUDA_H2D_DBLBUF", "COLI_CUDA_H2D_MADV",
            "CUDA_DENSE",
        ):
            self.assertNotIn(k, env, f"base env kept matrix knob {k}")

    def test_anchors_replay_and_fixture(self) -> None:
        # the harness runs on Windows too (Path() is OS-aware), so the
        # assertion has to use the platform-native separator rather than
        # the POSIX form, otherwise the test only passes on Linux.
        fixture = Path(__file__).resolve().parent.parent / "glm_bench_i4"
        env = _build_base_env(fixture, self._args())
        self.assertEqual(env.get("REPLAY"), "1")
        self.assertEqual(env.get("SNAP"), str(fixture))
        self.assertEqual(env.get("REF"), str(fixture / "ref_glm.json"))
        self.assertEqual(env.get("PROFILE"), "1")
        self.assertEqual(env.get("OMP_NUM_THREADS"), "1")


if __name__ == "__main__":
    unittest.main()
