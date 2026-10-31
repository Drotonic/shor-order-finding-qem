#!/usr/bin/env python3
"""CPU model/source tests; --gpu additionally tests only tiny synthetic states."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np

from vendor import reset_readout as model
from run_experiment import load_configuration, read_cache, outcome_counts, verify_chunk, atomic_npz

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
PINS = {
    "simulator_reference.py": "3e00444360affc759d419e71145f3728fcdb77eb98283107b31c9c174a15f170",
    "folding.py": "438208ff61a0fb59202b3454d78d6c7db82cb302d0f7d15730b2ebd6c7aa4358",
    "metrics.py": "83259721f73f0ec9143be08af0e8161bd10a2790ce1e1489e43341554f7275c1",
    "estimators.py": "65cb00d076c871ccb1cb63ab9dc05a2f9ed5c0cd8d589a4e5f5fa5d35097b993",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(node):
    return ast.dump(node, include_attributes=False)


def context(**kwargs):
    return {"seed": 20261020, "N": 119, "a": 16, "gate_error": .00005,
            "noise_scale": 1., "round_index": 0, "qubit": 8, **kwargs}


class SetupTests(unittest.TestCase):
    def test_frozen_sources_unchanged(self):
        for name, digest in PINS.items():
            self.assertEqual(sha(HERE/"vendor"/name), digest)
            original_name = "simulator.py" if name == "simulator_reference.py" else name
            self.assertEqual(sha(PAPER/"shor_error_mitigation/src/shor_qem"/original_name), digest)

    def test_exact_backend_scope_and_zero_error_path(self):
        original = ast.parse((HERE/"vendor/simulator_reference.py").read_text())
        changed = ast.parse((HERE/"vendor/simulator_noisy_reset.py").read_text())
        # All gates, RNG helpers, measurement, ideal reset, and other engines
        # remain syntactically identical. Only one engine and one import change.
        added = [n for n in changed.body if isinstance(n, ast.ImportFrom) and n.module == "reset_readout"]
        self.assertEqual(len(added), 1)
        changed.body.remove(added[0])
        fn = next(n for n in changed.body if isinstance(n, ast.FunctionDef) and n.name == "_run_gpu_engine_chunked_shotwise")
        self.assertEqual([x.arg for x in fn.args.args[-2:]], ["arithmetic_reset_readout_error", "reset_error_seed"])
        self.assertEqual(fn.args.defaults[-2].value, 0.)
        fn.args.args = fn.args.args[:-2]; fn.args.defaults = fn.args.defaults[:-2]
        self.assertIn("_validate_reset_probability", ast.unparse(fn.body[0]))
        self.assertIn("reset_error_seed", ast.unparse(fn.body[1]))
        del fn.body[:2]
        injections = []
        for n in ast.walk(fn):
            if isinstance(n, ast.For) and ast.unparse(n.target) == "r" and ast.unparse(n.iter) == "range(q, 2 * q + 1)":
                for index, statement in enumerate(n.body):
                    if isinstance(statement, ast.If) and ast.unparse(statement.test) == "arithmetic_reset_readout_error > 0":
                        injections.append((n, index, statement))
        self.assertEqual(len(injections), 1)
        parent, index, injection = injections[0]
        self.assertIn("reset_qubit_batch", ast.unparse(parent.body[index-1]))
        self.assertIn("qubit=r", ast.unparse(injection))
        self.assertIn("round_index=i", ast.unparse(injection))
        self.assertIn("shot_offset=shot_offset", ast.unparse(injection))
        self.assertIn("apply_gate_1q(psi, X_MAT, r", ast.unparse(injection))
        parent.body.pop(index)
        self.assertEqual(dump(changed), dump(original), "Unapproved source change outside arithmetic reset injection")

    def test_rng_chunk_invariance_and_shared_thresholds(self):
        full = model.error_uniforms(shots=257, shot_offset=1000, **context())
        pieces = np.r_[model.error_uniforms(shots=100, shot_offset=1000, **context()),
                       model.error_uniforms(shots=157, shot_offset=1100, **context())]
        np.testing.assert_array_equal(full, pieces)
        self.assertTrue(np.all((full >= 0) & (full < 1)))
        masks = [model.error_mask(p, shots=257, shot_offset=1000, **context()) for p in (0, .0001, .001, .005, .01, .02, 1)]
        for low, high in zip(masks[:-1], masks[1:]):
            self.assertTrue(np.all(low <= high))
        self.assertTrue(np.all(masks[0] == 0)); self.assertTrue(np.all(masks[-1] == 1))

    def test_domain_and_target_validation(self):
        keys = {model.context_key(**context(round_index=i, qubit=q)) for i in range(14) for q in range(8, 17)}
        self.assertEqual(len(keys), 126)
        for q in (0, 1, 7, 17):
            with self.assertRaises(ValueError):
                model.context_key(**context(qubit=q))
        for p in (-.01, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                model.error_mask(p, shots=2, shot_offset=0, **context())
        with self.assertRaises(ValueError):
            model.error_uniforms(shots=2, shot_offset=-1, **context())

    def test_no_global_numpy_rng_consumption(self):
        np.random.seed(71)
        before = np.random.get_state()
        model.error_mask(.01, shots=100, shot_offset=0, **context())
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_conditional_x_equivalence_and_cptp(self):
        identity = np.eye(2); X = np.array([[0, 1], [1, 0]])
        for m in (0, 1):
            for e in (0, 1):
                measured = identity[:, m]
                left = np.linalg.matrix_power(X, m ^ e) @ measured
                right = np.linalg.matrix_power(X, e) @ np.linalg.matrix_power(X, m) @ measured
                np.testing.assert_array_equal(left, right)
        # Arbitrary entangled two-qubit input; reset the first subsystem.
        v = np.array([1+1j, 2-1j, -3j, 2+2j]); v /= np.linalg.norm(v)
        rho = np.outer(v, v.conj())
        reduced = rho[:2, :2] + rho[2:, 2:]
        for p in (0, .0001, .001, .005, .01, .02, 1):
            channel = np.zeros((4, 4), complex)
            for m in (0, 1):
                for e, weight in ((0, 1-p), (1, p)):
                    K = np.outer(identity[:, e], identity[:, m])
                    big = np.kron(K, identity)
                    channel += weight * big @ rho @ big.conj().T
            expected = np.kron(np.diag([1-p, p]), reduced)
            np.testing.assert_allclose(channel, expected, atol=1e-15)
            np.testing.assert_allclose(channel[:2, :2]+channel[2:, 2:], reduced, atol=1e-15)

    def test_probability_and_percent_configuration(self):
        c = load_configuration()
        self.assertEqual(c["reset_error_percent"], [0., .01, .1, .5, 1., 2.])
        np.testing.assert_array_equal(np.array(c["reset_error_percent"])/100, [0, .0001, .001, .005, .01, .02])
        self.assertEqual(c["arithmetic_qubits"], list(range(8, 17)))

    def test_bits_chunk_save_verify_and_corruption(self):
        c = load_configuration()
        bits = np.zeros((3, 14), dtype=np.int32)
        bits[0, 0] = 1; bits[1, 13] = 1; bits[2, :] = 1
        outcomes, counts = outcome_counts(bits)
        np.testing.assert_array_equal(outcomes, [1, 8192, 16383])
        with tempfile.TemporaryDirectory(prefix="reset-readout-test-") as folder:
            path = Path(folder)/"chunk.npz"
            audit = model.fault_counts(.01, shots=3, shot_offset=7, seed=c["reset_error_seed"],
                                       N=119, a=16, gate_error=.00005, noise_scale=1.)
            atomic_npz(path, run_identity=np.asarray("test"), reset_error_probability=np.float64(.01),
                       shot_offset=np.int64(7), shots=np.int64(3), outcomes=outcomes, counts=counts,
                       reset_fault_counts_by_round_qubit=audit)
            verify_chunk(path, identity="test", pm=.01, offset=7, size=3, config=c)
            with self.assertRaises(AssertionError):
                verify_chunk(path, identity="wrong", pm=.01, offset=7, size=3, config=c)
            with self.assertRaises(FileExistsError):
                atomic_npz(path, bogus=np.ones(1))

    def test_compiled_operation_cache(self):
        c = load_configuration()
        path = PAPER/"shor_error_mitigation"/c["operation_cache_relative"]
        read_cache(path, c)

    def test_runner_completion_resume_with_mock_backend(self):
        # Only test orchestration/storage with fake all-zero outcomes. This
        # does not execute a quantum circuit and writes solely to a temp dir.
        import vendor
        import run_experiment as runner
        calls = []
        def fake_engine(method, rates, lambdas, N, a, shots, ops, **kwargs):
            calls.append((shots, kwargs["shot_offset"], kwargs["arithmetic_reset_readout_error"]))
            return [(0, np.zeros((shots, 14), dtype=np.int32))]
        fake_core = types.SimpleNamespace(GLOBAL_SEED=42, USE_FP32=True,
                                          _run_gpu_engine_chunked_shotwise=fake_engine)
        fake_cp = types.SimpleNamespace(__version__="mock-no-gpu",
            cuda=types.SimpleNamespace(Device=lambda _: types.SimpleNamespace(use=lambda: None),
                runtime=types.SimpleNamespace(getDeviceProperties=lambda _: {"name": b"mock-no-gpu"},
                                             runtimeGetVersion=lambda: 0, driverGetVersion=lambda: 0)))
        with tempfile.TemporaryDirectory(prefix="reset-runner-mock-") as folder:
            output = Path(folder)/"run"
            args = ["--execute", "--shots-per-condition", "3", "--chunk-shots", "2",
                    "--reset-error-percent", "0", "1", "--output-dir", str(output)]
            with patch.dict("sys.modules", {"cupy": fake_cp}), \
                 patch.object(vendor, "simulator_noisy_reset", fake_core, create=True), \
                 patch.object(runner, "read_cache", return_value={}), \
                 patch("builtins.print"):
                self.assertEqual(runner.main(args), 0)
                self.assertEqual(len(calls), 4)
                self.assertEqual(runner.main(args+["--resume"]), 0)
                self.assertEqual(len(calls), 4, "Resume generated duplicate chunks")
                with self.assertRaises(FileExistsError):
                    runner.main(args)
            manifest = json.loads((output/"manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(len(manifest["chunks"]), 4)
            self.assertEqual(len(manifest["conditions"]), 2)


def gpu_tests():
    """No full circuit: two-qubit pure states and the frozen primitive kernels."""
    import cupy as cp
    from vendor import simulator_noisy_reset as engine
    X = engine.X_MAT
    for target in (0, 1):
        for basis in range(4):
            for flipped in (0, 1):
                state = cp.zeros((4, 1), dtype=engine.c_type); state[basis] = 1
                state = engine.reset_qubit_batch(state, target, 2, 4, pre_rand=cp.asarray([.4], dtype=engine.r_type))
                state = engine.apply_gate_1q(state, X, target, cond_mask=cp.asarray([flipped], dtype=cp.int32))
                expected = (basis & ~(1 << target)) | (flipped << target)
                np.testing.assert_allclose(cp.asnumpy(cp.abs(state[:, 0])**2), np.eye(4)[expected], atol=1e-6)
    cp.cuda.get_current_stream().synchronize()
    print("PASS GPU primitive test:16 synthetic two-qubit reset/error truth-table cases; no compiled circuit shots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SetupTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.gpu:
        gpu_tests()
