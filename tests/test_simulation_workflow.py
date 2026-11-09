"""Preparation and restart checks using synthetic CPU backends, never CUDA."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from shor_qem import simulation_workflow as workflow


CONFIG = {"N": 3, "a": 2, "rates": [0., .001], "scales": [1., 1.2, 2., 3.],
          "shots_per_scale": 5, "baseline_shots": 12, "batch_size": 3, "seed": 17}


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    def compile_fixture(config):
        operations = {1 << i: [("unitary", [0], [np.eye(2, dtype=np.complex64)]),
                               ("cz", [0, 1], []), ("rzz", [1, 2], [.25])]
                      for i in range(config["bits"])}
        return operations, "synthetic-test-fixture"
    monkeypatch.setattr(workflow, "_compile", compile_fixture)
    workflow.prepare(CONFIG, tmp_path / "prepared")
    return tmp_path / "prepared"


class CPUBackend:
    calls = []
    fail_after = None
    environment = {"backend": "synthetic CPU fixture; not quantum simulation"}

    def __init__(self, config):
        self.config = config

    def generate(self, config, rate, lam, reset, offset, size, operations):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("simulated interruption")
        self.calls.append((rate, lam, reset, offset, size))
        outcomes = (np.arange(offset, offset + size, dtype=np.uint32) + round(rate * 1e6)
                    + round(lam * 10) + int(reset)) % (1 << config["bits"])
        return (outcomes[:, None] >> np.arange(config["bits"], dtype=np.uint32)) & 1


@pytest.fixture
def backend(monkeypatch):
    CPUBackend.calls = []
    CPUBackend.fail_after = None
    monkeypatch.setattr(workflow, "_GPUBackend", CPUBackend)
    return CPUBackend


def _rewrite(path, **changes):
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    values.update(changes)
    np.savez_compressed(path, **values)


def test_prepare_and_plan_are_cpu_only_and_read_only(prepared, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("GPU backend must not initialize while planning")
    monkeypatch.setattr(workflow, "_GPUBackend", forbidden)
    before = sorted(p.relative_to(prepared).as_posix() for p in prepared.rglob("*"))
    result = workflow.run(prepared)
    assert list(result) == ["0", "0.001"]
    assert before == sorted(p.relative_to(prepared).as_posix() for p in prepared.rglob("*"))
    manifest = json.loads((prepared / "manifest.json").read_text())
    assert manifest["config"]["bits"] == 4
    assert manifest["total_two_qubit_gates"] == 8
    assert manifest["effective_lambdas"] == [1., 1., 2., 3.]
    assert "order" not in manifest["config"]
    assert "cupy" not in sys.modules
    assert "shor_qem.simulator" not in sys.modules


@pytest.mark.parametrize("changes,match", [
    ({"N": 4}, "odd"), ({"a": 3}, "gcd"), ({"a": True}, "integer"),
    ({"bits": 5}, "frozen GPU"), ({"order": 2}, "known order"),
    ({"r": 2}, "known order"), ({"scales": [1., 2., 3.]}, "four"),
    ({"scales": [1., 1.01, 2., 3.]}, "collide"),
    ({"scales": [1., 2., 1.2, 3.]}, "increasing"),
    ({"rates": [.001, .00100000001]}, "collide"),
    ({"rates": [.001, float("nan")]}, "JSON"),
    ({"rates": [-.01]}, "finite"), ({"rates": []}, "nonempty"),
    ({"shots_per_scale": 0}, "integer"), ({"baseline_shots": 4}, "integer"),
    ({"baseline_shots": 2**30}, "signed32"), ({"batch_size": 0}, "integer"),
    ({"seed": 2**32}, "unsigned32"), ({"gpu_id": -1}, "integer"),
    ({"N": 4095, "a": 2}, "resource guard"),
])
def test_invalid_config_rejected_before_compilation(tmp_path, monkeypatch, changes, match):
    def forbidden(*args):
        raise AssertionError("invalid configuration reached the compiler")
    monkeypatch.setattr(workflow, "_compile", forbidden)
    with pytest.raises(ValueError, match=match):
        workflow.prepare({**CONFIG, **changes}, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_existing_preparation_never_overwritten(prepared):
    before = (prepared / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        workflow.prepare(CONFIG, prepared)
    assert (prepared / "manifest.json").read_bytes() == before


def test_execute_counts_budgets_independence_and_complete_resume(prepared, backend):
    paths = workflow.run(prepared, execute=True)
    assert len(backend.calls) == 30
    for path in paths.values():
        with np.load(path, allow_pickle=False) as archive:
            assert archive["scale_counts"].shape == (4, 16)
            np.testing.assert_array_equal(archive["scale_counts"].sum(axis=1), [5] * 4)
            assert archive["raw_counts"].sum() == 12
            assert archive["reset_extension_counts"].sum() == 7
            assert archive["scale_counts"][0].sum() + archive["reset_extension_counts"].sum() == 12
            provenance = json.loads(str(archive["provenance_json"]))
            assert provenance["known_order_used_for_synthesis"] is False
            assert provenance["raw_shot_start"] == 12
    for rate in CONFIG["rates"]:
        raw_ids, reset_ids = set(), set()
        for physical, lam, reset, offset, size in backend.calls:
            if physical != rate:
                continue
            if not reset:
                raw_ids.update(range(offset, offset + size))
            elif lam == 1.:
                reset_ids.update(range(offset, offset + size))
        assert raw_ids == set(range(12, 24))
        assert reset_ids == set(range(12))
        assert raw_ids.isdisjoint(reset_ids)
    before = list(backend.calls)
    assert workflow.run(prepared, execute=True, resume=True) == paths
    assert backend.calls == before
    assert workflow.run(prepared) == paths
    with pytest.raises(FileExistsError, match="resume"):
        workflow.run(prepared, execute=True)


def test_interruption_and_resume_never_repeats_completed_shots(prepared, backend):
    backend.fail_after = 5
    with pytest.raises(RuntimeError, match="interruption"):
        workflow.run(prepared, execute=True)
    initial = list(backend.calls)
    assert len(list((prepared / "checkpoints").rglob("*.npz"))) == 5
    backend.fail_after = None
    workflow.run(prepared, execute=True, resume=True)
    assert len(backend.calls) == 30
    assert backend.calls[:5] == initial
    assert len(set(backend.calls)) == len(backend.calls)


def test_unregistered_atomic_chunk_is_adopted(prepared, backend):
    backend.fail_after = 1
    with pytest.raises(RuntimeError):
        workflow.run(prepared, execute=True)
    state_path = prepared / "run_state.json"
    state = json.loads(state_path.read_text())
    state["chunk_hashes"] = {}  # Crash after exclusive chunk publish, before state update.
    state_path.write_text(json.dumps(state))
    backend.fail_after = None
    workflow.run(prepared, execute=True, resume=True)
    assert len(backend.calls) == 30


def test_partial_rate_export_and_unpublished_temporary_files(prepared, backend):
    backend.fail_after = 16
    with pytest.raises(RuntimeError):
        workflow.run(prepared, execute=True)
    assert (prepared / "counts" / "rate_000.npz").is_file()
    assert not (prepared / "counts" / "rate_001.npz").exists()
    temporary = prepared / "checkpoints" / "rate_001" / ".pending_aborted_write"
    temporary.write_bytes(b"unpublished, incomplete data")
    backend.fail_after = None
    workflow.run(prepared, execute=True, resume=True)
    assert len(backend.calls) == 30
    assert temporary.read_bytes() == b"unpublished, incomplete data"


def test_zero_extension_and_default_baseline(tmp_path, monkeypatch, backend):
    monkeypatch.setattr(workflow, "_compile", lambda config: (
        {1 << i: [("cz", [0, 1], [])] for i in range(config["bits"])}, "fixture"))
    prepared = tmp_path / "zero_extension"
    workflow.prepare({**CONFIG, "baseline_shots": 5}, prepared)
    for path in workflow.run(prepared, execute=True).values():
        with np.load(path) as archive:
            assert archive["reset_extension_counts"].sum() == 0
            assert archive["raw_counts"].sum() == 5
    config = {key: value for key, value in CONFIG.items() if key != "baseline_shots"}
    manifest = workflow.prepare(config, tmp_path / "default_baseline")
    assert manifest["config"]["baseline_shots"] == 20


def test_consistent_chunk_tampering_still_fails_registered_sha(prepared, backend):
    backend.fail_after = 1
    with pytest.raises(RuntimeError):
        workflow.run(prepared, execute=True)
    path = next((prepared / "checkpoints").rglob("*.npz"))
    outcomes = np.zeros(3, dtype=np.uint32)
    _rewrite(path, outcomes=outcomes, counts=np.bincount(outcomes, minlength=16).astype(np.int64))
    backend.fail_after = None
    with pytest.raises(ValueError, match="SHA256"):
        workflow.run(prepared, execute=True, resume=True)
    assert len(backend.calls) == 1


def test_resume_environment_mismatch_before_new_shots(prepared, backend):
    backend.fail_after = 1
    with pytest.raises(RuntimeError):
        workflow.run(prepared, execute=True)
    state_path = prepared / "run_state.json"
    state = json.loads(state_path.read_text())
    state["environment"] = {"backend": "different numerical environment"}
    state_path.write_text(json.dumps(state))
    backend.fail_after = None
    with pytest.raises(ValueError, match="environment"):
        workflow.run(prepared, execute=True, resume=True)
    assert len(backend.calls) == 1


@pytest.mark.parametrize("target", ["manifest", "operations", "checkpoint", "final", "missing", "unexpected"])
def test_corruption_rejected_before_gpu_initialization(prepared, backend, monkeypatch, target):
    workflow.run(prepared, execute=True)
    if target in {"manifest", "operations"}:
        path = prepared / ("manifest.json" if target == "manifest" else "operations.json")
        path.write_text(path.read_text() + " ")
        if target == "manifest":
            data = json.loads(path.read_text())
            data["config"]["seed"] += 1
            path.write_text(json.dumps(data))
    elif target == "checkpoint":
        path = sorted((prepared / "checkpoints").rglob("*.npz"))[0]
        with np.load(path) as archive:
            counts = archive["counts"].copy()
        counts[0] += 1
        _rewrite(path, counts=counts)
    elif target == "final":
        path = sorted((prepared / "counts").glob("*.npz"))[0]
        _rewrite(path, raw_counts=np.zeros(16, dtype=np.int64))
    elif target == "missing":
        sorted((prepared / "checkpoints").rglob("*.npz"))[0].unlink()
    else:
        (prepared / "checkpoints" / "unknown").mkdir()
    def forbidden(*args):
        raise AssertionError("corruption reached the GPU")
    monkeypatch.setattr(workflow, "_GPUBackend", forbidden)
    with pytest.raises(ValueError):
        workflow.run(prepared, execute=True, resume=True)


def test_resume_changed_manifest_rejected_even_with_recomputed_identity(prepared, backend):
    backend.fail_after = 1
    with pytest.raises(RuntimeError):
        workflow.run(prepared, execute=True)
    path = prepared / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.pop("manifest_identity")
    manifest["config"]["seed"] += 1
    manifest["manifest_identity"] = workflow._digest(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="resume configuration"):
        workflow.run(prepared, execute=True, resume=True)


def test_fusion_matches_frozen_cpu_functions():
    source = Path(workflow.__file__).with_name("simulator.py")
    nodes = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in {"get_gate_matrix", "fuse_shor_ops"}]
    scope = {"np": np, "np_c_type": np.complex64,
             "X_MAT": np.asarray([[0, 1], [1, 0]], dtype=np.complex64),
             "SX_MAT": np.asarray([[1+1j, 1-1j], [1-1j, 1+1j]], dtype=np.complex64) / 2,
             "SINGLE_QUBIT_GATES": {"x", "sx", "rz", "rx", "id"}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    operations = [("x", [0], []), ("rz", [0], [.2]), ("rx", [1], [.4]),
                  ("sx", [1], []), ("cz", [0, 1], []), ("id", [0], []),
                  ("rzz", [0, 2], [.7])]
    expected = scope["fuse_shor_ops"]({1: operations})[1]
    actual = workflow._fuse(operations)
    for (name, targets, parameters), (old_name, old_targets, old_parameters) in zip(actual, expected, strict=True):
        assert (name, targets) == (old_name, old_targets)
        np.testing.assert_array_equal(parameters, old_parameters)
    assert "cupy" not in sys.modules


def test_actual_cpu_arithmetic_preparation(tmp_path):
    pytest.importorskip("qiskit")
    manifest = workflow.prepare({**CONFIG, "rates": [.001]}, tmp_path / "actual")
    assert manifest["total_two_qubit_gates"] > 100
    assert manifest["dimensions"] == {"bits": 4, "qubits": 7, "outcomes": 16}
    encoded = json.loads((tmp_path / "actual" / "operations.json").read_text())
    assert set(encoded) == {"1", "2", "4", "8"}
    assert "order" not in manifest["config"]
    assert "cupy" not in sys.modules
    assert "shor_qem.simulator" not in sys.modules
    assert hashlib.sha256(Path(workflow.__file__).with_name("simulator.py").read_bytes()).hexdigest() == (
        "3e00444360affc759d419e71145f3728fcdb77eb98283107b31c9c174a15f170")
