"""Offline-only IBM lifecycle tests: no account lookup and no QPU submissions."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

pytest.importorskip("qiskit")
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qpy
from qiskit.providers.fake_provider import GenericBackendV2

from shor_qem import hardware_circuits
from shor_qem import ibm_runtime as ibm


@pytest.fixture
def backend():
    generic = GenericBackendV2(3, control_flow=True, seed=5)
    # This explicit non-simulator test double never reaches a network service.
    return SimpleNamespace(name="offline_test_qpu", num_qubits=3, target=generic.target, simulator=False)


@pytest.fixture
def circuit():
    qubits = QuantumRegister(3, "q")
    phase = ClassicalRegister(8, "phase")
    circuit = QuantumCircuit(qubits, phase)
    circuit.x(0)
    circuit.cx(0, 1)
    circuit.reset(2)
    circuit.measure(0, phase[0])
    with circuit.if_test((phase[0], 1)):
        circuit.x(1)
    circuit.measure(1, phase[1])
    return circuit


@pytest.fixture
def compiler(monkeypatch, circuit):
    calls = []

    def compile_pair(N, a, backend, seed=42, optimization_level=1):
        calls.append((N, a, seed, optimization_level))
        return circuit.copy(), circuit.copy(), {"offline_test": True}

    def fold_native(circuit, scale, seed):
        return circuit.copy(), float(scale), {"pattern_seed": seed, "per_shot_randomization": False}

    monkeypatch.setattr(hardware_circuits, "compile_pair", compile_pair)
    monkeypatch.setattr(hardware_circuits, "fold_native", fold_native)
    return calls


@pytest.fixture
def config(backend):
    return {"N": 15, "a": 2, "shots_per_scale": 11, "folding_patterns": 3,
            "backend_name": backend.name, "pubs_per_job": 4}


@pytest.fixture
def prepared(tmp_path, config, backend, compiler):
    manifest = ibm.prepare(config, tmp_path, backend)
    return tmp_path, manifest


class FakeBits:
    num_bits = 8

    def __init__(self, shots):
        self.shots = shots

    def get_counts(self):
        # Asymmetric to catch phase bit reversal: 01 is integer one, not two.
        return {"00000001": self.shots - 1, "00000010": 1}


class FakeJob:
    def __init__(self, identifier, pubs):
        self.identifier = identifier
        self.pubs = pubs
        self.state = "DONE"
        self.result_calls = 0

    def job_id(self):
        return self.identifier

    def status(self):
        return self.state

    def result(self):
        self.result_calls += 1
        return [SimpleNamespace(data={"phase": FakeBits(pub[2])}) for pub in self.pubs]


class FakeService:
    def __init__(self, backend):
        self.selected_backend = backend
        self.jobs = {}
        self.backend_calls = []
        self.job_calls = []

    def backend(self, name):
        self.backend_calls.append(name)
        assert name == self.selected_backend.name
        return self.selected_backend

    def job(self, identifier):
        self.job_calls.append(identifier)
        return self.jobs[identifier]


@pytest.fixture
def runtime(monkeypatch, backend):
    service = FakeService(backend)
    samplers = []

    class FakeSampler:
        def __init__(self, selected_backend, options):
            assert selected_backend is backend
            self.options = options
            self.calls = []
            samplers.append(self)

        def run(self, pubs):
            self.calls.append(pubs)
            identifier = f"fake-job-{len(service.jobs)}"
            job = FakeJob(identifier, pubs)
            service.jobs[identifier] = job
            return job

    monkeypatch.setattr(ibm, "_sampler_factory", FakeSampler)
    return service, samplers


def test_prepare_qpy_budget_and_pattern_scope(prepared, compiler):
    directory, manifest = prepared
    assert compiler == [(15, 2, 42, 1)]
    assert manifest["config"]["scales"] == [1, 1.4, 1.8, 2.2]
    assert manifest["noise_model"] == "backend_intrinsic_only_no_added_p2"
    assert not manifest["independent_fold_per_shot"]
    assert manifest["folding_pattern_scope"] == "per_scale_pattern_circuit"
    assert manifest["budget"] == {"reset_scale_shots": 44, "raw_shots": 44,
        "reset_lambda_one_shots": 11, "reset_extension_shots": 33,
        "reset_baseline_shots": 44, "total_physical_shots": 121, "planned_jobs": 6}
    assert len(manifest["allocations"]) == 12
    for scale_index in range(4):
        allocations = [item for item in manifest["allocations"] if item.get("scale_index") == scale_index]
        assert sum(item["shots"] for item in allocations) == 11
        assert [item["shots"] for item in allocations] == ([11] if scale_index == 0 else [4, 4, 3])
    with (directory / "circuits.qpy").open("rb") as handle:
        assert len(qpy.load(handle)) == len(manifest["allocations"])
    report = json.loads((directory / "compilation_report.json").read_text())
    assert report["compilation"] == {"offline_test": True}
    with pytest.raises(FileExistsError, match="preserve job provenance"):
        ibm.prepare(manifest["config"], directory, FakeService(None))


def test_effective_scales_are_shot_weighted(tmp_path, config, backend, compiler, monkeypatch):
    indices = {}

    def fold(circuit, scale, seed):
        index = indices.get(scale, 0)
        indices[scale] = index + 1
        return circuit.copy(), scale + index * 0.02, {}

    monkeypatch.setattr(hardware_circuits, "fold_native", fold)
    manifest = ibm.prepare(config, tmp_path, backend)
    expected = [1] + [scale + (4 * 0.02 + 3 * 0.04) / 11 for scale in [1.4, 1.8, 2.2]]
    np.testing.assert_allclose(manifest["effective_lambdas"], expected)


def test_single_shot_patterns_and_no_extension(tmp_path, config, backend, compiler):
    config.update(folding_patterns=20, reset_baseline_extension=False, baseline_shots=11)
    manifest = ibm.prepare(config, tmp_path, backend)
    assert manifest["independent_fold_per_shot"]
    assert manifest["budget"]["reset_extension_shots"] == 0
    assert manifest["budget"]["reset_baseline_shots"] == 11
    assert manifest["budget"]["raw_shots"] == 11
    assert all(item["shots"] == 1 for item in manifest["allocations"] if item.get("scale_index", 0) > 0)


@pytest.mark.parametrize("change,match", [
    ({"backend_name": ""}, "explicit backend_name"),
    ({"shots_per_scale": [11, 11]}, "integer"),
    ({"shots_per_scale": True}, "integer"),
    ({"baseline_shots": 10}, "at least"),
    ({"raw_shots": 99}, "matched Raw"),
    ({"scales": [1, 1]}, "strictly increasing"),
    ({"scales": [1, float("nan")]}, "finite"),
    ({"p2": 0.01}, "simulator-only"),
    ({"N": 16}, "odd N"),
    ({"a": 3}, "coprime"),
    ({"optimization_level": 4}, "optimization_level"),
    ({"QISKIT_IBM_TOKEN": "do-not-save"}, "Credentials"),
    ({"baseline_shots": 10_000_001}, "execution limit"),
    ({"reset_baseline_extension": False}, "matched baselines"),
])
def test_invalid_configuration_rejected_before_compile(tmp_path, config, backend, compiler, change, match):
    config.update(change)
    with pytest.raises(ValueError, match=match):
        ibm.prepare(config, tmp_path, backend)
    assert not compiler
    assert not (tmp_path / "manifest.json").exists()


def test_effective_quantization_collapse_rejected(tmp_path, config, backend, compiler, monkeypatch):
    monkeypatch.setattr(hardware_circuits, "fold_native", lambda circuit, scale, seed: (circuit.copy(), 1, {}))
    with pytest.raises(ValueError, match="quantization"):
        ibm.prepare(config, tmp_path, backend)
    assert not (tmp_path / "manifest.json").exists()


def test_validate_width_native_dynamic_and_explicit_name(backend, circuit):
    assert ibm.validate_backend(backend, circuit)["dynamic_circuits_validated"]
    with pytest.raises(ValueError, match="backend_name"):
        ibm.validate_backend(backend, backend_name="another-backend")
    with pytest.raises(ValueError, match="dynamic circuits"):
        ibm.validate_backend(GenericBackendV2(3, control_flow=False))
    with pytest.raises(ValueError, match="qubits"):
        ibm.validate_backend(backend, QuantumCircuit(4))
    circuit.h(0)
    with pytest.raises(ValueError, match="not native"):
        ibm.validate_backend(backend, circuit)


def test_dry_run_never_calls_service_or_creates_ledger(prepared):
    directory, manifest = prepared
    records = ibm.submit(directory, service=object())
    assert len(records) == len(manifest["jobs"])
    assert all(record["state"] == "planned" for record in records)
    assert not (directory / "jobs.json").exists()
    with pytest.raises(ValueError, match="explicit boolean"):
        ibm.submit(directory, service=object(), execute="False")
    with pytest.raises(ValueError, match="explicit boolean"):
        ibm.collect(directory, service=object(), wait="False")


def test_max_jobs_rejected_before_any_submission(prepared, runtime):
    directory, _ = prepared
    service, samplers = runtime
    with pytest.raises(ValueError, match="max_jobs"):
        ibm.submit(directory, service=service, execute=True, max_jobs=2)
    assert not service.backend_calls and not samplers
    assert not (directory / "jobs.json").exists()
    with pytest.raises(ValueError, match="10800"):
        ibm.submit(directory, service=service, execute=True, max_execution_time=10801)


def test_submit_persists_every_id_before_next_submission(prepared, runtime, monkeypatch):
    directory, manifest = prepared
    service, samplers = runtime
    original_factory = ibm._sampler_factory

    def factory(backend, options):
        sampler = original_factory(backend, options)
        original_run = sampler.run

        def run(pubs):
            records = json.loads((directory / "jobs.json").read_text())["records"]
            index = len(sampler.calls)
            assert records[index]["state"] == "submitting"
            assert all(item["job_id"] for item in records[:index])
            return original_run(pubs)

        sampler.run = run
        return sampler

    monkeypatch.setattr(ibm, "_sampler_factory", factory)
    records = ibm.submit(directory, service=service, execute=True, max_execution_time=120)
    assert [record["job_id"] for record in records] == [f"fake-job-{index}" for index in range(len(manifest["jobs"]))]
    assert samplers[0].options["max_execution_time"] == 120
    assert samplers[0].options["twirling"] == {"enable_gates": False, "enable_measure": False}
    assert sum(pub[2] for call in samplers[0].calls for pub in call) == manifest["budget"]["total_physical_shots"]
    assert all(job.result_calls == 0 for job in service.jobs.values())
    again = ibm.submit(directory, service=service, execute=True)
    assert again == records
    assert len(samplers) == 1  # No new primitive or backend lookup on restart.


def test_ambiguous_submit_never_retried_and_explicit_recovery(prepared, runtime, monkeypatch):
    directory, _ = prepared
    service, _ = runtime
    calls = []

    class AmbiguousSampler:
        def run(self, pubs):
            calls.append(pubs)
            raise ConnectionError("transport disconnected after request")

    monkeypatch.setattr(ibm, "_sampler_factory", lambda *_: AmbiguousSampler())
    with pytest.raises(ConnectionError):
        ibm.submit(directory, service=service, execute=True)
    ledger = json.loads((directory / "jobs.json").read_text())
    assert ledger["records"][0]["state"] == "ambiguous"
    assert "transport" not in (directory / "jobs.json").read_text()
    with pytest.raises(RuntimeError, match="Ambiguous"):
        ibm.submit(directory, service=service, execute=True)
    assert len(calls) == 1
    recovered = ibm.recover_submission(directory, 0, "verified-server-job")
    assert recovered["job_id"] == "verified-server-job" and recovered["operator_verified"]
    with pytest.raises(ValueError, match="Only an ambiguous"):
        ibm.recover_submission(directory, 0, "another-job")


def test_collect_uses_existing_ids_bit_order_and_portable_npz(prepared, runtime):
    directory, manifest = prepared
    service, _ = runtime
    records = ibm.submit(directory, service=service, execute=True)
    arrays = ibm.collect(directory, service=service)
    assert set(arrays) == {"scale_counts", "raw_counts", "reset_extension_counts", "effective_lambdas", "N", "a", "bits"}
    assert (int(arrays["N"]), int(arrays["a"]), int(arrays["bits"])) == (15, 2, 8)
    np.testing.assert_array_equal(arrays["scale_counts"].sum(axis=1), [11] * 4)
    assert arrays["raw_counts"].sum() == 44
    assert arrays["reset_extension_counts"].sum() == 33
    assert arrays["raw_counts"][1] == 43 and arrays["raw_counts"][2] == 1
    np.testing.assert_allclose(arrays["effective_lambdas"], manifest["effective_lambdas"])
    assert service.job_calls == [record["job_id"] for record in records]
    with np.load(directory / "counts.npz", allow_pickle=False) as saved:
        for key in arrays:
            np.testing.assert_array_equal(saved[key], arrays[key])
    cached = ibm.collect(directory, service=object())
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], cached[key])


def test_nonblocking_pending_and_explicit_wait(prepared, runtime):
    directory, _ = prepared
    service, _ = runtime
    ibm.submit(directory, service=service, execute=True)
    service.jobs["fake-job-0"].state = "RUNNING"
    assert ibm.collect(directory, service=service) is None
    assert service.jobs["fake-job-0"].result_calls == 0
    assert not (directory / "counts.npz").exists()
    arrays = ibm.collect(directory, service=service, wait=True)
    assert arrays["raw_counts"].sum() == 44
    assert service.jobs["fake-job-0"].result_calls == 1


def test_failed_job_not_resubmitted(prepared, runtime):
    directory, _ = prepared
    service, samplers = runtime
    ibm.submit(directory, service=service, execute=True)
    service.jobs["fake-job-0"].state = "ERROR"
    with pytest.raises(RuntimeError, match="automatic resubmission is forbidden"):
        ibm.collect(directory, service=service)
    ibm.submit(directory, service=service, execute=True)
    assert len(samplers) == 1
    assert not (directory / "counts.npz").exists()


def test_incomplete_plan_cannot_collect(prepared):
    directory, _ = prepared
    with pytest.raises(RuntimeError, match="saved IDs"):
        ibm.collect(directory, service=object())


def test_shot_mismatch_never_writes_aggregate(prepared, runtime, monkeypatch):
    directory, _ = prepared
    service, _ = runtime
    ibm.submit(directory, service=service, execute=True)
    monkeypatch.setattr(FakeBits, "get_counts", lambda self: {"00000000": self.shots + 1})
    with pytest.raises(ValueError, match="shot total"):
        ibm.collect(directory, service=service)
    assert not (directory / "counts.npz").exists()


def test_manifest_and_qpy_tampering_are_rejected(prepared, runtime):
    directory, _ = prepared
    service, _ = runtime
    ibm.submit(directory, service=service, execute=True)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["created_utc"] = "edited"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Manifest changed"):
        ibm.submit(directory, service=service, execute=True)
    with (directory / "circuits.qpy").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        ibm.submit(directory, service=service)


def test_stale_lock_fails_closed(prepared, runtime):
    directory, _ = prepared
    service, _ = runtime
    (directory / ".runtime.lock").touch()
    with pytest.raises(RuntimeError, match="locked"):
        ibm.submit(directory, service=service, execute=True)


def test_environment_credentials_never_saved(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", SimpleNamespace(QiskitRuntimeService=lambda **kwargs: calls.append(kwargs) or object()))
    monkeypatch.setenv("QISKIT_IBM_TOKEN", "test-secret")
    monkeypatch.setenv("QISKIT_IBM_INSTANCE", "test-instance")
    ibm.service_from_environment()
    assert calls == [{"channel": "ibm_quantum_platform", "token": "test-secret", "instance": "test-instance"}]
    monkeypatch.delenv("QISKIT_IBM_TOKEN")
    with pytest.raises(ValueError, match="QISKIT_IBM_TOKEN"):
        ibm.service_from_environment()
    assert len(calls) == 1
    prompts = []
    monkeypatch.setattr(ibm.getpass, "getpass", lambda prompt: prompts.append(prompt) or "securely-entered")
    ibm.service_from_environment(prompt=True)
    assert len(prompts) == 1
    assert calls[-1]["token"] == "securely-entered"


def test_secure_prompt_refuses_getpass_echo_fallback(monkeypatch):
    monkeypatch.delenv("QISKIT_IBM_TOKEN", raising=False)
    monkeypatch.delenv("QISKIT_IBM_INSTANCE", raising=False)
    fallback_reads = []

    def unsafe_getpass(prompt):
        warnings.warn("Can not control echo on the terminal", ibm.getpass.GetPassWarning)
        fallback_reads.append("echoed-input-would-have-been-read")
        return "must-never-be-read"

    monkeypatch.setattr(ibm.getpass, "getpass", unsafe_getpass)
    with pytest.raises(ValueError, match="Secure prompting is unavailable"):
        ibm.service_from_environment(prompt=True)
    assert not fallback_reads


def test_real_compiler_prepares_native_dynamic_qpy(tmp_path):
    backend = GenericBackendV2(7, control_flow=True, seed=1)
    manifest = ibm.prepare({"N": 3, "a": 2, "backend_name": backend.name,
                            "shots_per_scale": 2, "folding_patterns": 2}, tmp_path, backend)
    assert manifest["phase_bits"] == 4
    assert manifest["budget"]["total_physical_shots"] == 22
    assert np.all(np.diff(manifest["effective_lambdas"]) > 0)
    with (tmp_path / "circuits.qpy").open("rb") as handle:
        for circuit in qpy.load(handle):
            ibm.validate_backend(backend, circuit)


def test_real_sampler_v2_api_only_on_local_aer(tmp_path, monkeypatch):
    aer = pytest.importorskip("qiskit_aer")
    runtime = pytest.importorskip("qiskit_ibm_runtime")
    backend = aer.AerSimulator()
    # This construction is local and does not instantiate QiskitRuntimeService.
    manifest = ibm.prepare({"N": 3, "a": 2, "backend_name": backend.name,
                            "shots_per_scale": 2}, tmp_path, backend)
    service = FakeService(backend)
    with pytest.raises(ValueError, match="Offline/simulator"):
        ibm.submit(tmp_path, service=service, execute=True)
    with (tmp_path / "circuits.qpy").open("rb") as handle:
        circuits = qpy.load(handle)
    # Invoke IBM's primitive explicitly on local Aer, never on the cloud path.
    ledger = ibm._ledger(tmp_path, manifest)
    sampler = runtime.SamplerV2(mode=backend)
    allocations = {item["allocation_id"]: item for item in manifest["allocations"]}
    for record in ledger["records"]:
        pubs = [(circuits[allocations[identifier]["circuit_index"]], None, allocations[identifier]["shots"])
                for identifier in record["allocation_ids"]]
        job = sampler.run(pubs)
        service.jobs[job.job_id()] = job
        record.update(state="submitted", job_id=job.job_id())
    ibm._atomic_json(tmp_path / "jobs.json", ledger)
    arrays = ibm.collect(tmp_path, service=service, wait=True)
    assert arrays["scale_counts"].shape == (4, 16)
    np.testing.assert_array_equal(arrays["scale_counts"].sum(axis=1), [2] * 4)
    assert arrays["raw_counts"].sum() == manifest["budget"]["raw_shots"]


def test_offline_compile_only_cannot_submit(tmp_path, config, backend, compiler):
    config["offline_compile_only"] = True
    manifest = ibm.prepare(config, tmp_path, backend)
    assert manifest["config"]["offline_compile_only"]
    with pytest.raises(ValueError, match="Offline/simulator"):
        ibm.submit(tmp_path, service=object(), execute=True)
    assert not (tmp_path / "jobs.json").exists()


@pytest.mark.parametrize("change", [
    lambda manifest: manifest["allocations"][0].update(circuit_index=999),
    lambda manifest: manifest["allocations"][2].update(scale_index=-1),
    lambda manifest: manifest["jobs"][0]["allocation_ids"].append("pub-000000"),
    lambda manifest: manifest["budget"].update(total_physical_shots=1),
    lambda manifest: manifest.update(phase_bits=999),
])
def test_malformed_manifest_fails_before_service_calls(prepared, change):
    directory, manifest = prepared
    change(manifest)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        ibm.submit(directory, service=object(), execute=True)


def test_malformed_ledger_cannot_turn_submitted_job_into_retry(prepared, runtime):
    directory, _ = prepared
    service, _ = runtime
    ibm.submit(directory, service=service, execute=True)
    ledger = json.loads((directory / "jobs.json").read_text())
    ledger["records"][0]["job_id"] = None
    (directory / "jobs.json").write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="state and job ID"):
        ibm.submit(directory, service=service, execute=True)


def test_job_execution_cap_and_uniform_shots(tmp_path, config, backend, compiler):
    config.update(shots_per_scale=2_000_000, folding_patterns=1, baseline_shots=2_000_000)
    manifest = ibm.prepare(config, tmp_path, backend)
    assert manifest["budget"]["total_physical_shots"] == 10_000_000
    assert all(job["total_executions"] <= 10_000_000 for job in manifest["jobs"])
    allocations = {item["allocation_id"]: item for item in manifest["allocations"]}
    for job in manifest["jobs"]:
        assert {allocations[identifier]["shots"] for identifier in job["allocation_ids"]} == {job["shots_per_pub"]}
    # Six PUBs at two million shots cannot fit in one job, even with a high PUB cap.
    planned = ibm._plan_jobs([{"allocation_id": str(index), "shots": 2_000_000} for index in range(6)], 100)
    assert [job["total_executions"] for job in planned] == [10_000_000, 2_000_000]


def test_prepare_does_not_overwrite_unrelated_artifact(tmp_path, config, backend, compiler):
    path = tmp_path / "circuits.qpy"
    path.write_bytes(b"unrelated-user-data")
    with pytest.raises(FileExistsError, match="empty"):
        ibm.prepare(config, tmp_path, backend)
    assert path.read_bytes() == b"unrelated-user-data"
