"""Opt-in, restart-safe IBM SamplerV2 execution for automatic order finding.

Importing this module neither imports IBM Runtime nor opens an account.  Preparing
circuits is local; only ``submit(..., execute=True)`` can submit jobs.  A hardware
folding pattern is a circuit reused for its declared shots, not a fresh fold per
shot (unless the requested number of patterns equals the shot count).
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any
import warnings

import numpy as np


DEFAULT_SCALES = (1.0, 1.4, 1.8, 2.2)
MANIFEST_NAME = "manifest.json"
RECORDS_NAME = "jobs.json"
SCHEMA_VERSION = 1
# https://quantum.cloud.ibm.com/docs/en/guides/job-limits (checked 2026-11-10)
MAX_EXECUTIONS_PER_JOB = 10_000_000


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    lower = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < lower:
        raise ValueError(f"{name} must be an integer >= {lower}")
    return int(value)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Non-serializable provenance value: {type(value).__name__}")


def _reject_credentials(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(word in normalized for word in ("token", "password", "secret", "credential", "api_key")):
                raise ValueError("Credentials must be supplied through the environment or secure prompt, not configuration")
            _reject_credentials(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_credentials(item)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, json.dumps(_json_value(value), indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")


def _atomic_bytes(path: Path, value: bytes) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _fsync_directory(directory: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _directory_lock(directory: Path):
    """Fail closed on concurrent writers and stale locks left after a crash."""
    path = directory / ".runtime.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("Runtime directory is locked. Inspect jobs.json before manually removing a stale .runtime.lock") from exc
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)


def service_from_environment(*, prompt: bool = False):
    """Create an unsaved, explicitly credentialed service; never use saved accounts.

    ``prompt=True`` uses getpass only when an environment variable is absent.
    Credentials are never included in an artifact or returned provenance.
    """
    token = os.environ.get("QISKIT_IBM_TOKEN", "").strip()
    instance = os.environ.get("QISKIT_IBM_INSTANCE", "").strip()

    def secure_prompt(message: str) -> str:
        # getpass otherwise falls back to echoed stdin in some non-TTY contexts.
        # Turn that warning into an error *before* its fallback reads a secret.
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            try:
                return getpass.getpass(message).strip()
            except getpass.GetPassWarning:
                raise ValueError("Secure prompting is unavailable; use QISKIT_IBM_TOKEN and QISKIT_IBM_INSTANCE") from None

    if prompt and not token:
        token = secure_prompt("IBM Quantum API token (not saved): ")
    if prompt and not instance:
        instance = secure_prompt("IBM Quantum instance CRN (not saved): ")
    if not token or not instance:
        raise ValueError("Set QISKIT_IBM_TOKEN and QISKIT_IBM_INSTANCE, or request secure prompting")
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:
        raise ImportError("IBM execution requires the optional qiskit-ibm-runtime package") from exc
    return QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=instance)


def _backend_name(backend) -> str:
    name = getattr(backend, "name", None)
    return str(name() if callable(name) else name or "")


def _simulator_or_fake(backend) -> bool:
    module = type(backend).__module__
    if module.startswith(("qiskit_aer", "qiskit.providers.fake_provider", "qiskit_ibm_runtime.fake_provider")):
        return True
    if getattr(backend, "simulator", False) is True:
        return True
    configuration = getattr(backend, "configuration", None)
    return bool(configuration is not None and getattr(configuration(), "simulator", False))


def select_backend(service, name: str):
    """Resolve only the explicitly selected backend, never least_busy()."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("An explicit backend_name is required")
    backend = service.backend(name)
    if _backend_name(backend) != name:
        raise ValueError("The service returned a backend different from backend_name")
    return backend


def _walk_instructions(circuit, physical_qubits=None):
    if physical_qubits is None:
        physical_qubits = {bit: index for index, bit in enumerate(circuit.qubits)}
    for instruction in circuit.data:
        operation = instruction.operation
        qubits = tuple(physical_qubits[bit] for bit in instruction.qubits)
        yield operation, qubits
        for block in getattr(operation, "blocks", ()):
            yield from _walk_instructions(block, dict(zip(block.qubits, qubits)))


def validate_backend(backend, circuit=None, *, backend_name: str | None = None) -> dict:
    """Check width, reset/feed-forward support, and every native instruction.

    This accepts offline BackendV2 fakes.  It does not call backend.status(),
    properties(), or any cloud API.  Availability can still change at submission.
    """
    name = _backend_name(backend)
    if not name or (backend_name is not None and name != backend_name):
        raise ValueError("Backend does not match the explicit backend_name")
    width = getattr(backend, "num_qubits", None)
    if not isinstance(width, int) or width < 1:
        raise ValueError("A BackendV2 with an explicit positive num_qubits is required")
    target = getattr(backend, "target", None)
    names = set(getattr(target, "operation_names", ()))
    required = {"reset", "measure", "if_else"}
    if not required <= names:
        raise ValueError("Backend must support dynamic circuits: reset, measurement and if_else")
    if circuit is not None:
        if circuit.num_qubits > width:
            raise ValueError(f"Circuit needs {circuit.num_qubits} qubits but backend has {width}")
        if circuit.parameters:
            raise ValueError("Prepared circuits must not contain unbound parameters")
        for operation, qubits in _walk_instructions(circuit):
            if operation.name == "barrier":
                continue
            if operation.name not in names:
                raise ValueError(f"Circuit instruction {operation.name!r} is not native to backend")
            supported = getattr(target, "instruction_supported", None)
            if supported is not None:
                arguments = {} if getattr(operation, "blocks", ()) else {"qargs": qubits}
                if not supported(operation_name=operation.name, **arguments):
                    raise ValueError(f"Native instruction {operation.name!r} is unsupported on physical qubits {qubits}")
    return {"backend_name": name, "num_qubits": width, "native_operations": sorted(names),
            "simulator_or_fake": _simulator_or_fake(backend),
            "dynamic_circuits_validated": True}


def _phase_bits(circuit) -> int:
    registers = [register for register in circuit.cregs if register.name == "phase"]
    if len(registers) != 1:
        raise ValueError("Circuit must preserve exactly one classical register named 'phase'")
    return len(registers[0])


def _normalise_config(config: dict) -> dict:
    _reject_credentials(config)
    n = _positive_int(config.get("N"), "N")
    a = _positive_int(config.get("a"), "a")
    if n < 3 or n % 2 == 0 or not 1 < a < n or math.gcd(a, n) != 1:
        raise ValueError("Require odd N >= 3 and 1 < a < N coprime to N")
    scales = [float(value) for value in config.get("scales", DEFAULT_SCALES)]
    if len(scales) < 4 or scales[0] != 1 or not np.all(np.isfinite(scales)) or np.any(np.diff(scales) <= 0):
        raise ValueError("scales must start at 1 and be finite, strictly increasing (at least four)")
    shots = _positive_int(config.get("shots_per_scale", 1024), "shots_per_scale")
    baseline = _positive_int(config.get("baseline_shots", len(scales) * shots), "baseline_shots")
    if baseline < shots:
        raise ValueError("baseline_shots must be at least shots_per_scale")
    if max(shots, baseline) > MAX_EXECUTIONS_PER_JOB:
        raise ValueError("Each PUB must fit the IBM 10,000,000-execution limit; lower shots_per_scale/baseline_shots")
    if "raw_shots" in config and _positive_int(config["raw_shots"], "raw_shots") != baseline:
        raise ValueError("raw_shots must equal baseline_shots for a matched Raw baseline")
    if float(config.get("p2", 0)) != 0:
        raise ValueError("Hardware has intrinsic backend noise; p2 injection is simulator-only")
    name = config.get("backend_name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("An explicit backend_name is required")
    extension = config.get("reset_baseline_extension", True)
    if not isinstance(extension, bool):
        raise ValueError("reset_baseline_extension must be boolean")
    if not extension and baseline != shots:
        raise ValueError("Disabling the Reset extension requires baseline_shots=shots_per_scale for matched baselines")
    offline = config.get("offline_compile_only", False)
    if not isinstance(offline, bool):
        raise ValueError("offline_compile_only must be boolean")
    optimization = _positive_int(config.get("optimization_level", 1), "optimization_level", allow_zero=True)
    if optimization > 3:
        raise ValueError("optimization_level must be in [0,3]")
    return {"N": n, "a": a, "scales": scales, "shots_per_scale": shots,
            "baseline_shots": baseline, "reset_baseline_extension": extension,
            "folding_patterns": _positive_int(config.get("folding_patterns", 1), "folding_patterns"),
            "seed": _positive_int(config.get("seed", 42), "seed", allow_zero=True),
            "optimization_level": optimization, "backend_name": name,
            "offline_compile_only": offline,
            "pubs_per_job": _positive_int(config.get("pubs_per_job", 64), "pubs_per_job")}


def _plan_jobs(allocations: list[dict], pubs_per_job: int) -> list[dict]:
    """Use equal-shot PUBs and respect both job-size and execution-count limits.

    Different shot counts within one SamplerV2 job were deprecated in Runtime
    0.48.  Grouping here also makes the per-job execution budget explicit.
    """
    groups = {}
    for allocation in allocations:
        groups.setdefault(allocation["shots"], []).append(allocation["allocation_id"])
    jobs = []
    for shots, identifiers in groups.items():
        chunk = min(pubs_per_job, MAX_EXECUTIONS_PER_JOB // shots)
        if chunk < 1:
            raise ValueError("A PUB exceeds the IBM maximum executions per job")
        for start in range(0, len(identifiers), chunk):
            ids = identifiers[start:start + chunk]
            jobs.append({"job_index": len(jobs), "allocation_ids": ids,
                         "shots_per_pub": shots, "total_executions": len(ids) * shots})
    return jobs


def prepare(config: dict, output_dir: Path, backend) -> dict:
    """Locally compile all declared allocations and export QPY plus provenance.

    The Reset lambda-one block is shared with its matched-budget baseline, whose
    optional independent extension is ``baseline_shots - shots_per_scale``.  Raw
    gets ``baseline_shots`` independent shots.  Every fold pattern and its exact
    shot allocation is recorded; effective lambdas are shot-weighted actual gate
    expansion factors, not requested scale labels.
    """
    from qiskit import qpy
    from .hardware_circuits import compile_pair, fold_native

    normalized = _normalise_config(config)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError("Output must be empty; use a fresh directory to preserve job provenance")
    backend_info = validate_backend(backend, backend_name=normalized["backend_name"])
    raw, reset, compilation = compile_pair(normalized["N"], normalized["a"], backend,
                                          seed=normalized["seed"], optimization_level=normalized["optimization_level"])
    for circuit in (raw, reset):
        validate_backend(backend, circuit, backend_name=normalized["backend_name"])
    bits = _phase_bits(reset)
    if _phase_bits(raw) != bits:
        raise ValueError("Raw and Reset phase-register widths differ")
    if bits != 2 * normalized["N"].bit_length():
        raise ValueError("Phase-register width must equal 2*N.bit_length() for this circuit family")
    circuits, allocations, effective = [], [], []

    def allocate(circuit, role, shots, **extra):
        index = len(circuits)
        circuits.append(circuit)
        allocation = {"allocation_id": f"pub-{index:06d}", "circuit_index": index,
                      "role": role, "shots": shots, "phase_register": "phase", **extra}
        allocations.append(allocation)
        return allocation

    allocate(raw, "raw", normalized["baseline_shots"], requested_scale=1.0, effective_scale=1.0)
    for scale_index, scale in enumerate(normalized["scales"]):
        count = 1 if scale == 1 else min(normalized["folding_patterns"], normalized["shots_per_scale"])
        quotient, remainder = divmod(normalized["shots_per_scale"], count)
        weighted = 0.0
        for pattern_index in range(count):
            pattern_shots = quotient + (pattern_index < remainder)
            # No Python hash(): seed derivation is stable across interpreter runs.
            seed = int(np.random.SeedSequence([normalized["seed"], scale_index, pattern_index]).generate_state(1)[0])
            folded, actual, metadata = fold_native(reset, scale, seed=seed)
            actual = float(actual)
            if not math.isfinite(actual) or actual < 1:
                raise ValueError("fold_native returned an invalid effective scale")
            validate_backend(backend, folded, backend_name=normalized["backend_name"])
            if _phase_bits(folded) != bits:
                raise ValueError("Folding changed the phase-register width")
            allocate(folded, "reset_scale", pattern_shots, scale_index=scale_index,
                     requested_scale=scale, effective_scale=actual, pattern_index=pattern_index,
                     pattern_seed=seed, folding_metadata=_json_value(metadata))
            weighted += pattern_shots * actual
        effective.append(weighted / normalized["shots_per_scale"])
    if effective[0] != 1 or np.any(np.diff(effective) <= 0):
        raise ValueError("Effective scales must be strictly increasing; integer fold quantization collapsed the grid")
    extension_shots = normalized["baseline_shots"] - normalized["shots_per_scale"] if normalized["reset_baseline_extension"] else 0
    if extension_shots:
        allocate(reset.copy(), "reset_extension", extension_shots, requested_scale=1.0, effective_scale=1.0,
                 independent_sampling=True)
    jobs = _plan_jobs(allocations, normalized["pubs_per_job"])
    total_shots = sum(allocation["shots"] for allocation in allocations)
    manifest = {"schema_version": SCHEMA_VERSION, "source": "ibm_hardware", "created_utc": _now(),
                "config": normalized, "backend": backend_info, "phase_bits": bits,
                "phase_bit_order": "y = sum_i phase[i] * 2**i; IBM bitstrings display MSB first",
                "noise_model": "backend_intrinsic_only_no_added_p2", "effective_lambdas": effective,
                "folding_pattern_scope": "per_scale_pattern_circuit", "independent_fold_per_shot":
                normalized["folding_patterns"] >= normalized["shots_per_scale"],
                "compilation": _json_value(compilation), "allocations": allocations, "jobs": jobs,
                "budget": {"reset_scale_shots": len(normalized["scales"]) * normalized["shots_per_scale"],
                           "raw_shots": normalized["baseline_shots"], "reset_lambda_one_shots": normalized["shots_per_scale"],
                           "reset_extension_shots": extension_shots,
                           "reset_baseline_shots": normalized["shots_per_scale"] + extension_shots,
                           "total_physical_shots": total_shots, "planned_jobs": len(jobs)},
                "circuit_count": len(circuits), "qpy_file": "circuits.qpy", "counts_file": "counts.npz"}
    _reject_credentials(manifest)
    # Manifest is the final commit marker for a completely prepared directory.
    with tempfile.SpooledTemporaryFile() as handle:
        qpy.dump(circuits, handle)
        handle.seek(0)
        payload = handle.read()
    manifest["qpy_sha256"] = hashlib.sha256(payload).hexdigest()
    _atomic_bytes(directory / manifest["qpy_file"], payload)
    _atomic_json(directory / "compilation_report.json", {"backend": backend_info, "compilation": compilation,
                                                         "effective_lambdas": effective, "allocations": allocations})
    _atomic_json(directory / MANIFEST_NAME, manifest)
    return manifest


def _load_prepared(directory: Path) -> dict:
    manifest = _read_json(directory / MANIFEST_NAME)
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("source") != "ibm_hardware":
        raise ValueError("Unsupported IBM preparation manifest")
    # Paths are fixed, not trusted from a mutable manifest.
    if manifest.get("qpy_file") != "circuits.qpy" or manifest.get("counts_file") != "counts.npz":
        raise ValueError("Unexpected artifact path in manifest")
    if hashlib.sha256((directory / "circuits.qpy").read_bytes()).hexdigest() != manifest["qpy_sha256"]:
        raise ValueError("Prepared QPY checksum does not match manifest")
    _validate_manifest(manifest)
    return manifest


def _validate_manifest(manifest: dict) -> None:
    """Fail closed on malformed allocation/shot plans before any service call."""
    config = _normalise_config(manifest["config"])
    if config != manifest["config"]:
        raise ValueError("Manifest configuration is incomplete or inconsistent")
    if manifest["phase_bits"] != 2 * config["N"].bit_length():
        raise ValueError("Manifest phase width does not match N")
    if manifest["backend"]["backend_name"] != config["backend_name"]:
        raise ValueError("Manifest backend does not match configuration")
    allocations = manifest["allocations"]
    if not isinstance(allocations, list) or not allocations or manifest["circuit_count"] != len(allocations):
        raise ValueError("Manifest circuit count is inconsistent")
    for index, allocation in enumerate(allocations):
        if allocation["circuit_index"] != index or allocation["allocation_id"] != f"pub-{index:06d}":
            raise ValueError("Manifest allocation IDs or circuit indices are inconsistent")
        _positive_int(allocation["shots"], "allocation shots")
        if allocation["phase_register"] != "phase" or allocation["role"] not in {"raw", "reset_scale", "reset_extension"}:
            raise ValueError("Manifest allocation role/register is invalid")
        if not math.isfinite(allocation["effective_scale"]) or allocation["effective_scale"] < 1:
            raise ValueError("Manifest allocation has an invalid effective scale")
    raw = [item for item in allocations if item["role"] == "raw"]
    extension = [item for item in allocations if item["role"] == "reset_extension"]
    expected_extension = config["baseline_shots"] - config["shots_per_scale"] if config["reset_baseline_extension"] else 0
    if len(raw) != 1 or raw[0]["shots"] != config["baseline_shots"]:
        raise ValueError("Manifest Raw allocation differs from declared baseline")
    if len(extension) != bool(expected_extension) or sum(item["shots"] for item in extension) != expected_extension:
        raise ValueError("Manifest Reset extension differs from declared baseline")
    scales = [item for item in allocations if item["role"] == "reset_scale"]
    if any(not isinstance(item.get("scale_index"), int) or not 0 <= item["scale_index"] < len(config["scales"]) for item in scales):
        raise ValueError("Manifest scale index is out of bounds")
    actual = []
    for index, scale in enumerate(config["scales"]):
        patterns = [item for item in scales if item["scale_index"] == index]
        expected_patterns = 1 if index == 0 else min(config["folding_patterns"], config["shots_per_scale"])
        if (len(patterns) != expected_patterns or [item["pattern_index"] for item in patterns] != list(range(expected_patterns))
                or sum(item["shots"] for item in patterns) != config["shots_per_scale"]
                or any(item["requested_scale"] != scale for item in patterns)):
            raise ValueError("Manifest folding patterns differ from declared scale allocation")
        actual.append(sum(item["shots"] * item["effective_scale"] for item in patterns) / config["shots_per_scale"])
    effective = np.asarray(manifest["effective_lambdas"], dtype=float)
    if (effective.shape != (len(config["scales"]),) or not np.all(np.isfinite(effective))
            or effective[0] != 1 or np.any(np.diff(effective) <= 0) or not np.allclose(effective, actual, rtol=0, atol=1e-12)):
        raise ValueError("Manifest effective scales differ from shot-weighted allocations")
    expected_jobs = _plan_jobs(allocations, config["pubs_per_job"])
    if manifest["jobs"] != expected_jobs:
        raise ValueError("Manifest jobs do not partition all declared allocations exactly once")
    expected_budget = {"reset_scale_shots": len(config["scales"]) * config["shots_per_scale"],
                       "raw_shots": config["baseline_shots"], "reset_lambda_one_shots": config["shots_per_scale"],
                       "reset_extension_shots": expected_extension,
                       "reset_baseline_shots": config["shots_per_scale"] + expected_extension,
                       "total_physical_shots": sum(item["shots"] for item in allocations), "planned_jobs": len(expected_jobs)}
    if manifest["budget"] != expected_budget:
        raise ValueError("Manifest budget does not match all physical allocations")


def _ledger(directory: Path, manifest: dict) -> dict:
    path = directory / RECORDS_NAME
    digest = hashlib.sha256((directory / MANIFEST_NAME).read_bytes()).hexdigest()
    if path.exists():
        ledger = _read_json(path)
        if ledger.get("manifest_sha256") != digest:
            raise ValueError("Manifest changed after execution records were created")
        records = ledger.get("records", [])
        if len(records) != len(manifest["jobs"]):
            raise ValueError("Execution ledger does not match the complete job plan")
        seen_ids = set()
        for record, planned in zip(records, manifest["jobs"]):
            if any(record.get(key) != value for key, value in planned.items()):
                raise ValueError("Execution ledger job allocations differ from the manifest")
            state, identifier = record.get("state"), record.get("job_id")
            if state not in {"planned", "submitting", "ambiguous", "submitted", "completed", "failed"}:
                raise ValueError("Execution ledger contains an unknown state")
            needs_id = state in {"submitted", "completed", "failed"}
            if needs_id != (isinstance(identifier, str) and bool(identifier.strip())):
                raise ValueError("Execution ledger state and job ID are inconsistent")
            if identifier is not None and identifier in seen_ids:
                raise ValueError("Execution ledger contains duplicate job IDs")
            seen_ids.add(identifier)
        return ledger
    return {"schema_version": SCHEMA_VERSION, "manifest_sha256": digest,
            "records": [{**job, "state": "planned", "job_id": None} for job in manifest["jobs"]]}


def _sampler_factory(backend, options):
    try:
        from qiskit_ibm_runtime import SamplerV2
    except ImportError as exc:
        raise ImportError("IBM execution requires the optional qiskit-ibm-runtime package") from exc
    return SamplerV2(mode=backend, options=options)


def submit(prepared_dir: Path, *, service, execute: bool = False,
           max_execution_time: int = 300, max_jobs: int = 100) -> list[dict]:
    """Return the declared job plan, or submit it only with explicit execute=True.

    Each intent is durably recorded *before* the call and each returned job ID
    immediately afterwards.  An exception/crash in this interval is ambiguous:
    automatic retry is forbidden.  Use ``recover_submission`` with a verified
    server job ID, or inspect the IBM dashboard; do not blindly delete the ledger.
    Existing IDs are never resubmitted.  No result() call is made here.
    """
    if not isinstance(execute, bool):
        raise ValueError("execute must be an explicit boolean; only execute=True authorizes submission")
    directory = Path(prepared_dir)
    manifest = _load_prepared(directory)
    limit = _positive_int(max_jobs, "max_jobs")
    seconds = _positive_int(max_execution_time, "max_execution_time")
    if seconds > 10800:
        raise ValueError("max_execution_time must be <= 10800 seconds")
    if len(manifest["jobs"]) > limit:
        raise ValueError(f"Declared plan needs {len(manifest['jobs'])} jobs, exceeding max_jobs={limit}")
    if not execute:
        return _ledger(directory, manifest)["records"]
    if manifest["config"]["offline_compile_only"] or manifest["backend"]["simulator_or_fake"]:
        raise ValueError("Offline/simulator preparation cannot authorize QPU execution; prepare again for an explicit real IBM backend")
    from qiskit import qpy

    with _directory_lock(directory):
        ledger = _ledger(directory, manifest)
        ambiguous = [record for record in ledger["records"] if record["state"] in {"submitting", "ambiguous"}]
        if ambiguous:
            raise RuntimeError("Ambiguous in-flight submission; attach a verified job ID with recover_submission before continuing")
        pending = [record for record in ledger["records"] if not record.get("job_id")]
        if not pending:
            return ledger["records"]
        backend = select_backend(service, manifest["config"]["backend_name"])
        validate_backend(backend, backend_name=manifest["config"]["backend_name"])
        if _simulator_or_fake(backend):
            raise ValueError("IBM hardware submission requires a real backend, not a simulator or fake")
        with (directory / "circuits.qpy").open("rb") as handle:
            circuits = qpy.load(handle)
        if len(circuits) != manifest["circuit_count"]:
            raise ValueError("Prepared QPY circuit count differs from manifest")
        for circuit in circuits:
            validate_backend(backend, circuit, backend_name=manifest["config"]["backend_name"])
        sampler = _sampler_factory(backend, {"max_execution_time": seconds,
                                              "execution": {"meas_type": "classified"},
                                              "twirling": {"enable_gates": False, "enable_measure": False},
                                              "dynamical_decoupling": {"enable": False}})
        by_id = {item["allocation_id"]: item for item in manifest["allocations"]}
        for record in pending:
            pubs = [(circuits[by_id[allocation_id]["circuit_index"]], None, by_id[allocation_id]["shots"])
                    for allocation_id in record["allocation_ids"]]
            record.update(state="submitting", submitted_utc=_now(), max_execution_time=seconds)
            _atomic_json(directory / RECORDS_NAME, ledger)
            try:
                job = sampler.run(pubs)
                identifier = job.job_id()
                if not isinstance(identifier, str) or not identifier:
                    raise ValueError("Sampler did not return a nonempty job ID")
                record.update(job_id=identifier, state="submitted")
                _atomic_json(directory / RECORDS_NAME, ledger)
            except BaseException:
                # Do not store exception text: SDK errors may include credentials.
                if not record.get("job_id"):
                    record["state"] = "ambiguous"
                    _atomic_json(directory / RECORDS_NAME, ledger)
                raise
        return ledger["records"]


def recover_submission(prepared_dir: Path, job_index: int, job_id: str) -> dict:
    """Explicitly attach an independently verified ID to an ambiguous submission.

    The operator must verify backend, PUB order and shot allocations in IBM's job
    details.  This helper intentionally cannot reset an intent to 'planned'.
    """
    index = _positive_int(job_index, "job_index", allow_zero=True)
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("A verified nonempty job_id is required")
    directory = Path(prepared_dir)
    manifest = _load_prepared(directory)
    with _directory_lock(directory):
        ledger = _ledger(directory, manifest)
        matches = [record for record in ledger["records"] if record["job_index"] == index]
        if len(matches) != 1 or matches[0]["state"] not in {"submitting", "ambiguous"}:
            raise ValueError("Only an ambiguous submission can be recovered")
        if any(record.get("job_id") == job_id for record in ledger["records"]):
            raise ValueError("Job ID is already attached to this plan")
        record = matches[0]
        record.update(job_id=job_id, state="submitted", recovered_utc=_now(), operator_verified=True)
        _atomic_json(directory / RECORDS_NAME, ledger)
        return record


def _extract_counts(pub_result, *, register: str, bits: int, shots: int) -> dict[str, int]:
    data = pub_result.data
    try:
        bit_array = data[register]
    except (KeyError, TypeError):
        bit_array = getattr(data, register, None)
    if bit_array is None:
        raise ValueError(f"Sampler result is missing the {register!r} register")
    if getattr(bit_array, "num_bits", bits) != bits:
        raise ValueError("Sampler phase-register width differs from preparation")
    result = {}
    for key, value in bit_array.get_counts().items():
        if not isinstance(key, str) or len(key) != bits or any(bit not in "01" for bit in key):
            raise ValueError("Sampler phase counts must be fixed-width binary bitstrings")
        result[key] = _positive_int(value, "result count", allow_zero=True)
    if sum(result.values()) != shots:
        raise ValueError("Sampler shot total differs from declared allocation")
    return result


def collect(prepared_dir: Path, *, service, wait: bool = False) -> dict[str, np.ndarray] | None:
    """Retrieve saved IDs, atomically cache results and export portable arrays.

    By default, return None while any job is unfinished; never wait for it.
    ``wait=True`` explicitly permits the SDK's blocking result() call.  Collection
    never submits or recreates jobs.  ERROR/CANCELLED jobs fail without retry.
    """
    if not isinstance(wait, bool):
        raise ValueError("wait must be an explicit boolean")
    directory = Path(prepared_dir)
    manifest = _load_prepared(directory)
    with _directory_lock(directory):
        ledger = _ledger(directory, manifest)
        if any(not record.get("job_id") for record in ledger["records"]):
            raise RuntimeError("All declared jobs must have saved IDs before collection; no jobs were submitted")
        by_id = {item["allocation_id"]: item for item in manifest["allocations"]}
        complete = True
        for record in ledger["records"]:
            cached = directory / f"job-{record['job_index']:06d}-counts.json"
            if cached.exists():
                record["state"] = "completed"
                continue
            job = service.job(record["job_id"])
            status = str(job.status()).upper().split(".")[-1]
            record["last_status"] = status
            if status in {"ERROR", "CANCELLED", "CANCELED"}:
                record["state"] = "failed"
                _atomic_json(directory / RECORDS_NAME, ledger)
                raise RuntimeError(f"Job {record['job_id']} ended with {status}; automatic resubmission is forbidden")
            if not wait and status != "DONE":
                complete = False
                continue
            results = job.result()
            if len(results) != len(record["allocation_ids"]):
                raise ValueError("Sampler PUB count differs from declared job")
            counts = {allocation_id: _extract_counts(result, register="phase", bits=manifest["phase_bits"],
                                                    shots=by_id[allocation_id]["shots"])
                      for allocation_id, result in zip(record["allocation_ids"], results)}
            _atomic_json(cached, {"job_id": record["job_id"], "counts": counts})
            record.update(state="completed", collected_utc=_now())
            _atomic_json(directory / RECORDS_NAME, ledger)
        _atomic_json(directory / RECORDS_NAME, ledger)
        if not complete:
            return None
        dimension = 1 << manifest["phase_bits"]
        arrays = {"scale_counts": np.zeros((len(manifest["config"]["scales"]), dimension), dtype=np.int64),
                  "raw_counts": np.zeros(dimension, dtype=np.int64),
                  "reset_extension_counts": np.zeros(dimension, dtype=np.int64),
                  "effective_lambdas": np.asarray(manifest["effective_lambdas"], dtype=float),
                  "N": np.asarray(manifest["config"]["N"], dtype=np.int64),
                  "a": np.asarray(manifest["config"]["a"], dtype=np.int64),
                  "bits": np.asarray(manifest["phase_bits"], dtype=np.int64)}
        for record in ledger["records"]:
            cached = _read_json(directory / f"job-{record['job_index']:06d}-counts.json")
            if cached.get("job_id") != record["job_id"] or set(cached["counts"]) != set(record["allocation_ids"]):
                raise ValueError("Cached result does not match the recorded job")
            for allocation_id in record["allocation_ids"]:
                allocation = by_id[allocation_id]
                counts = cached["counts"][allocation_id]
                if sum(counts.values()) != allocation["shots"]:
                    raise ValueError("Cached result shot total differs from declared allocation")
                if allocation["role"] == "reset_scale":
                    destination = arrays["scale_counts"][allocation["scale_index"]]
                elif allocation["role"] == "raw":
                    destination = arrays["raw_counts"]
                else:
                    destination = arrays["reset_extension_counts"]
                for bitstring, count in counts.items():
                    if len(bitstring) != manifest["phase_bits"] or any(bit not in "01" for bit in bitstring):
                        raise ValueError("Cached phase count is not a fixed-width binary bitstring")
                    destination[int(bitstring, 2)] += _positive_int(count, "cached count", allow_zero=True)
        _atomic_npz(directory / "counts.npz", arrays)
        return arrays
