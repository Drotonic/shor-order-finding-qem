"""CPU preparation and explicitly requested, checkpointed GPU order finding.

Preparation compiles controlled modular multiplication from ``N`` and ``a``;
neither preparation nor execution computes, accepts, or uses a known order.
The retained simulator is imported only when a new batch actually executes.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
from typing import Any

import numpy as np

from .folding import FOLDING_PATTERN_SCOPE, FOLDING_SEED_FIELDS


BASIS = ["cz", "id", "rx", "rz", "rzz", "sx", "x", "measure", "reset"]
DEFAULT_SCALES = [1., 1.1, 1.2, 1.4, 1.8, 2.2, 2.6, 3.]
MAX_STATE_QUBITS = 25
MAX_HISTOGRAM_BYTES = 512 * 1024**2
SCHEMA_VERSION = 1


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _config(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary")
    result = json.loads(_canonical(config))
    # An order is never a compilation input, including an accidentally supplied one.
    if "order" in result or "r" in result:
        raise ValueError("known order is not a simulation input")
    N, a = _integer(result.get("N"), "N", 3), _integer(result.get("a"), "a", 2)
    if N % 2 == 0 or a >= N or math.gcd(a, N) != 1:
        raise ValueError("N must be odd and 1 < a < N with gcd(a,N)=1")
    bits = _integer(result.get("bits", 2 * N.bit_length()), "bits", 1)
    if bits != 2 * N.bit_length():
        raise ValueError("the frozen GPU engine requires bits=2*N.bit_length()")
    if 2 * N.bit_length() + 3 > MAX_STATE_QUBITS:
        raise ValueError(f"state exceeds the {MAX_STATE_QUBITS}-qubit resource guard")
    scales = result.get("scales", DEFAULT_SCALES)
    rates = result.get("rates")
    for name, values, lower, upper in (("scales", scales, 1., 3.), ("rates", rates, 0., 1.)):
        if not isinstance(values, list) or not values:
            raise ValueError(f"{name} must be a nonempty list")
        if any(isinstance(x, bool) or not isinstance(x, (float, int))
               or not math.isfinite(x) or not lower <= x <= upper for x in values):
            raise ValueError(f"{name} must contain finite numbers in [{lower},{upper}]")
        if values != sorted(set(values)):
            raise ValueError(f"{name} must be strictly increasing")
    scales, rates = list(map(float, scales)), list(map(float, rates))
    if len(scales) < 4 or scales[0] != 1.:
        raise ValueError("scales must start at 1 and contain at least four values")
    # These are the retained engine's seed encodings, not arbitrary tolerances.
    for name, values, multiplier in (("scales", scales, 10), ("rates", rates, 1_000_000_000)):
        if len({round(x * multiplier) for x in values}) != len(values):
            raise ValueError(f"{name} collide in the frozen engine's seed encoding")
    shots = _integer(result.get("shots_per_scale", 20_000), "shots_per_scale", 1)
    baseline = _integer(result.get("baseline_shots", len(scales) * shots), "baseline_shots", shots)
    if 2 * baseline > 2**31 - 1:
        raise ValueError("disjoint Raw/Reset shot IDs exceed the frozen signed32 seed range")
    batch = _integer(result.get("batch_size", 192), "batch_size", 1)
    seed = _integer(result.get("seed", 42), "seed")
    if seed > 2**32 - 1:
        raise ValueError("seed must fit unsigned32 (the retained gate-key seed width)")
    gpu = _integer(result.get("gpu_id", 0), "gpu_id")
    if len(rates) * (len(scales) + 2) * (1 << bits) * 8 > MAX_HISTOGRAM_BYTES:
        raise ValueError("dense count arrays exceed the 512 MiB resource guard")
    result.update(N=N, a=a, bits=bits, scales=scales, rates=rates,
                  shots_per_scale=shots, baseline_shots=baseline,
                  batch_size=batch, seed=seed, gpu_id=gpu)
    return result


def _fuse(operations: list) -> list:
    """Match the retained simulator's consecutive one-qubit fusion exactly."""
    x = np.asarray([[0, 1], [1, 0]], dtype=np.complex64)
    sx = np.asarray([[1+1j, 1-1j], [1-1j, 1+1j]], dtype=np.complex64) / 2

    def matrix(name, parameters):
        if name == "x":
            return x
        if name == "sx":
            return sx
        if name == "id":
            return np.eye(2, dtype=np.complex64)
        theta = float(parameters[0])
        if name == "rz":
            return np.diag([np.exp(-.5j * theta), np.exp(.5j * theta)]).astype(np.complex64)
        if name == "rx":
            return np.asarray([[np.cos(theta/2), -1j*np.sin(theta/2)],
                               [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=np.complex64)
        raise ValueError(f"unsupported compiled one-qubit gate: {name}")

    fused, index = [], 0
    while index < len(operations):
        name, targets, parameters = operations[index]
        if name in {"cz", "rzz"}:
            fused.append((name, targets, parameters))
            index += 1
            continue
        value = matrix(name, parameters)
        index += 1
        while index < len(operations):
            following, next_targets, params = operations[index]
            if following in {"cz", "rzz"} or next_targets != targets:
                break
            value = matrix(following, params) @ value
            index += 1
        fused.append(("unitary", targets, [value]))
    return fused


def _compile(config: dict) -> tuple[dict, str]:
    """Compile arithmetic on CPU, retaining even identity-residue rounds."""
    try:
        import qiskit
        from qiskit import transpile
        from .hardware_circuits import c_Ua
    except ImportError as exc:
        raise ImportError("CPU preparation requires qiskit; install the simulation extra") from exc
    blocks, operations = {}, {}
    for index in range(config["bits"]):
        power = 1 << index
        constant = pow(config["a"], power, config["N"])
        if constant not in blocks:
            circuit = c_Ua(config["N"].bit_length() + 1, constant, config["N"])
            compiled = transpile(circuit, basis_gates=BASIS, optimization_level=1,
                                 seed_transpiler=42, num_processes=1)
            raw = [(item.operation.name, [compiled.find_bit(q).index for q in item.qubits],
                    [float(p) for p in item.operation.params]) for item in compiled.data]
            blocks[constant] = _fuse(raw)
        operations[power] = blocks[constant]
    return operations, qiskit.__version__


def _encode_operations(operations: dict) -> dict:
    result = {}
    for power, block in operations.items():
        encoded = []
        for name, targets, parameters in block:
            row = {"name": name, "targets": list(targets)}
            if name == "unitary":
                matrix = np.asarray(parameters[0])
                row.update(real=matrix.real.tolist(), imag=matrix.imag.tolist())
            else:
                row["parameters"] = [float(p) for p in parameters]
            encoded.append(row)
        result[str(power)] = encoded
    return result


def _decode_operations(encoded: dict, config: dict) -> tuple[dict, list[int]]:
    if not isinstance(encoded, dict) or set(encoded) != {str(1 << i) for i in range(config["bits"])}:
        raise ValueError("operations must retain every IPE round")
    qubits, operations, counts = 2 * config["N"].bit_length() + 3, {}, []
    for power, block in encoded.items():
        if not isinstance(block, list) or not block:
            raise ValueError("operation blocks must be nonempty lists")
        decoded = []
        for row in block:
            if not isinstance(row, dict):
                raise ValueError("malformed operation")
            name, targets = row.get("name"), row.get("targets")
            if name not in {"unitary", "cz", "rzz"} or not isinstance(targets, list):
                raise ValueError("unsupported operation")
            expected = 1 if name == "unitary" else 2
            if len(targets) != expected or len(set(targets)) != expected:
                raise ValueError("malformed operation targets")
            if any(isinstance(q, bool) or not isinstance(q, int) or not 0 <= q < qubits for q in targets):
                raise ValueError("operation target out of range")
            if name == "unitary":
                real, imag = np.asarray(row.get("real")), np.asarray(row.get("imag"))
                if real.shape != (2, 2) or imag.shape != (2, 2):
                    raise ValueError("malformed one-qubit matrix")
                value = (real + 1j * imag).astype(np.complex64)
                if not np.isfinite(value).all() or not np.allclose(value.conj().T @ value, np.eye(2), atol=2e-5):
                    raise ValueError("compiled one-qubit matrix is not unitary")
                parameters = [value]
            else:
                parameters = row.get("parameters")
                if not isinstance(parameters, list) or len(parameters) != (1 if name == "rzz" else 0):
                    raise ValueError("malformed two-qubit parameters")
                if any(isinstance(p, bool) or not isinstance(p, (float, int)) or not math.isfinite(p) for p in parameters):
                    raise ValueError("nonfinite operation parameter")
            decoded.append((name, targets, parameters))
        operations[int(power)] = decoded
    for i in reversed(range(config["bits"])):
        counts.append(sum(name in {"cz", "rzz"} for name, _, _ in operations[1 << i]))
    if not sum(counts):
        raise ValueError("arithmetic circuit contains no noisy two-qubit gates")
    return operations, counts


def _effective(scales: list, counts: list) -> list:
    return [1. + 2. * sum(round((lam - 1.) * count / 2.) for count in counts) / sum(counts)
            for lam in scales]


def _atomic_json(path: Path, value: dict, *, exclusive: bool = False) -> None:
    _atomic(path, lambda handle: handle.write((_canonical(value) + "\n").encode()), exclusive=exclusive)


def _atomic(path: Path, writer, *, exclusive: bool = True) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".pending_", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(config: dict, output_dir: Path) -> dict:
    """Compile actual modular arithmetic without importing CUDA or running shots.

    ``output_dir`` must be absent or empty. Counts, checkpoints and their hashes
    are tied to the immutable manifest. No known order is required or accepted.
    """
    config = _config(config)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("preparation output must be absent or empty; refusing overwrite")
    operations, qiskit_version = _compile(config)
    encoded = _encode_operations(operations)
    _, counts = _decode_operations(encoded, config)
    source_root = Path(__file__).resolve().parent
    manifest = {
        "schema_version": SCHEMA_VERSION, "status": "prepared", "config": config,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dimensions": {"bits": config["bits"], "qubits": 2 * config["N"].bit_length() + 3,
                       "outcomes": 1 << config["bits"]},
        "compilation": {"basis": BASIS, "optimization_level": 1, "seed_transpiler": 42,
                        "qiskit_version": qiskit_version, "arithmetic": "controlled_modular_multiplication"},
        "per_round_gate_counts": counts, "total_two_qubit_gates": sum(counts),
        "effective_lambdas": _effective(config["scales"], counts),
        "source_hashes": {name: _sha(source_root / name) for name in
                          ("simulator.py", "folding.py", "hardware_circuits.py")},
        "seed_policy": "Reset IDs [0,baseline_shots); independent Raw IDs [baseline_shots,2*baseline_shots)",
        "folding_pattern_scope": FOLDING_PATTERN_SCOPE,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "operations.json", encoded, exclusive=True)
    manifest["operations"] = {"path": "operations.json", "sha256": _sha(output_dir / "operations.json")}
    manifest["manifest_identity"] = _digest(manifest)
    _atomic_json(output_dir / "manifest.json", manifest, exclusive=True)
    return manifest


def _load(prepared_dir: Path) -> tuple[dict, dict, str]:
    manifest_path = prepared_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest.pop("manifest_identity", None)
    if identity != _digest(manifest):
        raise ValueError("prepared manifest identity mismatch")
    manifest["manifest_identity"] = identity
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "prepared":
        raise ValueError("unsupported prepared manifest")
    config = _config(manifest.get("config"))
    if config != manifest["config"]:
        raise ValueError("manifest config is not normalized")
    expected_sources = {"simulator.py", "folding.py", "hardware_circuits.py"}
    if set(manifest.get("source_hashes", {})) != expected_sources:
        raise ValueError("missing pinned simulation sources")
    _verify_sources(manifest)
    info = manifest.get("operations", {})
    if info.get("path") != "operations.json" or _sha(prepared_dir / "operations.json") != info.get("sha256"):
        raise ValueError("compiled operations hash mismatch")
    operations, counts = _decode_operations(json.loads((prepared_dir / "operations.json").read_text()), config)
    if counts != manifest.get("per_round_gate_counts") or sum(counts) != manifest.get("total_two_qubit_gates"):
        raise ValueError("compiled gate counts differ from manifest")
    if _effective(config["scales"], counts) != manifest.get("effective_lambdas"):
        raise ValueError("effective lambdas differ from compiled folding cardinalities")
    expected_dimensions = {"bits": config["bits"], "qubits": 2 * config["N"].bit_length() + 3,
                           "outcomes": 1 << config["bits"]}
    if manifest.get("dimensions") != expected_dimensions:
        raise ValueError("manifest dimensions disagree with N")
    return manifest, operations, _sha(manifest_path)


def _verify_sources(manifest: dict) -> None:
    for name, expected in manifest["source_hashes"].items():
        if _sha(Path(__file__).resolve().parent / name) != expected:
            raise ValueError(f"pinned simulation source changed: {name}")


def _tasks(config: dict):
    for index, lam in enumerate(config["scales"]):
        yield f"scale_{index:03d}", lam, True, 0, config["shots_per_scale"]
    yield "raw", 1., False, config["baseline_shots"], config["baseline_shots"]
    yield "reset_extension", 1., True, config["shots_per_scale"], config["baseline_shots"] - config["shots_per_scale"]


def _batches(config: dict):
    for stream, lam, reset, start, shots in _tasks(config):
        for relative in range(0, shots, config["batch_size"]):
            offset, size = start + relative, min(config["batch_size"], shots - relative)
            yield stream, lam, reset, offset, size, f"{stream}_{offset:010d}_{size:010d}.npz"


def _histogram(bits: np.ndarray, size: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    bits = np.asarray(bits)
    if bits.shape != (size, width) or not np.all((bits == 0) | (bits == 1)):
        raise ValueError("engine output must contain exactly the requested L-column bit rows")
    outcomes = bits.astype(np.uint32) @ np.left_shift(np.uint32(1), np.arange(width, dtype=np.uint32))
    return outcomes, np.bincount(outcomes, minlength=1 << width).astype(np.int64)


def _read_chunk(path, identity, rate, stream, offset, size, width) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        for key, expected in {"manifest_sha256": identity, "rate": rate, "stream": stream,
                              "shot_offset": offset, "shots": size, "bits": width}.items():
            if key not in archive or archive[key].shape != () or archive[key].item() != expected:
                raise ValueError(f"checkpoint header mismatch: {key}: {path}")
        counts, outcomes = archive["counts"], archive["outcomes"]
        if (outcomes.shape != (size,) or outcomes.dtype != np.uint32
                or np.any(outcomes >= 1 << width) or counts.dtype != np.int64
                or counts.shape != (1 << width,) or np.any(counts < 0)
                or not np.array_equal(counts, np.bincount(outcomes, minlength=1 << width))):
            raise ValueError(f"invalid checkpoint outcomes/counts: {path}")
        return counts.copy()


class _GPUBackend:
    """Lazy adapter to the retained, unchanged shot-wise-folding simulator."""

    def __init__(self, config: dict):
        try:
            import cupy as cp
            engine = importlib.import_module("shor_qem.simulator")
        except ImportError as exc:
            raise ImportError("GPU execution requires a CUDA CuPy build and qiskit; install gpu-cuda12 or gpu-cuda13") from exc
        cp.cuda.Device(config["gpu_id"]).use()
        self.cp, self.engine = cp, engine
        if Path(engine.__file__).resolve() != Path(__file__).resolve().with_name("simulator.py"):
            raise ValueError("imported simulator is not the pinned package-local source")
        if not engine.USE_FP32:
            raise ValueError("this workflow requires the retained complex64 simulator")
        engine.GLOBAL_SEED = config["seed"]
        device = cp.cuda.runtime.getDeviceProperties(config["gpu_id"])["name"]
        if isinstance(device, bytes):
            device = device.decode()
        self.environment = {"python": platform.python_version(), "numpy": np.__version__,
                            "cupy": cp.__version__, "device": str(device),
                            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
                            "cuda_driver": cp.cuda.runtime.driverGetVersion(), "precision": "complex64"}

    def generate(self, config, rate, lam, reset, offset, size, operations):
        free_bytes, _ = self.cp.cuda.runtime.memGetInfo()
        per_shot = (1 << (2 * config["N"].bit_length() + 3)) * 8
        safe = int(free_bytes * .70 / 2.5 / per_shot)
        if size > safe:
            raise MemoryError(f"batch_size={size} exceeds current GPU memory allowance; use batch_size<={safe}")
        self.cp.cuda.get_current_stream().synchronize()
        result = self.engine._run_gpu_engine_chunked_shotwise(
            "random" if reset else "random_noareset", [rate], [lam],
            config["N"], config["a"], size, operations,
            shot_offset=offset, reset_ancilla=reset, folding_pattern_per_shot=reset)
        self.cp.cuda.get_current_stream().synchronize()
        if len(result) != 1 or result[0][0] != 0:
            raise ValueError("engine returned an unexpected configuration grid")
        return result[0][1]


@contextmanager
def _run_lock(prepared_dir: Path):
    with (prepared_dir / ".run.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another execution holds this prepared run's lock") from exc
        yield


def _expected_counts(config, manifest, totals, rate, identity) -> dict:
    return {"schema_version": np.int64(1), "N": np.int64(config["N"]), "a": np.int64(config["a"]),
            "bits": np.int64(config["bits"]), "rate": np.float64(rate),
            "scale_counts": np.stack([totals[f"scale_{i:03d}"] for i in range(len(config["scales"]))]),
            "raw_counts": totals["raw"], "reset_extension_counts": totals["reset_extension"],
            "nominal_lambdas": np.asarray(config["scales"]),
            "effective_lambdas": np.asarray(manifest["effective_lambdas"]),
            "manifest_sha256": np.asarray(identity),
            "provenance_json": np.asarray(_canonical({"manifest_sha256": identity,
                "generator": "retained_gpu_simulator", "known_order_used_for_synthesis": False,
                "seed": config["seed"], "folding_pattern_scope": FOLDING_PATTERN_SCOPE,
                "folding_seed_fields": FOLDING_SEED_FIELDS, "reset_scale_one_shared_with_baseline": True,
                "raw_shot_start": config["baseline_shots"], "reset_extension_shot_start": config["shots_per_scale"]}))}


def _validate_final(path: Path, expected: dict) -> None:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise ValueError(f"portable count archive fields differ: {path}")
        for name, value in expected.items():
            if archive[name].dtype != np.asarray(value).dtype or not np.array_equal(archive[name], value):
                raise ValueError(f"portable counts differ from verified checkpoints: {name}: {path}")


def _inspect(prepared_dir, manifest, identity, state):
    """Validate *all* existing chunks before any new GPU shot is requested."""
    config = manifest["config"]
    width = config["bits"]
    all_totals, pending, seen_chunks, seen_final = {}, [], set(), set()
    for rate_index, rate in enumerate(config["rates"]):
        folder = prepared_dir / "checkpoints" / f"rate_{rate_index:03d}"
        totals = {stream: np.zeros(1 << width, dtype=np.int64) for stream, *_ in _tasks(config)}
        gap = False
        for stream, lam, reset, offset, size, name in _batches(config):
            path = folder / name
            if path.exists():
                if gap:
                    raise ValueError("noncontiguous or overlapping checkpoint sequence")
                totals[stream] += _read_chunk(path, identity, rate, stream, offset, size, width)
                relative = path.relative_to(prepared_dir).as_posix()
                seen_chunks.add(relative)
                if relative in state.get("chunk_hashes", {}) and _sha(path) != state["chunk_hashes"][relative]:
                    raise ValueError(f"checkpoint SHA256 differs from run state: {relative}")
            else:
                gap = True
                pending.append((rate_index, rate, stream, lam, reset, offset, size, path))
        # A killed process can leave a not-yet-published temporary file. It has
        # no committed shots, and can be left untouched without blocking replay.
        actual = {p for p in folder.iterdir() if not p.name.startswith(".pending_")} if folder.exists() else set()
        expected_paths = {prepared_dir / name for name in seen_chunks if Path(name).parent == folder.relative_to(prepared_dir)}
        if actual != expected_paths:
            raise ValueError("unexpected checkpoint files; preserve and inspect before resuming")
        output = prepared_dir / "counts" / f"rate_{rate_index:03d}.npz"
        if output.exists():
            if gap:
                raise ValueError("portable final counts exist for an incomplete rate")
            _validate_final(output, _expected_counts(config, manifest, totals, rate, identity))
            relative = output.relative_to(prepared_dir).as_posix()
            seen_final.add(relative)
            if relative in state.get("final_hashes", {}) and _sha(output) != state["final_hashes"][relative]:
                raise ValueError("portable count SHA256 differs from run state")
        all_totals[rate_index] = totals
    for root_name, expected in (("checkpoints", {f"rate_{i:03d}" for i in range(len(config["rates"]))}),
                                ("counts", {Path(name).name for name in seen_final})):
        root = prepared_dir / root_name
        if root.exists() and any(path.name not in expected and not path.name.startswith(".pending_")
                                 for path in root.iterdir()):
            raise ValueError(f"unexpected {root_name} entries")
    if set(state.get("chunk_hashes", {})) - seen_chunks or set(state.get("final_hashes", {})) - seen_final:
        raise ValueError("run state references missing checkpoints/counts")
    return all_totals, pending, seen_chunks, seen_final


def run(prepared_dir: Path, *, execute: bool = False, resume: bool = False) -> dict[str, Path]:
    """Validate and plan, or explicitly execute uncompleted GPU batches.

    The default is read-only and never imports CuPy. Existing execution state
    requires ``resume=True``; no completed shot range is sampled or added twice.
    Return probability-string keys mapped to planned/final portable NPZ paths.
    """
    if not isinstance(execute, bool) or not isinstance(resume, bool):
        raise ValueError("execute and resume must be booleans")
    prepared_dir = Path(prepared_dir).resolve()
    manifest, operations, identity = _load(prepared_dir)
    config = manifest["config"]
    paths = {format(rate, ".12g"): prepared_dir / "counts" / f"rate_{i:03d}.npz"
             for i, rate in enumerate(config["rates"])}

    def inspect_state():
        state_path = prepared_dir / "run_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state.get("manifest_sha256") != identity or state.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("resume configuration or prepared-manifest SHA256 differs")
            if state.get("status") not in {"running", "complete"}:
                raise ValueError("unrecognized checkpoint run status")
            if not isinstance(state.get("chunk_hashes"), dict) or not isinstance(state.get("final_hashes"), dict):
                raise ValueError("malformed checkpoint run state")
        else:
            if (prepared_dir / "checkpoints").exists() or (prepared_dir / "counts").exists():
                raise ValueError("checkpoint/count artifacts exist without a recognized run state")
            state = {"schema_version": SCHEMA_VERSION, "manifest_sha256": identity,
                     "status": "running", "chunk_hashes": {}, "final_hashes": {}, "environment": None}
        return state_path, state

    if not execute:
        _, state = inspect_state()
        _inspect(prepared_dir, manifest, identity, state)
        return paths
    with _run_lock(prepared_dir):
        state_path, state = inspect_state()
        if state_path.exists() and not resume:
            raise FileExistsError("existing execution requires resume=True; refusing overwrite")
        totals, pending, seen_chunks, seen_final = _inspect(prepared_dir, manifest, identity, state)
        for name in seen_chunks:
            state["chunk_hashes"][name] = _sha(prepared_dir / name)
        for name in seen_final:
            state["final_hashes"][name] = _sha(prepared_dir / name)
        _atomic_json(state_path, state)
        backend, by_rate = None, {}
        for task in pending:
            by_rate.setdefault(task[0], []).append(task)
        for rate_index, rate in enumerate(config["rates"]):
            for _, _, stream, lam, reset, offset, size, path in by_rate.get(rate_index, []):
                if _sha(prepared_dir / "manifest.json") != identity:
                    raise ValueError("prepared manifest changed during execution")
                _verify_sources(manifest)
                if _sha(prepared_dir / "operations.json") != manifest["operations"]["sha256"]:
                    raise ValueError("compiled operations changed during execution")
                if backend is None:
                    backend = _GPUBackend(config)
                    if state["environment"] is not None and state["environment"] != backend.environment:
                        raise ValueError("resume numerical/GPU environment differs from saved run")
                    state["environment"] = backend.environment
                    _atomic_json(state_path, state)
                bits = backend.generate(config, rate, lam, reset, offset, size, operations)
                outcomes, counts = _histogram(bits, size, config["bits"])
                path.parent.mkdir(parents=True, exist_ok=True)
                _atomic(path, lambda handle: np.savez_compressed(handle,
                        manifest_sha256=np.asarray(identity), rate=np.float64(rate), stream=np.asarray(stream),
                        shot_offset=np.int64(offset), shots=np.int64(size), bits=np.int64(config["bits"]),
                        outcomes=outcomes, counts=counts))
                totals[rate_index][stream] += counts
                state["chunk_hashes"][path.relative_to(prepared_dir).as_posix()] = _sha(path)
                _atomic_json(state_path, state)
            (prepared_dir / "counts").mkdir(exist_ok=True)
            path = paths[format(rate, ".12g")]
            expected = _expected_counts(config, manifest, totals[rate_index], rate, identity)
            if not path.exists():
                _atomic(path, lambda handle: np.savez_compressed(handle, **expected))
            _validate_final(path, expected)
            state["final_hashes"][path.relative_to(prepared_dir).as_posix()] = _sha(path)
            _atomic_json(state_path, state)
        state["status"] = "complete"
        _atomic_json(state_path, state)
    return paths
