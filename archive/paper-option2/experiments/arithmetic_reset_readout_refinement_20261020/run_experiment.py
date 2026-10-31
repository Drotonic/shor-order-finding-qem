#!/usr/bin/env python3
"""Plan or explicitly execute the arithmetic-only reset-readout experiment.

Default: print a plan, without importing CUDA or running a quantum shot.
--execute is required for real trajectory generation. No cloud deployment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import time

sys.dont_write_bytecode = True
import numpy as np
from vendor.reset_readout import fault_counts

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
DEFAULT_COMPANION = PAPER / "shor_error_mitigation"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def atomic_json(path, value):
    fd, tmp = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(value, indent=2, allow_nan=False)+"\n")
        os.replace(tmp, path)
    finally:
        if Path(tmp).exists():
            Path(tmp).unlink()


def atomic_npz(path, **values):
    fd, tmp = tempfile.mkstemp(prefix=path.name+".", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(tmp, **values)
        # No overwrite even if a second producer races; the run lock is extra protection.
        os.link(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def load_configuration():
    config = json.loads((HERE/"config.json").read_text())
    assert (config["N"], config["a"], config["order"], config["measurement_bits"]) == (119, 16, 6, 14)
    assert config["two_qubit_error_probability"] == .00005 and config["noise_scale"] == 1
    assert config["arithmetic_qubits"] == list(range(8, 17)) and config["control_qubit"] == 0
    return config


def read_cache(path, config):
    """Only deserialize the pinned trusted operation cache, never arbitrary pickle."""
    if sha(path) != config["operation_cache_sha256"]:
        raise ValueError("Operation cache SHA256 differs from the approved original")
    with path.open("rb") as f:
        ops = pickle.load(f)
    assert set(ops) == {2**i for i in range(14)}
    noisy_count = 0
    for block in ops.values():
        assert isinstance(block, list)
        per_round = 0
        for name, qubits, parameters in block:
            assert all(0 <= int(q) < 17 for q in qubits)
            if name == "unitary":
                assert np.asarray(parameters[0]).shape == (2, 2)
            if name in ("cz", "rzz"):
                per_round += 1
        assert per_round == 3549
        noisy_count += per_round
    assert noisy_count == 49686
    return ops


def outcome_counts(bits):
    bits = np.asarray(bits)
    assert bits.ndim == 2 and bits.shape[1] == 14 and np.all((bits == 0) | (bits == 1))
    weights = np.left_shift(np.uint32(1), np.arange(14, dtype=np.uint32))
    outcomes = (bits.astype(np.uint32) @ weights).astype(np.uint16)
    counts = np.bincount(outcomes.astype(np.int64), minlength=16384).astype(np.int64)
    assert counts.sum() == len(bits)
    return outcomes, counts


def verify_chunk(path, *, identity, pm, offset, size, config):
    with np.load(path, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    assert str(data["run_identity"]) == identity and float(data["reset_error_probability"]) == pm
    assert int(data["shot_offset"]) == offset and int(data["shots"]) == size
    assert data["outcomes"].shape == (size,) and data["outcomes"].dtype == np.uint16
    assert np.all(data["outcomes"] < 16384)
    assert data["counts"].shape == (16384,) and data["counts"].dtype == np.int64
    assert np.array_equal(data["counts"], np.bincount(data["outcomes"], minlength=16384))
    expected = fault_counts(pm, shots=size, shot_offset=offset, seed=config["reset_error_seed"],
        N=119, a=16, gate_error=.00005, noise_scale=1.)
    assert np.array_equal(data["reset_fault_counts_by_round_qubit"], expected)
    return data


def arguments(argv=None):
    config = load_configuration()
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Actually generate new noisy quantum trajectories")
    mode.add_argument("--plan", action="store_true", help="Default: print plan only, no file/GPU writes")
    mode.add_argument("--validate-cache", action="store_true", help="Read/hash/check the pinned cache without CUDA or shots")
    p.add_argument("--reset-error-percent", nargs="+", type=float, default=config["reset_error_percent"],
                   help="Percent values, e.g. 0.2 0.3 0.4 (not probabilities)")
    p.add_argument("--shots-per-condition", type=int, default=20000)
    p.add_argument("--chunk-shots", type=int, default=48, help="Measured local GPU batch size")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=HERE/"results/pilot_20000")
    p.add_argument("--companion-root", type=Path, default=DEFAULT_COMPANION)
    p.add_argument("--operation-cache", type=Path)
    p.add_argument("--resume", action="store_true", help="Resume only matching, hash-verified chunks")
    args = p.parse_args(argv)
    if args.shots_per_condition < 1 or args.chunk_shots < 1 or args.gpu_id < 0:
        p.error("shot counts must be positive; GPU ID must be nonnegative")
    if args.shots_per_condition > 2**31-1:
        p.error("Frozen shot seeding uses signed32 shot IDs")
    if not args.reset_error_percent or len(set(args.reset_error_percent)) != len(args.reset_error_percent):
        p.error("Choose unique reset-error percentages")
    if any(v not in config["reset_error_percent"] for v in args.reset_error_percent):
        p.error("This refinement permits only 0,0.2,0.3,0.4 percent")
    args.reset_error_percent = sorted(args.reset_error_percent)
    args.operation_cache = (args.operation_cache or args.companion_root/config["operation_cache_relative"]).resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir in (Path("/"), Path("/tmp"), PAPER, HERE, Path.home()):
        p.error("Use a dedicated new output subdirectory")
    return args, config


def main(argv=None):
    args, config = arguments(argv)
    source_files = [HERE/"config.json", HERE/"run_experiment.py", *sorted((HERE/"vendor").glob("*.py"))]
    signature = {str(f.relative_to(HERE)): sha(f) for f in source_files}
    run_config = {"experiment": config, "reset_error_percent": args.reset_error_percent,
        "reset_error_probabilities": [v/100 for v in args.reset_error_percent],
        "shots_per_condition": args.shots_per_condition, "chunk_shots": args.chunk_shots,
        "source_sha256": signature, "operation_cache_sha256": config["operation_cache_sha256"]}
    identity = hashlib.sha256(canonical(run_config).encode()).hexdigest()
    plan = {"mode": "execute" if args.execute else "validate-cache" if args.validate_cache else "plan-only",
        "run_identity": identity, "configuration": run_config,
        "total_new_trajectory_shots_if_executed": len(args.reset_error_percent)*args.shots_per_condition,
        "operation_cache": str(args.operation_cache), "output_dir": str(args.output_dir),
        "control_error_probability": 0, "arithmetic_reset_targets": list(range(8, 17)),
        "new_shots_executed_by_plan": 0}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute and not args.validate_cache:
        return 0
    ops = read_cache(args.operation_cache, config)
    if args.validate_cache:
        print("PASS pinned operation cache:14rounds,49686two-qubit gates; noGPU/shots")
        return 0

    # GPU dependencies are intentionally imported only after explicit --execute.
    try:
        import cupy as cp
        from vendor import simulator_noisy_reset as engine
    except ImportError as exc:
        raise RuntimeError("GPU execution needs the CuPy/Qiskit environment; use qsim or the documented environment") from exc
    cp.cuda.Device(args.gpu_id).use()
    assert engine.GLOBAL_SEED == config["gate_measurement_seed"] == 42 and engine.USE_FP32
    device = cp.cuda.runtime.getDeviceProperties(args.gpu_id)["name"]
    if isinstance(device, bytes):
        device = device.decode()
    environment = {"python": sys.version, "numpy": np.__version__, "cupy": cp.__version__,
                   "gpu": str(device), "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
                   "cuda_driver": cp.cuda.runtime.driverGetVersion(), "dtype": "complex64"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.output_dir/".run.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = args.output_dir/"manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError("Run already exists; use --resume only with identical configuration")
        manifest = json.loads(manifest_path.read_text())
        assert manifest["run_identity"] == identity and manifest["configuration"] == run_config
        if manifest["environment"] != environment:
            raise RuntimeError("Resume requires the same recorded numerical environment")
    else:
        if any(p.name != ".run.lock" for p in args.output_dir.iterdir()):
            raise FileExistsError("Nonempty destination is not a recognized run")
        manifest = {"run_identity": identity, "configuration": run_config, "environment": environment,
            "created_utc": datetime.now(timezone.utc).isoformat(), "status": "running", "chunks": {}, "conditions": {}}
        atomic_json(manifest_path, manifest)
    start = time.monotonic()
    for percent in args.reset_error_percent:
        pm = percent/100
        label = "pm_"+format(pm, ".12g").replace(".", "p")
        folder = args.output_dir/label
        folder.mkdir(exist_ok=True)
        total_counts = np.zeros(16384, dtype=np.int64)
        total_faults = np.zeros((14, 9), dtype=np.int64)
        for offset in range(0, args.shots_per_condition, args.chunk_shots):
            size = min(args.chunk_shots, args.shots_per_condition-offset)
            file = folder/f"shots_{offset:09d}_{size:06d}.npz"
            relative = str(file.relative_to(args.output_dir))
            for name, digest in signature.items():
                assert sha(HERE/name) == digest, "Source changed during execution"
            if file.exists():
                if not args.resume:
                    raise FileExistsError(file)
                if relative in manifest["chunks"]:
                    assert sha(file) == manifest["chunks"][relative]["sha256"]
                data = verify_chunk(file, identity=identity, pm=pm, offset=offset, size=size, config=config)
            else:
                result = engine._run_gpu_engine_chunked_shotwise("random", [.00005], [1.], 119, 16,
                    size, ops, shot_offset=offset, reset_ancilla=True, folding_pattern_per_shot=True,
                    arithmetic_reset_readout_error=pm, reset_error_seed=config["reset_error_seed"])
                assert len(result) == 1 and result[0][0] == 0
                outcomes, counts = outcome_counts(result[0][1])
                faults = fault_counts(pm, shots=size, shot_offset=offset, seed=config["reset_error_seed"],
                    N=119, a=16, gate_error=.00005, noise_scale=1.)
                atomic_npz(file, run_identity=np.asarray(identity), reset_error_probability=np.float64(pm),
                    reset_error_percent=np.float64(percent), shot_offset=np.int64(offset), shots=np.int64(size),
                    outcomes=outcomes, counts=counts, reset_fault_counts_by_round_qubit=faults)
                data = verify_chunk(file, identity=identity, pm=pm, offset=offset, size=size, config=config)
            total_counts += data["counts"]
            total_faults += data["reset_fault_counts_by_round_qubit"]
            manifest["chunks"][relative] = {"sha256": sha(file), "shot_offset": offset, "shots": size,
                                             "reset_error_probability": pm}
            manifest["status"] = "running"
            manifest["elapsed_seconds_this_invocation"] = time.monotonic()-start
            atomic_json(manifest_path, manifest)
            print(f"p_reset={percent:g}%: {offset+size}/{args.shots_per_condition} shots", flush=True)
        assert total_counts.sum() == args.shots_per_condition
        aggregate = folder/"aggregate.npz"
        if not aggregate.exists():
            atomic_npz(aggregate, run_identity=np.asarray(identity), reset_error_probability=np.float64(pm),
                reset_error_percent=np.float64(percent), shots=np.int64(args.shots_per_condition),
                counts=total_counts, reset_fault_counts_by_round_qubit=total_faults)
        else:
            with np.load(aggregate, allow_pickle=False) as f:
                assert str(f["run_identity"]) == identity and np.array_equal(f["counts"], total_counts)
                assert np.array_equal(f["reset_fault_counts_by_round_qubit"], total_faults)
        manifest["conditions"][label] = {"reset_error_percent": percent, "reset_error_probability": pm,
            "shots": args.shots_per_condition, "aggregate": str(aggregate.relative_to(args.output_dir)),
            "sha256": sha(aggregate), "status": "complete"}
        atomic_json(manifest_path, manifest)
    manifest["status"] = "complete"
    manifest["elapsed_seconds_this_invocation"] = time.monotonic()-start
    atomic_json(manifest_path, manifest)
    print("Complete. No ZNE fits or bootstrap have been performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
