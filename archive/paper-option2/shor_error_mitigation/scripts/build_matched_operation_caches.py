#!/usr/bin/env python3
"""Build deterministic common-opt1 operation caches from configuration files."""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import platform
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import transpile

from shor_qem import simulator as core


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = (
    PROJECT_ROOT / "configs" / "n119_a13_matched.json",
    PROJECT_ROOT / "configs" / "n119_a50_matched.json",
    PROJECT_ROOT / "configs" / "n119_a16_matched.json",
)


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.stem + "_", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _compile(config: dict[str, Any]) -> tuple[dict[int, list[Any]], list[dict[str, int]]]:
    N = int(config["N"])
    a = int(config["a"])
    level = int(config["transpile_optimization_level"])
    seed = int(config["transpiler_seed"])
    basis = list(config["basis_gates"])
    rounds = 2 * N.bit_length()
    q = N.bit_length() + 1
    operation_map: dict[int, list[Any]] = {}
    per_round: list[dict[str, int]] = []

    for k_pow in sorted(2 ** (rounds - 1 - i) for i in range(rounds)):
        constant = pow(a, k_pow, N)
        circuit = transpile(
            core.c_Ua(q, constant, N),
            basis_gates=basis,
            optimization_level=level,
            seed_transpiler=seed,
        )
        raw_operations = [
            (
                instruction.operation.name,
                [circuit.find_bit(bit).index for bit in instruction.qubits],
                instruction.operation.params,
            )
            for instruction in circuit.data
        ]
        fused = core.fuse_shor_ops({k_pow: raw_operations})[k_pow]
        operation_map[k_pow] = fused
        counts = collections.Counter(operation[0] for operation in fused)
        per_round.append(
            {
                "k_pow": int(k_pow),
                "modular_constant": int(constant),
                "cz": int(counts["cz"]),
                "rzz": int(counts["rzz"]),
                "two_qubit": int(counts["cz"] + counts["rzz"]),
                "fused_one_qubit": int(counts["unitary"]),
            }
        )
    return operation_map, per_round


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build common-opt1 operation caches from explicit configs."
    )
    parser.add_argument("--configs", nargs="+", type=Path, default=list(DEFAULT_CONFIGS))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records: list[dict[str, Any]] = []
    totals: set[int] = set()
    for config_path in args.configs:
        resolved_config = config_path.resolve()
        config = json.loads(resolved_config.read_text(encoding="utf-8"))
        N = int(config["N"])
        a = int(config["a"])
        order = int(config["order"])
        if math.gcd(a, N) != 1:
            raise ValueError(f"{config['instance']}: a and N are not coprime")
        if pow(a, order, N) != 1 or any(
            pow(a, exponent, N) == 1 for exponent in range(1, order)
        ):
            raise ValueError(
                f"{config['instance']}: configured order {order} is not multiplicative"
            )
        destination = _project_path(config["ops_cache"])
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"{destination} exists; pass --overwrite to replace it")
        operation_map, per_round = _compile(config)
        total = sum(row["two_qubit"] for row in per_round)
        expected = int(config["two_qubit_gate_count"])
        if total != expected:
            raise ValueError(
                f"{config['instance']}: compiled {total} two-qubit gates, expected {expected}"
            )
        if {row["two_qubit"] for row in per_round} != {
            int(config["two_qubit_gates_per_round"])
        }:
            raise ValueError(f"{config['instance']}: round gate counts are not uniform")
        _atomic_pickle(destination, operation_map)
        totals.add(total)
        records.append(
            {
                "instance": config["instance"],
                "N": int(config["N"]),
                "a": int(config["a"]),
                "order": order,
                "config": _portable_path(resolved_config),
                "cache": _portable_path(destination),
                "cache_sha256": _sha256(destination),
                "optimization_level": int(config["transpile_optimization_level"]),
                "transpiler_seed": int(config["transpiler_seed"]),
                "two_qubit_gate_count": total,
                "per_round": per_round,
            }
        )
        print(f"{config['instance']}: {total} two-qubit gates -> {destination}")

    if len(records) > 1 and len(totals) != 1:
        raise ValueError(f"compiled instances are not gate-count matched: {sorted(totals)}")
    metadata = {
        "design": "common arithmetic template with Qiskit optimization_level=1",
        "gate_count_matched_across_instances": (
            len(records) > 1 and len(totals) == 1
        ),
        "uniform_two_qubit_count_across_rounds": True,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "qiskit": importlib.metadata.version("qiskit"),
        "records": records,
    }
    first_cache = _project_path(records[0]["cache"])
    metadata_path = first_cache.parent / "build_metadata.json"
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, metadata_path)
    print(f"metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
