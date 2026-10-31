#!/usr/bin/env python3
"""Pack one native matched-template shot record into portable C1 count parents.

Reads eight Reset scale NPZs and two ordered 160k baseline NPZs. The baseline
source-range records identify the retained initial 20k and seven subsequent
20k blocks; these recorded boundaries, not a histogram split, define parents.
No pickle, GPU, trajectory simulation, fitting, or bootstrap is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np

from shor_qem.folding import (
    FOLDING_ARCHIVE_SCHEMA_VERSION, FOLDING_PATTERN_SCOPE, FOLDING_SEED_FIELDS,
)
from shor_qem.portable_c1 import CountParents


NOMINAL_LAMBDAS = (1.0, 1.1, 1.2, 1.4, 1.8, 2.2, 2.6, 3.0)
TEMPLATE_GATE_COUNT = 49_686
TEMPLATE_ROUNDS = 14
GATES_PER_ROUND = 3_549
SCALE_SHOTS = 20_000
BASELINE_SHOTS = 160_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path, role: str) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    digest = _sha256(path)
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    if _sha256(path) != digest:
        raise ValueError(f"{path.name}: source changed while reading")
    return result, {"role": role, "filename": path.name, "sha256": digest}


def _scalar(data: dict[str, np.ndarray], key: str) -> Any:
    if key not in data or np.asarray(data[key]).shape != ():
        raise ValueError(f"{key}: expected scalar metadata")
    return np.asarray(data[key]).item()


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name}: expected integer")
    try:
        result = int(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name}: expected integer") from exc
    if result != value or result < minimum:
        raise ValueError(f"{name}: expected integer >= {minimum}")
    return result


def _integral_array(value: Any, name: str, shape: tuple[int, ...], maximum: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iuf":
        raise ValueError(f"{name}: invalid shape or numeric dtype; expected {shape}")
    if (not np.all(np.isfinite(array)) or np.any(array < 0)
            or np.any(array > maximum) or np.any(array != np.floor(array))):
        raise ValueError(f"{name}: entries must be finite nonnegative integers <= {maximum}")
    return array.astype(np.int64, copy=False)


def _metadata(data: dict[str, np.ndarray]) -> dict[str, Any]:
    values = {
        key: _integer(_scalar(data, key), key, minimum=1)
        for key in ("N", "a", "order", "measurement_bits", "two_qubit_gate_count")
    }
    values["global_seed"] = _integer(
        _scalar(data, "global_seed" if "global_seed" in data else "folding_seed"), "global_seed"
    )
    values["instance"] = _scalar(data, "instance")
    values["ops_cache_sha256"] = _scalar(data, "ops_cache_sha256")
    values["circuit_variant"] = _scalar(data, "circuit_variant")
    if values["N"] != 119 or values["two_qubit_gate_count"] != TEMPLATE_GATE_COUNT:
        raise ValueError("expected N=119 matched 49,686-two-qubit-gate template")
    if values["measurement_bits"] > 14:
        raise ValueError("measurement_bits must lie in 1..14")
    if values["circuit_variant"] != "gate_count_matched_common_opt1":
        raise ValueError("circuit_variant is not the matched template")
    if not isinstance(values["ops_cache_sha256"], str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", values["ops_cache_sha256"]
    ):
        raise ValueError("ops_cache_sha256 must be a 64-digit SHA256")
    if values["instance"] != f"n119_a{values['a']}":
        raise ValueError("instance metadata does not match N and a")
    a, order = values["a"], values["order"]
    if (a >= 119 or order >= 119 or math.gcd(a, 119) != 1 or pow(a, order, 119) != 1
            or any(pow(a, candidate, 119) == 1 for candidate in range(1, order))):
        raise ValueError("order metadata is not the multiplicative order of a modulo N")
    return values


def _same_metadata(actual: dict[str, Any], expected: dict[str, Any], name: str) -> None:
    for key in expected:
        if actual[key] != expected[key]:
            raise ValueError(f"{name}: inconsistent {key} metadata")


def _baseline(path: Path, mode: str, rate: float) -> tuple[dict[str, Any], np.ndarray, dict[str, str]]:
    data, source = _read(path, f"{mode}_baseline")
    metadata = _metadata(data)
    if _scalar(data, "schema_version") != 1:
        raise ValueError(f"{path.name}: expected native merged-baseline schema_version=1")
    if _scalar(data, "mode") != mode:
        raise ValueError(f"{path.name}: baseline mode must be {mode}")
    stored_rate = _scalar(data, "rate")
    if not isinstance(stored_rate, (int, float)) or not np.isclose(stored_rate, rate, rtol=0, atol=5e-12):
        raise ValueError(f"{path.name}: baseline rate differs from --rate")
    if _scalar(data, "lambda") != 1.0 or _scalar(data, "shots") != BASELINE_SHOTS:
        raise ValueError(f"{path.name}: baseline requires lambda=1 and 160000 shots")
    try:
        ranges = json.loads(_scalar(data, "source_ranges_json"))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{path.name}: invalid source_ranges_json") from exc
    if not isinstance(ranges, list) or len(ranges) != 8:
        raise ValueError(f"{path.name}: require eight recorded 20000-shot source ranges")
    for index, record in enumerate(ranges):
        if (not isinstance(record, dict)
                or _integer(record.get("shot_offset"), "source shot_offset") != index * SCALE_SHOTS
                or _integer(record.get("shots"), "source shots") != SCALE_SHOTS
                or not isinstance(record.get("source"), str) or not record["source"]):
            raise ValueError(f"{path.name}: source ranges must be ordered contiguous 20000-shot blocks")
    dimension = 1 << metadata["measurement_bits"]
    outcomes = _integral_array(data["outcomes"], "baseline outcomes", (BASELINE_SHOTS,), dimension - 1)
    counts = _integral_array(data["counts"], "baseline counts", (dimension,), BASELINE_SHOTS)
    if not np.array_equal(counts, np.bincount(outcomes, minlength=dimension)):
        raise ValueError(f"{path.name}: baseline counts disagree with outcomes")
    return metadata, outcomes, source


def pack_counts(*, reset_scales_dir: Path, raw_baseline: Path, reset_baseline: Path,
                rate: float, output: Path) -> Path:
    """Validate native records, then write a new five-parent count NPZ."""
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if output.suffix != ".npz":
        raise ValueError("--output must end in .npz")
    if isinstance(rate, (bool, np.bool_)) or not np.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("rate must be a finite probability in [0,1], not percent")
    metadata, raw, raw_source = _baseline(raw_baseline, "raw", rate)
    reset_metadata, reset, reset_source = _baseline(reset_baseline, "reset", rate)
    _same_metadata(reset_metadata, metadata, reset_baseline.name)
    dimension = 1 << metadata["measurement_bits"]
    sources = [raw_source, reset_source]
    rows, effective_lambdas = [], []
    for lam in NOMINAL_LAMBDAS:
        token = f"{lam:.1f}".replace(".", "p")
        path = reset_scales_dir / f"lambda_{token}-{token}.npz"
        data, source = _read(path, f"reset_scale_{lam:g}")
        _same_metadata(_metadata(data), metadata, path.name)
        if (_scalar(data, "schema_version") != FOLDING_ARCHIVE_SCHEMA_VERSION
                or _scalar(data, "folding_pattern_scope") != FOLDING_PATTERN_SCOPE
                or not np.array_equal(data["folding_seed_fields"], np.asarray(FOLDING_SEED_FIELDS))):
            raise ValueError(f"{path.name}: incompatible shot-wise folding schema, scope, or seed fields")
        if _scalar(data, "shots") != SCALE_SHOTS:
            raise ValueError(f"{path.name}: scale requires 20000 shots")
        rates, lambdas = np.asarray(data["rates"]), np.asarray(data["lambdas"])
        if (rates.ndim != 1 or rates.size == 0 or rates.dtype.kind not in "iuf"
                or not np.all(np.isfinite(rates)) or np.any(rates < 0) or np.any(rates > 1)
                or np.unique(rates).size != rates.size):
            raise ValueError(f"{path.name}: invalid rates array")
        if lambdas.shape != (1,) or lambdas.dtype.kind not in "iuf" or lambdas[0] != lam:
            raise ValueError(f"{path.name}: nominal lambda metadata differs from filename")
        indices = np.flatnonzero(np.isclose(rates, rate, rtol=0, atol=5e-12))
        if indices.size != 1:
            raise ValueError(f"{path.name}: require exactly one --rate entry")
        plan_hashes = np.asarray(data["fold_plan_sha256"])
        if (plan_hashes.shape != (rates.size, 1) or plan_hashes.dtype.kind != "U"
                or any(not re.fullmatch(r"[0-9a-fA-F]{64}", value) for value in plan_hashes.flat)):
            raise ValueError(f"{path.name}: invalid fold-plan SHA256 metadata")
        outcomes = _integral_array(data["outcomes"], "scale outcomes", (rates.size, 1, SCALE_SHOTS), dimension - 1)
        counts = _integral_array(data["counts"], "scale counts", (rates.size, 1, dimension), SCALE_SHOTS)
        for index in range(rates.size):
            if not np.array_equal(counts[index, 0], np.bincount(outcomes[index, 0], minlength=dimension)):
                raise ValueError(f"{path.name}: scale counts disagree with outcomes")
        folds = _integral_array(data["folded_gate_counts_by_round"], "folded_gate_counts_by_round",
                                (1, TEMPLATE_ROUNDS), GATES_PER_ROUND)
        expected_folds = round((lam - 1.0) * GATES_PER_ROUND / 2.0)
        if not np.all(folds == expected_folds):
            raise ValueError(f"{path.name}: fold counts disagree with matched-template nominal lambda")
        selected = int(indices[0])
        if lam == 1.0 and not np.array_equal(outcomes[selected, 0], reset[:SCALE_SHOTS]):
            raise ValueError("Reset lambda=1 outcomes differ from retained initial 20000 baseline shots")
        rows.append(counts[selected, 0])
        effective_lambdas.append(1.0 + 2.0 * float(folds.sum()) / TEMPLATE_GATE_COUNT)
        sources.append(source)
    parents = CountParents(
        reset_scale_counts=np.stack(rows),
        reset_extension_counts=np.bincount(reset[SCALE_SHOTS:], minlength=dimension),
        raw_initial_counts=np.bincount(raw[:SCALE_SHOTS], minlength=dimension),
        raw_extension_counts=np.bincount(raw[SCALE_SHOTS:], minlength=dimension),
        effective_lambdas=np.asarray(effective_lambdas),
    )
    provenance = {
        "operation": "pack native ordered shot records into C1 count parents",
        "sources": sources, "rate": float(rate), "nominal_lambdas": list(NOMINAL_LAMBDAS),
        "parent_boundary": "source_ranges_json: offsets 0,20000,...,140000; 20000 shots each",
        "initial_shots": SCALE_SHOTS, "extension_shots": BASELINE_SHOTS - SCALE_SHOTS,
        "effective_scale_rule": "1 + 2*sum(folded_gate_counts_by_round)/two_qubit_gate_count",
        "no_simulation_or_bootstrap": True,
    }
    arrays = {key: getattr(parents, key) for key in (
        "reset_scale_counts", "reset_extension_counts", "raw_initial_counts", "raw_extension_counts", "effective_lambdas"
    )}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        np.savez_compressed(handle, **arrays, **metadata, rate=np.float64(rate),
                            provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)))
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset-scales-dir", type=Path, required=True)
    parser.add_argument("--raw-baseline", type=Path, required=True)
    parser.add_argument("--reset-baseline", type=Path, required=True)
    parser.add_argument("--rate", type=float, required=True, help="p2 probability, e.g. 0.0001 = 0.01 percent")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = pack_counts(**vars(args))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Wrote portable count parents: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
