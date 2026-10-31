"""Safe ingestion of the historical shot-level pickle files.

The original simulator stored dictionaries whose values are NumPy arrays.  We
read only the three NumPy constructors needed by those files, validate every
array, and immediately convert the snapshot to a non-executable ``.npz`` count
archive.  Downstream analysis never needs to unpickle the historical files.
"""

from __future__ import annotations

import hashlib
import io
import json
import pickle
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that permits only the NumPy constructors in the snapshot."""

    _ALLOWED = {
        ("numpy._core.numeric", "_frombuffer"),
        ("numpy.core.numeric", "_frombuffer"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy", "dtype"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in self._ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked pickle global: {module}.{name}")


def restricted_load(path: Path) -> Any:
    """Load one historical pickle without allowing arbitrary globals."""

    with path.open("rb") as handle:
        return RestrictedUnpickler(handle).load()


def restricted_loads(payload: bytes) -> Any:
    """Bytes variant used by tests."""

    return RestrictedUnpickler(io.BytesIO(payload)).load()


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Read and minimally validate a project configuration file."""

    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)

    required = {
        "instance",
        "N",
        "a",
        "order",
        "measurement_bits",
        "rates",
        "lambdas",
        "shots_per_lambda",
    }
    if not config.get("require_all_lambdas_from_shotwise", False):
        required.update({"reset_data_glob", "no_reset_data_glob"})
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"missing config fields: {', '.join(missing)}")
    if int(config["measurement_bits"]) <= 0:
        raise ValueError("measurement_bits must be positive")
    if int(config["order"]) <= 0:
        raise ValueError("order must be positive")
    return config, config_path


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a path relative to this reproducibility project."""

    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _glob_project(pattern: str) -> list[Path]:
    path = Path(pattern)
    if path.is_absolute():
        parent = path.parent
        matches = sorted(parent.glob(path.name))
    else:
        matches = sorted(PROJECT_ROOT.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no files matched {pattern!r}")
    return matches


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def bits_to_outcomes(bits: np.ndarray) -> np.ndarray:
    """Convert rows of little-endian measured bits to integer outcomes.

    Historical column 0 is the least-significant bit, matching the simulator's
    ``sum(bit[j] * 2**j)`` convention.
    """

    values = np.asarray(bits)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D bit array, got shape {values.shape}")
    if values.shape[1] > 62:
        raise ValueError("at most 62 measured bits are supported")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("shot array contains values other than 0 and 1")
    powers = np.left_shift(np.int64(1), np.arange(values.shape[1], dtype=np.int64))
    return values.astype(np.int64, copy=False) @ powers


def bits_to_counts(bits: np.ndarray, measurement_bits: int) -> np.ndarray:
    values = np.asarray(bits)
    if values.shape[1:] != (measurement_bits,):
        raise ValueError(
            f"expected {measurement_bits} columns, got shape {values.shape}"
        )
    outcomes = bits_to_outcomes(values)
    return np.bincount(outcomes, minlength=1 << measurement_bits).astype(
        np.int32, copy=False
    )


def _as_float(value: Any, label: str) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    raise ValueError(f"{label} must be numeric, got {type(value).__name__}")


def _parse_record_key(key: Any) -> tuple[float, float]:
    if not isinstance(key, tuple) or len(key) != 2:
        raise ValueError(f"unexpected pickle key {key!r}; expected (rate, lambda)")
    return _as_float(key[0], "rate"), _as_float(key[1], "lambda")


def _nearest_index(value: float, expected: np.ndarray, label: str) -> int:
    index = int(np.argmin(np.abs(expected - value)))
    if not np.isclose(expected[index], value, rtol=0.0, atol=5e-12):
        raise ValueError(f"unexpected {label}={value}; expected one of {expected.tolist()}")
    return index


def _validate_shots(
    array: Any,
    *,
    shots: int,
    measurement_bits: int,
    source: Path,
    key: Any,
) -> np.ndarray:
    values = np.asarray(array)
    expected_shape = (shots, measurement_bits)
    if values.shape != expected_shape:
        raise ValueError(
            f"{source}: key {key!r} has shape {values.shape}, expected {expected_shape}"
        )
    if values.dtype.kind not in "biu":
        raise ValueError(f"{source}: key {key!r} has non-integral dtype {values.dtype}")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError(f"{source}: key {key!r} contains non-binary values")
    return values


def _load_count_grid(
    files: Iterable[Path],
    *,
    rates: np.ndarray,
    lambdas: np.ndarray,
    shots: int,
    measurement_bits: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    counts = np.full(
        (rates.size, lambdas.size, 1 << measurement_bits),
        -1,
        dtype=np.int32,
    )
    provenance: list[dict[str, Any]] = []

    for source in files:
        payload = restricted_load(source)
        if not isinstance(payload, dict):
            raise ValueError(f"{source}: top-level object is not a dictionary")
        provenance.append(
            {
                "path": str(source.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(source),
                "records": len(payload),
            }
        )
        for key, array in payload.items():
            rate, lam = _parse_record_key(key)
            rate_i = _nearest_index(rate, rates, "rate")
            lambda_i = _nearest_index(lam, lambdas, "lambda")
            if counts[rate_i, lambda_i, 0] != -1:
                raise ValueError(f"duplicate record for rate={rate}, lambda={lam}")
            values = _validate_shots(
                array,
                shots=shots,
                measurement_bits=measurement_bits,
                source=source,
                key=key,
            )
            counts[rate_i, lambda_i] = bits_to_counts(values, measurement_bits)

    missing = np.argwhere(counts[:, :, 0] < 0)
    if missing.size:
        labels = [
            f"(rate={rates[i]:g}, lambda={lambdas[j]:g})" for i, j in missing
        ]
        raise ValueError("missing shot records: " + ", ".join(labels))
    totals = counts.sum(axis=-1)
    if not np.all(totals == shots):
        raise ValueError("one or more count vectors do not sum to shots_per_lambda")
    return counts, provenance


def _extract_rate_from_filename(path: Path) -> float:
    match = re.search(r"_p([0-9]+(?:\.[0-9]+)?)\.pkl$", path.name)
    if match is None:
        raise ValueError(f"cannot infer noise rate from {path.name}")
    return float(match.group(1))


def _load_no_reset_counts(
    files: Iterable[Path],
    *,
    rates: np.ndarray,
    shots: int,
    measurement_bits: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Load no-reset files, requiring exactly one lambda=1 record per rate."""

    result = np.full((rates.size, 1 << measurement_bits), -1, dtype=np.int32)
    provenance: list[dict[str, Any]] = []
    for source in files:
        payload = restricted_load(source)
        if not isinstance(payload, dict) or len(payload) != 1:
            raise ValueError(f"{source}: expected exactly one no-reset record")
        provenance.append(
            {
                "path": str(source.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(source),
                "records": len(payload),
            }
        )
        key, array = next(iter(payload.items()))
        rate, lam = _parse_record_key(key)
        filename_rate = _extract_rate_from_filename(source)
        if not np.isclose(rate, filename_rate, rtol=0.0, atol=5e-12):
            raise ValueError(f"{source}: filename rate and key rate disagree")
        if not np.isclose(lam, 1.0, rtol=0.0, atol=5e-12):
            raise ValueError(f"{source}: no-reset record has lambda={lam}, expected 1")
        rate_i = _nearest_index(rate, rates, "rate")
        if result[rate_i, 0] != -1:
            raise ValueError(f"duplicate no-reset record for rate={rate}")
        values = _validate_shots(
            array,
            shots=shots,
            measurement_bits=measurement_bits,
            source=source,
            key=key,
        )
        result[rate_i] = bits_to_counts(values, measurement_bits)

    missing = np.flatnonzero(result[:, 0] < 0)
    if missing.size:
        raise ValueError(f"missing no-reset rates: {rates[missing].tolist()}")
    if not np.all(result.sum(axis=-1) == shots):
        raise ValueError("one or more no-reset count vectors have the wrong total")
    return result, provenance


def processed_path(config: dict[str, Any]) -> Path:
    return PROJECT_ROOT / "data" / "processed" / f"{config['instance']}_counts.npz"


def build_processed_counts(
    config_path: str | Path, *, overwrite: bool = False
) -> Path:
    """Validate raw files and create a safe, compact count archive."""

    config, resolved_config = load_config(config_path)
    destination = processed_path(config)
    if destination.exists() and not overwrite:
        return destination

    rates = np.asarray(config["rates"], dtype=np.float64)
    lambdas = np.asarray(config["lambdas"], dtype=np.float64)
    shots = int(config["shots_per_lambda"])
    measurement_bits = int(config["measurement_bits"])

    reset_counts, reset_sources = _load_count_grid(
        _glob_project(config["reset_data_glob"]),
        rates=rates,
        lambdas=lambdas,
        shots=shots,
        measurement_bits=measurement_bits,
    )
    no_reset_counts, no_reset_sources = _load_no_reset_counts(
        _glob_project(config["no_reset_data_glob"]),
        rates=rates,
        shots=shots,
        measurement_bits=measurement_bits,
    )

    metadata = {
        "schema_version": 1,
        "instance": config["instance"],
        "N": int(config["N"]),
        "a": int(config["a"]),
        "order": int(config["order"]),
        "measurement_bits": measurement_bits,
        "bit_order": "column_0_is_lsb",
        "shots_per_distribution": shots,
        "folding_design": config.get("folding_design"),
        "folding_pattern_replications": int(
            config.get("folding_pattern_replications", 1)
        ),
        "source_config": str(resolved_config.relative_to(PROJECT_ROOT)),
        "source_config_sha256": sha256_file(resolved_config),
        "reset_sources": reset_sources,
        "no_reset_sources": no_reset_sources,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        rates=rates,
        lambdas=lambdas,
        reset_counts=reset_counts,
        no_reset_counts=no_reset_counts,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return destination


def load_processed_counts(path: str | Path) -> dict[str, Any]:
    """Load a processed archive with pickle support explicitly disabled."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "rates",
            "lambdas",
            "reset_counts",
            "no_reset_counts",
            "metadata_json",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{source}: missing arrays {sorted(missing)}")
        result: dict[str, Any] = {
            "rates": archive["rates"].copy(),
            "lambdas": archive["lambdas"].copy(),
            "reset_counts": archive["reset_counts"].copy(),
            "no_reset_counts": archive["no_reset_counts"].copy(),
            "metadata": json.loads(str(archive["metadata_json"].item())),
        }
    return result
