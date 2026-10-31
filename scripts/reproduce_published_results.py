#!/usr/bin/env python3
"""Reproduce the published N=119 tables from the distributed completed caches.

Example (CPU only; no fitting or random draws):
    python scripts/reproduce_published_results.py \
        --data-root /path/to/extracted/shor-paper-data \
        --output-dir /path/to/new/reproduced-results

The data root contains ``bootstrap/c1_all_zero`` and
``bootstrap/order_success``.  This program computes every numeric summary from
the retained parent/scalar arrays and checks it against the distributed CSVs.
It never follows historical absolute paths embedded in archival provenance.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

PROFILE = "c1_beta2_all_ridges_zero_20261018"
METHODS = ("linear_zne", "quadratic_zne", "single_exponential_zne", "dual_exponential_zne")
METRICS = ("tvd", "normalized_grouped_peak_margin")
SUCCESS_ROUTES = ("raw_160k", "reset_160k", *METHODS)
CASES = {"n119_a13": (13, 4), "n119_a50": (50, 2), "n119_a16": (16, 6)}
COMMON_RATES = (1e-5, 2e-5, 3e-5, 4e-5, 5e-5, .0001, .0002, .0003, .0004, .0005, .0006)
EXTRA_R6 = (.00012, .00014, .00016, .00018)
REPLICATES = 10000
TOLERANCE = 1e-10


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, f"Duplicate JSON field: {key}")
        value[key] = item
    return value


def parse_json(text: str) -> dict:
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    value = json.loads(text, object_pairs_hook=_unique_object, parse_constant=reject_constant)
    require(isinstance(value, dict), "Expected a JSON object")
    return value


def text_scalar(array: Any, name: str) -> str:
    value = np.asarray(array)
    require(value.ndim == 0 and value.dtype.kind in "US", f"{name} must be a scalar string")
    item = value.item()
    return item.decode("utf-8") if isinstance(item, bytes) else item


def json_scalar(array: Any, name: str) -> dict:
    return parse_json(text_scalar(array, name))


def labels(array: Any, name: str) -> tuple[str, ...]:
    value = np.asarray(array)
    require(value.ndim == 1 and value.dtype.kind in "US", f"{name} must be a string-label vector")
    result = tuple(item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in value.tolist())
    require(len(result) > 0 and all(result) and len(set(result)) == len(result),
            f"{name} labels must be nonempty and unique")
    return result


def scalar_series(array: Any, route_labels: Sequence[str], quantity_labels: Sequence[str],
                  route: str, quantity: str, *, axes: Sequence[str], expected_replicates: int) -> np.ndarray:
    """Select a series using explicit labels, never an inferred axis position."""
    values = np.asarray(array)
    route_labels, quantity_labels, axes = tuple(route_labels), tuple(quantity_labels), tuple(axes)
    require(values.ndim == 3 and values.dtype.kind in "fiu", "Scalar cache must be a real three-axis array")
    require(len(set(axes)) == len(axes) == 3 and set(axes) == {"route", "quantity", "replicate"},
            "Ambiguous scalar axis labels")
    require(len(set(route_labels)) == len(route_labels) and len(set(quantity_labels)) == len(quantity_labels),
            "Ambiguous route or quantity labels")
    require(route in route_labels and quantity in quantity_labels, "Requested scalar label is absent")
    shape = {"route": len(route_labels), "quantity": len(quantity_labels), "replicate": expected_replicates}
    require(values.shape == tuple(shape[axis] for axis in axes), "Scalar array shape does not match labeled axes")
    canonical = np.transpose(values, tuple(axes.index(axis) for axis in ("route", "quantity", "replicate")))
    result = canonical[route_labels.index(route), quantity_labels.index(quantity)]
    require(np.all(np.isfinite(result)), "Locked-metric scalar values must all be finite")
    return result


def distribution_summary(samples: Any, point: float) -> dict:
    """The original linear-percentile summary, including tails and sign counts."""
    values = np.asarray(samples, dtype=float)
    require(values.ndim == 1 and values.size > 0 and np.isfinite(values).all() and np.isfinite(point),
            "A scalar summary needs a nonempty finite series and parent point")
    qs = np.quantile(values, (.001, .01, .025, .5, .975, .99, .999), method="linear")
    return dict(
        point=float(point), replicates=len(values), finite_replicates=len(values), nonfinite_replicates=0,
        numerical_tolerance=TOLERANCE, ci_low=float(qs[2]), ci_high=float(qs[4]),
        bootstrap_mean=float(np.mean(values)), bootstrap_median=float(qs[3]),
        minimum=float(values.min()), maximum=float(values.max()),
        q001=float(qs[0]), q01=float(qs[1]), q99=float(qs[5]), q999=float(qs[6]),
        exact_negative_count=int(np.count_nonzero(values < 0)),
        exact_zero_count=int(np.count_nonzero(values == 0)),
        exact_positive_count=int(np.count_nonzero(values > 0)),
        numerical_negative_count=int(np.count_nonzero(values < -TOLERANCE)),
        numerical_zero_count=int(np.count_nonzero(np.abs(values) <= TOLERANCE)),
        numerical_positive_count=int(np.count_nonzero(values > TOLERANCE)),
        below_central_interval_count=int(np.count_nonzero(values < qs[2])),
        above_central_interval_count=int(np.count_nonzero(values > qs[4])),
        point_outside_interval=bool(point < qs[2] or point > qs[4]),
        pointwise_interval_sign="positive" if qs[2] > TOLERANCE else "negative" if qs[4] < -TOLERANCE else "unresolved",
    )


def strict_order_mask(a: int, order: int, *, dimension: int = 16384, modulus: int = 119) -> np.ndarray:
    """Choose one bounded rational; do not reduce or search other denominators."""
    require(modulus == 119 and dimension == 16384 and (a, order) in CASES.values(),
            "This release uses the three N=119, 14-bit strict decoder masks")
    candidates = [Fraction(y, dimension).limit_denominator(modulus) for y in range(dimension)]
    mask = np.array([candidate.numerator != 0 and pow(a, candidate.denominator, modulus) == 1
                     and candidate.denominator == order for candidate in candidates], dtype=bool)
    require(int(mask.sum()) == {2: 69, 4: 70, 6: 48}[order], "Strict decoder mask changed")
    return mask


class DataReader:
    """Resolve only distributed files and verify their PUBLIC byte hashes."""
    def __init__(self, root: Path):
        self.root = root.resolve(strict=True)
        manifest_path = self.root / "bootstrap/publication_manifest.json"
        require(manifest_path.is_file(), "Missing bootstrap/publication_manifest.json")
        manifest = parse_json(manifest_path.read_text(encoding="utf-8"))
        require(manifest.get("configuration_id") == PROFILE and isinstance(manifest.get("files"), list),
                "Unknown bootstrap publication manifest")
        self.records = {}
        self.checked = {"bootstrap/publication_manifest.json": sha256(manifest_path)}
        for record in manifest["files"]:
            require(isinstance(record, dict) and isinstance(record.get("path"), str), "Invalid publication file record")
            relative = Path("bootstrap") / record["path"]
            require(not Path(record["path"]).is_absolute() and ".." not in relative.parts,
                    "Publication file path escapes the data root")
            require(relative.as_posix() not in self.records, "Duplicate publication file path")
            require(isinstance(record.get("public_sha256"), str) and len(record["public_sha256"]) == 64,
                    "Missing public file checksum")
            self.records[relative.as_posix()] = record

    def path(self, relative: str | Path) -> Path:
        path = (self.root / relative).resolve(strict=True)
        require(path.is_relative_to(self.root), "Input path escapes the data root")
        key = path.relative_to(self.root).as_posix()
        require(key in self.records, f"Input absent from publication manifest: {key}")
        if key not in self.checked:
            self.checked[key] = sha256(path)
            require(self.checked[key] == self.records[key]["public_sha256"], f"Input checksum differs: {key}")
        return path

    def read_json(self, relative: str | Path) -> dict:
        return parse_json(self.path(relative).read_text(encoding="utf-8"))

    def load_npz(self, relative: str | Path, required_keys: Sequence[str]) -> dict[str, np.ndarray]:
        with np.load(self.path(relative), allow_pickle=False) as archive:
            require(set(required_keys) <= set(archive.files), f"Missing NPZ fields in {relative}")
            return {name: archive[name] for name in required_keys}

    def read_csv(self, relative: str | Path) -> tuple[list[str], list[dict]]:
        with self.path(relative).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields, rows = list(reader.fieldnames or []), list(reader)
        require(fields and len(set(fields)) == len(fields) and all(None not in row for row in rows),
                "Missing or ambiguous CSV columns")
        return fields, rows


def expected_c1_jobs() -> set[tuple[str, str, float]]:
    return {(budget, instance, rate)
            for instance, (_, order) in CASES.items()
            for budget, rates in (("equal_shot", COMMON_RATES + (EXTRA_R6 if order == 6 else ())),
                                  ("gate_matched", (.0001,)))
            for rate in rates}


def row_key(row: Mapping[str, Any], fields: Sequence[str]) -> tuple:
    return tuple(float(row[name]) if name == "rate" else str(row[name]) for name in fields)


def compare_and_order(generated: Sequence[dict], reference: Sequence[dict], fields: Sequence[str], *, key_fields: Sequence[str]) -> list[dict]:
    """Use the supplied CSV only for verification and ordering, not its values."""
    index = {row_key(row, key_fields): row for row in generated}
    require(len(index) == len(generated), "Duplicate generated result rows")
    reference_keys = [row_key(row, key_fields) for row in reference]
    require(len(set(reference_keys)) == len(reference_keys) and set(reference_keys) == set(index),
            "Supplied and regenerated result scopes differ")
    ordered = []
    for original, key in zip(reference, reference_keys):
        actual = index[key]
        require(set(actual) == set(fields), "Generated result fields differ from the supplied table schema")
        for name in fields:
            require(str(actual[name]) == original[name], f"Regenerated value differs at {key}: {name}")
        ordered.append(actual)
    return ordered


def reproduce_c1(reader: DataReader) -> tuple[list[str], list[dict], dict]:
    base = Path("bootstrap/c1_all_zero")
    summary = reader.read_json(base / "aggregate/summary.json")
    require(summary.get("status") == "complete" and summary.get("publication_ready") is True
            and summary.get("completed_jobs") == 40 and summary.get("replicates_per_job") == REPLICATES,
            "The C1 cache summary is incomplete")
    require(summary.get("configuration_id") == PROFILE, "C1 configuration identity changed")
    directories = sorted((reader.root / base).glob("*/*/rate_*"))
    directories = [path for path in directories if path.is_dir()]
    require(len(directories) == 40, "Expected 40 C1 cache directories")
    rows, cases = [], {}
    keys = ("scalar_draws", "direct_draws", "routes", "quantities", "direct_routes", "direct_metrics",
            "identity", "source_job_identity", "configuration_json")
    for directory in directories:
        relative = directory.relative_to(reader.root)
        manifest = reader.read_json(relative / "manifest.json")
        config = manifest["configuration"]
        require(manifest.get("status") == "complete" and manifest.get("final_10000_contract") is True,
                "C1 case is not a complete final cache")
        budget, instance, rate = config["budget"], config["instance"], float(config["rate"])
        case_key = budget, instance, rate
        require(case_key in expected_c1_jobs() and case_key not in cases, "Unknown or duplicate C1 case")
        require(directory.parent.name == instance and directory.parent.parent.name == budget, "C1 folder metadata disagrees")
        require(config.get("configuration_id") == PROFILE and config.get("replicates") == REPLICATES
                and config.get("batch_size") == 25 and config.get("weight_exponent") == 2
                and config.get("ridges") == dict.fromkeys(METHODS, 0.), "C1 scientific profile changed")
        require(config.get("dimension") == 16384 and config.get("order") == CASES[instance][1], "C1 instance dimension/order changed")
        require(config.get("per_scale_shots") == (20000 if budget == "equal_shot" else 11188), "Unexpected circuit-shot budget")
        require(config.get("observed_support_reselected") is True and config.get("gamma_refitted_each_replicate") is True,
                "Unrecognized support or gamma selection contract")
        parent = reader.load_npz(relative / "parent_estimates.npz", keys)
        cache = reader.load_npz(relative / "scalar_bootstrap_draws.npz", (*keys, "scalar_axis_order", "replicates"))
        require(int(cache["replicates"]) == REPLICATES, "Incomplete C1 scalar series")
        require(json_scalar(parent["configuration_json"], "configuration_json") == config
                == json_scalar(cache["configuration_json"], "configuration_json"), "C1 embedded configurations disagree")
        identity = text_scalar(cache["identity"], "identity")
        source_identity = text_scalar(cache["source_job_identity"], "source_job_identity")
        require(identity == manifest["identity"] == text_scalar(parent["identity"], "identity"), "C1 parent/cache identity mismatch")
        require(source_identity == text_scalar(parent["source_job_identity"], "source_job_identity"), "C1 source identities differ")
        route_labels, quantity_labels = labels(cache["routes"], "routes"), labels(cache["quantities"], "quantities")
        direct_labels, metric_labels = labels(cache["direct_routes"], "direct_routes"), labels(cache["direct_metrics"], "direct_metrics")
        require(set(route_labels) == set(METHODS) and set(metric_labels) == set(METRICS), "Unrecognized C1 route/metric labels")
        for name, expected in (("routes", route_labels), ("quantities", quantity_labels), ("direct_routes", direct_labels), ("direct_metrics", metric_labels)):
            require(labels(parent[name], name) == expected and tuple(config[name]) == expected, f"Conflicting {name} metadata")
        expected_direct = {"raw_160k", "reset_160k", "raw_20k", "reset_20k"} if budget == "equal_shot" else {"reset_160k"}
        require(set(direct_labels) == expected_direct, "Unrecognized direct-reference budget")
        axes = labels(cache["scalar_axis_order"], "scalar_axis_order")
        require(axes == ("route", "quantity", "replicate"), "Unknown original C1 scalar schema")
        context = dict(configuration_id=PROFILE, budget=budget, instance=instance, rate=rate,
                       rate_percent=100 * rate, identity=identity)
        for route in route_labels:
            origin = ("exact_reuse_unchanged_method" if route != METHODS[-1] else
                      "exact_reuse_existing_zero_control" if config["reused_dual_zero_replicates_expected"] else
                      "new_dual_zero_refit_on_identical_draws")
            for metric in METRICS:
                values = scalar_series(cache["scalar_draws"], route_labels, quantity_labels, route, metric,
                                       axes=axes, expected_replicates=REPLICATES)
                point = scalar_series(parent["scalar_draws"], route_labels, quantity_labels, route, metric,
                                      axes=axes, expected_replicates=1)[0]
                rows.append({**context, "route": route, "quantity": metric, "origin": origin,
                             "source_mixed_identity": source_identity, **distribution_summary(values, point),
                             "metric": metric, "circuit_shots": config["zne_circuit_shots"], "role": ""})
        for route in direct_labels:
            if not route.endswith("_160k"):
                continue
            for metric in METRICS:
                values = scalar_series(cache["direct_draws"], direct_labels, metric_labels, route, metric,
                                       axes=axes, expected_replicates=REPLICATES)
                point = scalar_series(parent["direct_draws"], direct_labels, metric_labels, route, metric,
                                      axes=axes, expected_replicates=1)[0]
                rows.append({**context, "route": route, "quantity": metric, "origin": "exact_reuse_direct_baseline",
                             "source_mixed_identity": source_identity, **distribution_summary(values, point),
                             "metric": metric, "circuit_shots": 160000, "role": "scientific_direct_160k"})
        cases[case_key] = dict(relative=relative, identity=identity)
    require(set(cases) == expected_c1_jobs() and len(rows) == 474, "C1 coverage does not contain 444 equal-shot and 30 gate-matched rows")
    fields, expected = reader.read_csv(base / "aggregate/absolute_locked_metrics.csv")
    rows = compare_and_order(rows, expected, fields, key_fields=("budget", "instance", "rate", "route", "metric"))
    return fields, rows, cases


def success_summary(values: Any, point: float) -> dict:
    values = np.asarray(values, dtype=float)
    require(values.ndim == 1 and values.size > 0 and np.all(np.isfinite(values))
            and values.min() >= 0 and values.max() <= 1 + 1e-12 and np.isfinite(point)
            and 0 <= point <= 1 + 1e-12, "Invalid success-probability series")
    low, high = np.percentile(values, [2.5, 97.5], method="linear")
    return dict(point=float(point), ci_low=float(low), ci_high=float(high),
                bootstrap_mean=float(values.mean()), bootstrap_median=float(np.median(values)),
                minimum=float(values.min()), maximum=float(values.max()),
                numerical_zero_count=int(np.count_nonzero(np.abs(values) <= TOLERANCE)),
                numerical_one_count=int(np.count_nonzero(np.abs(values - 1) <= TOLERANCE)))


def reproduce_success(reader: DataReader, c1_cases: Mapping) -> tuple[list[str], list[dict]]:
    base = Path("bootstrap/order_success")
    summary = reader.read_json(base / "summary.json")
    require(summary.get("status") == "complete" and summary.get("equal_shot_conditions") == 37
            and summary.get("routes") == 6 and summary.get("replicates_per_condition") == REPLICATES,
            "Success bootstrap is incomplete")
    masks = {instance: strict_order_mask(a, order) for instance, (a, order) in CASES.items()}
    directories = sorted((reader.root / base).glob("*/rate_*"))
    directories = [path for path in directories if path.is_dir()]
    require(len(directories) == 37, "Expected 37 strict-success case directories")
    seen, rows = set(), []
    for directory in directories:
        relative = directory.relative_to(reader.root)
        manifest = reader.read_json(relative / "manifest.json")
        config = manifest["configuration"]
        instance, rate = config["instance"], float(config["rate"])
        case_key = "equal_shot", instance, rate
        require(case_key in c1_cases and case_key not in seen, "Unknown or duplicate success case")
        seen.add(case_key)
        require(manifest.get("status") == "complete" and manifest.get("replicates") == REPLICATES
                and config.get("expected_replicates") == REPLICATES and config.get("configuration_id") == PROFILE,
                "Incomplete or different-profile success case")
        require(directory.parent.name == instance and config["source_identity"] == c1_cases[case_key]["identity"],
                "Success and C1 cached parents do not match")
        cache = reader.load_npz(relative / "success_draws.npz", ("success_draws", "routes", "point_estimates", "identity"))
        route_labels = labels(cache["routes"], "routes")
        require(set(route_labels) == set(SUCCESS_ROUTES) and list(route_labels) == config["routes"], "Unknown success routes")
        require(cache["success_draws"].shape == (6, REPLICATES) and cache["point_estimates"].shape == (6,), "Unexpected success array axes")
        require(text_scalar(cache["identity"], "identity") == manifest["identity"], "Success cache identity mismatch")
        parent_relative = c1_cases[case_key]["relative"] / "parent_estimates.npz"
        parent = reader.load_npz(parent_relative, ("raw_full_original", "reset_reference", *("after__" + route for route in METHODS)))
        vectors = {"raw_160k": parent["raw_full_original"], "reset_160k": parent["reset_reference"],
                   **{route: parent["after__" + route] for route in METHODS}}
        parent_hash = reader.checked[parent_relative.as_posix()]
        a, order = CASES[instance]
        for ri, route in enumerate(route_labels):
            vector = vectors[route]
            require(vector.shape == (16384,) and np.all(np.isfinite(vector)) and vector.min() >= 0
                    and abs(vector.sum() - 1) < 1e-10, "Invalid stored parent probability vector")
            point = float(vector[masks[instance]].sum())
            require(point == float(cache["point_estimates"][ri]), "Strict-decoder parent score changed")
            statistics = success_summary(cache["success_draws"][ri], point)
            rows.append(dict(instance=instance, a=a, order=order, rate=rate, rate_percent=rate * 100,
                             route=route, **statistics, point_percent=point * 100,
                             ci_low_percent=statistics["ci_low"] * 100, ci_high_percent=statistics["ci_high"] * 100,
                             replicates=REPLICATES, source_parent_sha256=parent_hash))
    require(seen == {key for key in c1_cases if key[0] == "equal_shot"} and len(rows) == 222, "Incomplete success summary scope")
    fields, expected = reader.read_csv(base / "success_intervals.csv")
    rows = compare_and_order(rows, expected, fields, key_fields=("instance", "rate", "route"))
    return fields, rows


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reproduce(data_root: Path, output_dir: Path) -> dict:
    data_root, output_dir = Path(data_root).resolve(strict=True), Path(output_dir).resolve()
    require(not output_dir.exists(), "Output directory already exists; select a new directory")
    require(not output_dir.is_relative_to(data_root), "Output directory must be outside the immutable data root")
    reader = DataReader(data_root)
    c1_fields, c1_rows, cases = reproduce_c1(reader)
    success_fields, success_rows = reproduce_success(reader, cases)
    # Nothing is written until both products have passed every supplied-value check.
    output_dir.mkdir(parents=True, exist_ok=False)
    c1_path, success_path = output_dir / "absolute_locked_metrics.csv", output_dir / "success_intervals.csv"
    write_csv(c1_path, c1_fields, c1_rows)
    write_csv(success_path, success_fields, success_rows)
    report = dict(
        status="passed", scientific_profile=PROFILE,
        c1_completed_cases=len(cases), c1_absolute_rows=len(c1_rows),
        equal_shot_absolute_rows=sum(row["budget"] == "equal_shot" for row in c1_rows),
        gate_matched_absolute_rows=sum(row["budget"] == "gate_matched" for row in c1_rows),
        success_completed_cases=37, success_rows=len(success_rows), replicates_per_case=REPLICATES,
        all_generated_fields_equal_supplied_csv_strings=True,
        output_sha256={c1_path.name: sha256(c1_path), success_path.name: sha256(success_path)},
        byte_identical_to_supplied_csv={
            c1_path.name: sha256(c1_path) == reader.checked["bootstrap/c1_all_zero/aggregate/absolute_locked_metrics.csv"],
            success_path.name: sha256(success_path) == reader.checked["bootstrap/order_success/success_intervals.csv"],
        },
        input_public_sha256=reader.checked,
        new_quantum_shots=0, new_fits=0, new_bootstrap_draws=0,
        interval_method="original pointwise empirical-parent 2.5/97.5 linear percentiles",
        operations="completed scalar-cache reductions and strict-decoder sums of original parent probabilities",
    )
    with (output_dir / "reproduction_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, type=Path, help="Extracted data root containing bootstrap/ and measurements/")
    parser.add_argument("--output-dir", required=True, type=Path, help="New output directory outside the input data root")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    report = reproduce(args.data_root, args.output_dir)
    print(json.dumps({key: value for key, value in report.items() if key != "input_public_sha256"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
