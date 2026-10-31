"""Synthetic native-record conversion checks; no GPU or saved research data."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from shor_qem.portable_c1 import load_counts


SPEC = importlib.util.spec_from_file_location(
    "pack_simulator_counts", Path(__file__).resolve().parents[1] / "scripts/pack_simulator_counts.py"
)
packer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packer)


@pytest.fixture
def native_records(tmp_path):
    """D=16 schema fixture, not a simulated N=119 probability distribution."""
    metadata = {
        "N": 119, "a": 13, "order": 4, "measurement_bits": 4,
        "instance": "n119_a13", "two_qubit_gate_count": 49686,
        "ops_cache_sha256": "a" * 64, "circuit_variant": "gate_count_matched_common_opt1",
    }
    ranges = [{"shot_offset": i * 20000, "shots": 20000,
               "source": "original_run" if i == 0 else f"task_{i}"} for i in range(8)]
    rates = np.asarray([0.0001, 0.0002])
    initial_reset = np.arange(20000, dtype=np.int64) % 16
    raw = (np.arange(160000, dtype=np.int64) * 3 + 1) % 16
    reset = np.concatenate((initial_reset, np.arange(140000, dtype=np.int64) % 13))
    paths = {}
    for mode, outcomes in (("raw", raw), ("reset", reset)):
        path = tmp_path / f"{mode}_160k.npz"
        np.savez_compressed(path, **metadata, schema_version=1, global_seed=42, mode=mode, rate=0.0001,
                            **{"lambda": 1.0}, shots=160000, outcomes=outcomes,
                            counts=np.bincount(outcomes, minlength=16),
                            source_ranges_json=json.dumps(ranges))
        paths[f"{mode}_baseline"] = path
    scales = tmp_path / "scales"
    scales.mkdir()
    for lam in packer.NOMINAL_LAMBDAS:
        token = f"{lam:.1f}".replace(".", "p")
        outcomes = np.stack((initial_reset, (initial_reset + 1) % 16))[:, None, :]
        counts = np.stack([np.bincount(row[0], minlength=16) for row in outcomes])[:, None, :]
        folds = np.full((1, 14), round((lam - 1) * 3549 / 2), dtype=np.int64)
        np.savez_compressed(scales / f"lambda_{token}-{token}.npz", **metadata,
                            schema_version=packer.FOLDING_ARCHIVE_SCHEMA_VERSION,
                            folding_pattern_scope=packer.FOLDING_PATTERN_SCOPE,
                            folding_seed_fields=np.asarray(packer.FOLDING_SEED_FIELDS),
                            fold_plan_sha256=np.full((rates.size, 1), "b" * 64),
                            folding_seed=42, rates=rates, lambdas=np.asarray([lam]), shots=20000,
                            outcomes=outcomes, counts=counts, folded_gate_counts_by_round=folds)
    return {**paths, "reset_scales_dir": scales, "rate": 0.0001,
            "output": tmp_path / "packed/counts.npz"}


def rewrite(path, **changes):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays.update(changes)
    np.savez_compressed(path, **arrays)


def test_native_chain_preserves_counts_boundaries_and_effective_scales(native_records):
    output = packer.pack_counts(**native_records)
    parents = load_counts(output)
    assert parents.reset_scale_counts.shape == (8, 16)
    np.testing.assert_array_equal(parents.raw_initial_counts, np.full(16, 1250))
    with np.load(native_records["reset_baseline"], allow_pickle=False) as baseline:
        np.testing.assert_array_equal(parents.reset_extension_counts,
                                      np.bincount(baseline["outcomes"][20000:], minlength=16))
    np.testing.assert_allclose(parents.effective_lambdas, [
        1.0, 1.099746407438715, 1.2000563539025078, 1.4001127078050155,
        1.800225415610031, 2.199774584389969, 2.5998872921949845, 3.0,
    ], rtol=0, atol=1e-15)
    with np.load(output, allow_pickle=False) as archive:
        provenance = json.loads(str(archive["provenance_json"]))
        assert len(provenance["sources"]) == 10
        assert all(len(item["sha256"]) == 64 for item in provenance["sources"])
        assert str(native_records["output"].parent.parent) not in str(provenance)
        assert int(archive["a"]) == 13 and int(archive["order"]) == 4


@pytest.mark.parametrize("key,value,match", [
    ("N", 15, "N=119"), ("a", 50, "instance"), ("order", 2, "multiplicative"),
    ("measurement_bits", 15, "measurement_bits"),
    ("ops_cache_sha256", "b" * 64, "ops_cache_sha256"),
    ("global_seed", 43, "global_seed"), ("circuit_variant", "other", "circuit_variant"),
    ("shots", 159999, "160000"), ("rate", 0.0002, "rate"),
    ("mode", "raw", "mode"), ("lambda", 1.1, "lambda=1"),
    ("schema_version", 2, "schema_version"),
])
def test_baseline_metadata_rejected(native_records, key, value, match):
    rewrite(native_records["reset_baseline"], **{key: value})
    with pytest.raises(ValueError, match=match):
        packer.pack_counts(**native_records)
    assert not native_records["output"].exists()


@pytest.mark.parametrize("change", ["missing", "offset", "shots", "duplicate", "no_source"])
def test_retained_source_boundaries_required(native_records, change):
    with np.load(native_records["raw_baseline"], allow_pickle=False) as archive:
        ranges = json.loads(str(archive["source_ranges_json"]))
    if change == "missing":
        ranges.pop()
    elif change == "offset":
        ranges[1]["shot_offset"] = 10000
    elif change == "shots":
        ranges[1]["shots"] = 19999
    elif change == "duplicate":
        ranges[1] = ranges[0].copy()
    else:
        ranges[0].pop("source")
    rewrite(native_records["raw_baseline"], source_ranges_json=json.dumps(ranges))
    with pytest.raises(ValueError, match="source range"):
        packer.pack_counts(**native_records)


@pytest.mark.parametrize("target", ["raw_baseline", "reset_baseline", "scale"])
def test_histograms_must_match_shots(native_records, target):
    path = (native_records["reset_scales_dir"] / "lambda_1p2-1p2.npz"
            if target == "scale" else native_records[target])
    with np.load(path, allow_pickle=False) as archive:
        counts = archive["counts"].copy()
    counts.flat[0] += 1
    counts.flat[1] -= 1
    rewrite(path, counts=counts)
    with pytest.raises(ValueError, match="counts disagree"):
        packer.pack_counts(**native_records)


def test_reset_lambda_one_shared_parent_must_match(native_records):
    path = native_records["reset_scales_dir"] / "lambda_1p0-1p0.npz"
    with np.load(path, allow_pickle=False) as archive:
        outcomes = archive["outcomes"].copy()
    outcomes[0, 0, [0, 1]] = outcomes[0, 0, [1, 0]]  # Same histogram, wrong recorded parent.
    rewrite(path, outcomes=outcomes)
    with pytest.raises(ValueError, match="retained initial"):
        packer.pack_counts(**native_records)


@pytest.mark.parametrize("key,value,match", [
    ("folded_gate_counts_by_round", np.full((1, 14), 170), "fold counts"),
    ("folded_gate_counts_by_round", np.full((1, 14), 177.5), "integers"),
    ("lambdas", np.asarray([1.2]), "nominal lambda"),
    ("rates", np.asarray([0.0002, 0.0003]), "--rate"),
    ("shots", 20001, "20000"),
    ("folding_seed", 43, "global_seed"),
    ("folding_pattern_scope", "fixed_pattern", "shot-wise"),
    ("schema_version", 2, "shot-wise"),
    ("fold_plan_sha256", np.full((2, 1), "invalid"), "fold-plan"),
])
def test_scale_metadata_rejected(native_records, key, value, match):
    path = native_records["reset_scales_dir"] / "lambda_1p1-1p1.npz"
    rewrite(path, **{key: value})
    with pytest.raises(ValueError, match=match):
        packer.pack_counts(**native_records)


@pytest.mark.parametrize("target", ["raw_baseline", "scale"])
def test_outcomes_must_be_integral_and_in_range(native_records, target):
    path = (native_records["reset_scales_dir"] / "lambda_1p2-1p2.npz"
            if target == "scale" else native_records[target])
    with np.load(path, allow_pickle=False) as archive:
        outcomes = archive["outcomes"].astype(float)
    outcomes.flat[0] = 16 if target == "raw_baseline" else 0.5
    rewrite(path, outcomes=outcomes)
    with pytest.raises(ValueError, match="integers"):
        packer.pack_counts(**native_records)


def test_no_overwrite_and_cli(native_records):
    argv = []
    for key, value in native_records.items():
        argv.extend(("--" + key.replace("_", "-"), str(value)))
    assert packer.main(argv) == 0
    original = native_records["output"].read_bytes()
    with pytest.raises(FileExistsError):
        packer.pack_counts(**native_records)
    assert native_records["output"].read_bytes() == original


def test_missing_scale_does_not_create_output(native_records):
    native_records["reset_scales_dir"] = native_records["reset_scales_dir"] / "missing"
    with pytest.raises(FileNotFoundError):
        packer.pack_counts(**native_records)
    assert not native_records["output"].exists()
