"""Synthetic cache-reduction tests: no fitting, shots, GPU, or research inputs."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "reproduce_published_results",
    Path(__file__).resolve().parents[1] / "scripts/reproduce_published_results.py",
)
published = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(published)


def test_scalar_selection_uses_labels_and_explicit_axis_positions():
    canonical = np.arange(2 * 3 * 7, dtype=float).reshape(2, 3, 7)
    actual = published.scalar_series(
        canonical.transpose(2, 0, 1), ("second", "first"), ("gap", "other", "tvd"),
        "first", "tvd", axes=("replicate", "route", "quantity"), expected_replicates=7,
    )
    np.testing.assert_array_equal(actual, canonical[1, 2])


@pytest.mark.parametrize("change", ["axes", "shape", "duplicate", "missing", "nonfinite", "object"])
def test_scalar_selection_fails_closed_on_ambiguous_or_invalid_metadata(change):
    array = np.arange(24, dtype=float).reshape(2, 3, 4)
    routes, quantities = ("r0", "r1"), ("q0", "q1", "q2")
    kwargs = dict(axes=("route", "quantity", "replicate"), expected_replicates=4)
    route = "r1"
    if change == "axes":
        kwargs["axes"] = ("route", "route", "replicate")
    elif change == "shape":
        array = array.transpose(1, 0, 2)
    elif change == "duplicate":
        routes = ("r0", "r0")
    elif change == "missing":
        route = "absent"
    elif change == "nonfinite":
        array[1, 2, 1] = np.nan
    else:
        array = array.astype(object)
    with pytest.raises(ValueError):
        published.scalar_series(array, routes, quantities, route, "q2", **kwargs)


def test_distribution_summary_preserves_the_parent_and_all_exact_counts():
    values = np.array([-1., -1e-11, 0., 1e-11, 1.])
    summary = published.distribution_summary(values, 2.)
    assert summary["point"] == 2. and summary["bootstrap_mean"] == 0.
    assert summary["exact_negative_count"] == summary["exact_positive_count"] == 2
    assert summary["exact_zero_count"] == 1
    assert summary["numerical_negative_count"] == summary["numerical_positive_count"] == 1
    assert summary["numerical_zero_count"] == 3
    assert summary["point_outside_interval"] is True
    assert summary["pointwise_interval_sign"] == "unresolved"
    quantiles = np.quantile(values, [.001, .01, .025, .5, .975, .99, .999], method="linear")
    for name, q in zip(("q001", "q01", "ci_low", "bootstrap_median", "ci_high", "q99", "q999"), quantiles):
        assert summary[name] == q


@pytest.mark.parametrize("value,sign", [(1., "positive"), (-1., "negative"), (0., "unresolved")])
def test_distribution_sign_uses_numerical_tolerance(value, sign):
    assert published.distribution_summary(np.full(10, value), value)["pointwise_interval_sign"] == sign


@pytest.mark.parametrize("array,point", [(np.array([]), 0.), (np.array([np.nan]), 0.),
                                         (np.array([1.]), np.inf), (np.ones((2, 3)), 0.)])
def test_distribution_summary_rejects_nonfinite_or_wrong_axes(array, point):
    with pytest.raises(ValueError):
        published.distribution_summary(array, point)


def test_zero_width_percentiles_do_not_discard_large_rare_tails():
    values = np.r_[np.zeros(9786), np.full(30, .2), np.ones(184)]
    summary = published.success_summary(values, 0.)
    assert summary["ci_low"] == summary["ci_high"] == 0.
    assert summary["maximum"] == 1. and summary["bootstrap_mean"] > 0
    assert summary["numerical_zero_count"] == 9786
    assert summary["numerical_one_count"] == 184
    assert summary["point"] == 0.


@pytest.mark.parametrize("values", [np.array([-1e-4, .1]), np.array([.1, 1.001]), np.array([np.inf]), np.empty(0)])
def test_success_summary_rejects_invalid_probabilities(values):
    with pytest.raises(ValueError):
        published.success_summary(values, .2)


@pytest.mark.parametrize("a,order,count", [(50, 2, 69), (13, 4, 70), (16, 6, 48)])
def test_strict_decoder_uniform_reference_counts(a, order, count):
    mask = published.strict_order_mask(a, order)
    assert mask.shape == (16384,) and mask.dtype == bool
    assert int(mask.sum()) == count and not mask[0]


def test_final_condition_grid_includes_only_three_gate_matched_cases():
    grid = published.expected_c1_jobs()
    assert len(grid) == 40
    assert sum(budget == "equal_shot" for budget, _, _ in grid) == 37
    assert {item for item in grid if item[0] == "gate_matched"} == {
        ("gate_matched", instance, .0001) for instance in published.CASES
    }
    assert all(("equal_shot", "n119_a16", rate) in grid for rate in published.EXTRA_R6)


def test_json_scalar_parses_unicode_and_utf8_without_eval_or_pickle():
    for value in (np.asarray('{"field": 12}'), np.asarray(b'{"field": 12}')):
        assert published.json_scalar(value, "configuration_json") == {"field": 12}


@pytest.mark.parametrize("value", [np.asarray('{"x": 1, "x": 2}'), np.asarray('{"x": NaN}'),
                                    np.asarray('[1, 2]'), np.asarray(["{}"]), np.asarray({"x": 2}, dtype=object)])
def test_json_scalar_rejects_ambiguous_or_object_metadata(value):
    with pytest.raises(ValueError):
        published.json_scalar(value, "configuration_json")


def test_comparison_checks_every_generated_value_and_preserves_reference_order():
    fields = ("route", "rate", "point", "point_outside_interval")
    generated = [dict(route="a", rate=.0001, point=.2, point_outside_interval=True),
                 dict(route="b", rate=.0001, point=.4, point_outside_interval=False)]
    reference = [{name: str(row[name]) for name in fields} for row in reversed(generated)]
    actual = published.compare_and_order(generated, reference, fields, key_fields=("route", "rate"))
    assert [row["route"] for row in actual] == ["b", "a"]
    reference[0]["point"] = "0.5"
    with pytest.raises(ValueError, match="Regenerated value differs"):
        published.compare_and_order(generated, reference, fields, key_fields=("route", "rate"))


def test_comparison_rejects_duplicate_reference_keys():
    reference = [dict(route="x", rate="0.1", point="0.3")] * 2
    with pytest.raises(ValueError, match="scopes differ"):
        published.compare_and_order([dict(route="x", rate=.1, point=.3)], reference,
                                    ("route", "rate", "point"), key_fields=("route", "rate"))


@pytest.fixture
def synthetic_data_root(tmp_path):
    root = tmp_path / "data"
    directory = root / "bootstrap"
    directory.mkdir(parents=True)
    cache = directory / "synthetic.npz"
    np.savez_compressed(cache, values=np.arange(5), object_metadata=np.asarray({"not": "loaded"}, dtype=object))
    checksum = hashlib.sha256(cache.read_bytes()).hexdigest()
    manifest = dict(configuration_id=published.PROFILE,
                    files=[dict(path="synthetic.npz", public_sha256=checksum)])
    (directory / "publication_manifest.json").write_text(json.dumps(manifest))
    return root


def test_data_reader_uses_public_hash_and_never_loads_unrequested_object_array(synthetic_data_root):
    reader = published.DataReader(synthetic_data_root)
    arrays = reader.load_npz("bootstrap/synthetic.npz", ("values",))
    np.testing.assert_array_equal(arrays["values"], np.arange(5))
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        reader.load_npz("bootstrap/synthetic.npz", ("object_metadata",))


def test_data_reader_detects_tampered_cache(synthetic_data_root):
    cache = synthetic_data_root / "bootstrap/synthetic.npz"
    cache.write_bytes(cache.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checksum differs"):
        published.DataReader(synthetic_data_root).path("bootstrap/synthetic.npz")


def test_data_reader_rejects_path_escape_and_duplicate_file_records(synthetic_data_root):
    path = synthetic_data_root / "bootstrap/publication_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"] *= 2
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Duplicate publication"):
        published.DataReader(synthetic_data_root)
    manifest["files"] = [dict(path="../outside.npz", public_sha256="a" * 64)]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        published.DataReader(synthetic_data_root)


def test_existing_or_in_dataset_output_directory_is_refused(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        published.reproduce(data_root, existing)
    with pytest.raises(ValueError, match="outside the immutable data root"):
        published.reproduce(data_root, data_root / "new-output")
    assert not (data_root / "new-output").exists()


def test_cli_requires_explicit_input_and_output_paths(tmp_path):
    with pytest.raises(SystemExit):
        published.arguments([])
    args = published.arguments(["--data-root", str(tmp_path / "data"), "--output-dir", str(tmp_path / "out")])
    assert args.data_root == tmp_path / "data" and args.output_dir == tmp_path / "out"
