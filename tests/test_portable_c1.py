"""Small synthetic-only tests: no saved experiment data, simulator, or GPU."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from shor_qem import portable_c1 as portable


@pytest.fixture
def count_arrays():
    grid = np.zeros((8, 16), dtype=np.int64)
    for index in range(8):
        grid[index, [0, 4, 8, 12]] = [6000 - 100 * index, 5000, 4500, 4500 + 100 * index]
    reset_extension = np.zeros(16, dtype=np.int64)
    reset_extension[[0, 4, 8, 12, 15]] = [40_000, 35_000, 30_000, 30_000, 5000]
    raw_initial = np.full(16, 1000, dtype=np.int64)
    raw_initial[0] += 4000
    raw_extension = np.full(16, 8000, dtype=np.int64)
    raw_extension[0] += 12_000
    return {
        "reset_scale_counts": grid,
        "reset_extension_counts": reset_extension,
        "raw_initial_counts": raw_initial,
        "raw_extension_counts": raw_extension,
        "effective_lambdas": np.asarray([1, 1.1, 1.2, 1.4, 1.8, 2.2, 2.6, 3]),
    }


@pytest.fixture
def parents(count_arrays):
    return portable.CountParents(**count_arrays)


def test_strict_counts_schema_and_totals(tmp_path, parents, count_arrays):
    path = tmp_path / "counts.npz"
    np.savez(path, **count_arrays, synthetic_demo_only=True)
    loaded = portable.load_counts(path)
    assert loaded.dimension == 16 and loaded.bits == 4
    np.testing.assert_array_equal(loaded.reset_scale_counts, parents.reset_scale_counts)
    assert loaded.direct_probabilities()["reset_160k"][15] > 0
    assert loaded.reset_parents[:, 15].sum() == 0
    with pytest.raises(ValueError, match="missing separate parent"):
        np.savez(tmp_path / "merged.npz", reset_160k=np.ones(16))
        portable.load_counts(tmp_path / "merged.npz")


@pytest.mark.parametrize("key,change,match", [
    ("reset_scale_counts", lambda a: a.astype(float) + 0.25, "integer"),
    ("reset_extension_counts", lambda a: np.where(np.arange(16) == 15, -1, a), "nonnegative"),
    ("raw_initial_counts", lambda a: a + 1, "exactly 20000"),
    ("raw_extension_counts", lambda a: a[:-1], "shape"),
    ("reset_scale_counts", lambda a: a[:7], "shape"),
    ("reset_scale_counts", lambda a: a[:, :15], "power of two"),
    ("raw_extension_counts", lambda a: np.full(16, np.nan), "finite"),
    ("effective_lambdas", lambda a: a[::-1], "increasing"),
    ("effective_lambdas", lambda a: np.ones(8), "increasing"),
    ("effective_lambdas", lambda a: np.asarray([1, 1.1, 1.2, 1.4, 1.8, 2.2, 2.6, np.inf]), "finite"),
])
def test_reject_malformed_parents(count_arrays, key, change, match):
    count_arrays[key] = change(count_arrays[key])
    with pytest.raises(ValueError, match=match):
        portable.CountParents(**count_arrays)


def test_suite_and_support_only_reset(parents):
    fixed, families = portable.build_suite(parents.effective_lambdas)
    assert set(fixed) == set(portable.METHODS[:2])
    assert families[portable.METHODS[2]].candidate_count == 200
    assert families[portable.METHODS[3]].candidate_count == 19_900
    assert all(family.ridge == 0 and family.weight_exponent == 2 for family in families.values())
    result = portable.reconstruct(parents, 4)
    arrays = result["arrays"]
    support = np.any(parents.reset_scale_counts > 0, axis=0)
    assert arrays["support_sizes"].tolist() == [4]
    for method in portable.METHODS:
        after, before = arrays["after__" + method], arrays["before__" + method]
        assert np.all(after[~support] == 0) and np.all(before[~support] == 0)
        assert np.min(after) >= 0
        assert after.sum() == pytest.approx(1, abs=1e-12)
        assert after[15] == 0  # Positive extension-only bin must not enter C1.
        assert result["paired_reset_improvements"][method]["tvd"] == pytest.approx(
            result["points"]["reset_160k"]["tvd"] - result["points"][method]["tvd"])
        assert result["paired_reset_improvements"][method]["normalized_grouped_peak_margin"] == pytest.approx(
            result["points"][method]["normalized_grouped_peak_margin"]
            - result["points"]["reset_160k"]["normalized_grouped_peak_margin"])
    # Evaluation labels cannot affect fitting, selected rates, or C1 support.
    other_label = portable.reconstruct(parents, 2)
    for method in portable.METHODS:
        np.testing.assert_array_equal(arrays["after__" + method], other_label["arrays"]["after__" + method])
    np.testing.assert_array_equal(arrays["selected_indices"], other_label["arrays"]["selected_indices"])


def test_each_replicate_recomputes_its_union(parents):
    grid = np.zeros((2, 8, 16))
    grid[:, :, 0] = 0.6
    grid[:, :, 4] = 0.4
    grid[0, 0, 7] = 0.01
    grid[0, 0, 4] -= 0.01
    direct = {key: np.stack([value, value]) for key, value in parents.direct_probabilities().items()}
    arrays = portable.evaluate_batch(grid, direct, portable.build_suite(parents.effective_lambdas), portable._reference(4, 4))
    assert arrays["support_sizes"].tolist() == [3, 2]
    masks = np.unpackbits(arrays["packed_observed_support"], axis=1, count=16, bitorder="little").astype(bool)
    assert masks[0, 7] and not masks[1, 7]
    assert not np.any(masks[:, 15])


def test_paired_draw_exactly_matches_original_helper(parents):
    # This import uses only the frozen CPU analysis helpers, not a simulator.
    from shor_qem.bias_variance_revision import _draw_paired_batch
    own = portable.draw_paired_batch(parents, portable.bootstrap_seed_streams(123), 3)
    rngs = portable.bootstrap_seed_streams(123)
    original = _draw_paired_batch(
        reset_parents=parents.reset_parents,
        reset_extension_parent=parents.reset_extension_counts / 140_000,
        raw_initial_parent=parents.raw_initial_counts / 20_000,
        raw_extension_parent=parents.raw_extension_counts / 140_000,
        reset_rng=rngs[0], reset_extension_rng=rngs[1], raw_rng=rngs[2], raw_extension_rng=rngs[3],
        size=3, lambda_one_index=0,
    )
    assert own.keys() == original.keys()
    for key in own:
        np.testing.assert_array_equal(own[key], original[key])
    assert np.shares_memory(own["reset_20k"], own["reset_grid"])
    np.testing.assert_allclose(own["reset_160k"] * 160_000,
                               own["reset_grid"][:, 0] * 20_000 + own["reset_extension_counts"], atol=1e-11)
    np.testing.assert_allclose(own["raw_160k"] * 160_000,
                               own["raw_20k_counts"] + own["raw_extension_counts"], atol=1e-11)


def test_rng_stream_independence(parents, count_arrays):
    original = portable.draw_paired_batch(parents, portable.bootstrap_seed_streams(9), 3)
    # Changing both Raw blocks must not perturb Reset or any ZNE scale draw.
    count_arrays["raw_initial_counts"] = np.roll(count_arrays["raw_initial_counts"], 1)
    count_arrays["raw_extension_counts"] = np.roll(count_arrays["raw_extension_counts"], 1)
    changed = portable.draw_paired_batch(portable.CountParents(**count_arrays), portable.bootstrap_seed_streams(9), 3)
    for key in ("reset_grid", "reset_20k", "reset_160k", "reset_extension_counts"):
        np.testing.assert_array_equal(changed[key], original[key])
    assert not np.array_equal(changed["raw_160k"], original["raw_160k"])


def test_fresh_bootstrap_seed_point_and_paired_percentiles(parents):
    result = portable.bootstrap(parents, 4, replicates=5, seed=12, batch_size=2, confidence=0.9)
    repeat = portable.bootstrap(parents, 4, replicates=5, seed=12, batch_size=2, confidence=0.9)
    other = portable.bootstrap(parents, 4, replicates=5, seed=13, batch_size=2, confidence=0.9)
    assert result["metadata"]["analysis"] == portable.FRESH_BOOTSTRAP_LABEL
    assert result["metadata"]["historical_replay"] is False
    assert result["arrays"]["scalar_draws"].shape == (4, 3, 5)
    for key in result["arrays"]:
        np.testing.assert_array_equal(result["arrays"][key], repeat["arrays"][key])
    assert not np.array_equal(result["arrays"]["direct_draws"], other["arrays"]["direct_draws"])
    point = portable.reconstruct(parents, 4)
    assert result["point"]["points"] == point["points"]
    for ri, method in enumerate(portable.METHODS):
        for mi, metric in enumerate(portable.METRICS):
            interval = result["intervals"]["paired_reset_improvements"][method][metric]
            draw = result["arrays"]["paired_reset_improvements"][ri, mi]
            np.testing.assert_allclose([interval["ci_low"], interval["ci_high"]], np.quantile(draw, [.05, .95]))
            assert interval["point"] == point["paired_reset_improvements"][method][metric]


@pytest.mark.parametrize("kwargs", [{"replicates": 0}, {"batch_size": 0}, {"seed": -1}, {"confidence": 1}, {"replicates": 2.5}])
def test_bootstrap_reject_invalid_parameters(parents, kwargs):
    with pytest.raises(ValueError):
        portable.bootstrap(parents, 4, **kwargs)


def test_strict_s6_decoder_and_truth_label_independence():
    # Independent exhaustive nearest-rational enumeration, with exact integer
    # cross-multiplication, checks Fraction.limit_denominator for every bin.
    # Ties prefer the smaller reduced denominator, then the lower rational.
    outcomes = 1 << 14
    y = np.arange(outcomes, dtype=np.int64)
    best_error, best_d = y.copy(), np.ones(outcomes, dtype=np.int64)
    best_p, best_q = np.zeros(outcomes, dtype=np.int64), np.ones(outcomes, dtype=np.int64)
    for denominator in range(1, 120):
        lower = y * denominator // outcomes
        for numerator in (lower, lower + 1):
            error = np.abs(numerator * outcomes - y * denominator)
            divisor = np.gcd(numerator, denominator)
            p, q = numerator // divisor, denominator // divisor
            left, right = error * best_d, best_error * denominator
            tie = (q < best_q) | ((q == best_q) & (p < best_p))
            update = (left < right) | ((left == right) & tie)
            best_error[update], best_d[update] = error[update], denominator
            best_p[update], best_q[update] = p[update], q[update]
    expected = [(13, 4, 70), (50, 2, 69), (16, 6, 48)]
    for a, order, accepted in expected:
        selected, mask = portable.exact_order_mask(119, a, 14, true_order=order)
        assert selected[0] == 0 and not mask[0]
        assert mask.sum() == accepted
        independent = np.asarray([q if p != 0 and pow(a, int(q), 119) == 1 else 0
                                  for p, q in zip(best_p, best_q)])
        np.testing.assert_array_equal(selected, independent)
        np.testing.assert_array_equal(mask, independent == order)
        wrong = portable.evaluation_order_mask(selected, order + 1)
        assert not np.any(wrong)
    # q=8 is a modularly verified multiple of true r=4. It MUST remain eight,
    # not be prime-reduced or incorrectly labelled exact-order success.
    selected, mask = portable.exact_order_mask(119, 13, 14, true_order=4)
    assert selected[outcomes // 8] == 8 and not mask[outcomes // 8]
    assert selected[1] == 0  # Nearest candidate is zero numerator / denominator1.
    # No candidate of a four-bit outcome has denominator6. Trying arbitrary
    # multiples or alternative convergents must not change the strict result.
    selected = portable.decoder_selected_orders(119, 16, 4)
    assert not np.any(selected)
    with pytest.raises(ValueError, match="coprime"):
        portable.decoder_selected_orders(119, 7, 4)


def test_cli_new_directories_and_decode_probabilities(tmp_path, count_arrays):
    path = tmp_path / "counts.npz"
    np.savez(path, **count_arrays)
    output = tmp_path / "point"
    args = ["reconstruct", "--input", str(path), "--order", "4", "--output-dir", str(output)]
    assert portable.main(args) == 0
    assert {p.name for p in output.iterdir()} == {"points.json", "parent_estimates.npz", "manifest.json"}
    assert json.loads((output / "manifest.json").read_text())["cpu_only"] is True
    saved = (output / "points.json").read_bytes()
    with pytest.raises(SystemExit) as exc:
        portable.main(args)
    assert exc.value.code == 2
    assert (output / "points.json").read_bytes() == saved
    decoded = tmp_path / "decoded"
    assert portable.main(["decode", "--input", str(output / "parent_estimates.npz"), "--N", "119", "--a", "13",
                          "--bits", "4", "--order", "4", "--outputdir", str(decoded)]) == 0
    summary = json.loads((decoded / "decoder.json").read_text())
    assert summary["true_order_used_during_decoding"] is False
    assert len(summary["P_exact"]) == 8
    assert not any(name.startswith("before__") for name in summary["P_exact"])
    with pytest.raises(SystemExit):
        portable.main(["decode", "--input", str(output / "parent_estimates.npz"), "--N", "119", "--a", "13",
                       "--bits", "4", "--order", "4", "--key", "before__linear_zne",
                       "--outputdir", str(tmp_path / "bad-decode")])
    assert not (tmp_path / "bad-decode").exists()


def test_cli_explicit_small_bootstrap_and_gate_rejection(tmp_path, count_arrays):
    path = tmp_path / "counts.npz"
    np.savez(path, **count_arrays)
    output = tmp_path / "fresh"
    assert portable.main(["bootstrap", "--input", str(path), "--order", "4", "--replicates", "2",
                          "--seed", "3", "--outputdir", str(output)]) == 0
    interval = json.loads((output / "intervals.json").read_text())
    assert interval["metadata"]["replicates"] == 2
    assert interval["metadata"]["historical_replay"] is False
    with pytest.raises(SystemExit):
        portable.main(["reconstruct", "--input", str(path), "--order", "4", "--budget", "gate_matched",
                       "--output-dir", str(tmp_path / "unsupported")])
    assert not (tmp_path / "unsupported").exists()


def test_synthetic_generator_is_labeled_reproducible_and_exclusive(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_synthetic_counts.py"
    spec = importlib.util.spec_from_file_location("make_synthetic_counts", script)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    arrays = generator.synthetic_counts(bits=4, order=4, seed=1)
    repeat = generator.synthetic_counts(bits=4, order=4, seed=1)
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], repeat[key])
    assert bool(arrays["synthetic_demo_only"])
    assert arrays["reset_scale_counts"].shape == (8, 16)
    output = tmp_path / "toy.npz"
    assert generator.main(["--output", str(output), "--bits", "4", "--order", "4"]) == 0
    saved = output.read_bytes()
    with pytest.raises(SystemExit):
        generator.main(["--output", str(output)])
    assert output.read_bytes() == saved
