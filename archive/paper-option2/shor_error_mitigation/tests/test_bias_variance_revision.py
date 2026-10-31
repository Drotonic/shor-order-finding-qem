import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from shor_qem.bias_variance_revision import (
    DIRECT_EXTENSION_SHOTS,
    DIRECT_SHOTS,
    FROZEN_BATCH_SIZE,
    InstanceSpec,
    L2_PROJECTION_NONEXPANSIVENESS_TOLERANCE,
    NOMINAL_LAMBDAS,
    PREPROJECTION_VARIANCE_NEGATIVE_TOLERANCE,
    ROUTES,
    VARIANCE_MC_BLOCK_SIZE,
    ZNE_SHOTS_PER_SCALE,
    _StreamingVectorMoments,
    analyse_instance,
    _bootstrap_seed_streams,
    _coefficient_rows,
    _draw_paired_batch,
)
from verify_bias_variance_revision import (
    AuditLog,
    CACHE_SCHEMA_VERSION,
    DEFAULT_SEED,
    METRICS,
    RECOMPUTED_ARITHMETIC_ATOL,
    RECOMPUTED_ARITHMETIC_RTOL,
    ZNE_MODELS,
    _audit_optional_common_rate_tables,
    _b20_prefix_mismatches,
    _is_nonfinal_two_instance_pilot_table,
    _mc_diagnostic_available,
    _load_and_audit_cache,
    _load_and_audit_sufficient_statistics,
    _recomputed_arithmetic_close,
    _valid_first_baseline_source,
)
from shor_qem.bootstrap_analysis import (
    ZNE_METHODS,
    _fit_zne_batch,
    _fit_zne_batch_detailed,
)
from shor_qem.estimators import (
    DEFAULT_RIDGE,
    build_exponential_grid_family,
    polynomial_intercept_coefficients,
)


class BiasVarianceRevisionTests(unittest.TestCase):
    def test_first_baseline_source_accepts_only_exact_canonical_parent(self):
        common = {
            "instance": "n119_a16",
            "mode": "reset",
            "rate": 0.00012,
            "shot_offset": 0,
            "shots": 20_000,
        }
        self.assertTrue(_valid_first_baseline_source("original_run", **common))
        canonical = "n119_a16__reset__r000120000__l10__s000000_n020000"
        self.assertTrue(_valid_first_baseline_source(canonical, **common))

        rejected = (
            "n119_a15__reset__r000120000__l10__s000000_n020000",
            "n119_a16__raw__r000120000__l10__s000000_n020000",
            "n119_a16__reset__r000140000__l10__s000000_n020000",
            "n119_a16__reset__r000120000__l10__s020000_n020000",
            "n119_a16__reset__r000120000__l11__s000000_n020000",
            "prefix_n119_a16__reset__r000120000__l10__s000000_n020000",
        )
        for source in rejected:
            with self.subTest(source=source):
                self.assertFalse(_valid_first_baseline_source(source, **common))

        self.assertFalse(
            _valid_first_baseline_source(canonical, **{**common, "shot_offset": 1})
        )
        self.assertFalse(
            _valid_first_baseline_source(canonical, **{**common, "shot_offset": 0.5})
        )
        self.assertFalse(
            _valid_first_baseline_source(canonical, **{**common, "shots": 19_999})
        )
        self.assertFalse(
            _valid_first_baseline_source(
                "original_run", **{**common, "shot_offset": 20_000}
            )
        )

    def test_full_table_scan_ignores_only_explicit_nonfinal_2inst_archives(self):
        columns = [
            "instance",
            "method",
            "reference",
            "rate",
            "rate_percent",
            "tvd_reduction_median",
        ]
        canonical = {
            "instance": "n119_a13",
            "method": "linear_zne",
            "reference": "reset_160k",
            "rate": "0.0001",
            "rate_percent": "0.01",
            "tvd_reduction_median": "0.125",
        }
        corrupt = {**canonical, "tvd_reduction_median": "999"}

        def write(path: Path, rows: list[dict[str, str]]) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archived_summary = (
                root / "pilot_summary_2inst" / "common_0p01_paired_nominal.csv"
            )
            archived_render = (
                root
                / "pilot_render_2inst"
                / "tables"
                / "common_0p01_paired_nominal.csv"
            )
            final_table = root / "summary" / "common_0p01_paired_nominal.csv"
            write(archived_summary, [corrupt])
            write(archived_render, [{**corrupt, "instance": "unknown_instance"}])
            write(final_table, [canonical])

            self.assertTrue(
                _is_nonfinal_two_instance_pilot_table(archived_summary, root)
            )
            self.assertTrue(
                _is_nonfinal_two_instance_pilot_table(archived_render, root)
            )
            self.assertFalse(_is_nonfinal_two_instance_pilot_table(final_table, root))
            self.assertFalse(
                _is_nonfinal_two_instance_pilot_table(
                    root / "pilot_summary_full" / archived_summary.name, root
                )
            )

            clean_log = AuditLog()
            _audit_optional_common_rate_tables(
                clean_log,
                root,
                [canonical],
                allow_partial=False,
                audited_instances={"n119_a13"},
            )
            self.assertFalse(clean_log.hard_failures, clean_log.checks)

            # A similarly named but non-exempt final/full table remains strict.
            write(final_table, [{**canonical, "instance": "unknown_instance"}])
            corrupt_log = AuditLog()
            _audit_optional_common_rate_tables(
                corrupt_log,
                root,
                [canonical],
                allow_partial=False,
                audited_instances={"n119_a13"},
            )
            self.assertTrue(corrupt_log.hard_failures)

    def test_recomputed_arithmetic_tolerance_is_scale_aware_and_bounded(self):
        unit_budget = RECOMPUTED_ARITHMETIC_ATOL + RECOMPUTED_ARITHMETIC_RTOL
        self.assertTrue(
            _recomputed_arithmetic_close(1.0, 1.0 + 0.9 * unit_budget)
        )
        self.assertFalse(
            _recomputed_arithmetic_close(1.0, 1.0 + 1.1 * unit_budget)
        )

        # This brackets the 4.52e-9 production recomputation drift while a
        # scientifically meaningful 1e-6 unit-scale corruption is rejected.
        self.assertTrue(_recomputed_arithmetic_close(1.0, 1.0 + 4.6e-9))
        self.assertFalse(_recomputed_arithmetic_close(1.0, 1.0 + 1.0e-6))

        # Near zero the absolute, not the unit-scale relative, tolerance wins.
        self.assertTrue(_recomputed_arithmetic_close(1.0e-12, 5.0e-12))
        self.assertFalse(_recomputed_arithmetic_close(1.0e-12, 2.0e-11))
        self.assertTrue(
            _recomputed_arithmetic_close(
                0.008294279205425359,
                0.008294276995419488,
                scale_floor=1.0,
            )
        )
        self.assertFalse(
            _recomputed_arithmetic_close(0.008, 0.008001, scale_floor=1.0)
        )
        self.assertFalse(
            _recomputed_arithmetic_close(0.0, 0.0, scale_floor=-1.0)
        )
        self.assertFalse(_recomputed_arithmetic_close(float("nan"), float("nan")))

    def test_partial_common_rate_audit_ignores_only_known_unaudited_instances(self):
        columns = [
            "instance",
            "method",
            "reference",
            "rate",
            "rate_percent",
            "tvd_reduction_median",
        ]
        canonical = {
            "instance": "n119_a13",
            "method": "linear_zne",
            "reference": "reset_160k",
            "rate": "0.0001",
            "rate_percent": "0.01",
            "tvd_reduction_median": "0.125",
        }
        preserved_pilot = {
            **canonical,
            "instance": "n119_a16",
            "tvd_reduction_median": "0.333",
        }

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            table = root / "pilot_summary_full" / "common_0p01_paired_nominal.csv"
            table.parent.mkdir(parents=True)
            with table.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows([canonical, preserved_pilot])

            partial_log = AuditLog()
            _audit_optional_common_rate_tables(
                partial_log,
                root,
                [canonical],
                allow_partial=True,
                audited_instances={"n119_a13"},
            )
            self.assertFalse(partial_log.hard_failures, partial_log.checks)

            full_log = AuditLog()
            _audit_optional_common_rate_tables(
                full_log,
                root,
                [canonical],
                allow_partial=False,
                audited_instances={"n119_a13"},
            )
            self.assertTrue(full_log.hard_failures)

            # Partial mode still rejects corruption for an audited instance.
            with table.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(
                    [{**canonical, "tvd_reduction_median": "0.126"}, preserved_pilot]
                )
            corrupt_log = AuditLog()
            _audit_optional_common_rate_tables(
                corrupt_log,
                root,
                [canonical],
                allow_partial=True,
                audited_instances={"n119_a13"},
            )
            self.assertTrue(corrupt_log.hard_failures)

    @staticmethod
    def _synthetic_suite():
        lambdas = NOMINAL_LAMBDAS.copy()
        fixed = {
            "linear_zne": polynomial_intercept_coefficients(
                lambdas, degree=1, ridge=0.0, weight_exponent=2.0
            ),
            "quadratic_zne": polynomial_intercept_coefficients(
                lambdas,
                degree=2,
                ridge=DEFAULT_RIDGE,
                weight_exponent=2.0,
            ),
        }
        exponential = {
            "single_exponential_zne": build_exponential_grid_family(
                lambdas,
                components=1,
                single_grid=np.array([0.05, 0.5, 2.0]),
                ridge=DEFAULT_RIDGE,
                weight_exponent=2.0,
            ),
            "dual_exponential_zne": build_exponential_grid_family(
                lambdas,
                components=2,
                dual_grid=np.array([0.0, 0.5, 1.5]),
                ridge=DEFAULT_RIDGE,
                weight_exponent=2.0,
            ),
        }

        # Valid but deliberately non-smooth scale records make projection
        # active, so the test exercises the pre/post-projection diagnostics.
        batch, outcomes = 3, 6
        distributions = np.full(
            (batch, lambdas.size, outcomes), 0.016, dtype=np.float64
        )
        for batch_i in range(batch):
            for scale_i in range(lambdas.size):
                distributions[
                    batch_i, scale_i, (scale_i + 2 * batch_i) % outcomes
                ] = 0.92
        np.testing.assert_allclose(distributions.sum(axis=2), 1.0)
        return distributions, fixed, exponential

    def test_frozen_nominal_q2_polynomial_coefficients_and_checkpoints(self):
        expected = {
            "linear_zne": np.array(
                [
                    0.79049259711852848,
                    0.53298305472818952,
                    0.34675444119700732,
                    0.10620418037186356,
                    -0.11548525001877265,
                    -0.19762492017572303,
                    -0.22763861201777738,
                    -0.23568549120331680,
                ]
            ),
            "quadratic_zne": np.array(
                [
                    2.9859288848201668,
                    1.0514672637993732,
                    -0.16445941917397749,
                    -1.3476254289696767,
                    -1.5418876176971248,
                    -0.84239281753186135,
                    0.016902084714924433,
                    0.84206690080473123,
                ]
            ),
        }
        expected_moments = {
            "linear_zne": np.array(
                [
                    1.0,
                    0.0,
                    -2.8477945795891686,
                    -10.751795647164020,
                ]
            ),
            # The frozen quadratic fit uses a 1e-8 ridge, so its first three
            # moment constraints are close to, but not exactly, (1, 0, 0).
            "quadratic_zne": np.array(
                [
                    0.99999985076655484,
                    1.8368123466672159e-7,
                    -4.8651108919273156e-8,
                    5.4741518147000683,
                ]
            ),
        }
        expected_l2_squared = {
            "linear_zne": 1.2002268936363849,
            "quadratic_zne": 15.660901339941471,
        }
        expected_effective_shots = {
            "linear_zne": 16663.515961890374,
            "quadratic_zne": 1277.0657043213801,
        }

        fixed = {
            "linear_zne": polynomial_intercept_coefficients(
                NOMINAL_LAMBDAS, degree=1, ridge=0.0, weight_exponent=2.0
            ),
            "quadratic_zne": polynomial_intercept_coefficients(
                NOMINAL_LAMBDAS,
                degree=2,
                ridge=DEFAULT_RIDGE,
                weight_exponent=2.0,
            ),
        }
        rows = _coefficient_rows(
            {"instance": "nominal_q2_test"}, NOMINAL_LAMBDAS, fixed
        )

        self.assertEqual(len(rows), 2 * NOMINAL_LAMBDAS.size)
        for method, degree in (("linear_zne", 1), ("quadratic_zne", 2)):
            np.testing.assert_allclose(
                fixed[method], expected[method], rtol=2.0e-13, atol=2.0e-13
            )
            moments = np.array(
                [
                    np.dot(fixed[method], NOMINAL_LAMBDAS**power)
                    for power in range(4)
                ]
            )
            np.testing.assert_allclose(
                moments, expected_moments[method], rtol=2.0e-12, atol=2.0e-12
            )

            method_rows = [row for row in rows if row["method"] == method]
            self.assertEqual(len(method_rows), NOMINAL_LAMBDAS.size)
            np.testing.assert_allclose(
                [row["coefficient"] for row in method_rows], expected[method]
            )
            for row in method_rows:
                self.assertEqual(row["degree"], degree)
                self.assertAlmostEqual(
                    row["coefficient_l2_norm_squared"],
                    expected_l2_squared[method],
                    places=12,
                )
                self.assertAlmostEqual(
                    row["equal_covariance_effective_shots"],
                    expected_effective_shots[method],
                    places=9,
                )
                self.assertAlmostEqual(
                    row["equal_covariance_variance_ratio_vs_reset_20k"],
                    expected_l2_squared[method],
                    places=12,
                )
                self.assertAlmostEqual(
                    row["equal_covariance_variance_ratio_vs_reset_160k"],
                    8.0 * expected_l2_squared[method],
                    places=11,
                )
                self.assertAlmostEqual(
                    row["equal_covariance_effective_shots"],
                    ZNE_SHOTS_PER_SCALE
                    / row["coefficient_l2_norm_squared"],
                    places=10,
                )

    def test_production_engine_rejects_nonfrozen_batch_size_before_io(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = InstanceSpec(
                key="synthetic",
                config=root / "missing_config.json",
                run_dir=root / "missing_run",
                no_reset_dir=root / "missing_raw",
            )
            with self.assertRaisesRegex(ValueError, "requires batch_size=25"):
                analyse_instance(
                    spec,
                    output_dir=root / "out",
                    replicates=10,
                    batch_size=FROZEN_BATCH_SIZE - 1,
                    confidence=0.95,
                    seed=DEFAULT_SEED,
                    requested_rates=None,
                    scale_mode="nominal",
                )

    def test_mc_diagnostic_single_block_smoke_and_multiblock_production_contract(self):
        self.assertFalse(_mc_diagnostic_available(1, float("nan")))
        self.assertTrue(_mc_diagnostic_available(20, 1.25e-7))

        with self.assertRaisesRegex(ValueError, "single-block.*NaN"):
            _mc_diagnostic_available(1, 0.0)
        with self.assertRaisesRegex(ValueError, "finite positive SE"):
            _mc_diagnostic_available(20, float("nan"))
        with self.assertRaisesRegex(ValueError, "finite positive SE"):
            _mc_diagnostic_available(20, 0.0)
        with self.assertRaisesRegex(ValueError, "block count must be positive"):
            _mc_diagnostic_available(0, float("nan"))

    def test_detailed_fit_is_equivalent_to_legacy_fit(self):
        distributions, fixed, exponential = self._synthetic_suite()
        detailed_projected, detailed_diagnostics, _, _ = (
            _fit_zne_batch_detailed(
                distributions,
                fixed_coefficients=fixed,
                exponential_families=exponential,
            )
        )
        legacy_projected, legacy_diagnostics = _fit_zne_batch(
            distributions,
            fixed_coefficients=fixed,
            exponential_families=exponential,
        )

        self.assertEqual(set(detailed_projected), set(ZNE_METHODS))
        self.assertEqual(set(legacy_projected), set(ZNE_METHODS))
        for method in ZNE_METHODS:
            np.testing.assert_allclose(
                detailed_projected[method],
                legacy_projected[method],
                rtol=0.0,
                atol=0.0,
            )
            self.assertEqual(
                set(detailed_diagnostics[method]),
                set(legacy_diagnostics[method]),
            )
            for name in detailed_diagnostics[method]:
                np.testing.assert_array_equal(
                    detailed_diagnostics[method][name],
                    legacy_diagnostics[method][name],
                )

    def test_simplex_and_nonlinear_diagnostic_contract(self):
        distributions, fixed, exponential = self._synthetic_suite()
        projected, diagnostics, unconstrained, selection = (
            _fit_zne_batch_detailed(
                distributions,
                fixed_coefficients=fixed,
                exponential_families=exponential,
            )
        )
        batch, _, outcomes = distributions.shape
        common_diagnostics = {
            "normalization_error_before_projection",
            "negative_mass_before_projection",
            "negative_entry_count_before_projection",
            "negative_entry_fraction_before_projection",
            "l2_projection_correction",
            "projection_active",
        }

        for method in ZNE_METHODS:
            before = unconstrained[method]
            after = projected[method]
            self.assertEqual(before.shape, (batch, outcomes))
            self.assertEqual(after.shape, (batch, outcomes))
            self.assertTrue(np.all(np.isfinite(before)))
            self.assertTrue(np.all(np.isfinite(after)))
            self.assertTrue(np.all(after >= -1.0e-14))
            np.testing.assert_allclose(after.sum(axis=1), 1.0, atol=1.0e-14)
            self.assertTrue(common_diagnostics.issubset(diagnostics[method]))
            for name in diagnostics[method]:
                self.assertEqual(np.asarray(diagnostics[method][name]).shape, (batch,))

            np.testing.assert_allclose(
                diagnostics[method]["normalization_error_before_projection"],
                np.abs(before.sum(axis=1) - 1.0),
            )
            np.testing.assert_allclose(
                diagnostics[method]["negative_mass_before_projection"],
                -np.minimum(before, 0.0).sum(axis=1),
            )
            negative_count = np.count_nonzero(before < 0.0, axis=1)
            np.testing.assert_array_equal(
                diagnostics[method]["negative_entry_count_before_projection"],
                negative_count.astype(np.float64),
            )
            np.testing.assert_allclose(
                diagnostics[method]["negative_entry_fraction_before_projection"],
                negative_count / outcomes,
            )
            np.testing.assert_allclose(
                diagnostics[method]["l2_projection_correction"],
                np.linalg.norm(after - before, axis=1),
            )
            np.testing.assert_array_equal(
                diagnostics[method]["projection_active"], np.ones(batch)
            )
            simplex_target = np.full(outcomes, 1.0 / outcomes)
            before_l2 = np.linalg.norm(before - simplex_target[None, :], axis=1)
            after_l2 = np.linalg.norm(after - simplex_target[None, :], axis=1)
            self.assertTrue(np.all(after_l2 <= before_l2 + 2.0e-14))

        nonlinear_specs = {
            "single_exponential_zne": 1,
            "dual_exponential_zne": 2,
        }
        for method, components in nonlinear_specs.items():
            family = exponential[method]
            selected = selection[method]
            self.assertEqual(selected["selected_parameters"].shape, (batch, components))
            for name, values in selected.items():
                expected_shape = (
                    (batch, components)
                    if name == "selected_parameters"
                    else (batch,)
                )
                self.assertEqual(np.asarray(values).shape, expected_shape)

            indices = selected["selected_index"]
            self.assertTrue(np.all(indices >= 0))
            self.assertTrue(np.all(indices < family.candidate_count))
            self.assertTrue(
                np.all(selected["selected_normal_condition_number"] >= 1.0)
            )
            self.assertTrue(
                np.all(np.isfinite(selected["selected_normal_condition_number"]))
            )
            self.assertTrue(np.all(selected["runner_up_score_gap"] >= -1.0e-12))
            self.assertTrue(
                np.all(selected["near_tie_candidate_count"] >= 1)
            )
            self.assertTrue(
                np.all(
                    selected["near_tie_candidate_count"]
                    <= family.candidate_count
                )
            )
            self.assertTrue(
                np.all(selected["penalized_weighted_residual"] >= 0.0)
            )
            np.testing.assert_array_equal(
                selected["any_boundary"],
                selected["lower_boundary"] | selected["upper_boundary"],
            )

        single = selection["single_exponential_zne"]
        np.testing.assert_allclose(
            diagnostics["single_exponential_zne"]["selected_gamma"],
            single["selected_parameters"][:, 0],
        )
        dual = selection["dual_exponential_zne"]
        self.assertTrue(
            np.all(
                dual["selected_parameters"][:, 0]
                < dual["selected_parameters"][:, 1]
            )
        )
        np.testing.assert_allclose(
            dual["selected_minimum_rate_separation"],
            dual["selected_parameters"][:, 1]
            - dual["selected_parameters"][:, 0],
        )
        np.testing.assert_allclose(
            diagnostics["dual_exponential_zne"]["selected_gamma_1"],
            dual["selected_parameters"][:, 0],
        )
        np.testing.assert_allclose(
            diagnostics["dual_exponential_zne"]["selected_gamma_2"],
            dual["selected_parameters"][:, 1],
        )

    def test_rate_keyed_paired_batch_preserves_exact_overlap(self):
        # One-hot parents remove multinomial noise and expose the block algebra
        # exactly.  All ZNE models consume the single returned reset_grid.
        reset_parents = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        seeds_a = _bootstrap_seed_streams(91, "n119_a13", 1.0e-4)
        seeds_b = _bootstrap_seed_streams(91, "n119_a13", 1.0e-4)
        self.assertEqual(
            [seed.generate_state(4).tolist() for seed in seeds_a],
            [seed.generate_state(4).tolist() for seed in seeds_b],
        )
        self.assertNotEqual(
            seeds_a[0].generate_state(4).tolist(),
            _bootstrap_seed_streams(91, "n119_a13", 2.0e-4)[0]
            .generate_state(4)
            .tolist(),
        )

        batch = _draw_paired_batch(
            reset_parents=reset_parents,
            reset_extension_parent=np.array([0.0, 1.0, 0.0]),
            raw_initial_parent=np.array([0.0, 0.0, 1.0]),
            raw_extension_parent=np.array([1.0, 0.0, 0.0]),
            reset_rng=np.random.default_rng(seeds_a[0]),
            reset_extension_rng=np.random.default_rng(seeds_a[1]),
            raw_rng=np.random.default_rng(seeds_a[2]),
            raw_extension_rng=np.random.default_rng(seeds_a[3]),
            size=5,
            lambda_one_index=0,
        )
        self.assertTrue(np.shares_memory(batch["reset_20k"], batch["reset_grid"]))
        np.testing.assert_array_equal(batch["reset_20k"], batch["reset_grid"][:, 0])
        expected_reset160 = (
            ZNE_SHOTS_PER_SCALE * batch["reset_20k"]
            + batch["reset_extension_counts"]
        ) / DIRECT_SHOTS
        np.testing.assert_array_equal(batch["reset_160k"], expected_reset160)
        expected_raw160 = (
            batch["raw_20k_counts"] + batch["raw_extension_counts"]
        ) / DIRECT_SHOTS
        np.testing.assert_array_equal(batch["raw_160k"], expected_raw160)
        np.testing.assert_array_equal(
            batch["reset_extension_counts"].sum(axis=1),
            np.full(5, DIRECT_EXTENSION_SHOTS),
        )

    @staticmethod
    def _paired_prefix_run(replicates: int) -> dict[str, np.ndarray]:
        reset_parents = np.asarray(
            [
                [0.61, 0.21, 0.11, 0.07],
                [0.55, 0.24, 0.13, 0.08],
                [0.49, 0.27, 0.15, 0.09],
                [0.43, 0.29, 0.17, 0.11],
                [0.38, 0.31, 0.19, 0.12],
                [0.34, 0.32, 0.21, 0.13],
                [0.31, 0.33, 0.22, 0.14],
                [0.28, 0.34, 0.23, 0.15],
            ],
            dtype=np.float64,
        )
        reset_extension_parent = np.asarray([0.36, 0.30, 0.20, 0.14])
        raw_initial_parent = np.asarray([0.24, 0.28, 0.31, 0.17])
        raw_extension_parent = np.asarray([0.22, 0.27, 0.32, 0.19])
        seeds = _bootstrap_seed_streams(20260722, "n119_a16", 1.4e-4)
        rngs = [np.random.default_rng(seed) for seed in seeds]
        chunks: dict[str, list[np.ndarray]] = {
            "reset_grid": [],
            "reset_20k": [],
            "reset_160k": [],
            "raw_20k": [],
            "raw_160k": [],
        }
        for start in range(0, replicates, FROZEN_BATCH_SIZE):
            size = min(FROZEN_BATCH_SIZE, replicates - start)
            batch = _draw_paired_batch(
                reset_parents=reset_parents,
                reset_extension_parent=reset_extension_parent,
                raw_initial_parent=raw_initial_parent,
                raw_extension_parent=raw_extension_parent,
                reset_rng=rngs[0],
                reset_extension_rng=rngs[1],
                raw_rng=rngs[2],
                raw_extension_rng=rngs[3],
                size=size,
                lambda_one_index=0,
            )
            for name in chunks:
                chunks[name].append(np.asarray(batch[name]).copy())
        return {name: np.concatenate(values, axis=0) for name, values in chunks.items()}

    def test_frozen_batch25_exact_prefixes_b2k_b5k_b10k_b20k(self):
        extended = self._paired_prefix_run(20_000)
        for prefix in (2_000, 5_000, 10_000):
            standalone = self._paired_prefix_run(prefix)
            for name, values in standalone.items():
                np.testing.assert_array_equal(
                    values,
                    extended[name][:prefix],
                    err_msg=f"{name} is not an exact B{prefix} seed prefix",
                )

    def test_streaming_sufficient_statistics_reconstruct_and_reject_negative_variance(self):
        rng = np.random.default_rng(7123)
        replicates = 100
        target = np.asarray([0.40, 0.30, 0.20, 0.10], dtype=np.float64)
        draws = np.empty((len(ROUTES), replicates, target.size), dtype=np.float64)
        for route_i in range(len(ROUTES)):
            draws[route_i] = (
                target[None, :]
                + 0.01 * route_i
                + rng.normal(scale=0.025 + 0.002 * route_i, size=(replicates, target.size))
            )

        mean = np.empty((1, len(ROUTES), target.size), dtype=np.float64)
        mean_squared_norm = np.empty((1, len(ROUTES)), dtype=np.float64)
        centered_sum_squares = np.empty((1, len(ROUTES)), dtype=np.float64)
        population_variance = np.empty((1, len(ROUTES)), dtype=np.float64)
        block_traces = np.empty((1, len(ROUTES), 1), dtype=np.float64)
        for route_i in range(len(ROUTES)):
            moments = _StreamingVectorMoments.empty(target.size)
            for start in range(0, replicates, FROZEN_BATCH_SIZE):
                moments.update(draws[route_i, start : start + FROZEN_BATCH_SIZE])
            direct_mean = np.mean(draws[route_i], axis=0)
            direct_centered = draws[route_i] - direct_mean[None, :]
            direct_m2 = float(np.square(direct_centered).sum())
            np.testing.assert_allclose(moments.mean, direct_mean, rtol=0.0, atol=5e-16)
            self.assertAlmostEqual(moments.centered_sum_squares, direct_m2, places=14)
            mean[0, route_i] = moments.mean
            centered_sum_squares[0, route_i] = moments.centered_sum_squares
            population_variance[0, route_i] = moments.population_variance_trace(
                negative_tolerance=PREPROJECTION_VARIANCE_NEGATIVE_TOLERANCE
            )
            mean_squared_norm[0, route_i] = float(
                np.square(draws[route_i]).sum(axis=1).mean()
            )
            block_traces[0, route_i, 0] = (
                moments.unbiased_variance_trace(
                    negative_tolerance=PREPROJECTION_VARIANCE_NEGATIVE_TOLERANCE
                )
            )

        with TemporaryDirectory() as temporary:
            instance_dir = Path(temporary)
            path = instance_dir / "preprojection_sufficient_statistics.npz"

            def write(centered: np.ndarray) -> None:
                np.savez_compressed(
                    path,
                    schema_version=np.asarray(1, dtype=np.int64),
                    rates=np.asarray([1.0e-4]),
                    routes=np.asarray(ROUTES),
                    target=target,
                    replicates=np.asarray(replicates, dtype=np.int64),
                    mean_vectors=mean,
                    mean_squared_norms=mean_squared_norm,
                    centered_sum_squares=centered,
                    population_variance_traces=population_variance,
                    block_unbiased_variance_traces=block_traces,
                    block_counts=np.asarray([[replicates]], dtype=np.int64),
                    variance_mc_block_size=np.asarray(
                        VARIANCE_MC_BLOCK_SIZE, dtype=np.int64
                    ),
                    variance_negative_tolerance=np.asarray(
                        PREPROJECTION_VARIANCE_NEGATIVE_TOLERANCE,
                        dtype=np.float64,
                    ),
                    streaming_algorithm=np.asarray(
                        "parallel_chan_welford_float64"
                    ),
                )

            write(centered_sum_squares)
            log = AuditLog()
            reconstructed = _load_and_audit_sufficient_statistics(
                log,
                instance_dir,
                "synthetic",
                np.asarray([1.0e-4]),
                replicates,
            )
            self.assertIsNotNone(reconstructed)
            self.assertFalse(log.hard_failures)
            np.testing.assert_allclose(
                reconstructed["variance"], population_variance, rtol=5e-13, atol=5e-15
            )
            np.testing.assert_allclose(
                reconstructed["mse"],
                reconstructed["bias"] + reconstructed["variance"],
                rtol=0.0,
                atol=0.0,
            )
            np.testing.assert_allclose(
                reconstructed["conditional_squared_displacement"],
                reconstructed["mse"],
                rtol=5e-12,
                atol=5e-14,
            )

            invalid = centered_sum_squares.copy()
            invalid[0, 0] = -10.0 * PREPROJECTION_VARIANCE_NEGATIVE_TOLERANCE
            write(invalid)
            failed_log = AuditLog()
            self.assertIsNone(
                _load_and_audit_sufficient_statistics(
                    failed_log,
                    instance_dir,
                    "synthetic",
                    np.asarray([1.0e-4]),
                    replicates,
                )
            )
            self.assertTrue(failed_log.hard_failures)

    def test_l2_production_cache_contract_and_violation_detection(self):
        replicates = 100
        rates = np.asarray([1.0e-4], dtype=np.float64)
        draws = np.zeros(
            (1, len(ROUTES), len(METRICS), replicates), dtype=np.float64
        )
        draws[:, :, 0, :] = 0.2
        draws[:, :, 1, :] = 0.1
        draws[:, :, 2, :] = 0.4
        pre_l2 = np.full((1, len(ZNE_MODELS), replicates), 0.20)
        post_l2 = np.full_like(pre_l2, 0.15)
        gamma_zero = np.zeros((1, replicates), dtype=np.float64)
        gamma_zero[:, ::3] = 1.0

        with TemporaryDirectory() as temporary:
            instance_dir = Path(temporary)
            path = instance_dir / "bootstrap_cache.npz"

            def write(post: np.ndarray) -> None:
                np.savez_compressed(
                    path,
                    schema_version=np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int64),
                    rates=rates,
                    nominal_lambdas=NOMINAL_LAMBDAS,
                    fit_lambdas=NOMINAL_LAMBDAS,
                    routes=np.asarray(ROUTES),
                    metrics=np.asarray(METRICS),
                    postprojection_draws=draws,
                    ideal_grouped_peak_margin=np.asarray(0.25),
                    zne_methods=np.asarray(ZNE_MODELS),
                    preprojection_l2_squared_to_ideal=pre_l2,
                    postprojection_l2_squared_to_ideal=post,
                    post_minus_pre_l2_squared_to_ideal=post - pre_l2,
                    dual_gamma_1_zero=gamma_zero,
                    l2_nonexpansiveness_tolerance=np.asarray(
                        L2_PROJECTION_NONEXPANSIVENESS_TOLERANCE
                    ),
                    batch_size=np.asarray(FROZEN_BATCH_SIZE, dtype=np.int64),
                    replicates=np.asarray(replicates, dtype=np.int64),
                    seed=np.asarray(DEFAULT_SEED, dtype=np.int64),
                )

            write(post_l2)
            log = AuditLog()
            cache = _load_and_audit_cache(
                log, instance_dir, "synthetic", rates, replicates
            )
            self.assertIsNotNone(cache, msg=str(log.checks))
            self.assertFalse(log.hard_failures)

            violating = pre_l2.copy()
            violating[0, 0, 7] += 2.0 * L2_PROJECTION_NONEXPANSIVENESS_TOLERANCE
            write(violating)
            failed_log = AuditLog()
            self.assertIsNone(
                _load_and_audit_cache(
                    failed_log, instance_dir, "synthetic", rates, replicates
                )
            )
            self.assertTrue(failed_log.hard_failures)

    def test_b20_cache_comparison_support_is_bit_exact(self):
        rng = np.random.default_rng(881)
        rates = np.asarray([1.0e-4, 2.0e-4])
        nominal = {
            "rates": rates,
            "postprojection_draws": rng.normal(
                size=(2, len(ROUTES), len(METRICS), 10_000)
            ),
            "preprojection_l2_squared_to_ideal": rng.random(
                (2, len(ZNE_MODELS), 10_000)
            ),
            "postprojection_l2_squared_to_ideal": rng.random(
                (2, len(ZNE_MODELS), 10_000)
            ),
            "post_minus_pre_l2_squared_to_ideal": rng.normal(
                size=(2, len(ZNE_MODELS), 10_000)
            ),
            "dual_gamma_1_zero": rng.integers(0, 2, size=(2, 10_000)).astype(float),
        }
        b20: dict[str, np.ndarray] = {"rates": rates.copy()}
        for name, values in nominal.items():
            if name == "rates":
                continue
            tail_shape = values.shape[:-1] + (10_000,)
            b20[name] = np.concatenate(
                [values, rng.normal(size=tail_shape)], axis=-1
            )
        self.assertEqual(_b20_prefix_mismatches(nominal, b20), [])
        b20["postprojection_draws"][1, 0, 0, 9_999] += 1.0
        mismatches = _b20_prefix_mismatches(nominal, b20)
        self.assertIn("0.00020000000000000001/postprojection_draws", mismatches)


if __name__ == "__main__":
    unittest.main()
