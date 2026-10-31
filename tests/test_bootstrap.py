import unittest

import numpy as np

from shor_qem.bootstrap_analysis import _bootstrap_rate, _build_estimator_suite
from shor_qem.estimators import (
    build_exponential_grid_family,
    polynomial_intercept_coefficients,
)
from shor_qem.metrics import ideal_distribution, ideal_indices


class BootstrapTests(unittest.TestCase):
    def test_estimator_suite_propagates_weight_exponent(self):
        lambdas = np.array([1.0, 1.2, 1.8, 3.0])
        fixed, exponential = _build_estimator_suite(
            lambdas, weight_exponent=0.0
        )
        np.testing.assert_allclose(
            fixed["linear_zne"],
            polynomial_intercept_coefficients(
                lambdas, 1, weight_exponent=0.0
            ),
        )
        self.assertEqual(
            exponential["single_exponential_zne"].weight_exponent, 0.0
        )

    def test_seeded_bootstrap_is_reproducible_and_returns_valid_shapes(self):
        reset = np.array(
            [
                [0.70, 0.20, 0.08, 0.02],
                [0.65, 0.22, 0.10, 0.03],
                [0.60, 0.24, 0.12, 0.04],
            ]
        )
        raw = np.array([0.68, 0.20, 0.09, 0.03])
        lambdas = np.array([1.0, 2.0, 3.0])
        fixed_coefficients = {
            "linear_zne": polynomial_intercept_coefficients(lambdas, 1),
            "quadratic_zne": polynomial_intercept_coefficients(
                lambdas, 2, ridge=1.0e-8
            ),
        }
        exponential_families = {
            "single_exponential_zne": build_exponential_grid_family(
                lambdas,
                1,
                single_grid=np.array([0.1, 0.5, 1.0]),
            ),
            "dual_exponential_zne": build_exponential_grid_family(
                lambdas,
                2,
                dual_grid=np.array([0.0, 0.5, 1.0]),
            ),
        }
        kwargs = dict(
            reset_probabilities=reset,
            raw_probabilities=raw,
            shots=200,
            fixed_coefficients=fixed_coefficients,
            exponential_families=exponential_families,
            replicates=7,
            batch_size=3,
            target=ideal_distribution(2, 2),
            support=ideal_indices(2, 2),
        )
        first, first_diag = _bootstrap_rate(
            **kwargs, seed_sequence=np.random.SeedSequence(91)
        )
        second, second_diag = _bootstrap_rate(
            **kwargs, seed_sequence=np.random.SeedSequence(91)
        )
        for method in first:
            for metric in first[method]:
                self.assertEqual(first[method][metric].shape, (7,))
                np.testing.assert_array_equal(
                    first[method][metric], second[method][metric]
                )
        for method in first_diag:
            for name in first_diag[method]:
                self.assertEqual(first_diag[method][name].shape, (7,))
                np.testing.assert_array_equal(
                    first_diag[method][name], second_diag[method][name]
                )

    def test_bootstrap_accepts_cost_matched_high_shot_baselines(self):
        reset = np.array(
            [
                [0.70, 0.20, 0.08, 0.02],
                [0.65, 0.22, 0.10, 0.03],
                [0.60, 0.24, 0.12, 0.04],
            ]
        )
        raw = np.array([0.68, 0.20, 0.09, 0.03])
        extension = np.array([0.69, 0.21, 0.08, 0.02])
        lambdas = np.array([1.0, 2.0, 3.0])
        fixed, exponential = _build_estimator_suite(lambdas)
        sampled, _ = _bootstrap_rate(
            reset_probabilities=reset,
            raw_probabilities=raw,
            shots=200,
            raw_shots=600,
            reset_baseline_shots=600,
            reset_extension_probabilities=extension,
            reset_extension_shots=400,
            lambda_one_index=0,
            fixed_coefficients=fixed,
            exponential_families=exponential,
            replicates=9,
            batch_size=4,
            seed_sequence=np.random.SeedSequence(117),
            target=ideal_distribution(2, 2),
            support=ideal_indices(2, 2),
        )
        for method in sampled:
            for values in sampled[method].values():
                self.assertEqual(values.shape, (9,))
                self.assertTrue(np.all(np.isfinite(values)))


if __name__ == "__main__":
    unittest.main()
