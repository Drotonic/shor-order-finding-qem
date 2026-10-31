import unittest

import numpy as np

from shor_qem.estimators import (
    build_exponential_grid_family,
    fit_exponential_grid_batch,
    objective_weights,
    polynomial_intercept_coefficients,
)


class PolynomialEstimatorTests(unittest.TestCase):
    def test_weight_exponent_zero_gives_equal_objective_weights(self):
        lambdas = np.array([1.0, 1.2, 1.8, 3.0])
        np.testing.assert_allclose(
            objective_weights(lambdas, exponent=0.0), np.ones(lambdas.size)
        )

    def test_linear_moment_constraints_hold_for_multiple_weight_exponents(self):
        lambdas = np.array([1.0, 1.1, 1.4, 2.2, 3.0])
        for exponent in (-2.0, 0.0, 2.0, 4.0):
            coefficients = polynomial_intercept_coefficients(
                lambdas, 1, weight_exponent=exponent
            )
            self.assertAlmostEqual(float(coefficients.sum()), 1.0, places=13)
            self.assertAlmostEqual(
                float(coefficients @ lambdas), 0.0, places=13
            )

    def test_quadratic_intercept_is_exact_without_ridge(self):
        lambdas = np.array([1.0, 1.5, 2.0, 3.0])
        values = 0.2 + 0.3 * lambdas - 0.04 * lambdas**2
        coefficients = polynomial_intercept_coefficients(lambdas, 2)
        self.assertAlmostEqual(float(coefficients @ values), 0.2, places=13)


class ExponentialEstimatorTests(unittest.TestCase):
    def test_single_exponential_recovers_grid_rate_and_zero_noise_values(self):
        lambdas = np.array([1.0, 1.4, 2.0, 3.0])
        floor = 0.25
        amplitudes = np.array([0.35, -0.10, 0.05, -0.30])
        rate = 0.7
        probabilities = floor + np.exp(-rate * lambdas[:, None]) * amplitudes
        family = build_exponential_grid_family(
            lambdas,
            1,
            single_grid=np.array([0.2, rate, 1.3]),
            ridge=0.0,
        )
        estimate, diagnostics = fit_exponential_grid_batch(
            probabilities, family, uniform_floor=floor
        )
        self.assertAlmostEqual(float(diagnostics["selected_parameters"][0]), rate)
        np.testing.assert_allclose(estimate, floor + amplitudes, atol=1e-13)

    def test_dual_candidate_order_matches_nested_historical_grid(self):
        family = build_exponential_grid_family(
            np.array([1.0, 2.0, 3.0]),
            2,
            dual_grid=np.array([0.0, 0.5, 1.0]),
        )
        np.testing.assert_allclose(
            family.parameters,
            [[0.0, 0.5], [0.0, 1.0], [0.5, 1.0]],
        )

    def test_fast_dual_search_matches_direct_penalized_wls(self):
        rng = np.random.default_rng(83)
        lambdas = np.array([1.0, 1.4, 2.2, 3.0])
        values = rng.dirichlet(np.ones(7), size=lambdas.size)
        ridge = 1.0e-8
        family = build_exponential_grid_family(
            lambdas,
            2,
            dual_grid=np.array([0.0, 0.4, 1.1, 2.0]),
            ridge=ridge,
        )
        actual, diagnostics = fit_exponential_grid_batch(values, family)

        weights = objective_weights(lambdas)
        centered = values - 1.0 / values.shape[1]
        objectives = []
        direct_estimates = []
        for rates in family.parameters:
            design = np.exp(-lambdas[:, None] * rates[None, :])
            normal = design.T @ (weights[:, None] * design)
            normal += ridge * np.eye(2)
            amplitudes = np.linalg.solve(
                normal, design.T @ (weights[:, None] * centered)
            )
            residual = centered - design @ amplitudes
            objectives.append(
                np.sum(weights[:, None] * residual**2)
                + ridge * np.sum(amplitudes**2)
            )
            direct_estimates.append(1.0 / values.shape[1] + amplitudes.sum(axis=0))
        expected_index = int(np.argmin(objectives))
        self.assertEqual(int(diagnostics["selected_index"]), expected_index)
        np.testing.assert_allclose(
            actual, direct_estimates[expected_index], atol=2e-12
        )


if __name__ == "__main__":
    unittest.main()
