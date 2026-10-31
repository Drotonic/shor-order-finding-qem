import unittest

import numpy as np

from shor_qem.metrics import (
    distribution_metrics,
    ideal_distribution,
    ideal_indices,
    linear_zne,
    main_lobe_groups,
    simplex_projection,
    simplex_projection_rows,
    weighted_linear_intercept_coefficients,
)


class SimplexTests(unittest.TestCase):
    def test_known_projection(self):
        actual = simplex_projection(np.array([0.8, 0.3, -0.1]))
        np.testing.assert_allclose(actual, [0.75, 0.25, 0.0], atol=1e-14)

    def test_identity_on_probability_vector(self):
        vector = np.array([0.2, 0.3, 0.5])
        np.testing.assert_allclose(simplex_projection(vector), vector, atol=1e-14)

    def test_projection_is_valid(self):
        rng = np.random.default_rng(3)
        actual = simplex_projection(rng.normal(size=100))
        self.assertGreaterEqual(float(actual.min()), 0.0)
        self.assertAlmostEqual(float(actual.sum()), 1.0, places=14)

    def test_vectorized_rows_match_scalar_projection(self):
        rng = np.random.default_rng(17)
        values = rng.normal(size=(7, 31))
        expected = np.stack([simplex_projection(row) for row in values])
        np.testing.assert_allclose(
            simplex_projection_rows(values), expected, atol=1e-14
        )


class EstimatorTests(unittest.TestCase):
    def test_wls_coefficients_reproduce_constant_and_zero_intercept_line(self):
        lambdas = np.array([1.0, 1.1, 1.2, 1.4, 1.8, 2.2, 2.6, 3.0])
        h = weighted_linear_intercept_coefficients(lambdas)
        self.assertAlmostEqual(float(h.sum()), 1.0, places=13)
        self.assertAlmostEqual(float(h @ lambdas), 0.0, places=13)

    def test_linear_zne_returns_simplex_distribution(self):
        lambdas = np.array([1.0, 2.0, 3.0])
        distributions = np.array([[0.7, 0.3], [0.6, 0.4], [0.5, 0.5]])
        projected, unconstrained, diagnostics = linear_zne(distributions, lambdas)
        np.testing.assert_allclose(projected, [0.8, 0.2], atol=1e-13)
        np.testing.assert_allclose(unconstrained, [0.8, 0.2], atol=1e-13)
        self.assertAlmostEqual(diagnostics["negative_mass_before_projection"], 0.0)


class MetricTests(unittest.TestCase):
    def test_ideal_distribution_metrics(self):
        probabilities = ideal_distribution(4, 4)
        metrics = distribution_metrics(probabilities, 4, 4)
        self.assertAlmostEqual(metrics.tvd, 0.0)
        self.assertAlmostEqual(metrics.support_margin, 0.25)
        self.assertTrue(np.isinf(metrics.ideal_to_spurious_ratio))
        self.assertAlmostEqual(metrics.ideal_support_mass, 1.0)

    def test_margin_sign_matches_ratio_threshold(self):
        probabilities = np.zeros(16)
        probabilities[[0, 4, 8, 12]] = [0.20, 0.19, 0.18, 0.17]
        probabilities[3] = 0.10
        probabilities[5] = 0.16
        metrics = distribution_metrics(probabilities, 4, 4)
        self.assertGreater(metrics.support_margin, 0.0)
        self.assertGreater(metrics.ideal_to_spurious_ratio, 1.0)

    def test_non_grid_aligned_order_uses_exact_qpe_distribution(self):
        probabilities = ideal_distribution(10, 6)
        np.testing.assert_array_equal(
            ideal_indices(10, 6), [0, 171, 341, 512, 683, 853]
        )
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=14)
        self.assertGreater(np.count_nonzero(probabilities > 1e-12), 6)
        metrics = distribution_metrics(probabilities, 10, 6)
        self.assertAlmostEqual(metrics.tvd, 0.0, places=14)
        self.assertGreater(metrics.support_margin, 0.0)
        self.assertAlmostEqual(
            metrics.normalized_grouped_peak_margin, 1.0, places=14
        )
        self.assertGreater(metrics.ideal_to_spurious_ratio, 1.0)
        self.assertLess(metrics.ideal_support_mass, 1.0)

    def test_main_lobe_groups_reduce_to_singletons_on_grid(self):
        groups = main_lobe_groups(4, 4)
        self.assertEqual([group.tolist() for group in groups], [[0], [4], [8], [12]])

    def test_n15_a2_main_lobe_groups(self):
        groups = main_lobe_groups(8, 4)
        self.assertEqual(
            [group.tolist() for group in groups],
            [[0], [64], [128], [192]],
        )

    def test_grouped_margin_equals_original_support_margin_on_grid(self):
        rng = np.random.default_rng(27)
        probabilities = rng.random(1 << 10)
        probabilities /= probabilities.sum()
        for order in (2, 4):
            support = ideal_indices(10, order)
            outside = probabilities.copy()
            outside[support] = -np.inf
            original_margin = (
                probabilities[support].min() - outside.max()
            )
            metrics = distribution_metrics(probabilities, 10, order)
            self.assertAlmostEqual(
                metrics.grouped_peak_margin, original_margin, places=15
            )

    def test_main_lobe_groups_collect_two_bins_off_grid(self):
        groups_r3 = main_lobe_groups(10, 3)
        self.assertEqual(
            [group.tolist() for group in groups_r3],
            [[0], [341, 342], [682, 683]],
        )
        groups_r6 = main_lobe_groups(10, 6)
        self.assertEqual(
            [group.tolist() for group in groups_r6],
            [[0], [170, 171], [341, 342], [512], [682, 683], [853, 854]],
        )

    def test_n119_a16_main_lobe_groups(self):
        groups = main_lobe_groups(14, 6)
        self.assertEqual(
            [group.tolist() for group in groups],
            [
                [0],
                [2730, 2731],
                [5461, 5462],
                [8192],
                [10922, 10923],
                [13653, 13654],
            ],
        )

    def test_grouped_margin_matches_group_mass_definition(self):
        probabilities = np.zeros(1 << 10)
        probabilities[0] = 0.10
        probabilities[[341, 342]] = [0.05, 0.06]
        probabilities[[682, 683]] = [0.04, 0.05]
        probabilities[100] = 0.08
        metrics = distribution_metrics(probabilities, 10, 3)
        self.assertAlmostEqual(metrics.grouped_peak_margin, 0.01)
        self.assertAlmostEqual(metrics.minimum_group_probability, 0.09)
        self.assertAlmostEqual(metrics.maximum_outside_group_probability, 0.08)


if __name__ == "__main__":
    unittest.main()
