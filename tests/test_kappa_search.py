import math
import unittest

from scout.calib.search import solve_kappa


class KappaSearchTests(unittest.TestCase):
    def test_nonlinear_response_that_makes_ratio_updates_oscillate(self):
        result = solve_kappa(lambda k: k ** 2, target=9, initial=1, band=.01)
        self.assertTrue(result["converged"])
        self.assertLessEqual(abs(result["C_mean"] - 9), .09)
        self.assertIn(result["kappa"], [p["kappa"] for p in result["history"]])

    def test_unreachable_flat_response_stops_at_bound(self):
        result = solve_kappa(lambda k: 1, target=4, initial=1, upper=8)
        self.assertFalse(result["converged"])
        self.assertEqual(result["termination_reason"], "bound_or_resolution")
        self.assertEqual([p["kappa"] for p in result["history"]], [1, 2, 4, 8])

    def test_discontinuity_does_not_report_a_false_solution(self):
        result = solve_kappa(lambda k: 1 if k < 3 else 8, target=4, initial=1, max_probes=8)
        self.assertFalse(result["converged"])
        self.assertEqual(len(result["history"]), 8)
        self.assertEqual(result["C_mean"], 1)

    def test_one_probe_returns_measured_initial_value(self):
        result = solve_kappa(lambda k: k, target=8, initial=2, max_probes=1)
        self.assertEqual(result["kappa"], 2)
        self.assertEqual(result["C_mean"], 2)
        self.assertFalse(result["converged"])

    def test_invalid_measurement_is_rejected(self):
        with self.assertRaises(ValueError):
            solve_kappa(lambda k: math.nan, target=1, initial=1)

    def test_unknown_direction_finds_decreasing_response(self):
        result = solve_kappa(lambda k: 1 / k, target=.1, initial=1,
                             direction="unknown", max_probes=16)
        self.assertTrue(result["converged"])
        self.assertLessEqual(abs(result["C_mean"] - .1), .01)

    def test_unknown_direction_can_find_a_target_below_a_local_maximum(self):
        result = solve_kappa(lambda k: math.exp(-math.log(k) ** 2),
                             target=.3, initial=1, direction="unknown")
        self.assertTrue(result["converged"])


if __name__ == "__main__":
    unittest.main()
