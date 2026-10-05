"""Numerical tests run only on an allocated cluster worker."""

import unittest

import jax
import jax.numpy as jnp

from experiments.solver_in_loop.conditioned_ic_fd import conditioned_ic_fd


class ConditionedICFDTest(unittest.TestCase):
    """Check diagnostic units and derivative semantics independently."""

    def test_affine_large_baseline_and_reported_units(self):
        """Field centering preserves derivatives and RMS perturbation units."""
        initial = jnp.ones((8, 8, 1, 2), dtype=jnp.float32) * 2
        result = conditioned_ic_fd(lambda x: 3 * x + 8, initial)
        self.assertAlmostEqual(result["direction_rms"], 1, places=6)
        self.assertAlmostEqual(result["direction_l2"], initial.size**0.5, places=5)
        self.assertEqual(len(result["sweep"]), 7)
        for row in result["sweep"][:3]:
            self.assertLess(row["relative_error"], 0.002)
            self.assertAlmostEqual(
                row["realized_positive_rms"] / row["epsilon_rms"], 1, places=3
            )
        self.assertNotIn("admitted", result)

    def test_float64_nonlinear_convergence(self):
        """Central differences converge for a smooth nonlinear field map."""
        previous = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", True)
        try:
            initial = jnp.linspace(0.1, 0.9, 32, dtype=jnp.float64)
            result = conditioned_ic_fd(lambda x: x**3, initial)
        finally:
            jax.config.update("jax_enable_x64", previous)
        errors = [row["relative_error"] for row in result["sweep"]]
        self.assertLess(errors[2], errors[0] / 50)
        self.assertLess(errors[4], 1e-7)

    def test_invalid_steps_rejected(self):
        """Invalid perturbations fail before numerical evaluation."""
        for values in [(), (0,), (-1,), (float("nan"),)]:
            with self.assertRaises(ValueError):
                conditioned_ic_fd(lambda x: x, jnp.ones((2,)), epsilons=values)


if __name__ == "__main__":
    unittest.main()
