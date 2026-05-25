"""Tests for ``plot=dict[str, callable]`` multi-view support on ``add_experiment``.

The composite plot wrapper that ``Problem.add_experiment(plot={...})`` produces
must:

  * Call every view in declaration order.
  * Forward only the kwargs each view's signature declares (so a view written
    as ``def f(cfg)`` doesn't blow up when the runner passes ``suffix=``).
  * Survive one view raising and keep going.
  * Expose the union of ``{exp_key, sub_keys, results, suffix}`` parameters
    on its synthesised signature, so ``_register_plot``'s inspect-based
    wrapping picks the right (single-leaf vs. sweep) injection path.
"""

from __future__ import annotations

import unittest

from mosaic.benchmarks.core.config import Problem


class ComposePlotViewsTests(unittest.TestCase):
    def test_calls_every_view_in_order(self) -> None:
        called: list[str] = []

        def view_a(cfg):
            called.append("a")

        def view_b(cfg):
            called.append("b")

        composite = Problem._compose_plot_views(
            "key", {"first": view_a, "second": view_b}
        )
        composite(cfg=None)
        self.assertEqual(called, ["a", "b"])

    def test_filters_kwargs_per_view_signature(self) -> None:
        seen: dict[str, dict] = {}

        def needs_exp_key(cfg, *, exp_key):
            seen["exp_key_view"] = {"exp_key": exp_key}

        def needs_sub_keys(cfg, *, sub_keys):
            seen["sub_keys_view"] = {"sub_keys": sub_keys}

        def plain(cfg):
            seen["plain"] = {}

        composite = Problem._compose_plot_views(
            "k",
            {
                "exp_key_view": needs_exp_key,
                "sub_keys_view": needs_sub_keys,
                "plain": plain,
            },
        )
        composite(cfg=None, exp_key="tgv", sub_keys=["a", "b"], suffix="_debug")
        # Each view sees only what it declared.
        self.assertEqual(seen["exp_key_view"], {"exp_key": "tgv"})
        self.assertEqual(seen["sub_keys_view"], {"sub_keys": ["a", "b"]})
        self.assertEqual(seen["plain"], {})

    def test_one_view_raising_doesnt_skip_others(self) -> None:
        called: list[str] = []

        def bad(cfg):
            called.append("bad")
            raise RuntimeError("boom")

        def good(cfg):
            called.append("good")

        composite = Problem._compose_plot_views("k", {"bad": bad, "good": good})
        composite(cfg=None)
        self.assertEqual(called, ["bad", "good"])

    def test_synth_signature_union(self) -> None:
        import inspect

        def takes_exp_key(cfg, *, exp_key):
            pass

        def takes_sub_keys(cfg, *, sub_keys):
            pass

        composite = Problem._compose_plot_views(
            "k", {"a": takes_exp_key, "b": takes_sub_keys}
        )
        params = set(inspect.signature(composite).parameters)
        # The synthesised signature must expose the union so _register_plot's
        # inspect picks both injection paths.
        self.assertIn("exp_key", params)
        self.assertIn("sub_keys", params)


if __name__ == "__main__":
    unittest.main()
