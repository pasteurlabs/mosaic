# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``coords`` typed sweep position on ``Experiment``.

``Problem.add_experiment(..., coords={...})`` annotates an experiment with
its position in a parameter space — borrowed from the asie ``Campaign``
API. Auto-populated for variant fan-out (``runs=[{name: ...}, ...]``)
with ``variant=<name>``; user-supplied ``coords`` is merged in (user wins
on key collision).

Mosaic's internal-sweep path (``physics={"N": [16, 32, 64]}`` on a
``sweep_mode="default"`` kernel) collapses the axis into a single
runtime-sweep ``Experiment`` rather than fanning out, so coords are not
auto-derived for that case — the user can still attach a manual
``coords={"regime": "diffusive"}`` though, and it stays attached.
"""

from __future__ import annotations

import unittest

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.experiment import kernel


@kernel(sweep_mode="none")
def _noop(t, ctx) -> dict:
    return {"metrics": {}}


@kernel(sweep_mode="default")
def _noop_sweep(t, ctx) -> dict:
    return {"metrics": {}}


def _make_problem() -> Problem:
    return Problem(
        name="test",
        tesseract_dir="navier-stokes-grid",
        output_key="result",
        ic_key="v0",
        solvers=[],
        make_ic={"default": lambda **kw: None},
        make_inputs=lambda spec, ic, **kw: {},
        error_fn=lambda *a, **k: 0.0,
    )


class CoordsRegistrationTests(unittest.TestCase):
    def test_no_user_coords_means_empty(self) -> None:
        p = _make_problem()
        p.add_experiment("fwd/baseline", _noop, physics={"N": 32})
        self.assertEqual(p.experiments["fwd/baseline"].coords, {})

    def test_user_coords_attached_to_single_leaf(self) -> None:
        p = _make_problem()
        p.add_experiment(
            "fwd/baseline",
            _noop,
            physics={"N": 32},
            coords={"regime": "diffusive"},
        )
        self.assertEqual(p.experiments["fwd/baseline"].coords, {"regime": "diffusive"})

    def test_user_coords_attached_to_runtime_sweep_experiment(self) -> None:
        # Internal sweep — mosaic registers ONE Experiment that iterates
        # at runtime. The user coords still attach (as a constant label
        # for the whole sweep).
        p = _make_problem()
        p.add_experiment(
            "fwd/sweep_N",
            _noop_sweep,
            physics={"N": [16, 32, 64]},
            coords={"regime": "diffusive"},
        )
        # No sub_keys are created for the internal sweep.
        self.assertIn("fwd/sweep_N", p.experiments)
        self.assertEqual(p.experiments["fwd/sweep_N"].coords, {"regime": "diffusive"})

    def test_variant_fan_out_auto_tags_with_variant_name(self) -> None:
        p = _make_problem()
        p.add_experiment(
            "fwd/agreement",
            _noop,
            runs=[
                {"name": "tgv", "ic": {"name": "default"}, "physics": {"N": 32}},
                {"name": "mm", "ic": {"name": "default"}, "physics": {"N": 32}},
            ],
        )
        subs = {
            k: v for k, v in p.experiments.items() if k.startswith("fwd/agreement/")
        }
        self.assertEqual(len(subs), 2)
        self.assertEqual({v.coords["variant"] for v in subs.values()}, {"tgv", "mm"})

    def test_variant_fan_out_merges_user_coords(self) -> None:
        # User coords carry across every variant (e.g. a constant regime
        # label). Variant name still auto-tags.
        p = _make_problem()
        p.add_experiment(
            "fwd/agreement",
            _noop,
            runs=[
                {"name": "tgv", "ic": {"name": "default"}, "physics": {"N": 32}},
                {"name": "mm", "ic": {"name": "default"}, "physics": {"N": 32}},
            ],
            coords={"regime": "diffusive"},
        )
        for k, v in p.experiments.items():
            if not k.startswith("fwd/agreement/"):
                continue
            self.assertEqual(v.coords.get("regime"), "diffusive")
            self.assertIn(v.coords["variant"], ("tgv", "mm"))

    def test_user_coords_override_auto_variant_tag(self) -> None:
        # User-provided coords win on key collision (rare, but the
        # precedence is documented).
        p = _make_problem()
        p.add_experiment(
            "fwd/agreement",
            _noop,
            runs=[
                {"name": "tgv", "ic": {"name": "default"}, "physics": {"N": 32}},
            ],
            coords={"variant": "manual_override"},
        )
        # Single-variant runs list goes through the single-leaf path,
        # so coords is just the user-provided dict.
        self.assertEqual(
            p.experiments["fwd/agreement"].coords, {"variant": "manual_override"}
        )


if __name__ == "__main__":
    unittest.main()
