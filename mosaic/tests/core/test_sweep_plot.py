"""Tests for ``Problem.add_sweep_plot`` aggregator-by-coord plot registration.

``add_sweep_plot(name, fn, *, group_by=..., filter=...)`` walks every
experiment with non-empty :attr:`Experiment.coords`, applies ``filter``,
partitions survivors by ``group_by``, and calls ``fn(payload, group)``
once per partition.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.experiment import kernel


@kernel(sweep_mode="none")
def _noop(t, ctx) -> dict:
    return {}


def _make_problem(name: str = "test") -> Problem:
    return Problem(
        name=name,
        tesseract_dir="navier-stokes-grid",
        output_key="result",
        ic_key="v0",
        solvers=[],
        make_ic={"default": lambda **kw: None},
        make_inputs=lambda spec, ic, **kw: {},
        error_fn=lambda *a, **k: 0.0,
    )


def _seed_results(results_root: Path, problem: str, cells: dict[str, dict]) -> None:
    """Write a ``result.json`` for each ``"<suite>/<rest>"`` key in *cells*."""
    for full_key, result in cells.items():
        suite, _, rest = full_key.partition("/")
        d = results_root / problem / suite / rest
        d.mkdir(parents=True, exist_ok=True)
        (d / "result.json").write_text(json.dumps(result))


class SweepPlotDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.results_root = Path(self._tmp.name) / "results"
        self._env = mock.patch.dict(
            "os.environ", {"MOSAIC_RESULTS_DIR": str(self.results_root)}
        )
        self._env.start()

    def tearDown(self) -> None:
        self._env.stop()
        self._tmp.cleanup()

    def test_no_grouping_one_partition(self) -> None:
        p = _make_problem()
        # Three experiments with coords.
        for n in (16, 32, 64):
            p.add_experiment(
                f"fwd/sweep_N_{n}", _noop, physics={"N": n}, coords={"N": n}
            )
        _seed_results(
            self.results_root,
            p.name,
            {
                "fwd/sweep_N_16": {"err": 0.1},
                "fwd/sweep_N_32": {"err": 0.05},
                "fwd/sweep_N_64": {"err": 0.02},
            },
        )

        calls: list[tuple[list, dict]] = []
        p.add_sweep_plot("by_N", lambda payload, group: calls.append((payload, group)))
        # Invoke the registered runner with the cfg.
        p.plot_fns["_extra/sweep/by_N"](p)

        self.assertEqual(len(calls), 1)
        payload, group = calls[0]
        self.assertEqual(group, {})
        self.assertEqual({c["coords"]["N"] for c in payload}, {16, 32, 64})

    def test_group_by_partitions_payload(self) -> None:
        p = _make_problem()
        # 2 regimes × 2 N values.
        for regime in ("diffusive", "turbulent"):
            for n in (16, 32):
                p.add_experiment(
                    f"fwd/{regime}_{n}",
                    _noop,
                    physics={"N": n},
                    coords={"N": n, "regime": regime},
                )
                _seed_results(
                    self.results_root,
                    p.name,
                    {f"fwd/{regime}_{n}": {"err": 0.5}},
                )

        groups: dict[tuple, list] = {}

        def fn(payload, group):
            groups[tuple(group.items())] = payload

        p.add_sweep_plot("by_regime", fn, group_by="regime")
        p.plot_fns["_extra/sweep/by_regime"](p)

        self.assertEqual(len(groups), 2)
        diffusive = groups[(("regime", "diffusive"),)]
        turbulent = groups[(("regime", "turbulent"),)]
        # group_by key is stripped from each cell's coords (it's identical in
        # the partition; the ``group`` arg carries it).
        self.assertEqual({c["coords"]["N"] for c in diffusive}, {16, 32})
        self.assertNotIn("regime", diffusive[0]["coords"])
        self.assertEqual({c["coords"]["N"] for c in turbulent}, {16, 32})

    def test_filter_drops_non_matching_cells(self) -> None:
        p = _make_problem()
        for regime in ("diffusive", "turbulent"):
            p.add_experiment(
                f"fwd/{regime}",
                _noop,
                physics={"N": 16},
                coords={"regime": regime},
            )
            _seed_results(self.results_root, p.name, {f"fwd/{regime}": {"err": 0.5}})

        seen_coords: list[dict] = []
        p.add_sweep_plot(
            "diffusive_only",
            lambda payload, group: seen_coords.extend(c["coords"] for c in payload),
            filter={"regime": "diffusive"},
        )
        p.plot_fns["_extra/sweep/diffusive_only"](p)

        # Only the diffusive cell survived the filter.
        self.assertEqual(seen_coords, [{"regime": "diffusive"}])

    def test_skips_experiments_without_results(self) -> None:
        p = _make_problem()
        p.add_experiment("fwd/has_result", _noop, coords={"N": 16})
        p.add_experiment("fwd/missing_result", _noop, coords={"N": 32})
        # Only seed one of the two.
        _seed_results(self.results_root, p.name, {"fwd/has_result": {"err": 0.1}})

        payloads: list[list] = []
        p.add_sweep_plot("any", lambda payload, group: payloads.append(payload))
        p.plot_fns["_extra/sweep/any"](p)

        self.assertEqual(len(payloads), 1)
        self.assertEqual(len(payloads[0]), 1)
        self.assertEqual(payloads[0][0]["exp_key"], "fwd/has_result")


if __name__ == "__main__":
    unittest.main()
