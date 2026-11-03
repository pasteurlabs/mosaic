# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""An experiment's own reference generator, run by ``mosaic reference``."""

import contextlib
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from mosaic.benchmarks.cli import reference as cli_reference
from mosaic.benchmarks.core import runner
from mosaic.benchmarks.core.reference import ConvergedSpec


@dataclass
class _Config:
    name: str = "toy"
    domain_extent: float = 1.0
    output_key: str = "result"
    solvers: list = field(
        default_factory=lambda: [
            SimpleNamespace(name="Other"),
            SimpleNamespace(name="Ref"),
        ]
    )

    def make_inputs(self, solver: str, v: np.ndarray, **kwargs) -> dict:
        return {"solver": solver, "v": v, **kwargs}


def test_generator_runs_the_reference_solver_and_stores_its_arrays(
    monkeypatch, tmp_path
):
    built, containers, calls = [], [], []
    monkeypatch.setattr(
        runner, "build_all", lambda cfg: built.append(cfg.solvers) or {"Ref": "ref:tag"}
    )

    @contextlib.contextmanager
    def tracked(tag, gpus, docker_args):
        containers.append(tag)
        yield "tesseract"

    def apply(t, inputs, output_key):
        calls.append(inputs)
        return inputs["v"] * 2

    monkeypatch.setattr(runner, "_tracked_tesseract", tracked)
    monkeypatch.setattr(runner, "safe_apply", apply)
    path = tmp_path / "toy.npz"
    monkeypatch.setattr(cli_reference, "reference_path", lambda domain, key: path)

    def generate(advance, log):
        return {"doubled": advance(np.ones(3), dt=0.1, steps=5)}

    run = {"physics": {"nu": 0.5}, "reference": generate}
    cli_reference._generate_trajectories(
        _Config(), "forward/toy", run, ConvergedSpec(solver="Ref", high_n=64)
    )

    assert [[s.name for s in b] for b in built] == [["Ref"]]
    assert containers == ["ref:tag"]
    assert calls[0]["solver"] == "Ref"
    assert {k: calls[0][k] for k in ("nu", "dt", "steps", "domain_extent")} == {
        "nu": 0.5,
        "dt": 0.1,
        "steps": 5,
        "domain_extent": 1.0,
    }
    np.testing.assert_array_equal(np.load(path)["doubled"], 2 * np.ones(3))


def test_from_results_accepts_spaces_in_the_experiment_list(monkeypatch, tmp_path):
    """``-e "forward/baseline, forward/agreement"`` generates both references."""
    from typer.testing import CliRunner

    from mosaic.benchmarks.cli import app
    from mosaic.benchmarks.problems import get_config

    saved = []
    monkeypatch.setattr(
        cli_reference,
        "save_reference",
        lambda domain, key, refs, sweep_values=None: saved.append(key) or tmp_path,
    )
    names = [s.name for s in get_config("thermal-mesh").solvers][:2]
    for exp in ("baseline", "agreement"):
        exp_dir = tmp_path / "thermal-mesh" / "forward" / exp
        exp_dir.mkdir(parents=True)
        np.savez(exp_dir / "fields.npz", **{f"{n}_0": np.ones(3) for n in names})
        (exp_dir / "result.json").write_text('{"sweep": {"key": "nx", "values": [8]}}')

    result = CliRunner().invoke(
        app,
        [
            "reference",
            "-p",
            "thermal-mesh",
            "-e",
            "forward/baseline, forward/agreement",
            "--from-results",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert saved == ["forward/baseline", "forward/agreement"]
