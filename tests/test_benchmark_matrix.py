"""Targeted CI matrices must include only real, runnable experiment cells."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / ".github/scripts/generate-benchmark-matrix.py"
spec = importlib.util.spec_from_file_location("benchmark_matrix", SCRIPT)
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


@pytest.fixture
def configs(monkeypatch):
    cpu = SimpleNamespace(name="CPU", uses_gpu=False)
    gpu = SimpleNamespace(name="GPU", uses_gpu=True)
    cfg = SimpleNamespace(
        experiments={
            "optimization/solver_in_loop": object(),
            "forward/accuracy": object(),
        },
        solvers=[cpu, gpu],
        solver=lambda name: {"CPU": cpu, "GPU": gpu}[name],
    )
    other = SimpleNamespace(
        experiments={"forward/accuracy": object()},
        solvers=[gpu],
        solver=lambda name: gpu,
    )
    monkeypatch.setattr(matrix, "PROBLEMS", ["ns-grid", "other"])
    monkeypatch.setattr(
        matrix, "get_config", lambda name: {"ns-grid": cfg, "other": other}[name]
    )
    monkeypatch.setattr(
        matrix,
        "active_solvers",
        lambda cfg, suite, experiment: (
            [s.name for s in cfg.solvers] if suite == "forward" else ["GPU"]
        ),
    )
    return cfg


def run(monkeypatch, capsys, *args):
    monkeypatch.setattr("sys.argv", [str(SCRIPT), *args])
    matrix.main()
    return json.loads(capsys.readouterr().out)["include"]


def test_targeted_experiment_drops_unrelated_suites_problems_and_hardware(
    configs, monkeypatch, capsys
):
    assert run(monkeypatch, capsys, "--experiments", "solver_in_loop") == [
        {"suite": "optimization", "problem": "ns-grid", "hardware": "gpu"}
    ]


def test_default_all_preserves_suite_wide_matrix(configs, monkeypatch, capsys):
    cells = run(monkeypatch, capsys)
    assert len(cells) == 12
    assert {"suite": "forward", "problem": "ns-grid", "hardware": "cpu"} in cells


@pytest.mark.parametrize(
    "selector", ["optimization/solver_in_loop", "a,b", "", "a b", "$(foo)"]
)
def test_invalid_selector_fails_before_dispatch(configs, monkeypatch, capsys, selector):
    with pytest.raises(SystemExit) as exc:
        run(monkeypatch, capsys, "--experiments", selector)
    assert exc.value.code == 2
    assert not capsys.readouterr().out


@pytest.mark.parametrize(
    "args",
    [
        ["--experiments", "missing"],
        ["--experiments", "solver_in_loop", "--suites", "forward"],
        ["--experiments", "solver_in_loop", "--problems", "other"],
        ["--experiments", "solver_in_loop", "--solvers", "CPU"],
    ],
)
def test_empty_targeted_matrix_is_an_error(configs, monkeypatch, capsys, args):
    with pytest.raises(SystemExit) as exc:
        run(monkeypatch, capsys, *args)
    assert exc.value.code == 2
    assert not capsys.readouterr().out


def test_default_ics_has_no_solver_matrix(configs, monkeypatch, capsys):
    assert run(monkeypatch, capsys, "--suites", "ics") == []


def test_real_ns_grid_corrector_has_cpu_and_gpu_cells(monkeypatch, capsys):
    cells = run(
        monkeypatch,
        capsys,
        "--problems",
        "ns-grid",
        "--suites",
        "optimization",
        "--experiments",
        "solver_in_loop",
    )
    assert cells == [
        {"suite": "optimization", "problem": "ns-grid", "hardware": "gpu"},
        {"suite": "optimization", "problem": "ns-grid", "hardware": "cpu"},
    ]


@pytest.mark.parametrize("experiment", ["solver_in_loop", "all"])
def test_workflow_execution_qualifies_experiment_selector(tmp_path, experiment):
    """Execute the actual workflow shell with a stub CLI, then parse its arguments."""
    import os
    import subprocess
    import textwrap

    from mosaic.benchmarks.cli._helpers import _parse_experiments_path

    workflow = (SCRIPT.parents[1] / "workflows/benchmark-run.yml").read_text()
    step = workflow.split("      - name: Run ${{ matrix.suite }}", 1)[1]
    shell = textwrap.dedent(
        step.split("        run: |\n", 1)[1].split("        timeout-minutes:", 1)[0]
    )
    for key, value in {
        "suite": "optimization",
        "problem": "ns-grid",
        "hardware": "gpu",
    }.items():
        shell = shell.replace("${{ matrix." + key + " }}", value)
    shell = shell.replace("${{ inputs.solvers }}", "")
    stub = tmp_path / "uv"
    stub.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$ARGV_FILE"\n')
    stub.chmod(0o755)
    argv_file = tmp_path / "argv"
    subprocess.run(
        ["bash", "-eu", "-c", shell],
        check=True,
        env=dict(
            os.environ,
            PATH=f"{tmp_path}:{os.environ['PATH']}",
            EXPERIMENT=experiment,
            ARGV_FILE=str(argv_file),
        ),
    )
    argv = argv_file.read_text().splitlines()
    selector = argv[argv.index("--experiments") + 1]
    assert _parse_experiments_path(selector) == (
        ("optimization", "solver_in_loop", None)
        if experiment != "all"
        else (None, None, None)
    )
