# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`mosaic reference` — generate precomputed reference solutions.

Generates reference NPZ files for experiments listed in
``PRECOMPUTED_EXPERIMENTS``. These references decouple single-solver CI
runs from needing all peers present.

Most experiments use a **consensus** reference (trimmed mean across all
solvers). Experiments in ``CONVERGED_REFERENCES`` instead use a
**converged** reference: one spectral solver run at high resolution and
spectrally downsampled to the benchmark grid — the true continuous
solution, not a peer average that would share their common-mode bias.

Two modes:

1. **From existing results** (``--from-results``): reads field snapshots
   from a ``mosaic-results/`` directory and extracts the consensus
   arrays. Fast, no solver runs needed — just needs prior results.
   (Consensus experiments only; converged references must be run.)

2. **By running solvers** (default): builds + runs the solver(s) for the
   target experiments and writes the NPZ — trimmed mean for consensus
   experiments, high-N spectral run for converged ones.
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from mosaic.benchmarks.cli import app
from mosaic.benchmarks.core.console import console
from mosaic.benchmarks.core.reference import (
    PRECOMPUTED_EXPERIMENTS,
    ConvergedSpec,
    converged_spec,
    extract_references_from_fields,
    save_reference,
    spectral_downsample,
)

logger = logging.getLogger(__name__)


@app.command()
def reference(
    problems: str = typer.Option(
        "all",
        "--problems",
        "-p",
        help="Comma-separated domain(s) or 'all'. "
        "Only domains with consensus experiments are processed.",
    ),
    experiments: str = typer.Option(
        "all",
        "--experiments",
        "-e",
        help="Comma-separated experiment keys (e.g. 'forward/baseline') or 'all'.",
    ),
    from_results: str | None = typer.Option(
        None,
        "--from-results",
        help="Path to a mosaic-results/ directory. Extract references from "
        "existing fields.npz snapshots instead of running solvers.",
    ),
) -> None:
    """Generate precomputed reference solutions for consensus experiments.

    References are trimmed-mean fields computed across all solvers. Once
    checked in, single-solver CI runs can compute errors without needing
    the full solver ensemble.

    Examples:
        # Extract from existing benchmark results
        mosaic reference --from-results mosaic-results/

        # Extract for one domain only
        mosaic reference -p structural-mesh --from-results mosaic-results/

        # Run solvers to generate fresh references (requires built images)
        mosaic reference -p thermal-mesh
    """
    # Determine which domains to process.
    if problems == "all":
        domains = list(PRECOMPUTED_EXPERIMENTS.keys())
    else:
        domains = [p.strip() for p in problems.split(",")]
        # Filter to domains that actually have precomputed experiments.
        unknown = [d for d in domains if d not in PRECOMPUTED_EXPERIMENTS]
        if unknown:
            console.print(
                f"[yellow]WARN[/] No precomputed-reference experiments for: "
                f"{', '.join(unknown)}."
            )
        domains = [d for d in domains if d in PRECOMPUTED_EXPERIMENTS]

    if not domains:
        console.print("[yellow]No domains with precomputed references to process.[/]")
        raise typer.Exit()

    # Determine which experiments to process.
    exp_filter = None if experiments == "all" else set(experiments.split(","))

    if from_results is not None:
        _generate_from_results(Path(from_results), domains, exp_filter)
    else:
        _generate_by_running(domains, exp_filter)


def _generate_from_results(
    results_dir: Path,
    domains: list[str],
    exp_filter: set[str] | None,
) -> None:
    """Extract references from existing mosaic-results/ field snapshots."""
    if not results_dir.is_dir():
        console.print(f"[red]Results directory not found:[/] {results_dir}")
        raise typer.Exit(code=1)

    total = 0
    for domain in domains:
        exps = PRECOMPUTED_EXPERIMENTS[domain]
        if exp_filter is not None:
            exps = [e for e in exps if e in exp_filter]

        for exp_key in exps:
            # The fields.npz lives at:
            #   results_dir/<domain>/<suite>/<exp_name>/fields.npz
            suite, _, exp_name = exp_key.partition("/")
            fields_path = results_dir / domain / suite / exp_name / "fields.npz"

            if not fields_path.exists():
                console.print(
                    f"  [yellow]SKIP[/] {domain}/{exp_key}: "
                    f"no fields.npz at {fields_path}"
                )
                continue

            # Determine the number of sweep values from the result.json.
            result_path = fields_path.parent / "result.json"
            n_sweep = _count_sweep_values(result_path, fields_path)
            if n_sweep == 0:
                console.print(
                    f"  [yellow]SKIP[/] {domain}/{exp_key}: "
                    "could not determine sweep values"
                )
                continue

            refs = extract_references_from_fields(fields_path, n_sweep)
            if not refs:
                console.print(
                    f"  [yellow]SKIP[/] {domain}/{exp_key}: "
                    "no consensus arrays found in fields.npz"
                )
                continue

            # Read sweep values from the fields.npz for provenance.
            sweep_values = _read_sweep_values(fields_path)
            path = save_reference(domain, exp_key, refs, sweep_values)
            console.print(
                f"  [green]OK[/] {domain}/{exp_key}: "
                f"{len(refs)} reference(s) → "
                f"{path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}"
            )
            total += 1

    console.print(f"\n[bold]Generated {total} reference file(s).[/]")


def _count_sweep_values(result_path: Path, fields_path: Path) -> int:
    """Determine the number of sweep values for an experiment.

    Tries the result.json first (schema_version=1 has sweep.values),
    then falls back to counting consensus_* keys in the fields.npz.
    """
    import json

    import numpy as np

    if result_path.exists():
        try:
            with open(result_path) as f:
                result = json.load(f)
            sweep = result.get("sweep")
            if sweep and "values" in sweep:
                return len(sweep["values"])
        except Exception:
            logger.debug(
                "reading sweep count from %s failed", result_path, exc_info=True
            )

    # Fallback: count consensus_* keys in the NPZ.
    try:
        with np.load(str(fields_path), allow_pickle=False) as data:
            return sum(1 for k in data if k.startswith("consensus_"))
    except Exception:
        return 0


def _read_sweep_values(fields_path: Path) -> list | None:
    """Read the sweep_values array from a fields.npz, if present."""
    import numpy as np

    try:
        with np.load(str(fields_path), allow_pickle=False) as data:
            if "sweep_values" in data:
                return data["sweep_values"].tolist()
    except Exception:
        logger.debug("reading sweep_values from %s failed", fields_path, exc_info=True)
    return None


def _experiment_run(cfg: object, exp_key: str) -> dict | None:
    """Recover the normalised run dict for an experiment.

    ``Experiment.params`` is only a metadata manifest; the actual run config
    (ic / physics / sweep) is captured in the experiment closure as ``runs``.
    Returns the first (single-sweep) run dict, or None.
    """
    exp = cfg.experiments.get(exp_key)
    if exp is None or exp.fn is None:
        return None
    for default in getattr(exp.fn, "__defaults__", ()) or ():
        if isinstance(default, dict) and isinstance(default.get("runs"), list):
            runs = default["runs"]
            return runs[0] if runs else None
    return None


def _experiment_run_params(cfg: object, exp_key: str) -> dict | None:
    """Extract sweep/IC/physics from an experiment registration.

    Returns a dict with ``sweep_key``, ``sweep_values``, ``ic_name``, ``seed``,
    ``phys`` — or None (after printing why) if the experiment is unusable for
    reference generation.
    """
    if exp_key not in cfg.experiments:
        console.print(f"    [yellow]SKIP[/] experiment {exp_key!r} not registered")
        return None

    run = _experiment_run(cfg, exp_key)
    if run is None:
        console.print("    [yellow]SKIP[/] could not recover run config")
        return None

    sweep_cfg = run.get("sweep", {})
    sweep_key = sweep_cfg.get("key")
    sweep_values = sweep_cfg.get("values", [])
    if not sweep_key or not sweep_values:
        console.print("    [yellow]SKIP[/] no sweep configured")
        return None

    ic_cfg = run.get("ic", {})
    return {
        "sweep_key": sweep_key,
        "sweep_values": sweep_values,
        "ic_name": ic_cfg.get("name", next(iter(cfg.make_ic))),
        "seed": ic_cfg.get("seed", 0),
        "phys": run.get("physics", {}),
    }


def _apply_via_image(tag: str, inputs: dict, output_key: str):
    """Open a built image as a (CPU-only) Tesseract and run one forward pass.

    ``safe_apply`` needs a live ``Tesseract`` object, not an image tag, so the
    tag is opened through the same tracked-container context the runner uses.
    ``gpus=None`` runs CPU-only (no ``--gpus`` flag).
    """
    from mosaic.benchmarks.core.resources import container_memory_args
    from mosaic.benchmarks.core.runner import _tracked_tesseract, safe_apply

    docker_args = ["--no-healthcheck", *container_memory_args()]
    with _tracked_tesseract(tag, None, docker_args) as t:
        return safe_apply(t, inputs, output_key)


def _generate_consensus(cfg: object, exp_key: str, rp: dict) -> None:
    """Trimmed-mean consensus reference: run all solvers, average per sweep value."""
    import numpy as np

    from mosaic.benchmarks.core.runner import build_all
    from mosaic.benchmarks.core.utils import trimmed_mean

    sweep_key, sweep_values = rp["sweep_key"], rp["sweep_values"]

    console.print("    Building solvers...")
    tags = build_all(cfg)

    solver_outputs: dict[str, dict] = {s.name: {} for s in cfg.solvers}
    for i, val in enumerate(sweep_values):
        curr_phys = {**rp["phys"], sweep_key: val, "domain_extent": cfg.domain_extent}
        ic = cfg.make_ic[rp["ic_name"]](
            L=cfg.domain_extent, seed=rp["seed"], **curr_phys
        )
        for s in cfg.solvers:
            try:
                inputs_s = cfg.make_inputs(s.name, ic, **curr_phys)
                result = _apply_via_image(tags[s.name], inputs_s, cfg.output_key)
                if result is not None:
                    norm = s.normalize_output
                    out = norm(result) if norm is not None else result
                    solver_outputs[s.name][i] = np.asarray(out)
                    console.print(f"    [green]✓[/] {s.name} @ {sweep_key}={val}")
                else:
                    console.print(
                        f"    [yellow]✗[/] {s.name} @ {sweep_key}={val}: apply failed"
                    )
            except Exception as e:
                console.print(f"    [red]✗[/] {s.name} @ {sweep_key}={val}: {e}")

    refs: dict[int, np.ndarray] = {}
    for i, val in enumerate(sweep_values):
        arrays = [
            solver_outputs[s.name][i]
            for s in cfg.solvers
            if i in solver_outputs[s.name]
        ]
        if len(arrays) < 2:
            console.print(
                f"    [yellow]WARN[/] sweep {sweep_key}={val}: "
                f"only {len(arrays)} solver(s), need >= 2 for consensus"
            )
            if arrays:
                refs[i] = np.asarray(arrays[0])
            continue
        refs[i] = np.asarray(trimmed_mean(arrays))

    _save(cfg, exp_key, refs, sweep_values)


def _generate_converged(
    cfg: object, exp_key: str, rp: dict, spec: ConvergedSpec
) -> None:
    """Converged reference: run one spectral solver at high N, spectrally downsample."""
    from dataclasses import replace

    import numpy as np

    from mosaic.benchmarks.core.runner import build_all

    sweep_key, sweep_values = rp["sweep_key"], rp["sweep_values"]
    solver = next((s for s in cfg.solvers if s.name == spec.solver), None)
    if solver is None:
        console.print(f"    [red]✗[/] reference solver {spec.solver!r} not found")
        return

    n_bench = int(rp["phys"]["N"])
    console.print(
        f"    Converged reference via {spec.solver} at N={spec.high_n} "
        f"→ N={n_bench} (spectral truncation)"
    )
    console.print(f"    Building {spec.solver}...")
    # Only the reference solver is needed — build it alone rather than the
    # whole ensemble.
    tags = build_all(replace(cfg, solvers=[solver]))

    refs: dict[int, np.ndarray] = {}
    for i, val in enumerate(sweep_values):
        # Integrate at the high resolution, then truncate back to the grid the
        # benchmark (and the stored reference) lives on.
        curr_phys = {
            **rp["phys"],
            "N": spec.high_n,
            sweep_key: val,
            "domain_extent": cfg.domain_extent,
        }
        ic = cfg.make_ic[rp["ic_name"]](
            L=cfg.domain_extent, seed=rp["seed"], **curr_phys
        )
        try:
            inputs_s = cfg.make_inputs(solver.name, ic, **curr_phys)
            result = _apply_via_image(tags[solver.name], inputs_s, cfg.output_key)
        except Exception as e:
            console.print(f"    [red]✗[/] {spec.solver} @ {sweep_key}={val}: {e}")
            continue
        if result is None:
            console.print(
                f"    [yellow]✗[/] {spec.solver} @ {sweep_key}={val}: apply failed"
            )
            continue
        norm = solver.normalize_output
        hi = np.asarray(norm(result) if norm is not None else result)
        refs[i] = spectral_downsample(hi, n_bench).astype(np.float32)
        console.print(f"    [green]✓[/] {spec.solver} @ {sweep_key}={val}")

    _save(cfg, exp_key, refs, sweep_values)


def _save(cfg: object, exp_key: str, refs: dict, sweep_values: list) -> None:
    """Persist generated references, or report that none were produced."""
    if refs:
        path = save_reference(cfg.name, exp_key, refs, sweep_values)
        console.print(f"    [green]OK[/] {len(refs)} reference(s) → {path}")
    else:
        console.print("    [red]FAIL[/] no references generated")


def _generate_by_running(
    domains: list[str],
    exp_filter: set[str] | None,
) -> None:
    """Generate references by running solvers.

    Consensus experiments run the full ensemble and store a trimmed mean;
    converged experiments (``CONVERGED_REFERENCES``) run one spectral solver
    at high resolution and spectrally downsample to the benchmark grid.
    """
    from mosaic.benchmarks.problems import get_config

    console.print("[bold]Generating references by running solvers...[/]\n")

    for domain in domains:
        cfg = get_config(domain)
        exps = PRECOMPUTED_EXPERIMENTS[domain]
        if exp_filter is not None:
            exps = [e for e in exps if e in exp_filter]

        for exp_key in exps:
            console.print(f"  [bold]{domain}/{exp_key}[/]")
            rp = _experiment_run_params(cfg, exp_key)
            if rp is None:
                continue
            spec = converged_spec(domain, exp_key)
            if spec is not None:
                _generate_converged(cfg, exp_key, rp, spec)
            else:
                _generate_consensus(cfg, exp_key, rp)

    console.print("\n[bold]Done.[/]")
