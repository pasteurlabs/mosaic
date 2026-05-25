"""`mosaic` template / domain commands: validate-domain, new-domain,
validate-template, templates.
"""

from __future__ import annotations

import typer

from mosaic.benchmarks.cli import app
from mosaic.benchmarks.cli._helpers import _repo_root
from mosaic.benchmarks.core.console import console, print_rule


@app.command("validate-domain")
def validate_domain_cmd(
    problem: str = typer.Argument(help="Problem name to validate (e.g. 'ns-grid')."),
) -> None:
    """Validate a registered problem domain's Problem.

    Checks solver metadata, tesseract directories, suite defaults structure,
    ad_strategy values, and output_key against the schema module.
    """
    from mosaic.benchmarks.problems import get_config

    cfg = get_config(problem)
    n_checks = 0
    n_ok = 0

    # 1. Problem.validate()
    n_checks += 1
    try:
        cfg.validate()
        console.print("[green]  OK[/green]  Problem.validate()")
        n_ok += 1
    except ValueError as exc:
        console.print(f"[red]FAIL[/red]  Problem.validate():\n{exc}")

    # 2. Solver directories exist
    for spec in cfg.solvers:
        n_checks += 1
        solver_dir = cfg.tesseract_dir / spec.dir
        if solver_dir.is_dir():
            console.print(f"[green]  OK[/green]  solver dir: {spec.dir}/")
            n_ok += 1
        else:
            console.print(f"[red]FAIL[/red]  solver dir missing: {solver_dir}")

    # 3. Check output_key against schema module (best-effort)
    n_checks += 1
    try:
        # Schema modules live under the canonical tesseract directory name
        # (e.g. "navier-stokes-grid" → tesseract_shared.problems.navier_stokes_grid),
        # which is shared across CLI aliases like ns-grid / ns-3d-grid.
        slug = cfg.tesseract_dir.name.replace("-", "_")
        import importlib

        mod = importlib.import_module(f"tesseract_shared.problems.{slug}")
        if hasattr(mod, "OutputSchema"):
            out_fields = set(mod.OutputSchema.model_fields.keys())
            if cfg.output_key in out_fields:
                console.print(
                    f"[green]  OK[/green]  output_key {cfg.output_key!r} in OutputSchema"
                )
                n_ok += 1
            else:
                console.print(
                    f"[red]FAIL[/red]  output_key {cfg.output_key!r} not in "
                    f"OutputSchema fields: {sorted(out_fields)}"
                )
        else:
            console.print(
                "  [dim]SKIP[/dim]  output_key schema check (no OutputSchema found)"
            )
            n_ok += 1  # not a failure
    except ImportError:
        console.print(
            "  [dim]SKIP[/dim]  output_key schema check (could not import schema module)"
        )
        n_ok += 1  # not a failure

    # Summary
    if n_ok == n_checks:
        console.print(f"\n[green]All {n_checks} checks passed for {problem!r}.[/green]")
    else:
        console.print(
            f"\n[red]{n_checks - n_ok} of {n_checks} checks failed for {problem!r}.[/red]"
        )
        raise typer.Exit(1)


@app.command("new-domain")
def new_domain(
    name: str = typer.Argument(help="Name for the new domain (e.g. 'my-flow')."),
    from_template: str = typer.Option(
        ...,
        "--from-template",
        "-t",
        help="Template to scaffold from. Use 'mosaic templates' to list available templates.",
    ),
) -> None:
    """Scaffold a new benchmark domain from a template."""
    from mosaic.templates.scaffold import load_template, scaffold_domain

    tpl = load_template(from_template)
    created = scaffold_domain(name, tpl, target_dir=_repo_root() / "mosaic")
    print_rule(f"scaffolded domain: {name}")
    for role, path in created.items():
        console.print(f"  {role}: [green]{path.relative_to(_repo_root())}[/green]")
    console.print(
        f"\nNext steps:\n"
        f"  1. Edit the generated schemas and problem config\n"
        f"  2. Add a solver in mosaic/tesseracts/{name}/\n"
        f"  3. Run [bold]mosaic validate-domain {name}[/bold] to verify\n"
    )


@app.command("validate-template")
def validate_template_cmd(
    template: str = typer.Argument(help="Template name or path to a YAML file."),
) -> None:
    """Validate a task template against its schema module."""
    from mosaic.templates.scaffold import load_template, validate_template

    tpl = load_template(template)
    errors = validate_template(tpl)
    if errors:
        console.print(f"[red]{len(errors)} error(s) in template {tpl.name!r}:[/red]")
        for err in errors:
            console.print(f"  [red]- {err}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Template {tpl.name!r} is valid.[/green]")


@app.command("templates")
def list_templates_cmd(
    show: str | None = typer.Option(
        None,
        "--show",
        help="Show full details for a specific template (suites, physics defaults, ICs).",
    ),
) -> None:
    """List available task templates, or show details for a specific template."""
    from mosaic.templates.scaffold import list_templates, load_template

    if show:
        tpl = load_template(show)
        console.print(f"\n[bold]{tpl.name}[/bold]")
        console.print(f"  {tpl.description.strip()}\n")
        console.print(f"  [dim]schema:[/dim]  {tpl.schema_module}")
        console.print(f"  [dim]output_key:[/dim]  {tpl.output_key}")
        console.print(f"  [dim]ic_key:[/dim]  {tpl.ic_key}")
        console.print(f"  [dim]resolution_key:[/dim]  {tpl.resolution_key}")
        if tpl.physics_defaults:
            console.print("\n  [bold]Physics defaults:[/bold]")
            for k, v in tpl.physics_defaults.items():
                console.print(f"    {k}: {v}")
        if tpl.ic_defaults:
            console.print("\n  [bold]IC defaults:[/bold]")
            for k, v in tpl.ic_defaults.items():
                console.print(f"    {k}: {v}")
        for suite_name, suite_data in [
            ("forward", tpl.forward),
            ("gradient", tpl.gradient),
            ("cost", tpl.cost),
            ("optimization", tpl.optimization),
        ]:
            if suite_data:
                console.print(f"\n  [bold]{suite_name}:[/bold]")
                for exp_name in suite_data:
                    n_runs = (
                        len(suite_data[exp_name])
                        if isinstance(suite_data[exp_name], list)
                        else 1
                    )
                    console.print(f"    {exp_name}  ({n_runs} run(s))")
        console.print()
        return

    templates = list_templates()
    if not templates:
        console.print("[dim]No templates found.[/dim]")
        return
    for name in templates:
        tpl = load_template(name)
        console.print(f"  [bold]{name}[/bold]  {tpl.description.strip()}")
    console.print("\n  [dim]Use --show <name> for full details.[/dim]")
