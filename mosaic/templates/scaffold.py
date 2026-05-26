"""Template loader, validator, and scaffolding for new benchmark domains.

Usage (via CLI)::

    mosaic new-domain my-flow --from-template ns-periodic
    mosaic validate-template mosaic/templates/ns-periodic.yaml

Or programmatically::

    from mosaic.templates.scaffold import list_templates, load_template, scaffold_domain

    templates = list_templates()
    tpl = load_template("ns-periodic")
    scaffold_domain("my-flow", tpl, target_dir=Path("mosaic"))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_TEMPLATE_DIR = Path(__file__).parent


@dataclass
class DomainTemplate:
    """Validated starting configuration for a benchmark domain."""

    name: str
    description: str
    schema_module: str
    output_key: str
    ic_key: str = "ic"
    domain_extent: float = 1.0
    resolution_key: str = "N"
    category_label: str = ""
    bc_description: str = ""
    physics_defaults: dict[str, Any] = field(default_factory=dict)
    ic_defaults: dict[str, Any] = field(default_factory=dict)
    forward: dict[str, Any] = field(default_factory=dict)
    gradient: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    optimization: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None


def list_templates() -> list[str]:
    """Return names of available built-in templates."""
    return sorted(
        p.stem for p in _TEMPLATE_DIR.glob("*.yaml") if not p.name.startswith("_")
    )


def load_template(name_or_path: str) -> DomainTemplate:
    """Load a template by name (built-in) or file path.

    Raises ``FileNotFoundError`` if the template is not found, and
    ``ValueError`` if required fields are missing.
    """
    path = Path(name_or_path)
    if not path.exists():
        path = _TEMPLATE_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        available = list_templates()
        raise FileNotFoundError(
            f"Template {name_or_path!r} not found. "
            f"Available: {available}. "
            f"Or provide a path to a YAML file."
        )
    with open(path) as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: expected a YAML mapping, got {type(doc).__name__}")
    for key in ("name", "schema_module", "output_key"):
        if key not in doc:
            raise ValueError(f"{path}: missing required field {key!r}")
    return DomainTemplate(
        name=doc["name"],
        description=doc.get("description", ""),
        schema_module=doc["schema_module"],
        output_key=doc["output_key"],
        ic_key=doc.get("ic_key", "ic"),
        domain_extent=doc.get("domain_extent", 1.0),
        resolution_key=doc.get("resolution_key", "N"),
        category_label=doc.get("category_label", ""),
        bc_description=doc.get("bc_description", ""),
        physics_defaults=doc.get("physics_defaults", {}),
        ic_defaults=doc.get("ic_defaults", {}),
        forward=doc.get("forward", {}),
        gradient=doc.get("gradient", {}),
        cost=doc.get("cost", {}),
        optimization=doc.get("optimization", {}),
        source_path=path,
    )


def validate_template(tpl: DomainTemplate) -> list[str]:
    """Validate a template, returning a list of error messages (empty = valid).

    Checks:
    - Schema module is importable and has InputSchema / OutputSchema
    - output_key and ic_key are fields on the schema
    - Suite defaults have valid structure
    """
    errors: list[str] = []

    # Check schema module
    try:
        import importlib

        mod = importlib.import_module(tpl.schema_module)
    except ImportError as exc:
        errors.append(f"cannot import schema module {tpl.schema_module!r}: {exc}")
        return errors

    for cls_name in ("InputSchema", "OutputSchema"):
        if not hasattr(mod, cls_name):
            errors.append(f"{tpl.schema_module} has no {cls_name}")

    # Check output_key is on OutputSchema
    if hasattr(mod, "OutputSchema"):
        out_fields = set(mod.OutputSchema.model_fields.keys())
        if tpl.output_key not in out_fields:
            errors.append(
                f"output_key={tpl.output_key!r} not in OutputSchema fields: "
                f"{sorted(out_fields)}"
            )

    # Check ic_key is on InputSchema
    if hasattr(mod, "InputSchema"):
        in_fields = set(mod.InputSchema.model_fields.keys())
        if tpl.ic_key not in in_fields:
            errors.append(
                f"ic_key={tpl.ic_key!r} not in InputSchema fields: {sorted(in_fields)}"
            )

    # Check suite defaults have list-of-dicts structure
    for suite_name, suite_defaults in [
        ("forward", tpl.forward),
        ("gradient", tpl.gradient),
        ("optimization", tpl.optimization),
        ("cost", tpl.cost),
    ]:
        for exp_name, exp_runs in suite_defaults.items():
            if not isinstance(exp_runs, list):
                errors.append(
                    f"{suite_name}.{exp_name}: expected a list of run dicts, "
                    f"got {type(exp_runs).__name__}"
                )

    return errors


# ── Codegen helpers ──────────────────────────────────────────────────────────


def _render_experiment_todos(tpl: DomainTemplate) -> str:
    """Emit a ``# TODO:`` block per experiment in the template.

    Each entry shows the suggested ``ic`` / ``physics`` / ``fd`` / ``optim``
    payload from the template as a comment, so the user can copy-paste it
    into a real ``problem.add_experiment(...)`` call.
    """
    lines: list[str] = []
    for suite_name, suite_data in [
        ("forward", tpl.forward),
        ("gradient", tpl.gradient),
        ("cost", tpl.cost),
        ("optimization", tpl.optimization),
    ]:
        if not suite_data:
            continue
        lines.append(f"# {suite_name}/")
        for exp_name, runs in suite_data.items():
            lines.append(f"#   TODO: add_experiment({suite_name}/{exp_name})")
            for run in runs:
                for k, v in run.items():
                    lines.append(f"#       {k} = {v!r}")
        lines.append("")
    return "\n".join(lines)


# ── Scaffold entry point ─────────────────────────────────────────────────────


def scaffold_domain(
    domain_name: str,
    tpl: DomainTemplate,
    *,
    target_dir: Path = Path("mosaic"),
) -> dict[str, Path]:
    """Generate the minimum file tree for a new benchmark domain.

    Creates:

    * ``tesseracts/tesseract_shared/problems/<domain>/{schemas,__init__}.py``
      — schema stubs (where solvers import their canonical InputSchema /
      OutputSchema).
    * ``tesseracts/<domain>/`` — empty, ready for solver dirs.
    * ``benchmarks/problems/<domain>/{__init__,config}.py`` — the
      ``Problem`` instance, the IC generator, the ``make_inputs`` callable,
      and ``# TODO:`` blocks per template experiment, all in ``config.py``.

    Two-file Problem packages keep the scaffold lean; once the domain
    grows, ``config.py`` can be split into ``ics.py`` / ``physics.py`` /
    ``experiments.py`` by hand.

    Returns a dict of generated file paths keyed by role.
    """
    slug = domain_name.replace("-", "_")
    created: dict[str, Path] = {}

    # 1. Schema stubs under ``tesseract_shared/problems/<slug>``.
    schema_dir = target_dir / "tesseracts" / "tesseract_shared" / "problems" / slug
    schema_dir.mkdir(parents=True, exist_ok=True)

    schema_init = schema_dir / "__init__.py"
    schema_init.write_text(
        '__all__ = ["InputSchema", "OutputSchema"]\n\n'
        "from .schemas import InputSchema, OutputSchema\n"
    )
    created["schema_init"] = schema_init

    schema_path = schema_dir / "schemas.py"
    schema_path.write_text(
        f'"""Canonical InputSchema / OutputSchema for {domain_name} tesseracts.\n'
        f"\n"
        f"Generated from template: {tpl.name}\n"
        f'"""\n\n'
        f"from pydantic import BaseModel\n\n\n"
        f"# TODO: declare your domain's canonical fields here. Every solver\n"
        f"# subclasses these so cross-solver comparison is well-defined. See\n"
        f"# docs/tutorial.qmd Part B §4 for the Array / Differentiable typing.\n"
        f"class InputSchema(BaseModel):\n"
        f'    """Inputs for {domain_name} solvers."""\n\n'
        f"    pass\n\n\n"
        f"class OutputSchema(BaseModel):\n"
        f'    """Outputs for {domain_name} solvers."""\n\n'
        f"    pass\n"
    )
    created["schemas"] = schema_path

    # 2. Empty tesseract directory — solver subdirs land here.
    tess_dir = target_dir / "tesseracts" / domain_name
    tess_dir.mkdir(parents=True, exist_ok=True)
    created["tesseracts_dir"] = tess_dir

    # 3. Two-file Problem package: __init__ stub + everything in config.py.
    pkg_dir = target_dir / "benchmarks" / "problems" / slug
    pkg_dir.mkdir(parents=True, exist_ok=True)

    pkg_init = pkg_dir / "__init__.py"
    pkg_init.write_text(
        f'"""Problem package for {domain_name}, generated from template: {tpl.name}. See :mod:`.config`."""\n'
    )
    created["problem_init"] = pkg_init

    n_default = tpl.physics_defaults.get("N", 8)
    ic_default_name = tpl.ic_defaults.get("name", "default")
    category_label = tpl.category_label or domain_name
    description = tpl.description.strip()
    bc_description = tpl.bc_description.strip()
    experiment_todos = _render_experiment_todos(tpl)
    experiment_todos_block = (
        "# Template-suggested experiments (uncomment + flesh out as you wire each one up):\n"
        + "\n".join(
            f"# {line[2:] if line.startswith('# ') else line}"
            for line in experiment_todos.splitlines()
        )
        if experiment_todos.strip()
        else "# (template has no suite defaults; add experiments as you implement them.)"
    )

    config_path = pkg_dir / "config.py"
    config_path.write_text(
        f'"""Problem definition for {domain_name} (generated from template: {tpl.name}).\n'
        f"\n"
        f"Holds the IC generator, ``make_inputs``, the :class:`Problem` instance,\n"
        f"and the per-experiment ``problem.add_experiment(...)`` registrations.\n"
        f"Split into ``ics.py`` / ``physics.py`` / ``experiments.py`` when the\n"
        f"file gets too large to navigate.\n"
        f'"""\n\n'
        f"from __future__ import annotations\n\n"
        f"import numpy as np\n\n"
        f"from mosaic.benchmarks.core.config import Problem, SolverSpec, discover_solvers\n"
        f"from mosaic.benchmarks.core.utils import l2_error_rel\n"
        f"from mosaic.benchmarks.problems.shared.plots.ics import plot_ic\n"
        f"from mosaic.benchmarks.problems.shared.plots.solver_styles import apply_styles\n\n"
        f"_TESSERACT_SLUG = {domain_name!r}\n\n\n"
        f"# ── Initial conditions ───────────────────────────────────────────────\n"
        f"# TODO: replace this stub with the canonical IC generator(s) for\n"
        f"# {domain_name}. Each generator has signature\n"
        f"# ``(L: float, seed: int, **physics) -> array``.\n"
        f"def _default_ic(L: float = 1.0, seed: int = 0, **physics) -> np.ndarray:\n"
        f"    N = physics.get('N', {n_default})\n"
        f"    return np.zeros(N, dtype=np.float32)\n\n\n"
        f"# ── make_inputs ──────────────────────────────────────────────────────\n"
        f"# TODO: assemble the dict passed to each solver's ``apply`` from the IC\n"
        f"# and physics parameters. Per-solver overrides come from\n"
        f"# ``spec.input_overrides``.\n"
        f"def make_inputs(spec: SolverSpec, ic: np.ndarray, **physics) -> dict:\n"
        f"    base = {{{tpl.ic_key!r}: ic}}\n"
        f"    return {{**base, **spec.input_overrides}}\n\n\n"
        f"# ── Solver discovery ─────────────────────────────────────────────────\n"
        f"# tesseract_config.yaml files under mosaic/tesseracts/{domain_name}/<solver>/\n"
        f"# are the source of truth; plot styling is applied to each spec.\n"
        f"_SOLVERS: dict[str, SolverSpec] = discover_solvers(_TESSERACT_SLUG)\n"
        f"apply_styles(_SOLVERS)\n"
        f"# TODO: merge any per-(solver, problem) overrides here, e.g.:\n"
        f'#   _SOLVERS["my_solver"].input_overrides = {{...}}\n\n\n'
        f"# ── Problem ──────────────────────────────────────────────────────────\n"
        f"problem = Problem(\n"
        f"    name={domain_name!r},\n"
        f"    category_label={category_label!r},\n"
        f"    description={description!r},\n"
        f"    bc_description={bc_description!r},\n"
        f"    tesseract_dir=_TESSERACT_SLUG,\n"
        f"    solvers=list(_SOLVERS.values()),\n"
        f"    make_inputs=make_inputs,\n"
        f"    error_fn=l2_error_rel,\n"
        f"    output_key={tpl.output_key!r},\n"
        f"    ic_key={tpl.ic_key!r},\n"
        f"    domain_extent={tpl.domain_extent},\n"
        f"    resolution_key={tpl.resolution_key!r},\n"
        f")\n\n"
        f"problem.add_ic(\n"
        f"    {ic_default_name!r},\n"
        f"    fn=_default_ic,\n"
        f'    description="TODO: describe this initial condition.",\n'
        f'    plot_params={{"N": {n_default}}},\n'
        f"    plot=plot_ic,\n"
        f")\n\n\n"
        f"# ── Experiments ──────────────────────────────────────────────────────\n"
        f"# Each problem.add_experiment(key, kernel, ...) call takes:\n"
        f"#   plot=callable | {{view: callable, ...}}\n"
        f"#   coords={{...}}                   typed sweep position\n"
        f"#   status_check=[...]              cell-status callables\n"
        f"# Aggregator plots that partition cells by coord are registered via\n"
        f"# problem.add_sweep_plot(name, fn, group_by=..., filter=...).\n"
        f"#\n"
        f"{experiment_todos_block}\n\n\n"
        f'__all__ = ["problem"]\n'
    )
    created["problem_config"] = config_path
    created["problem_pkg"] = pkg_dir

    return created
