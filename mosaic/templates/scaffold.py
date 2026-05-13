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


# Shared-runner mapping table used by ``scaffold_domain`` when translating
# per-suite YAML defaults into ``problem.add_experiment(...)`` calls. Keys
# are ``<suite>/<exp_name>`` slugs; values are ``(runner_attr, plot_attr)``
# pairs where the attrs are exported from the corresponding
# ``mosaic.benchmarks.problems.shared.<suite>`` and
# ``mosaic.benchmarks.problems.shared.plots.<suite>`` modules.
#
# Optimization experiments are intentionally absent: their runners are
# problem-specific (drag_opt, topopt, conductivity_recovery, …) and the
# scaffold emits a TODO comment block instead of trying to fabricate one.
_RUNNER_TABLE: dict[str, tuple[str, str]] = {
    "forward/agreement": ("agreement", "plot_agreement"),
    "forward/baseline": ("agreement", "plot_agreement"),
    "forward/physical_laws": ("physical_laws", "plot_physical_laws"),
    "gradient/fd_check": ("fd_check", "plot_fd_check"),
    "gradient/param_sweep": ("param_sweep", "plot_param_sweep"),
    "gradient/horizon_sweep": ("param_sweep", "plot_horizon_sweep"),
    "gradient/jacobian_svd": ("jacobian_svd", "plot_jacobian_svd"),
    "cost/spatial_cost": ("spatial_cost", "plot_cost"),
    "cost/temporal_cost": ("temporal_cost", "plot_cost"),
    "cost/vjp_cost": ("vjp_cost", "plot_cost"),
}

# Reverse index: which shared module exports each runner / plot symbol. Used
# by the codegen to emit a minimal import block.
_RUNNER_MODULES: dict[str, str] = {
    "agreement": "mosaic.benchmarks.problems.shared.forward",
    "physical_laws": "mosaic.benchmarks.problems.shared.forward",
    "fd_check": "mosaic.benchmarks.problems.shared.gradient",
    "param_sweep": "mosaic.benchmarks.problems.shared.gradient",
    "jacobian_svd": "mosaic.benchmarks.problems.shared.gradient",
    "spatial_cost": "mosaic.benchmarks.problems.shared.cost",
    "temporal_cost": "mosaic.benchmarks.problems.shared.cost",
    "vjp_cost": "mosaic.benchmarks.problems.shared.cost",
}
_PLOT_MODULES: dict[str, str] = {
    "plot_agreement": "mosaic.benchmarks.problems.shared.plots.forward",
    "plot_physical_laws": "mosaic.benchmarks.problems.shared.plots.forward",
    "plot_fd_check": "mosaic.benchmarks.problems.shared.plots.gradient",
    "plot_param_sweep": "mosaic.benchmarks.problems.shared.plots.gradient",
    "plot_horizon_sweep": "mosaic.benchmarks.problems.shared.plots.gradient",
    "plot_jacobian_svd": "mosaic.benchmarks.problems.shared.plots.gradient",
    "plot_cost": "mosaic.benchmarks.problems.shared.plots.cost",
}


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


def _py_repr(value: Any, indent: int = 0) -> str:
    """Render a YAML-loaded value as a multi-line Python literal.

    Dicts are formatted one key per line at ``indent + 4`` spaces; nested
    dicts recurse with deeper indent. Lists of primitives stay on one line.
    Strings/numbers/bools use plain ``repr``.
    """
    if isinstance(value, dict):
        if not value:
            return "{}"
        inner_indent = " " * (indent + 4)
        close_indent = " " * indent
        parts = [
            f"{inner_indent}{k!r}: {_py_repr(v, indent + 4)}," for k, v in value.items()
        ]
        return "{\n" + "\n".join(parts) + f"\n{close_indent}}}"
    if isinstance(value, list):
        return repr(value)
    return repr(value)


def _emit_add_experiment(
    key: str, run: dict, runner_attr: str, plot_attr: str, indent: int = 4
) -> str:
    """Emit a single ``problem.add_experiment(...)`` call.

    ``run`` carries the experiment kwargs (``ic``, ``physics``, ``fd``,
    ``optim``, ``cost``, etc.). Inline-list values inside ``physics``
    auto-detect as sweep axes at the framework level — no ``sweep:``
    shape needed.
    """
    pad = " " * indent
    inner_pad = " " * (indent + 4)
    lines = [f"{pad}problem.add_experiment("]
    lines.append(f"{inner_pad}{key!r},")
    lines.append(f"{inner_pad}{runner_attr},")
    for k, v in run.items():
        lines.append(f"{inner_pad}{k}={_py_repr(v, indent + 4)},")
    lines.append(f"{inner_pad}plot={plot_attr},")
    lines.append(f"{pad})")
    return "\n".join(lines)


def _emit_optimization_todo(exp_name: str, runs: list[dict], indent: int = 4) -> str:
    """Emit a TODO comment block for an optimisation experiment.

    Optimisation runners are problem-specific (e.g. ``drag_opt`` for
    ns-grid, ``topopt`` for structural-mesh, ``conductivity_recovery``
    for thermal-mesh) so the scaffold cannot pick a runner; it just dumps
    the template's suggested config alongside a pointer to how the
    in-tree problems wire theirs up.
    """
    pad = " " * indent
    lines = [f"{pad}# TODO: optimisation experiment {exp_name!r} from template."]
    lines.append(f"{pad}# Define a runner in .optimization (see e.g.")
    lines.append(
        f"{pad}# mosaic.benchmarks.problems.structural_mesh.optimization.topopt)"
    )
    lines.append(f"{pad}# and register it here. Suggested config:")
    for run in runs:
        for k, v in run.items():
            # Render as a single flat repr so the comment stays on one line;
            # multi-line dict literals would need # prefixing on every line.
            lines.append(f"{pad}#   {k} = {v!r}")
    return "\n".join(lines)


def _render_experiments_module(tpl: DomainTemplate, domain_name: str) -> str:
    """Translate ``tpl.{forward,gradient,cost,optimization}`` into a register()
    function body. Returns the full ``experiments.py`` source.
    """
    runners_used: set[str] = set()
    plots_used: set[str] = set()
    blocks: list[str] = []

    for suite_name, suite_data in [
        ("forward", tpl.forward),
        ("gradient", tpl.gradient),
        ("cost", tpl.cost),
        ("optimization", tpl.optimization),
    ]:
        if not suite_data:
            continue
        section_lines = [f"    # {suite_name.capitalize()}"]
        for exp_name, runs in suite_data.items():
            full_key = f"{suite_name}/{exp_name}"
            if suite_name == "optimization":
                section_lines.append(_emit_optimization_todo(exp_name, runs))
                continue
            mapping = _RUNNER_TABLE.get(full_key)
            if mapping is None:
                section_lines.append(
                    f"    # TODO: experiment {full_key!r} has no shared runner in"
                    f" _RUNNER_TABLE; register manually."
                )
                continue
            runner_attr, plot_attr = mapping
            runners_used.add(runner_attr)
            plots_used.add(plot_attr)
            if len(runs) == 1:
                section_lines.append(
                    _emit_add_experiment(full_key, runs[0], runner_attr, plot_attr)
                )
            else:
                # Multi-variant: pass the run list straight through; the
                # runner fans out at <full_key>/<variant_name>.
                multi_run = {"runs": runs}
                section_lines.append(
                    _emit_add_experiment(full_key, multi_run, runner_attr, plot_attr)
                )
        blocks.append("\n\n".join(section_lines))

    # Build import block grouped by module so the generated file is tidy.
    import_lines: list[str] = []
    runner_by_module: dict[str, list[str]] = {}
    plot_by_module: dict[str, list[str]] = {}
    for attr in sorted(runners_used):
        runner_by_module.setdefault(_RUNNER_MODULES[attr], []).append(attr)
    for attr in sorted(plots_used):
        plot_by_module.setdefault(_PLOT_MODULES[attr], []).append(attr)
    for mod in sorted(runner_by_module):
        import_lines.append(f"from {mod} import {', '.join(runner_by_module[mod])}")
    for mod in sorted(plot_by_module):
        import_lines.append(f"from {mod} import {', '.join(plot_by_module[mod])}")
    imports = "\n".join(import_lines)

    body = (
        "\n\n".join(blocks)
        if blocks
        else (
            "    # TODO: register experiments via problem.add_experiment(...).\n"
            "    pass"
        )
    )

    return (
        f'"""Experiment + plot registrations for {domain_name}.\n'
        f"\n"
        f"Exposes :func:`register(problem)` which the package's ``config.py``\n"
        f"calls after building the canonical :class:`Problem` instance, so the\n"
        f"experiment / plot registries live on a single ``Problem``.\n"
        f"\n"
        f"Generated from template: {tpl.name}\n"
        f'"""\n'
        f"\n"
        f"from __future__ import annotations\n"
        f"\n"
        f"from typing import TYPE_CHECKING\n"
        f"\n"
        f"{imports}\n"
        f"\n"
        f"if TYPE_CHECKING:\n"
        f"    from mosaic.benchmarks.core.config import Problem\n"
        f"\n"
        f"\n"
        f"def register(problem: Problem) -> None:\n"
        f'    """Populate ``problem.experiments`` / ``problem.plot_fns``."""\n'
        f"{body}\n"
    )


# ── Scaffold entry point ─────────────────────────────────────────────────────


def scaffold_domain(
    domain_name: str,
    tpl: DomainTemplate,
    *,
    target_dir: Path = Path("mosaic"),
) -> dict[str, Path]:
    """Generate files for a new benchmark domain from a template.

    Creates:
    - ``tesseracts/tesseract_shared/problems/<domain>/schemas.py`` (stub)
    - ``tesseracts/tesseract_shared/problems/<domain>/__init__.py``
    - ``tesseracts/<domain>/`` (empty, ready for solver dirs)
    - ``benchmarks/problems/<domain>/`` (Problem package with the canonical
      five-file split: ``__init__.py``, ``config.py``, ``ics.py``,
      ``physics.py``, ``experiments.py``)

    Returns a dict of generated file paths keyed by role.
    """
    slug = domain_name.replace("-", "_")
    created: dict[str, Path] = {}

    # 1. Schema stubs — live under ``tesseracts/tesseract_shared/problems/<slug>``
    #    so the Python import path ``tesseract_shared.problems.<slug>`` resolves
    #    against the package shipped from ``mosaic/tesseracts/tesseract_shared``.
    schema_dir = target_dir / "tesseracts" / "tesseract_shared" / "problems" / slug
    schema_dir.mkdir(parents=True, exist_ok=True)

    init_path = schema_dir / "__init__.py"
    init_path.write_text(
        '__all__ = ["InputSchema", "OutputSchema"]\n\n'
        "from .schemas import InputSchema, OutputSchema\n"
    )
    created["schema_init"] = init_path

    schema_path = schema_dir / "schemas.py"
    schema_path.write_text(
        f'"""Canonical InputSchema / OutputSchema for {domain_name} tesseracts.\n'
        f"\n"
        f"Generated from template: {tpl.name}\n"
        f'"""\n\n'
        f"from pydantic import BaseModel\n\n\n"
        f"class InputSchema(BaseModel):\n"
        f'    """Inputs for {domain_name} solvers."""\n\n'
        f"    # TODO: define input fields\n"
        f"    pass\n\n\n"
        f"class OutputSchema(BaseModel):\n"
        f'    """Outputs for {domain_name} solvers."""\n\n'
        f"    # TODO: define output fields\n"
        f"    pass\n"
    )
    created["schemas"] = schema_path

    # 2. Tesseract directory
    tess_dir = target_dir / "tesseracts" / domain_name
    tess_dir.mkdir(parents=True, exist_ok=True)
    created["tesseracts_dir"] = tess_dir

    # 3. Problem config package — five-file layout under benchmarks/problems/<slug>/.
    problems_dir = target_dir / "benchmarks" / "problems"
    pkg_dir = problems_dir / slug
    pkg_dir.mkdir(parents=True, exist_ok=True)

    n_default = tpl.physics_defaults.get("N", 8)
    ic_default_name = tpl.ic_defaults.get("name", "default")

    (pkg_dir / "ics.py").write_text(
        f'"""Initial-condition generators.\n\n'
        f"Each generator has signature ``(L: float, seed: int, **physics) -> array``.\n"
        f"Register them on the problem via ``problem.add_ic(name, fn, ...)`` in\n"
        f"``config.py`` — there is no module-level IC dict.\n"
        f'"""\n\n'
        f"from __future__ import annotations\n\n"
        f"import numpy as np\n\n\n"
        f"def _default_ic(L: float = 1.0, seed: int = 0, **physics) -> np.ndarray:\n"
        f'    """Generate an initial condition. TODO: implement."""\n'
        f"    N = physics.get('N', {n_default})\n"
        f"    return np.zeros(N, dtype=np.float32)\n"
    )

    (pkg_dir / "physics.py").write_text(
        f'"""Input factory + diagnostics. ``make_inputs`` receives a ``SolverSpec``\n'
        f"directly; the per-solver ``input_overrides`` are merged into the returned\n"
        f'dict so the user does not need to import ``config`` here."""\n\n'
        f"from __future__ import annotations\n\n"
        f"import numpy as np\n\n"
        f"from mosaic.benchmarks.core.config import SolverSpec\n\n\n"
        f"def make_inputs(spec: SolverSpec, ic: np.ndarray, **physics) -> dict:\n"
        f'    """Build solver inputs from IC and physics parameters. TODO: implement."""\n'
        f'    base = {{"{tpl.ic_key}": ic}}\n'
        f"    return {{**base, **spec.input_overrides}}\n\n\n"
        f"DIAGNOSTICS: dict = {{}}\n"
    )

    (pkg_dir / "experiments.py").write_text(
        _render_experiments_module(tpl, domain_name)
    )

    init_path = pkg_dir / "__init__.py"
    init_path.write_text(
        f'"""Problem package for {domain_name}, generated from template: {tpl.name}. See :mod:`.config`."""\n'
    )

    category_label = tpl.category_label or domain_name
    description = tpl.description.strip()
    bc_description = tpl.bc_description.strip()

    config_path = pkg_dir / "config.py"
    config_path.write_text(
        f'"""Solver discovery, canonical :class:`Problem`, and per-solver overrides for {domain_name}."""\n\n'
        f"from __future__ import annotations\n\n"
        f"from mosaic.benchmarks.core.config import Problem, SolverSpec, discover_solvers\n"
        f"from mosaic.benchmarks.core.utils import l2_error_rel\n"
        f"from mosaic.benchmarks.problems.shared.plots.ics import plot_ic\n"
        f"from mosaic.benchmarks.problems.shared.plots.solver_styles import apply_styles\n\n"
        f"from .experiments import register as _register_experiments\n"
        f"from .ics import _default_ic\n"
        f"from .physics import DIAGNOSTICS, make_inputs  # noqa: F401\n\n"
        f'_TESSERACT_SLUG = "{domain_name}"\n\n\n'
        f"# Auto-discover solvers from tesseract_config.yaml metadata.mosaic blocks.\n"
        f"_SOLVERS: dict[str, SolverSpec] = discover_solvers(_TESSERACT_SLUG)\n"
        f"apply_styles(_SOLVERS)\n\n"
        f"# Merge domain-specific overrides here, e.g.:\n"
        f'# _SOLVERS["my_solver"].input_overrides = {{...}}\n\n'
        f"problem = Problem(\n"
        f'    name="{domain_name}",\n'
        f'    category_label="{category_label}",\n'
        f'    description="{description}",\n'
        f'    bc_description="{bc_description}",\n'
        f"    tesseract_dir=_TESSERACT_SLUG,\n"
        f"    solvers=list(_SOLVERS.values()),\n"
        f"    make_inputs=make_inputs,\n"
        f"    error_fn=l2_error_rel,\n"
        f'    output_key="{tpl.output_key}",\n'
        f'    ic_key="{tpl.ic_key}",\n'
        f"    domain_extent={tpl.domain_extent},\n"
        f'    resolution_key="{tpl.resolution_key}",\n'
        f"    status_checks={{}},\n"
        f")\n\n"
        f"# ── Initial conditions ─────────────────────────────────────────────\n"
        f"problem.add_ic(\n"
        f'    "{ic_default_name}",\n'
        f"    fn=_default_ic,\n"
        f'    description="TODO: describe this initial condition.",\n'
        f'    plot_params={{"N": {n_default}}},\n'
        f"    plot=plot_ic,\n"
        f")\n\n"
        f"_register_experiments(problem)\n\n"
        f'__all__ = ["problem"]\n'
    )
    created["problem_init"] = init_path
    created["problem_config"] = config_path
    created["problem_pkg"] = pkg_dir

    return created
