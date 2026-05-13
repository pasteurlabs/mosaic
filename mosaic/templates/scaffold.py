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
    ]:
        for exp_name, exp_runs in suite_defaults.items():
            if not isinstance(exp_runs, list):
                errors.append(
                    f"{suite_name}.{exp_name}: expected a list of run dicts, "
                    f"got {type(exp_runs).__name__}"
                )

    return errors


def scaffold_domain(
    domain_name: str,
    tpl: DomainTemplate,
    *,
    target_dir: Path = Path("mosaic"),
) -> dict[str, Path]:
    """Generate files for a new benchmark domain from a template.

    Creates:
    - ``tesseract_shared/problems/<domain>/schemas.py`` (stub)
    - ``tesseract_shared/problems/<domain>/__init__.py``
    - ``tesseracts/<domain>/`` (empty, ready for solver dirs)
    - ``benchmarks/problems/<domain>.py`` (Problem stub)

    Returns a dict of generated file paths keyed by role.
    """
    slug = domain_name.replace("-", "_")
    created: dict[str, Path] = {}

    # 1. Schema stubs
    schema_dir = target_dir / "tesseract_shared" / "problems" / slug
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
        f"from pydantic import BaseModel, Field\n"
        f"from tesseract_core.runtime import Array, Differentiable, Float32\n\n\n"
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

    # 3. Problem config package — four-file layout under benchmarks/problems/<slug>/.
    #    __init__.py holds the canonical CONFIG; experiments/ics/physics are
    #    the other three.
    problems_dir = target_dir / "benchmarks" / "problems"
    pkg_dir = problems_dir / slug
    pkg_dir.mkdir(parents=True, exist_ok=True)

    n_default = tpl.physics_defaults.get("N", 8)

    (pkg_dir / "ics.py").write_text(
        f'"""Initial-condition generators and the ``MAKE_IC`` registry."""\n\n'
        f"from __future__ import annotations\n\n"
        f"import numpy as np\n\n"
        f"from mosaic.benchmarks.core.config import IcSpec\n\n\n"
        f"def _default_ic(seed: int = 0, **physics) -> np.ndarray:\n"
        f'    """Generate an initial condition. TODO: implement."""\n'
        f"    N = physics.get('N', {n_default})\n"
        f"    return np.zeros(N, dtype=np.float32)\n\n\n"
        f"MAKE_IC: dict[str, IcSpec] = {{\n"
        f'    "default": IcSpec(fn=_default_ic, description="TODO", plot_params={{}}),\n'
        f"}}\n"
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
        f'"""Experiment + plot registrations for {domain_name}.\n\n'
        f"Exposes :func:`register(problem)` which the package ``__init__`` calls\n"
        f"after building the canonical :class:`Problem` instance, so closure\n"
        f"deps and the experiment/plot registries live on a single ``Problem``.\n"
        f'"""\n\n'
        f"from __future__ import annotations\n\n"
        f"from typing import TYPE_CHECKING\n\n"
        f"if TYPE_CHECKING:\n"
        f"    from mosaic.benchmarks.core.config import Problem\n\n\n"
        f"def register(problem: Problem) -> None:\n"
        f'    """Populate ``problem.experiments`` / ``problem.plot_fns``."""\n'
        f"    # TODO: register experiments with problem.add_experiment(...).\n"
        f"    #   from mosaic.benchmarks.problems.shared.forward import run_agreement\n"
        f'    #   problem.add_experiment("forward/baseline", run_agreement, runs=[{{...}}])\n\n'
        f"    # TODO: register IC visualisations:\n"
        f"    #   for ic_name, ic_spec in problem.make_ic.items():\n"
        f"    #       problem.add_ic(ic_name, ic_spec.plot_params)\n"
    )

    init_path = pkg_dir / "__init__.py"
    init_path.write_text(
        f'"""Problem package for {domain_name}, generated from template: {tpl.name}. See :mod:`.config`."""\n'
    )

    config_path = pkg_dir / "config.py"
    config_path.write_text(
        f'"""Solver discovery, canonical :class:`Problem`, and per-solver overrides for {domain_name}."""\n\n'
        f"from __future__ import annotations\n\n"
        f"from mosaic.benchmarks.core.config import Problem, SolverSpec, discover_solvers\n"
        f"from mosaic.benchmarks.core.utils import l2_error_rel\n"
        f"from mosaic.benchmarks.problems.shared.plots.solver_styles import apply_styles\n\n"
        f"from .experiments import register as _register_experiments\n"
        f"from .ics import MAKE_IC\n"
        f"from .physics import DIAGNOSTICS, make_inputs\n\n"
        f'_TESSERACT_SLUG = "{domain_name}"\n\n\n'
        f"# Auto-discover solvers from tesseract_config.yaml metadata.mosaic blocks.\n"
        f"_SOLVERS: dict[str, SolverSpec] = discover_solvers(_TESSERACT_SLUG)\n"
        f"apply_styles(_SOLVERS)\n\n"
        f"# Merge domain-specific overrides here, e.g.:\n"
        f'# _SOLVERS["my_solver"].input_overrides = {{...}}\n\n'
        f"problem = Problem(\n"
        f'    name="{domain_name}",\n'
        f"    tesseract_dir=_TESSERACT_SLUG,\n"
        f"    solvers=list(_SOLVERS.values()),\n"
        f"    make_ic=MAKE_IC,\n"
        f"    make_inputs=make_inputs,\n"
        f"    error_fn=l2_error_rel,\n"
        f'    output_key="{tpl.output_key}",\n'
        f'    ic_key="{tpl.ic_key}",\n'
        f"    domain_extent={tpl.domain_extent},\n"
        f'    resolution_key="{tpl.resolution_key}",\n'
        f'    description="{tpl.description.strip()}",\n'
        f")\n\n"
        f"_register_experiments(problem)\n"
    )
    created["problem_init"] = init_path
    created["problem_config"] = config_path
    created["problem_pkg"] = pkg_dir

    return created
