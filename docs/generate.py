#!/usr/bin/env python3
"""Generate docs/solvers.qmd from tesseract source files and SolverSpec metadata.

Usage:
    python docs/generate.py           # writes docs/solvers.qmd
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required: pip install pyyaml")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TESSERACTS = ROOT / "mosaic" / "tesseracts"
OUTPUT = Path(__file__).resolve().parent / "solvers.qmd"

# Maps (domain_dir, solver_dir) to the solver key used in problem configs.
# Populated lazily by _load_solver_specs().
_SOLVER_SPECS: dict[tuple[str, str], dict] | None = None

# Per-physics-dir category label (e.g. "Navier\u2013Stokes (Grid)"), harvested from
# each Problem.category_label. Populated by _load_solver_specs().
_CATEGORY_LABELS: dict[str, str] = {}


# ──────────────────────────────────────────────────────────────────────────────
# SolverSpec loading (optional — gracefully degrades without full install)
# ──────────────────────────────────────────────────────────────────────────────


def _load_solver_specs() -> dict[tuple[str, str], dict]:
    """Try to load SolverSpec metadata from problem configs.

    Also harvests each ``Problem.category_label`` into ``_CATEGORY_LABELS``
    keyed by tesseract-dir basename, so the generated page's section headings
    are sourced from the problem configs rather than from this file.
    """
    global _SOLVER_SPECS, _CATEGORY_LABELS
    if _SOLVER_SPECS is not None:
        return _SOLVER_SPECS

    _SOLVER_SPECS = {}
    try:
        from mosaic.benchmarks.problems import PROBLEMS, get_config
    except ImportError:
        return _SOLVER_SPECS

    for prob in PROBLEMS:
        try:
            cfg = get_config(prob)
        except Exception:
            continue
        domain_dir = cfg.tesseract_dir.name
        # First non-empty category_label per tesseract dir wins. Multiple
        # Problems (e.g. ns-grid + ns-3d-grid) can share a dir.
        label = cfg.category_label
        if label and not _CATEGORY_LABELS.get(domain_dir):
            _CATEGORY_LABELS[domain_dir] = label

        # cfg.exclusions is keyed by ``<experiment>``→``<solver_name>``→
        # ``Exclusion``. Pivot to a per-solver view so the docs template
        # can render one section per solver.
        per_solver_excl: dict[str, dict[str, dict]] = {}
        per_solver_anom: dict[str, dict[str, dict]] = {}
        for exp_key, by_solver in cfg.exclusions.items():
            for sname, exc in by_solver.items():
                payload = {"category": exc.category.value, "reason": exc.reason}
                target = (
                    per_solver_anom
                    if exc.category.value == "anomaly_explained"
                    else per_solver_excl
                )
                target.setdefault(sname, {})[exp_key] = payload

        for spec in cfg.solvers:
            key = (domain_dir, spec.dir)
            exc_dict = per_solver_excl.get(spec.name, {})
            anom_dict = per_solver_anom.get(spec.name, {})
            _SOLVER_SPECS[key] = {
                "name": spec.name,
                "scheme": spec.scheme,
                "backend_lang": spec.backend,
                "ad_strategy": spec.ad_strategy,
                "description": spec.description,
                "doc_url": spec.doc_url,
                "family": spec.family,
                "uses_gpu": spec.uses_gpu,
                "differentiable": spec.differentiable,
                "internal_dtype": spec.internal_dtype,
                "exclusions": exc_dict,
                "explained_anomalies": anom_dict,
            }
    return _SOLVER_SPECS


def _get_spec(physics: str, backend: str) -> dict | None:
    """Get SolverSpec data for a given physics/backend pair."""
    specs = _load_solver_specs()
    return specs.get((physics, backend))


# ──────────────────────────────────────────────────────────────────────────────
# Schema parsing (unchanged)
# ──────────────────────────────────────────────────────────────────────────────


def _extract_description(call_node: ast.Call | None) -> str:
    if call_node is None:
        return ""
    for kw in call_node.keywords:
        if kw.arg == "description":
            raw = ast.unparse(kw.value)
            if raw.startswith(("'", '"')):
                try:
                    return ast.literal_eval(raw)
                except Exception:
                    pass
            return raw
    return ""


def _parse_type(annotation: str) -> tuple[str, bool]:
    """Return (display_type, is_differentiable)."""
    m = re.match(r"Differentiable\[(.+)\]$", annotation, re.DOTALL)
    if m:
        return m.group(1), True
    return annotation, False


def _parse_fields(source: str, class_name: str) -> list[dict]:
    """Extract annotated fields from a single class definition."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    fields = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if not (
                    isinstance(item, ast.AnnAssign)
                    and isinstance(item.target, ast.Name)
                ):
                    continue
                raw_type = ast.unparse(item.annotation)
                display_type, differentiable = _parse_type(raw_type)
                call = item.value if isinstance(item.value, ast.Call) else None
                fields.append(
                    {
                        "name": item.target.id,
                        "type": display_type,
                        "differentiable": differentiable,
                        "description": _extract_description(call),
                    }
                )
    return fields


# Cache: domain module name → parsed canonical schema source text
_CANONICAL_CACHE: dict[str, str] = {}


def _canonical_schema_source(physics_dir: str) -> str | None:
    """Return the source text of the canonical schemas.py for a physics domain.

    The schema module follows the convention
    ``mosaic/tesseracts/tesseract_shared/problems/<dir with hyphens swapped
    for underscores>/schemas.py``.
    """
    module = physics_dir.replace("-", "_") if physics_dir else None
    if module is None:
        return None
    if module not in _CANONICAL_CACHE:
        schema_path = (
            ROOT
            / "mosaic"
            / "tesseracts"
            / "tesseract_shared"
            / "problems"
            / module
            / "schemas.py"
        )
        if not schema_path.exists():
            _CANONICAL_CACHE[module] = ""
        else:
            _CANONICAL_CACHE[module] = schema_path.read_text()
    return _CANONICAL_CACHE[module] or None


def _extract_make_differentiable_fields(source: str, class_name: str) -> set[str]:
    """Return the field names a solver promotes to ``Differentiable`` via
    ``make_differentiable(_Canonical*Schema, [...])`` in its class bases.

    Looks for patterns like::

        class InputSchema(make_differentiable(_CanonicalInputSchema, ["rho"])):
            ...

    The second argument is collected literally so the wrapped fields are
    marked differentiable in the merged schema even though the solver's own
    source has no ``Differentiable[...]`` annotation on them.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    fields: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        for base in node.bases:
            if not (
                isinstance(base, ast.Call)
                and isinstance(base.func, ast.Name)
                and base.func.id == "make_differentiable"
                and len(base.args) >= 2
            ):
                continue
            field_list = base.args[1]
            if isinstance(field_list, (ast.List, ast.Tuple)):
                for elt in field_list.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        fields.add(elt.value)
    return fields


def parse_schema(source: str, class_name: str, physics_dir: str = "") -> list[dict]:
    """Parse schema fields, including inherited canonical fields.

    When ``InputSchema`` or ``OutputSchema`` in the solver source inherits from
    ``_CanonicalInputSchema`` / ``_CanonicalOutputSchema``, the canonical
    fields are resolved from ``tesseract_shared/problems/<domain>/schemas.py``
    and prepended so that the full interface is documented. Fields promoted
    via ``make_differentiable(...)`` in the solver's class bases are marked
    differentiable on the merged result.
    """
    solver_fields = _parse_fields(source, class_name)
    solver_names = {f["name"] for f in solver_fields}
    diff_promoted = _extract_make_differentiable_fields(source, class_name)

    # Resolve canonical parent fields.
    canonical_source = _canonical_schema_source(physics_dir) if physics_dir else None
    if canonical_source is None:
        merged = solver_fields
    else:
        # The canonical module defines InputSchema / OutputSchema directly (not
        # _CanonicalInputSchema), so parse with the bare class_name.
        canonical_fields = _parse_fields(canonical_source, class_name)
        merged = [f for f in canonical_fields if f["name"] not in solver_names]
        merged.extend(solver_fields)

    if diff_promoted:
        for f in merged:
            if f["name"] in diff_promoted:
                f["differentiable"] = True
    return merged


def load_solver(path: Path) -> dict | None:
    config_path = path / "tesseract_config.yaml"
    api_path = path / "tesseract_api.py"
    if not config_path.exists() or not api_path.exists():
        return None
    with open(config_path) as f:
        config = yaml.safe_load(f)
    source = api_path.read_text()
    physics, backend = path.parent.name, path.name
    spec = _get_spec(physics, backend)

    # The YAML metadata.mosaic block is the authoritative per-container
    # source; SolverSpec is used only as a fallback when a key is missing.
    yaml_mosaic = (config.get("metadata") or {}).get("mosaic") or {}

    return {
        "name": config.get("name", f"{physics}/{backend}"),
        "version": config.get("version", ""),
        "description": (config.get("description") or "").strip(),
        "physics": physics,
        "backend": backend,
        # Category label comes from each Problem.category_label (harvested
        # by _load_solver_specs); fall back to the physics dir name if missing.
        "category": _CATEGORY_LABELS.get(physics, physics),
        "inputs": parse_schema(source, "InputSchema", physics),
        "outputs": parse_schema(source, "OutputSchema", physics),
        "path": f"mosaic/tesseracts/{physics}/{backend}",
        # Enriched fields: YAML metadata wins, SolverSpec is fallback.
        "display_name": yaml_mosaic.get("name")
        or (spec["name"] if spec else config.get("name", backend)),
        "scheme": yaml_mosaic.get("scheme") or (spec["scheme"] if spec else ""),
        "backend_lang": yaml_mosaic.get("backend")
        or (spec["backend_lang"] if spec else ""),
        "ad_strategy": yaml_mosaic.get(
            "ad_strategy", spec["ad_strategy"] if spec else None
        ),
        "doc_url": yaml_mosaic.get("doc_url") or (spec["doc_url"] if spec else ""),
        "discretization": yaml_mosaic.get("discretization", ""),
        "numerics": yaml_mosaic.get("numerics", ""),
        "family": yaml_mosaic.get("family") or (spec["family"] if spec else ""),
        "uses_gpu": yaml_mosaic.get("uses_gpu", spec["uses_gpu"] if spec else None),
        "differentiable": yaml_mosaic.get(
            "differentiable", spec["differentiable"] if spec else None
        ),
        "internal_dtype": yaml_mosaic.get("internal_dtype")
        or (spec["internal_dtype"] if spec else ""),
        "exclusions": spec["exclusions"] if spec else {},
        "explained_anomalies": spec["explained_anomalies"] if spec else {},
    }


def collect_solvers() -> dict[str, list[dict]]:
    solvers: dict[str, list[dict]] = {}
    for physics_dir in sorted(TESSERACTS.iterdir()):
        if not physics_dir.is_dir():
            continue
        for backend_dir in sorted(physics_dir.iterdir()):
            if not backend_dir.is_dir():
                continue
            solver = load_solver(backend_dir)
            if solver:
                cat = solver["category"]
                solvers.setdefault(cat, []).append(solver)
    return solvers


# ──────────────────────────────────────────────────────────────────────────────
# Markdown generation
# ──────────────────────────────────────────────────────────────────────────────

AD_LABELS = {
    "autodiff": "AD: autodiff",
    "adjoint": "AD: adjoint",
    "hybrid": "AD: hybrid",
    None: "AD: forward-only",
}

# Expand discretization codes for the plain-text "Numerics:" line that appears
# on every solver card.
_DISCR_FULL = {
    "FD": "Finite Difference",
    "FV": "Finite Volume",
    "FE": "Finite Element",
    "LBM": "Lattice Boltzmann",
    "Spectral": "Spectral",
}

# Bootstrap badge classes used on every card. One colour per category so the
# three pill types are visually distinct without overloading hues; every label
# uses white text (which is the Bootstrap default on bg-primary / bg-dark /
# bg-success).
_LANG_BADGE_CLASS = "bg-dark"  # all language badges
_AD_BADGE_CLASS = "bg-primary"  # all AD-strategy badges
_GPU_BADGE_CLASS = "bg-success"  # only present on GPU-capable solvers


def _badge(text: str, classes: str) -> str:
    """Render a Bootstrap pill badge as a Pandoc bracketed span.

    Quarto/Pandoc turns ``[text]{.badge .bg-primary}`` into
    ``<span class="badge bg-primary">text</span>``, which the flatly theme
    styles as a coloured pill label.
    """
    cls = " ".join("." + c for c in classes.split())
    return f"[{text}]{{.badge {cls}}}"


def render_field_table(fields: list[dict], label: str) -> str:
    if not fields:
        return f"**{label}:** \u2014\n"
    checkmark = "\u2713"
    rows = [
        f"| `{f['name']}` | `{f['type']}` | {checkmark if f['differentiable'] else ''} | {f['description']} |"
        for f in fields
    ]
    header = (
        "| Field | Type | \u2202 | Description |\n|-------|------|---|-------------|\n"
    )
    return f"**{label}**\n\n{header}" + "\n".join(rows) + "\n"


def render_solver(solver: dict) -> str:
    name = solver["display_name"] or solver["name"]
    anchor = f"{solver['physics']}-{solver['backend']}"
    lines = [f"### {name} {{#{anchor}}}", ""]

    # Method row: plain-text "Numerics: <Finite Volume>, <PISO, BDF1>".
    numerics_parts: list[str] = []
    if solver["discretization"]:
        numerics_parts.append(
            _DISCR_FULL.get(solver["discretization"], solver["discretization"])
        )
    if solver["numerics"]:
        numerics_parts.append(solver["numerics"])
    if numerics_parts:
        lines.append("**Numerics:** " + ", ".join(numerics_parts))
        lines.append("")

    # Implementation row: language, AD strategy, GPU badge (only when present).
    impl_badges: list[str] = []
    if solver["backend_lang"]:
        impl_badges.append(_badge(solver["backend_lang"], _LANG_BADGE_CLASS))
    ad_label = AD_LABELS.get(solver["ad_strategy"], solver["ad_strategy"])
    if ad_label:
        impl_badges.append(_badge(ad_label, _AD_BADGE_CLASS))
    if solver["uses_gpu"] is True:
        impl_badges.append(_badge("GPU", _GPU_BADGE_CLASS))
    if impl_badges:
        lines.append(" ".join(impl_badges))
        lines.append("")

    if solver["doc_url"]:
        lines.append(f"**Upstream docs:** [{solver['doc_url']}]({solver['doc_url']})")
        lines.append("")

    lines.append(f"**Path:** `{solver['path']}`")
    lines.append("")

    if solver["name"]:
        lines.append(f"**Image:** `{solver['name']}`")
        lines.append("")

    # Two-sentence description from the YAML's top-level ``description:``.
    if solver["description"]:
        lines.append(solver["description"])
        lines.append("")

    # Exclusions
    if solver["exclusions"]:
        lines.append("::: {.callout-note}")
        lines.append("#### Excluded experiments")
        lines.append("")
        for key, val in solver["exclusions"].items():
            reason = val.get("reason", val) if isinstance(val, dict) else val
            lines.append(f"- **{key}**: {reason}")
        lines.append(":::")
        lines.append("")

    # Collapsible schema tables
    lines += [
        '::: {.callout-note collapse="true"}',
        "#### Inputs / Outputs",
        "",
        render_field_table(solver["inputs"], "Inputs"),
        render_field_table(solver["outputs"], "Outputs"),
        ":::",
        "",
    ]

    return "\n".join(lines)


def render_category(category: str, solvers: list[dict]) -> str:
    body = "\n".join(render_solver(s) for s in solvers)
    return f"## {category}\n\n{body}"


def generate_qmd(categories: dict[str, list[dict]]) -> str:
    frontmatter = "---\ntitle: Solver Reference\n---\n\n"
    header = (
        f"Each solver card shows its numerical scheme, AD strategy, and "
        f"Tesseract schema; per-(solver, problem) exclusions and explained "
        f"anomalies come from the problem configs and are listed alongside. "
        f"Click **Inputs / Outputs** on any solver to expand its field tables. "
        f"The \u2202 column marks fields that support automatic differentiation "
        f"(VJP/JVP).\n\n"
        f"**Legend.** Each card shows a **Numerics:** line — the discretization "
        f"followed by the numerical method — and three "
        f"badge categories: "
        f"{_badge('language', _LANG_BADGE_CLASS)} (backend / runtime), "
        f"{_badge('AD: strategy', _AD_BADGE_CLASS)} "
        f"(autodiff / adjoint / hybrid / forward-only), and "
        f"{_badge('GPU', _GPU_BADGE_CLASS)} on GPU-capable solvers.\n\n"
    )
    sections = "\n\n".join(
        render_category(cat, solvers) for cat, solvers in categories.items()
    )
    return frontmatter + header + sections + "\n"


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    categories = collect_solvers()
    if not categories:
        sys.exit(f"No solvers found under {TESSERACTS}")

    new_qmd = generate_qmd(categories)
    OUTPUT.write_text(new_qmd, encoding="utf-8")
    total = sum(len(v) for v in categories.values())
    print(f"Generated {OUTPUT} ({total} solvers, {len(categories)} categories)")


if __name__ == "__main__":
    main()
