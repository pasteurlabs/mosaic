#!/usr/bin/env python3

# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the solver reference from tesseract source files and SolverSpec metadata.

Usage:
    python docs/scripts/generate.py   # writes docs/_build/_generated/solvers.qmd
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

DOCS_DIR = Path(__file__).resolve().parents[1]
ROOT = DOCS_DIR.parent
sys.path.insert(0, str(ROOT))


def _load_views():
    """Load docs/scripts/intervention_views.py whether this runs as a script or a module."""
    import importlib.util

    name = "_mosaic_intervention_views"
    if name not in sys.modules:
        path = Path(__file__).resolve().parent / "intervention_views.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


views = _load_views()
TESSERACTS = ROOT / "mosaic" / "tesseracts"
OUTPUT = DOCS_DIR / "_build" / "_generated" / "solvers.qmd"

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
    global _SOLVER_SPECS
    if _SOLVER_SPECS is not None:
        return _SOLVER_SPECS

    _SOLVER_SPECS = {}
    try:
        from mosaic.benchmarks.core.utils import exclusion_lookup
        from mosaic.benchmarks.problems import PROBLEMS, get_config
    except ImportError:
        return _SOLVER_SPECS

    for prob in PROBLEMS:
        try:
            cfg = get_config(prob)
        except Exception:  # noqa: S112 — skip problems that don't import in the docs env
            continue
        domain_dir = cfg.tesseract_dir.name
        # First non-empty category_label per tesseract dir wins. Multiple
        # Problems (e.g. ns-grid + ns-3d-grid) can share a dir.
        label = cfg.category_label
        if label and not _CATEGORY_LABELS.get(domain_dir):
            _CATEGORY_LABELS[domain_dir] = label

        for spec in cfg.solvers:
            key = (domain_dir, spec.dir)
            entry = _SOLVER_SPECS.setdefault(
                key,
                {
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
                    "exclusions": {},
                    "explained_anomalies": {},
                    "setups_run": set(),
                },
            )
            excl = {
                k: e
                for k, e in cfg.exclusions.get(spec.key, {}).items()
                if e.category.value != "anomaly_explained"
            }
            # A solver runs a setup if any of the setup's experiments is not
            # excluded for it. ``ics`` only generates initial conditions.
            for exp_key in cfg.experiments:
                suite, _, exp_name = exp_key.partition("/")
                setup = views.iv.setup_of(prob, exp_key)
                if (
                    setup
                    and suite != "ics"
                    and exclusion_lookup(excl, suite, exp_name) is None
                ):
                    entry["setups_run"].add(setup.key)
            # cfg.exclusions is keyed by ``spec.key``→``<experiment key>``→
            # ``Exclusion``. Several Problems can share a tesseract dir (and
            # thus a solver section), so entries are grouped by problem name.
            for exp_key, exc in cfg.exclusions.get(spec.key, {}).items():
                target = (
                    entry["explained_anomalies"]
                    if exc.category.value == "anomaly_explained"
                    else entry["exclusions"]
                )
                target.setdefault(prob, {})[exp_key] = {
                    "category": exc.category.value,
                    "reason": exc.reason,
                    # The setups that own experiments under this key.
                    "setups": list(
                        dict.fromkeys(
                            setup.key
                            for e in cfg.experiments
                            if (e == exp_key or e.startswith(exp_key + "/"))
                            and (setup := views.iv.setup_of(prob, e))
                        )
                    ),
                }
    return _SOLVER_SPECS


def _get_spec(physics: str, backend: str) -> dict | None:
    """Get SolverSpec data for a given physics/backend pair."""
    specs = _load_solver_specs()
    return specs.get((physics, backend))


# ──────────────────────────────────────────────────────────────────────────────
# Schema parsing
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
                except Exception:  # noqa: S110 — fall back to the raw source text
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
    ``mosaic/mosaic_shared/problems/<dir with hyphens swapped
    for underscores>/schemas.py``.
    """
    module = physics_dir.replace("-", "_") if physics_dir else None
    if module is None:
        return None
    if module not in _CANONICAL_CACHE:
        schema_path = (
            ROOT / "mosaic" / "mosaic_shared" / "problems" / module / "schemas.py"
        )
        if not schema_path.exists():
            _CANONICAL_CACHE[module] = ""
        else:
            _CANONICAL_CACHE[module] = schema_path.read_text(encoding="utf-8")
    return _CANONICAL_CACHE[module] or None


def _extract_make_differentiable_fields(source: str, class_name: str) -> set[str]:
    """Return the field names a solver promotes to ``Differentiable`` via
    ``make_differentiable(_Canonical*Schema, [...])`` in its class bases.

    Looks for patterns like::

        class InputSchema(make_differentiable(_CanonicalInputSchema, ["rho"])): ...

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
            if isinstance(field_list, ast.List | ast.Tuple):
                for elt in field_list.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        fields.add(elt.value)
    return fields


def parse_schema(source: str, class_name: str, physics_dir: str = "") -> list[dict]:
    """Parse schema fields, including inherited canonical fields.

    When ``InputSchema`` or ``OutputSchema`` in the solver source inherits from
    ``_CanonicalInputSchema`` / ``_CanonicalOutputSchema``, the canonical
    fields are resolved from ``mosaic_shared/problems/<domain>/schemas.py``
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
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    source = api_path.read_text(encoding="utf-8")
    physics, backend = path.parent.name, path.name
    spec = _get_spec(physics, backend)

    # The YAML metadata.mosaic block is the authoritative per-container
    # source; SolverSpec is used only as a fallback when a key is missing.
    yaml_mosaic = (config.get("metadata") or {}).get("mosaic") or {}
    setups = views.iv.DOMAIN_SETUPS.get(physics, ())
    record, _ = views.iv.parse_record(yaml_mosaic, setups or None)

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
        "record": record,
        "setups": setups,
        # Without the problem configs, assume the solver runs every setup.
        "setups_run": tuple(s for s in setups if not spec or s in spec["setups_run"]),
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

_REPO_URL = "https://github.com/pasteurlabs/mosaic"

_DISCR_FULL = {
    "FD": "Finite differences",
    "FV": "Finite volumes",
    "FE": "Finite elements",
    "LBM": "Lattice Boltzmann",
}

# Per-solver checks written by `generate_results.py`, keyed by
# section anchor. Optional: sections render without them if absent.
SCORES = DOCS_DIR / "_build" / "_generated" / "scores.json"
_CHECK_SUITES = ("forward", "gradient", "optimization")
# Same words as the check states on the results pages.
_CHECK_LABELS = {
    "pass": "Pass",
    "flagged": "Flagged",
    "fail": "Fail",
    "blocked": "Blocked",
    "na": "Not supported",
    "notrun": "Not run",
}
_EXCLUSION_LABELS = {
    "categorical": "not supported",
    "infeasible": "infeasible",
    "not_implemented": "not implemented",
    "unstable": "unstable",
    "upstream_bug": "upstream bug",
    "wip": "work in progress",
    "unspecified": "unspecified",
}


def _load_scores() -> dict:
    try:
        import json

        return json.loads(SCORES.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _anchor(solver: dict) -> str:
    return f"{solver['physics']}-{solver['backend']}"


def _setup_link(setup: str) -> str:
    """The setup's label, linked to its results page."""
    label = views.iv.SETUPS[setup].label
    return f"[{label}](results/{setup}.qmd)"


def _join(items: list[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    return " and ".join([", ".join(items[:-1]), items[-1]] if len(items) > 1 else items)


def _discretization(solver: dict) -> str:
    code = solver["discretization"]
    return f"{_DISCR_FULL[code]} ({code})" if code in _DISCR_FULL else code


def _ad_strategy(solver: dict) -> str:
    return solver["ad_strategy"] or "none (forward only)"


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


def render_index(categories: dict[str, list[dict]]) -> list[str]:
    """One row per solver, linking to its section."""
    lines = [
        "| Domain | Solver | Discretization | Backend | AD strategy | GPU |",
        "|---|---|---|---|---|:-:|",
    ]
    for category, solvers in categories.items():
        domain = f"[{category}](#{solvers[0]['physics']})"
        for s in solvers:
            gpu = "\u2713" if s["uses_gpu"] is True else ""
            lines.append(
                f"| {domain} | [{s['display_name'] or s['name']}](#{_anchor(s)}) "
                f"| {_discretization(s)} | {s['backend_lang']} "
                f"| {_ad_strategy(s)} | {gpu} |"
            )
            domain = ""
    lines.append("")
    return lines


def render_facts(solver: dict) -> list[str]:
    facts = [
        ("Discretization", _discretization(solver)),
        ("Method", solver["numerics"]),
        ("Backend", solver["backend_lang"]),
        ("AD strategy", _ad_strategy(solver)),
        ("GPU", "yes" if solver["uses_gpu"] is True else "no"),
        ("Upstream documentation", views.linkify(solver["doc_url"])),
        ("Source", f"[`{solver['path']}`]({_REPO_URL}/tree/main/{solver['path']})"),
    ]
    return [f"- **{k}:** {v}" for k, v in facts if v] + [""]


def render_checks(solver: dict, score: list[dict] | None) -> list[str]:
    """The setups the solver runs, with its check states when available."""
    setups = list(solver["setups_run"])
    lines = [f"#### Setups and checks {{#{_anchor(solver)}-checks}}", ""]
    if not setups:
        lines += ["This solver runs none of the setups.", ""]
    else:
        record = solver["record"]
        suites = _CHECK_SUITES if score else ()
        by_setup = {v["key"]: v["states"] for v in score or ()}
        header = ["Setup", "Effort", *(s.capitalize() for s in suites)]
        lines += [
            "| " + " | ".join(header) + " |",
            "|---|" + ":-:|" * (len(header) - 1),
        ]
        for setup in setups:
            level = (record.effort(setup) or views.BASELINE_LEVEL) if record else None
            effort = f"[{level}]{{.eff .eff-{level}}}" if level else views.NOT_RECORDED
            # A setup missing from the results was not run at all.
            states = by_setup.get(setup, dict.fromkeys(suites, "notrun"))
            cells = [effort] + [
                f"[{_CHECK_LABELS[states[s]]}]{{.ck .ck-{states[s]}}}"
                if s in states
                else "\u2014"
                for s in suites
            ]
            lines.append(f"| {_setup_link(setup)} | " + " | ".join(cells) + " |")
        lines.append("")

    # Exclusions with the same category and reason (one reason for several
    # experiments, or for both Navier\u2013Stokes problems) are listed once.
    merged: dict[tuple[str, str], tuple[list[str], list[str]]] = {}
    for by_key in solver["exclusions"].values():
        for key, val in by_key.items():
            keys, hit = merged.setdefault((val["category"], val["reason"]), ([], []))
            if key not in keys:
                keys.append(key)
            hit += [s for s in val["setups"] if s not in hit]
    if merged:
        lines += ["**Excluded experiments**", ""]
        order = list(solver["setups"])
        for (category, reason), (keys, hit) in sorted(
            merged.items(),
            key=lambda m: min((order.index(s) for s in m[1][1]), default=len(order)),
        ):
            where = _join([views.iv.SETUPS[s].label for s in hit])
            label = _EXCLUSION_LABELS.get(category, category)
            lines.append(
                f"- {_join([f'`{k}`' for k in keys])}"
                + (f" on {where}" if len(order) > 1 and where else "")
                + f" ({label}): {views.linkify(reason).rstrip('.')}."
            )
        lines.append("")
    return lines


def render_usage(solver: dict) -> list[str]:
    return [
        f"#### Use it in your code {{#{_anchor(solver)}-usage}}",
        "",
        (
            f"`tesseract build {solver['path']}` builds the image "
            f"`{solver['name']}`. To load it and call it from JAX, follow "
            "[Use a solver in your code](standalone.qmd)."
        ),
        "",
        '::: {.callout-note collapse="true" appearance="simple"}',
        "#### Inputs and outputs",
        "",
        "Fields marked \u2202 are differentiable.",
        "",
        render_field_table(solver["inputs"], "Inputs"),
        render_field_table(solver["outputs"], "Outputs"),
        ":::",
        "",
    ]


def render_solver(solver: dict, scores: dict) -> str:
    name = solver["display_name"] or solver["name"]
    lines = [f"### {name} {{#{_anchor(solver)}}}", ""]
    if solver["description"]:
        lines += [solver["description"], ""]
    lines += render_facts(solver)
    lines += views.developer_block(solver["record"])
    lines += render_checks(solver, scores.get(_anchor(solver)))
    lines += render_usage(solver)
    return "\n".join(lines)


def render_domain(category: str, solvers: list[dict], scores: dict) -> str:
    domain = solvers[0]["physics"]
    schema = f"mosaic/mosaic_shared/problems/{domain.replace('-', '_')}/schemas.py"
    setups = _join([_setup_link(s) for s in solvers[0]["setups"]])
    intro = (
        f"The solvers in this domain share the schema in "
        f"[`{schema}`]({_REPO_URL}/blob/main/{schema})"
        + (f" and are benchmarked on {setups}." if setups else ".")
    )
    body = "\n".join(render_solver(s, scores) for s in solvers)
    return f"## {category} {{#{domain}}}\n\n{intro}\n\n{body}"


def generate_qmd(categories: dict[str, list[dict]]) -> str:
    scores = _load_scores()
    intro = [
        "---",
        "title: Solvers",
        "---",
        "",
        (
            "One section per solver: its method and backend, what it took to "
            "run it in Mosaic, its checks on each setup, and how to call it from "
            "your own code. Solvers that share an input and output schema form "
            "a domain."
        ),
        "",
        "## Effort levels {#effort-levels}",
        "",
        (
            "Every solver has an intervention record that lists everything done "
            "to run it beyond using the library as intended, with the reason and "
            "evidence for each item. Each intervention gets a level by its "
            "mechanism, and a solver's effort on a setup is the highest level "
            "among the interventions it requires there."
        ),
        "",
        "| Level | Mechanism | Example |",
        "|---|---|---|",
        (
            "| [L0]{.eff .eff-L0} | Nothing beyond expected tweaks, meaning the public API with "
            "its defaults or documented options | Setting a documented solver "
            "tolerance |"
        ),
        (
            "| [L1]{.eff .eff-L1} | Undocumented configuration, found by reading the source "
            "| An option that only the library's source code mentions |"
        ),
        (
            "| [L2]{.eff .eff-L2} | User code on the library's public primitives | A boundary "
            "condition or VJP rule that the library does not provide |"
        ),
        (
            "| [L3]{.eff .eff-L3} | Changes to the solver itself | A monkeypatch, a private API "
            "or a patched source file |"
        ),
        "",
        "What counts as intended use depends on the solver's abstraction level:",
        "",
        "- `application`: running a program as documented.",
        "- `framework`: expressing the setup through the library's API.",
        "- `kernel-toolkit`: building on the library's primitives.",
        "",
        (
            "The record also lists *deviations*, the places where a solver runs "
            "a substitute for part of the setup, with their effect on the "
            "results. An upstream fix or a better configuration can lower a "
            "solver's effort, so "
            "[contributions](contributing.qmd#tuning-an-existing-solver) are "
            "welcome on every item."
        ),
        "",
        "## All solvers {#all-solvers}",
        "",
    ]
    intro += render_index(categories)
    sections = [render_domain(c, s, scores) for c, s in categories.items()]
    return "\n".join(intro) + "\n" + "\n\n".join(sections) + "\n"


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    categories = collect_solvers()
    if not categories:
        sys.exit(f"No solvers found under {TESSERACTS}")

    new_qmd = generate_qmd(categories)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(new_qmd, encoding="utf-8")
    total = sum(len(v) for v in categories.values())
    print(f"Generated {OUTPUT} ({total} solvers, {len(categories)} categories)")


if __name__ == "__main__":
    main()
