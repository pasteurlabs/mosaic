#!/usr/bin/env python3
"""Generate docs/solvers.qmd from tesseract source files.

Usage:
    python docs/generate.py           # writes docs/solvers.qmd
    python docs/generate.py --check   # exit 1 if solvers.qmd is stale
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
TESSERACTS = ROOT / "mosaic" / "tesseracts"
OUTPUT = Path(__file__).resolve().parent / "solvers.qmd"

CATEGORY_LABELS = {
    "navier-stokes-grid": "Navier\u2013Stokes (Grid)",
    "structural-mesh": "Structural Mechanics",
    "thermal-mesh": "Heat Conduction",
}

BACKEND_LABELS = {
    "jax-cfd": "JAX",
    "jax-fem": "JAX",
    "exponax": "JAX",
    "phiflow": "PhiFlow / JAX",
    "xlb": "XLB / JAX",
    "fenics": "FEniCS",
    "fenics-heat": "FEniCS",
    "firedrake": "Firedrake",
    "firedrake-heat": "Firedrake",
    "warp-ns": "NVIDIA Warp",
    "openfoam": "OpenFOAM",
    "incompressible-navier-stokes-jl": "Julia / Zygote",
    "pict": "PyTorch / PICT",
    "topopt-jl": "Julia / TopOpt.jl",
    "dealii": "C++ / deal.II",
    "dealii-heat": "C++ / deal.II",
    "torch-fem": "PyTorch / torch-fem",
}


# ──────────────────────────────────────────────────────────────────────────────
# Parsing
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


def parse_schema(source: str, class_name: str) -> list[dict]:
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


def load_solver(path: Path) -> dict | None:
    config_path = path / "tesseract_config.yaml"
    api_path = path / "tesseract_api.py"
    if not config_path.exists() or not api_path.exists():
        return None
    with open(config_path) as f:
        config = yaml.safe_load(f)
    source = api_path.read_text()
    physics, backend = path.parent.name, path.name
    return {
        "name": config.get("name", f"{physics}/{backend}"),
        "version": config.get("version", ""),
        "description": (config.get("description") or "").strip(),
        "physics": physics,
        "backend": backend,
        "backend_label": BACKEND_LABELS.get(backend, backend),
        "category": CATEGORY_LABELS.get(physics, physics),
        "inputs": parse_schema(source, "InputSchema"),
        "outputs": parse_schema(source, "OutputSchema"),
        "path": f"mosaic/tesseracts/{physics}/{backend}",
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


def render_field_table(fields: list[dict], label: str) -> str:
    if not fields:
        return f"**{label}:** —\n"
    rows = [
        f"| `{f['name']}` | `{f['type']}` | {'✓' if f['differentiable'] else ''} | {f['description']} |"
        for f in fields
    ]
    header = "| Field | Type | ∂ | Description |\n|-------|------|---|-------------|\n"
    return f"**{label}**\n\n{header}" + "\n".join(rows) + "\n"


def render_solver(solver: dict) -> str:
    desc_short = solver["description"].split("\n\n")[0].replace("\n", " ").strip()
    backend = solver["backend_label"]

    inputs_md = render_field_table(solver["inputs"], "Inputs")
    outputs_md = render_field_table(solver["outputs"], "Outputs")

    lines = [
        f"### `{solver['name']}`",
        "",
        f"**Backend:** {backend} &nbsp;·&nbsp; **Path:** `{solver['path']}`",
        "",
    ]
    if desc_short:
        lines += [desc_short, ""]

    # Quarto collapsible callout
    lines += [
        '::: {.callout-note collapse="true"}',
        "#### Inputs / Outputs",
        "",
        inputs_md,
        outputs_md,
        ":::",
        "",
    ]

    return "\n".join(lines)


def render_category(category: str, solvers: list[dict]) -> str:
    body = "\n".join(render_solver(s) for s in solvers)
    return f"## {category}\n\n{body}"


def generate_qmd(categories: dict[str, list[dict]]) -> str:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total = sum(len(v) for v in categories.values())
    total_diff = sum(
        1
        for v in categories.values()
        for s in v
        for f in s["inputs"] + s["outputs"]
        if f["differentiable"]
    )

    frontmatter = "---\ntitle: Solver Reference\n---\n\n"
    header = (
        f"> Auto-generated {now} &nbsp;·&nbsp; "
        f"{total} solvers &nbsp;·&nbsp; "
        f"{len(categories)} physics domains &nbsp;·&nbsp; "
        f"{total_diff} differentiable fields\n\n"
        f"Click **Inputs / Outputs** on any solver to expand its field tables. "
        f"The ∂ column marks fields that support automatic differentiation (VJP/JVP).\n\n"
    )
    sections = "\n\n".join(
        render_category(cat, solvers) for cat, solvers in categories.items()
    )
    return frontmatter + header + sections + "\n"


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    check_mode = "--check" in sys.argv

    categories = collect_solvers()
    if not categories:
        sys.exit(f"No solvers found under {TESSERACTS}")

    new_qmd = generate_qmd(categories)

    if check_mode:
        if OUTPUT.exists() and OUTPUT.read_text() == new_qmd:
            print("docs/solvers.qmd is up to date.")
            return
        sys.exit(
            "docs/solvers.qmd is stale. Run `python docs/generate.py` to regenerate."
        )

    OUTPUT.write_text(new_qmd, encoding="utf-8")
    total = sum(len(v) for v in categories.values())
    print(f"Generated {OUTPUT} ({total} solvers, {len(categories)} categories)")


if __name__ == "__main__":
    main()
