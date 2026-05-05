#!/usr/bin/env python3
"""Generate docs/solvers.qmd from tesseract source files and SolverSpec metadata.

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
sys.path.insert(0, str(ROOT))
TESSERACTS = ROOT / "mosaic" / "tesseracts"
OUTPUT = Path(__file__).resolve().parent / "solvers.qmd"

CATEGORY_LABELS = {
    "navier-stokes-grid": "Navier\u2013Stokes (Grid)",
    "structural-mesh": "Structural Mechanics",
    "thermal-mesh": "Heat Conduction",
}

# Maps (domain_dir, solver_dir) to the solver key used in problem configs.
# Populated lazily by _load_solver_specs().
_SOLVER_SPECS: dict[tuple[str, str], dict] | None = None

# Per-solver narrative documentation sourced from the paper appendix.
# Keys match the solver_dir name (not the display name).
SOLVER_DOCS: dict[str, str] = {
    "jax-cfd": (
        "JAX-native incompressible flow solver on a staggered MAC grid with "
        "finite-difference advection and spectral (FFT) pressure projection. "
        "Gradients via JAX source-transformation AD. "
        "The spectral pressure solve requires periodic BCs in all spatial "
        "directions, so JAX-CFD is included in periodic benchmarks (TGV, "
        "multimode agreement) but excluded from the cylinder-wake experiment."
    ),
    "phiflow": (
        "Semi-Lagrangian advection with pressure projection. "
        "Supports PyTorch, JAX, and TensorFlow backends, enabling gradient "
        "computation through the same simulation code across AD frameworks."
    ),
    "incompressible-navier-stokes-jl": (
        "IncompressibleNavierStokes.jl: Julia finite-difference "
        "pressure-projection solver. Differentiates through the time loop "
        "via Zygote.jl reverse-mode AD. CPU only."
    ),
    "xlb": (
        "JAX-native lattice Boltzmann solver supporting D2Q9 and D3Q19 "
        "stencils on GPU. Gradients via source-transformation AD through the "
        "collision-streaming loop. Note: the LBM recovers incompressible "
        "Navier\u2013Stokes only to O(Ma\u00b2) via Chapman\u2013Enskog "
        "expansion, giving an intrinsic compressibility error floor at fine grids."
    ),
    "pict": (
        "GPU-accelerated differentiable incompressible Navier\u2013Stokes "
        "solver built on PyTorch with custom CUDA kernels implementing the "
        "PISO algorithm. Supports reverse-mode AD through the full time loop "
        "and handles multi-block curvilinear grids. The Mosaic Tesseract "
        "exposes periodic, lid-driven cavity, inflow/channel, and "
        "cylinder-wake modes via an 8-block ring topology."
    ),
    "warp-ns": (
        "Implements the IPCS projection scheme in NVIDIA Warp CUDA kernels "
        "with an FFT-based spectral Poisson solver. Gradients via the "
        "wp.Tape kernel-level VJP. Custom solver: the framework provides "
        "differentiable kernels but no ready-made incompressible flow solver, "
        "so FE assembly and time integration were implemented using Warp "
        "primitives. Represents the integration cost when a kernel toolkit "
        "provides no built-in solver."
    ),
    "exponax": (
        "Integrates the 3D Navier\u2013Stokes equations with an exponential "
        "time-differencing Runge\u2013Kutta (ETDRK) spectral scheme, "
        "enforcing incompressibility to machine precision by construction. "
        "Gradients via JAX source-transformation AD."
    ),
    "openfoam": (
        "Runs the icoFoam incompressible PISO solver as a forward-only "
        "reference baseline. No reverse-mode AD available. Used as the "
        "reference solver for fluid benchmark domains."
    ),
    "fenics": (
        "Finite-element solver using P1 elements for structural problems. "
        "dolfin-adjoint automates the discrete adjoint by replaying the "
        "forward tape."
    ),
    "fenics-heat": (
        "Finite-element solver using P1 elements for thermal problems. "
        "dolfin-adjoint automates the discrete adjoint by replaying the "
        "forward tape."
    ),
    "firedrake": (
        "Mirrors the FEniCS P1/CG1 formulation for structural problems. "
        "Differentiates via firedrake-adjoint, providing an independent "
        "tape-based adjoint implementation for cross-validation."
    ),
    "firedrake-heat": (
        "Mirrors the FEniCS P1/CG1 formulation for thermal problems. "
        "Differentiates via firedrake-adjoint, providing an independent "
        "tape-based adjoint implementation for cross-validation."
    ),
    "jax-fem": (
        "Solves heat conduction and linear elasticity with trilinear HEX8 "
        "finite elements in JAX. Gradients via AD through the assembled system."
    ),
    "topopt-jl": (
        "SIMP topology optimization for linear elasticity with HEX8 elements "
        "in Julia, using analytical adjoint sensitivities."
    ),
    "dealii": (
        "Solves structural problems with Q1 elements using the industry-grade "
        "C++ finite-element library deal.II. Gradients via hand-derived "
        "analytical adjoint sensitivities. Used as the reference solver for "
        "structural and thermal domains."
    ),
    "dealii-heat": (
        "Solves thermal problems with Q1 elements using the industry-grade "
        "C++ finite-element library deal.II. Gradients via hand-derived "
        "analytical adjoint sensitivities. Used as the reference solver for "
        "the thermal domain."
    ),
    "torch-fem": (
        "Solves heat conduction with linear finite elements in PyTorch. "
        "Gradients via torch.autograd through the assembled system."
    ),
}

# Known solver limitations from benchmarking (paper appendix C.3).
SOLVER_LIMITATIONS: dict[str, list[str]] = {
    "pict": [
        "Viscosity is not differentiable: PISOtorch_diff treats viscosity as "
        "a static scalar. Differentiation w.r.t. viscosity would require "
        "upstream changes to the C++/CUDA kernels.",
    ],
    "jax-cfd": [
        "Spectral pressure solve requires periodic boundary conditions. "
        "Excluded from channel/cylinder-wake experiments.",
    ],
    "warp-ns": [
        "Spectral pressure solve requires periodic boundary conditions. "
        "Excluded from channel/cylinder-wake experiments.",
    ],
    "incompressible-navier-stokes-jl": [
        "Viscosity gradient uses finite differences (Zygote returns "
        "NoTangent for the diffusion operator). FD gradients diverge as "
        "rollout length grows.",
        "Brinkman penalization incompatible with spectral pressure solve. "
        "Excluded from drag optimization.",
    ],
    "xlb": [
        "Intrinsic O(Ma\u00b2) compressibility error: the LBM recovers "
        "incompressible NS only approximately. At N=128, dt=0.01, "
        "Ma \u2248 0.2, giving ~4% error floor.",
        "BGK collision instability at low viscosity: relaxation time "
        "approaches the stability boundary as viscosity decreases.",
    ],
}


# ──────────────────────────────────────────────────────────────────────────────
# SolverSpec loading (optional — gracefully degrades without full install)
# ──────────────────────────────────────────────────────────────────────────────


def _load_solver_specs() -> dict[tuple[str, str], dict]:
    """Try to load SolverSpec metadata from problem configs."""
    global _SOLVER_SPECS
    if _SOLVER_SPECS is not None:
        return _SOLVER_SPECS

    _SOLVER_SPECS = {}
    try:
        from mosaic.benchmarks.problems import PROBLEMS, get_config
    except ImportError:
        return _SOLVER_SPECS

    # Map from domain CLI names to tesseract directory names.
    domain_dir_map = {
        "ns-grid": "navier-stokes-grid",
        "ns-3d-grid": "navier-stokes-grid",
        "structural-mesh": "structural-mesh",
        "thermal-mesh": "thermal-mesh",
    }

    for prob in PROBLEMS:
        try:
            cfg = get_config(prob)
        except Exception:
            continue
        domain_dir = domain_dir_map.get(prob, prob)
        for _solver_key, spec in cfg.solvers.items():
            key = (domain_dir, spec.dir)
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
                "exclusions": spec.exclusions,
                "explained_anomalies": spec.explained_anomalies,
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
    spec = _get_spec(physics, backend)

    return {
        "name": config.get("name", f"{physics}/{backend}"),
        "version": config.get("version", ""),
        "description": (config.get("description") or "").strip(),
        "physics": physics,
        "backend": backend,
        "category": CATEGORY_LABELS.get(physics, physics),
        "inputs": parse_schema(source, "InputSchema"),
        "outputs": parse_schema(source, "OutputSchema"),
        "path": f"mosaic/tesseracts/{physics}/{backend}",
        # Enriched fields from SolverSpec
        "display_name": spec["name"] if spec else config.get("name", backend),
        "scheme": spec["scheme"] if spec else "",
        "backend_lang": spec["backend_lang"] if spec else "",
        "ad_strategy": spec["ad_strategy"] if spec else None,
        "doc_url": spec["doc_url"] if spec else "",
        "family": spec["family"] if spec else "",
        "uses_gpu": spec["uses_gpu"] if spec else None,
        "differentiable": spec["differentiable"] if spec else None,
        "internal_dtype": spec["internal_dtype"] if spec else "",
        "exclusions": spec["exclusions"] if spec else {},
        "explained_anomalies": spec["explained_anomalies"] if spec else {},
        "narrative": SOLVER_DOCS.get(backend, ""),
        "limitations": SOLVER_LIMITATIONS.get(backend, []),
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
    "autodiff": "Reverse-mode AD",
    "adjoint": "Discrete adjoint",
    "hybrid": "Hybrid (analytic rules + AD)",
    None: "None (forward only)",
}


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
    lines = [f"### {name}", ""]

    # Metadata badges
    badges = []
    if solver["scheme"]:
        badges.append(f"**Scheme:** {solver['scheme']}")
    if solver["backend_lang"]:
        badges.append(f"**Language:** {solver['backend_lang']}")
    ad = AD_LABELS.get(solver["ad_strategy"], solver["ad_strategy"])
    if ad:
        badges.append(f"**Gradients:** {ad}")
    if solver["internal_dtype"]:
        badges.append(f"**Precision:** {solver['internal_dtype']}")
    if solver["uses_gpu"] is not None:
        badges.append(f"**GPU:** {'Yes' if solver['uses_gpu'] else 'No'}")
    if badges:
        lines.append(" &nbsp;\u00b7&nbsp; ".join(badges))
        lines.append("")

    if solver["doc_url"]:
        lines.append(f"**Upstream docs:** [{solver['doc_url']}]({solver['doc_url']})")
        lines.append("")

    lines.append(f"**Path:** `{solver['path']}`")
    lines.append("")

    # Narrative description (from paper)
    if solver["narrative"]:
        lines.append(solver["narrative"])
        lines.append("")

    # Known limitations
    if solver["limitations"]:
        lines.append("::: {.callout-warning}")
        lines.append("#### Known limitations")
        lines.append("")
        for lim in solver["limitations"]:
            lines.append(f"- {lim}")
        lines.append(":::")
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
        f"> Auto-generated {now} &nbsp;\u00b7&nbsp; "
        f"{total} solvers &nbsp;\u00b7&nbsp; "
        f"{len(categories)} physics domains &nbsp;\u00b7&nbsp; "
        f"{total_diff} differentiable fields\n\n"
        f"Each solver card shows its numerical scheme, AD strategy, known "
        f"limitations, and Tesseract schema. "
        f"Click **Inputs / Outputs** on any solver to expand its field tables. "
        f"The \u2202 column marks fields that support automatic differentiation "
        f"(VJP/JVP).\n\n"
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
