"""Tests for docs/generate.py: solver reference generation."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def test_generate_collects_solvers():
    """generate.py should find all tesseract backends."""
    import sys

    sys.path.insert(0, str(ROOT))
    from docs.generate import collect_solvers

    categories = collect_solvers()
    assert len(categories) >= 3, f"Expected >= 3 categories, got {len(categories)}"
    total = sum(len(v) for v in categories.values())
    assert total >= 14, f"Expected >= 14 solvers, got {total}"


def test_generate_produces_valid_qmd():
    """Generated QMD should have frontmatter and solver sections."""
    import sys

    sys.path.insert(0, str(ROOT))
    from docs.generate import collect_solvers, generate_qmd

    categories = collect_solvers()
    qmd = generate_qmd(categories)
    assert qmd.startswith("---\ntitle:")
    assert "## " in qmd  # has category headings
    assert "### " in qmd  # has solver headings


def test_solver_specs_loaded():
    """SolverSpec metadata should enrich the generated output.

    Every discovered solver must carry the metadata that the generated page
    relies on: a non-empty scheme, a non-empty description, a known backend
    language, and one of the documented AD strategies. A missing value on any
    solver means that solver's card would render with placeholder text.
    """
    import sys

    sys.path.insert(0, str(ROOT))
    from docs.generate import collect_solvers

    categories = collect_solvers()
    all_solvers = [s for solvers in categories.values() for s in solvers]

    assert all_solvers, "collect_solvers returned no solvers at all"

    valid_ad = {"autodiff", "adjoint", "hybrid", None}
    missing: list[str] = []
    for s in all_solvers:
        ident = f"{s['physics']}/{s['backend']}"
        if not s.get("scheme"):
            missing.append(f"{ident}: empty scheme")
        if not s.get("description"):
            missing.append(f"{ident}: empty description")
        if not s.get("backend_lang"):
            missing.append(f"{ident}: empty backend_lang")
        if s.get("ad_strategy") not in valid_ad:
            missing.append(f"{ident}: invalid ad_strategy {s.get('ad_strategy')!r}")

    assert not missing, "Solver metadata gaps:\n  " + "\n  ".join(missing)
