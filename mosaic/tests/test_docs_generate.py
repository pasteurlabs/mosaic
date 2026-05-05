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
    """SolverSpec metadata should enrich the generated output."""
    import sys

    sys.path.insert(0, str(ROOT))
    from docs.generate import collect_solvers

    categories = collect_solvers()
    all_solvers = [s for solvers in categories.values() for s in solvers]

    # At least some solvers should have enriched data
    with_scheme = [s for s in all_solvers if s.get("scheme")]
    assert len(with_scheme) >= 10, "Most solvers should have scheme metadata"

    with_narrative = [s for s in all_solvers if s.get("narrative")]
    assert len(with_narrative) >= 10, "Most solvers should have narrative docs"
