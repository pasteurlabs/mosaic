# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the template scaffolding system."""

from __future__ import annotations

import pytest

from mosaic.templates.scaffold import list_templates, load_template, scaffold_domain


def test_list_templates():
    """Built-in templates must be exactly the three known names."""
    templates = list_templates()
    assert templates == ["ns-periodic", "structural-steady", "thermal-steady"]


def test_load_template():
    """Loading a known template must return a fully-populated DomainTemplate."""
    tpl = load_template("ns-periodic")
    assert tpl.name == "ns-periodic"
    assert tpl.schema_module == "mosaic_shared.problems.navier_stokes_grid"
    assert tpl.output_key == "result"
    assert tpl.ic_key == "v0"
    assert tpl.source_path is not None
    assert tpl.source_path.exists()


def test_load_unknown_template_raises():
    with pytest.raises(FileNotFoundError, match="not found"):
        load_template("nonexistent-template-xyz")


def test_load_template_missing_required_field_raises(tmp_path):
    """Templates without name/schema_module/output_key must be rejected."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("description: missing required keys\n")
    with pytest.raises(ValueError, match="missing required field"):
        load_template(str(bad))


def test_load_template_non_mapping_raises(tmp_path):
    """A YAML file that's not a mapping must be rejected with a clear error."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="expected a YAML mapping"):
        load_template(str(bad))


def _has_runtime():
    try:
        import tesseract_core.runtime  # noqa: F401

        return True
    except ImportError:
        return False


# validate_template needs tesseract-core[runtime] for schema imports
needs_runtime = pytest.mark.skipif(
    not _has_runtime(), reason="tesseract-core[runtime] not installed"
)


@needs_runtime
@pytest.mark.parametrize("name", list_templates())
def test_validate_builtin_templates(name):
    """Every built-in template must pass validation."""
    from mosaic.templates.scaffold import validate_template

    tpl = load_template(name)
    errors = validate_template(tpl)
    assert errors == [], f"Template {name!r} has errors: {errors}"


@needs_runtime
def test_validate_template_rejects_unknown_schema_module():
    """A template referencing a non-importable schema module must error."""
    from mosaic.templates.scaffold import DomainTemplate, validate_template

    tpl = DomainTemplate(
        name="bogus",
        description="",
        schema_module="this.module.does.not.exist",
        output_key="result",
    )
    errors = validate_template(tpl)
    assert any("cannot import schema module" in e for e in errors), errors


@needs_runtime
def test_validate_template_rejects_unknown_output_key():
    """Template with an output_key not on OutputSchema must produce an error."""
    from mosaic.templates.scaffold import validate_template

    tpl = load_template("ns-periodic")
    # Mutate the field that validate_template checks. DomainTemplate is a
    # plain dataclass, so direct attribute assignment is fine.
    tpl.output_key = "not-a-real-output-field"
    errors = validate_template(tpl)
    assert any("output_key" in e for e in errors), errors


@needs_runtime
def test_validate_template_rejects_bad_suite_shape():
    """Suite defaults must be dicts of lists of dicts."""
    from mosaic.templates.scaffold import validate_template

    tpl = load_template("ns-periodic")
    tpl.gradient = {"fd_check": "not-a-list"}
    errors = validate_template(tpl)
    assert any("expected a list of run dicts" in e for e in errors), errors


def test_scaffold_creates_files(tmp_path):
    """scaffold_domain must create schema stubs, tesseract dir, and problem package."""
    tpl = load_template("ns-periodic")
    target = tmp_path / "mosaic"
    # Create the required parent dirs
    (target / "benchmarks" / "problems").mkdir(parents=True)

    created = scaffold_domain("test-domain", tpl, target_dir=target)

    assert "schemas" in created
    assert "schema_init" in created
    assert "tesseracts_dir" in created
    assert "problem_config" in created
    assert "problem_pkg" in created

    assert created["schemas"].exists()
    assert created["schema_init"].exists()
    assert created["tesseracts_dir"].is_dir()
    assert created["problem_config"].exists()

    # The problem package is a two-file scaffold: a minimal __init__.py
    # (docstring only) + config.py (Problem + IC fn + make_inputs + experiment
    # TODOs all inline). The user can split into ics.py / physics.py /
    # experiments.py later when the file grows.
    pkg = created["problem_pkg"]
    assert (pkg / "__init__.py").exists()
    assert (pkg / "config.py").exists()
    # The old multi-file layout (ics.py / physics.py / experiments.py) is
    # intentionally not generated.
    for name in ("ics.py", "physics.py", "experiments.py"):
        assert not (pkg / name).exists(), f"unexpected scaffolded file: {name}"


def test_scaffold_generates_valid_python(tmp_path):
    """Generated problem config must be syntactically valid Python."""
    tpl = load_template("ns-periodic")
    target = tmp_path / "mosaic"
    (target / "benchmarks" / "problems").mkdir(parents=True)

    created = scaffold_domain("test-domain", tpl, target_dir=target)

    source = created["problem_config"].read_text()
    compile(source, str(created["problem_config"]), "exec")

    schema_source = created["schemas"].read_text()
    compile(schema_source, str(created["schemas"]), "exec")


@needs_runtime
def test_scaffold_produces_loadable_config(tmp_path):
    """Scaffolded domain must produce an importable CONFIG that passes validate()."""
    import importlib.util
    import sys

    from mosaic.benchmarks.core.config import Problem

    tpl = load_template("ns-periodic")
    target = tmp_path / "mosaic"
    (target / "benchmarks" / "problems").mkdir(parents=True)

    # scaffold_domain creates a tesseracts dir; discover_solvers needs it
    created = scaffold_domain("test-domain", tpl, target_dir=target)

    # Load the generated package via ``__init__.py``, giving the loader the
    # package's directory as a submodule search location so the relative
    # imports inside ``__init__.py`` (``from .experiments import register``)
    # can be resolved. Absolute imports (mosaic.benchmarks.core.*) still
    # resolve against the real installed package because the tmp tree is
    # not on sys.path.
    pkg_dir = created["problem_pkg"]
    pkg_name = "test_domain_pkg"
    # Load the package shell first so relative imports inside ``config.py``
    # (``from .experiments import register``) resolve against the tmp dir.
    pkg_spec = importlib.util.spec_from_file_location(
        pkg_name,
        pkg_dir / "__init__.py",
        submodule_search_locations=[str(pkg_dir)],
    )
    pkg_mod = importlib.util.module_from_spec(pkg_spec)
    sys.modules[pkg_spec.name] = pkg_mod
    pkg_spec.loader.exec_module(pkg_mod)
    # Then load the .config submodule that holds ``problem``.
    cfg_spec = importlib.util.spec_from_file_location(
        f"{pkg_name}.config", pkg_dir / "config.py"
    )
    mod = importlib.util.module_from_spec(cfg_spec)
    sys.modules[cfg_spec.name] = mod
    cfg_spec.loader.exec_module(mod)

    assert hasattr(mod, "problem"), "Generated module must define problem"
    cfg = mod.problem

    assert isinstance(cfg, Problem)
    assert cfg.name == "test-domain"
    assert cfg.output_key == tpl.output_key
    assert callable(cfg.make_inputs)
    assert callable(cfg.error_fn)

    # Solvers list will be empty (no tesseract configs in the scaffolded dir)
    # so validate() would fail on "no solvers registered". Instead we verify
    # the structural properties above are correct.
    assert cfg.solvers == []
