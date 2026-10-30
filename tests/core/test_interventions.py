# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Intervention records: the format check that runs on every tesseract."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from mosaic.benchmarks.core.config import TESSERACTS_DIR, discover_solvers
from mosaic.benchmarks.core.interventions import (
    check_markers,
    check_tesseract,
    parse_record,
)

TESSERACT_DIRS = sorted(
    p.parent for p in TESSERACTS_DIR.glob("*/*/tesseract_config.yaml")
)

# Tesseracts whose record is in place; the others follow in a later change.
RECORDED = {
    "navier-stokes-grid/openfoam",
    "navier-stokes-grid/warp-ns",
    "navier-stokes-grid/xlb",
    "structural-mesh/fenics-structural",
    "thermal-mesh/fenics-heat",
}


def _name(d: Path) -> str:
    return f"{d.parent.name}/{d.name}"


@pytest.mark.parametrize("tesseract_dir", TESSERACT_DIRS, ids=_name)
def test_record_and_markers_are_valid(tesseract_dir: Path) -> None:
    record, problems = check_tesseract(tesseract_dir)
    assert not problems, "\n".join(problems)
    if _name(tesseract_dir) in RECORDED:
        assert record is not None, f"{_name(tesseract_dir)} has no record"


def test_discovered_specs_carry_the_record() -> None:
    specs = discover_solvers("navier-stokes-grid")
    assert specs["openfoam"].record is not None
    assert specs["openfoam"].record.abstraction == "application"


VALID = {
    "abstraction": "framework",
    "interventions": [
        {
            "id": "custom-bc",
            "level": "L2",
            "necessity": "required",
            "what": "Writes the outlet by hand.",
            "why": {
                "ns-2d-cylinder": {
                    "consequences": ["cannot-express"],
                    "symptom": "No pressure outlet in the library.",
                    "evidence": "Library docs.",
                }
            },
            "provenance": "mosaic-team",
        },
        {
            "id": "fast-solver",
            "level": "L0",
            "necessity": "choice",
            "tuned": True,
            "what": "Uses a direct solver.",
            "why": {
                "ns-2d-periodic": {
                    "consequences": ["too-slow"],
                    "symptom": "2x slower otherwise.",
                    "evidence": "#1 before/after table.",
                }
            },
            "provenance": "contributor",
        },
        {
            "id": "old-flag",
            "level": "L3",
            "necessity": "required",
            "active": False,
            "what": "Patches a module.",
            "why": {
                "ns-2d-periodic": {
                    "consequences": ["crash"],
                    "symptom": "Fails on import.",
                    "evidence": "#2",
                }
            },
            "provenance": "mosaic-team",
        },
    ],
    "deviations": [
        {
            "id": "advective-outflow",
            "setups": ["ns-2d-cylinder"],
            "what": "Advective outflow instead of p = 0.",
            "consequences": ["cannot-express"],
            "effect": "Not measured.",
        }
    ],
}
SETUPS = ("ns-2d-periodic", "ns-2d-cylinder", "ns-3d")


def test_valid_record_parses() -> None:
    record, problems = parse_record(VALID, SETUPS)
    assert problems == []
    assert record.abstraction == "framework"
    assert [i.id for i in record.interventions_for("ns-2d-periodic")] == ["fast-solver"]
    assert record.deviations_for("ns-2d-cylinder")[0].id == "advective-outflow"
    assert record.deviations_for("ns-3d") == []


def test_effort_is_highest_required_active_level() -> None:
    record, _ = parse_record(VALID, SETUPS)
    assert record.effort("ns-2d-cylinder") == "L2"
    # The L0 entry is a choice and the L3 one inactive, so nothing counts.
    assert record.effort("ns-2d-periodic") is None


def test_no_record_keys_means_no_record() -> None:
    assert parse_record({"name": "X"}, SETUPS) == (None, [])


def _mutated(path: list, value: object) -> dict:
    meta = copy.deepcopy(VALID)
    target = meta
    for key in path[:-1]:
        target = target[key]
    if value is _DELETE:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return meta


_DELETE = object()
_FIRST = ["interventions", 0]
_REASON = [*_FIRST, "why", "ns-2d-cylinder"]


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (["abstraction"], "library", "abstraction"),
        ([*_FIRST, "level"], "L4", "not accepted"),
        ([*_FIRST, "level"], "L5", "level"),
        ([*_FIRST, "necessity"], "maybe", "necessity"),
        ([*_FIRST, "provenance"], "someone", "provenance"),
        ([*_FIRST, "id"], "Custom_BC", "kebab-case"),
        ([*_FIRST, "what"], " ", "`what` is empty"),
        ([*_FIRST, "scope"], ["all"], "unknown key `scope`"),
        ([*_FIRST, "why"], {}, "at least one setup"),
        ([*_FIRST, "tuned"], "yes", "`tuned` must be"),
        ([*_REASON, "consequences"], ["slow"], "consequence `slow`"),
        ([*_REASON, "consequences"], [], "`consequences` is empty"),
        ([*_REASON, "symptom"], "", "`symptom` is empty"),
        # Above L0, a stated necessity needs evidence.
        ([*_REASON, "evidence"], _DELETE, "`evidence` is empty"),
        # A tuned value needs its measurement, even at L0.
        (["interventions", 1, "why", "ns-2d-periodic", "evidence"], "", "evidence"),
        (["deviations", 0, "setups"], ["ns-4d"], "unknown setup `ns-4d`"),
        (["deviations", 0, "effect"], "", "`effect` is empty"),
        (["deviations", 0, "id"], "custom-bc", "used 2 times"),
    ],
)
def test_invalid_records_are_reported(path: list, value: object, message: str) -> None:
    record, problems = parse_record(_mutated(path, value), SETUPS)
    assert record is None
    assert any(message in p for p in problems), problems


def test_unknown_setup_in_reason() -> None:
    meta = copy.deepcopy(VALID)
    meta["interventions"][0]["why"]["cantilever"] = meta["interventions"][0]["why"].pop(
        "ns-2d-cylinder"
    )
    _, problems = parse_record(meta, SETUPS)
    assert any("setup `cantilever`: unknown setup" in p for p in problems)


def test_unknown_necessity_needs_no_consequence_or_evidence() -> None:
    meta = _mutated([*_FIRST, "necessity"], "unknown")
    reason = meta["interventions"][0]["why"]["ns-2d-cylinder"]
    reason["consequences"] = []
    del reason["evidence"]
    record, problems = parse_record(meta, SETUPS)
    assert problems == []
    assert record.interventions[0].reason_unknown


def _tesseract(tmp_path: Path, meta: dict, source: str) -> Path:
    d = tmp_path / "navier-stokes-grid" / "solver"
    d.mkdir(parents=True)
    doc = {"name": "solver", "metadata": {"mosaic": {"name": "S", **meta}}}
    (d / "tesseract_config.yaml").write_text(yaml.safe_dump(doc))
    (d / "tesseract_api.py").write_text(source)
    return d


MARKED_SOURCE = """\
x = 1
# mosaic-intervention: custom-bc
y = 2
# mosaic-intervention-end: custom-bc
"""


def test_markers_match_records(tmp_path: Path) -> None:
    d = _tesseract(tmp_path, VALID, MARKED_SOURCE + "# mosaic-intervention: old-flag\n")
    _, problems = check_tesseract(d)
    # old-flag is L3 but its region is never closed.
    assert len(problems) == 2, problems
    assert "never closed" in problems[0]
    assert "`old-flag` has no `mosaic-intervention` markers" in problems[1]


def test_marker_without_record(tmp_path: Path) -> None:
    d = _tesseract(tmp_path, {}, MARKED_SOURCE)
    problems = check_markers(None, d, "x")
    assert problems == ["x: marker `custom-bc` has no intervention record"]


def test_end_marker_without_begin(tmp_path: Path) -> None:
    d = _tesseract(tmp_path, {}, "// mosaic-intervention-end: custom-bc\n")
    problems = check_markers(None, d, "x")
    assert any("without a begin" in p for p in problems)
