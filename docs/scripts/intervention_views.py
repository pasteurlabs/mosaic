# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the solvers' intervention records for the docs pages.

Two views of the same records (``metadata.mosaic`` in each
``tesseract_config.yaml``, format in ``mosaic/benchmarks/core/interventions.py``):

* results pages (``generate_results.py``): an effort and a deviations column
  per setup, with the detail on hover, and a filter by effort level;
* solver reference (``generate.py``): every intervention and deviation of a
  solver, with its reason, evidence and upstream link.

The records module is loaded from its file, so the docs build does not import
the ``mosaic`` package and its compute stack.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
TESSERACTS = ROOT / "mosaic" / "tesseracts"
_REPO_URL = "https://github.com/pasteurlabs/mosaic"


def _load_records_module() -> ModuleType:
    name = "_mosaic_interventions"
    if name in sys.modules:
        return sys.modules[name]
    path = ROOT / "mosaic" / "benchmarks" / "core" / "interventions.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look the module up while it loads
    spec.loader.exec_module(module)
    return module


iv = _load_records_module()

EFFORT_LABEL = "Effort"
DEVIATIONS_LABEL = "Deviations"
# The effort of a solver that needs no intervention beyond the baseline.
BASELINE_LEVEL = "L0"
NOT_RECORDED = "not recorded"

# Explains the two columns on a results page without check tables.
EFFORT_EXPLAINER = (
    "**Effort** is the highest level, from L0 to L3, among the interventions a "
    "solver *needs* on this setup, and **deviations** counts the places where "
    "it runs a substitute for part of the setup. The "
    "[solver reference](../solvers.qmd#effort-levels) explains the levels and "
    "lists every intervention."
)


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def attr(text: str) -> str:
    """Make ``text`` safe as a quoted Pandoc attribute inside a pipe table."""
    return (
        _one_line(text)
        .replace("\\", "/")
        .replace('"', "'")
        .replace("|", "/")
        .replace("{", "(")
        .replace("}", ")")
    )


def tip(text: str, title: str) -> str:
    """A table cell showing ``text`` with ``title`` on hover."""
    return f'[{text}]{{.m-tip title="{attr(title)}"}}'


def records_for(setup: str) -> dict[str, object]:
    """Records of the tesseracts serving ``setup``, keyed by display name."""
    domain = iv.domain_of(setup)
    return iv.load_records(TESSERACTS, domain) if domain else {}


def title(item: object) -> str:
    """An intervention's or deviation's title, or its id if it has none."""
    return item.title or item.id


def effort_parts(record: object | None, setup: str) -> tuple[str, str]:
    """The effort of one solver on one setup, as ``(text, detail)``.

    ``detail`` is empty when there is nothing to show on hover.
    """
    if record is None:
        return NOT_RECORDED, ""
    required = [i for i in record.interventions_for(setup) if i.necessity == "required"]
    level = record.effort(setup)
    if level is None:
        return BASELINE_LEVEL, (
            f"{BASELINE_LEVEL}, {iv.LEVEL_NAMES[BASELINE_LEVEL]}. "
            "No intervention is required on this setup."
        )
    detail = "; ".join(
        f"{i.level} {title(i)}"
        for i in sorted(required, key=lambda i: iv.LEVELS.index(i.level), reverse=True)
    )
    return level, f"{level}, {iv.LEVEL_NAMES[level]}. Required: {detail}."


def deviations_parts(record: object | None, setup: str) -> tuple[str, str]:
    """The setup deviations of one solver on one setup, as ``(text, detail)``."""
    if record is None:
        return NOT_RECORDED, ""
    devs = record.deviations_for(setup)
    if not devs:
        return "—", ""
    detail = " ".join(f"{title(d)}. Effect: {d.effect}" for d in devs)
    return str(len(devs)), detail


def deviation_items(record: object | None, setup: str) -> list[tuple[str, str]]:
    """The setup deviations of one solver on one setup, as ``(title, effect)``."""
    if record is None:
        return []
    return [(title(d), _one_line(d.effect)) for d in record.deviations_for(setup)]


def _hover_cell(parts: tuple[str, str]) -> str:
    text, detail = parts
    return tip(text, detail) if detail else text


def effort_cell(record: object | None, setup: str) -> str:
    """The effort column for one solver on one setup."""
    return _hover_cell(effort_parts(record, setup))


def deviations_cell(record: object | None, setup: str) -> str:
    """The deviations column for one solver on one setup."""
    return _hover_cell(deviations_parts(record, setup))


def interventions_cell(record: object | None, setup: str) -> str:
    """A count of the interventions on one setup, listed on hover."""
    if record is None:
        return NOT_RECORDED
    items = record.interventions_for(setup)
    if not items:
        return "—"
    unknown = sum(i.reason_unknown for i in items)
    text = f"{len(items)}" + (f" ({unknown} unknown reason)" if unknown else "")
    detail = "; ".join(f"{title(i)} ({i.level}, {i.necessity})" for i in items)
    return tip(text, detail)


# ── Solver reference ─────────────────────────────────────────────────────────

_PR_REF = re.compile(r"(?<![\w/#])#(\d+)\b")
_URL = re.compile(r"(?<![<(])(https?://[^\s<>()]+[^\s<>().,;:])")


def linkify(text: str) -> str:
    """Turn ``#123`` into links to this repository and wrap bare URLs."""
    text = _URL.sub(r"<\1>", _one_line(text))
    return _PR_REF.sub(rf"[#\1]({_REPO_URL}/pull/\1)", text)


def _setup_label(key: str) -> str:
    setup = iv.SETUPS.get(key)
    return setup.label if setup else key


def _consequences(reason: object) -> str:
    return ", ".join(c.replace("-", " ") for c in reason.consequences)


def _intervention_item(i: object) -> list[str]:
    """One intervention: title, what was done and why per setup, with the
    notes, evidence and upstream link behind a disclosure."""
    tags = [i.necessity] + (["tuned"] if i.tuned else [])
    lines = [
        "::: {.iv-item}",
        f"[{title(i)}]{{.iv-title}} [{i.level}]{{.iv-level}} "
        + " ".join(f"[{t}]{{.iv-tag}}" for t in tags),
        "",
        linkify(i.what),
        "",
    ]
    # Setups with the same reason are listed once.
    grouped: dict[object, list[str]] = {}
    for s, r in i.why.items():
        grouped.setdefault(r, []).append(_setup_label(s))
    lines += [
        f"- *{', '.join(names)}*"
        + (f" ({_consequences(r)})" if r.consequences else "")
        + f": {linkify(r.symptom)}"
        for r, names in grouped.items()
    ]
    lines.append("")
    details = []
    for r, names in grouped.items():
        parts = [linkify(r.notes)] if r.notes else []
        if r.evidence:
            parts.append(f"Evidence: {linkify(r.evidence)}")
        if parts:
            details.append(f"- *{', '.join(names)}*: {' '.join(parts)}")
    if i.upstream:
        details.append(f"- Upstream: {linkify(i.upstream)}")
    details.append(f"- Record id: `{i.id}`")
    lines += [
        '<details class="iv-details"><summary>Details</summary>',
        "",
        *details,
        "",
        "</details>",
        ":::",
        "",
    ]
    return lines


def developer_block(record: object | None) -> list[str]:
    """Markdown lines describing a solver's record on the solver reference."""
    if record is None:
        return [
            "**Interventions:** not recorded yet.",
            "",
        ]
    lines = [
        "::: {.callout-note}",
        "#### Interventions and deviations",
        "",
        f"Abstraction level: **{record.abstraction}**.",
        "",
    ]
    active = [i for i in record.interventions if i.active]
    if active:
        for i in active:
            lines += _intervention_item(i)
    else:
        lines += ["No interventions are recorded.", ""]

    if record.deviations:
        lines += ["**Setup deviations**", ""]
        for d in record.deviations:
            where = ", ".join(_setup_label(s) for s in d.setups)
            lines.append(
                f"- **{title(d)}** ({where}): {linkify(d.what)} "
                f"*Effect:* {linkify(d.effect)}"
                + (f" *Evidence:* {linkify(d.evidence)}" if d.evidence else "")
            )
        lines.append("")

    inactive = [i for i in record.interventions if not i.active]
    if inactive:
        lines += [
            "**Inactive** (present in the code, but exercised by no experiment): "
            + ", ".join(title(i) for i in inactive),
            "",
        ]
    lines += [":::", ""]
    return lines
