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

ROOT = Path(__file__).resolve().parent.parent
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

# Explains the two columns; shown once per results page.
EFFORT_EXPLAINER = (
    "**Effort** is the highest level among the interventions a solver *needs* "
    "on this setup: L0 nothing beyond expected tweaks (the public API with its "
    "defaults or documented options, such as a tolerance), L1 undocumented "
    "configuration, L2 user code on the library's public primitives, and L3 "
    "changes to the solver itself (monkeypatches, private APIs, patched "
    "sources). **Deviations** counts the places where a solver runs a "
    "substitute for the prescribed setup. Each solver's entry in the "
    "[solver reference](solvers.qmd) lists its interventions and deviations "
    "with their reasons and evidence."
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
        f"{i.level} {i.id}"
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
    detail = " ".join(f"{d.id}: {d.what} Effect: {d.effect}" for d in devs)
    return str(len(devs)), detail


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
    detail = "; ".join(f"{i.id} ({i.level}, {i.necessity})" for i in items)
    return tip(text, detail)


# ── Solver reference ─────────────────────────────────────────────────────────

_PR_REF = re.compile(r"(?<![\w/#])#(\d+)\b")
_URL = re.compile(r"(?<![<(])(https?://[^\s<>()]+[^\s<>().,;:])")


def linkify(text: str) -> str:
    """Turn ``#123`` into links to this repository and wrap bare URLs."""
    text = _URL.sub(r"<\1>", _one_line(text))
    return _PR_REF.sub(rf"[#\1]({_REPO_URL}/pull/\1)", text)


def _cell(text: str) -> str:
    return linkify(text).replace("|", "\\|")


def _setup_label(key: str) -> str:
    setup = iv.SETUPS.get(key)
    return setup.label if setup else key


def mentioned_setups(record: object, setups: tuple[str, ...]) -> tuple[str, ...]:
    """The setups of ``setups`` that the record says anything about."""
    named = {s for i in record.interventions for s in i.why}
    named |= {s for d in record.deviations for s in d.setups}
    return tuple(s for s in setups if s in named)


def summary_line(record: object, setups: tuple[str, ...]) -> str:
    """One line: effort per setup and how many records have no known reason."""
    setups = mentioned_setups(record, setups)
    if not setups:
        return f"**Effort:** {BASELINE_LEVEL} on every setup"
    efforts = ", ".join(
        f"{record.effort(s) or BASELINE_LEVEL} on {_setup_label(s)}" for s in setups
    )
    n = len(record.interventions)
    unknown = sum(i.reason_unknown for i in record.interventions)
    d = len(record.deviations)
    counts = (
        f"{n} intervention{'s' if n != 1 else ''}"
        + (f" ({unknown} with unknown reason)" if unknown else "")
        + f" and {d} setup deviation{'s' if d != 1 else ''}"
    )
    return f"**Effort:** {efforts}. {counts.capitalize()}."


def developer_block(record: object | None, setups: tuple[str, ...]) -> list[str]:
    """Markdown lines describing a solver's record on the solver reference."""
    if record is None:
        return [
            "**Interventions:** not recorded yet.",
            "",
        ]
    lines = [
        summary_line(record, setups),
        "",
        "::: {.callout-note}",
        "#### Interventions and deviations",
        "",
        (
            f"Abstraction level: **{record.abstraction}**. The effort level counts "
            "only interventions marked `required`. Each item is an invitation to "
            "contribute, since an upstream fix or a better configuration moves the "
            "solver down the ladder."
        ),
        "",
    ]
    active = [i for i in record.interventions if i.active]
    if active:
        lines += [
            "| Intervention | Level | Necessity | Why, per setup | Upstream |",
            "|---|---|---|---|---|",
        ]
        for i in active:
            name = f"`{i.id}`" + (" (tuned)" if i.tuned else "")
            # Setups that share one reason (YAML anchors) are listed once.
            grouped: dict[object, list[str]] = {}
            for s, r in i.why.items():
                grouped.setdefault(r, []).append(_setup_label(s))
            why = "<br>".join(
                f"*{', '.join(names)}*"
                + (f" ({', '.join(r.consequences)})" if r.consequences else "")
                + f": {_cell(r.symptom)}"
                + (f" Evidence: {_cell(r.evidence)}" if r.evidence else "")
                for r, names in grouped.items()
            )
            lines.append(
                f"| {name}: {_cell(i.what)} | {i.level} | {i.necessity} | {why} "
                f"| {_cell(i.upstream) if i.upstream else '—'} |"
            )
        lines.append("")
    else:
        lines += ["No interventions are recorded.", ""]

    if record.deviations:
        lines += ["**Setup deviations**", ""]
        for d in record.deviations:
            where = ", ".join(_setup_label(s) for s in d.setups)
            lines.append(
                f"- `{d.id}` ({where}): {linkify(d.what)} "
                f"*Effect:* {linkify(d.effect)}"
                + (f" *Evidence:* {linkify(d.evidence)}" if d.evidence else "")
            )
        lines.append("")

    unknown = [i for i in active if i.reason_unknown]
    if unknown:
        lines += [
            (
                "**Reason unknown.** Nobody has measured whether these are "
                "needed. A measurement, or removing them, would settle it."
            ),
            "",
        ]
        lines += [f"- `{i.id}` ({i.level}): {linkify(i.what)}" for i in unknown]
        lines.append("")

    inactive = [i for i in record.interventions if not i.active]
    if inactive:
        lines += [
            "**Inactive** (present in the code, but exercised by no experiment): "
            + ", ".join(f"`{i.id}`" for i in inactive),
            "",
        ]
    lines += [":::", ""]
    return lines
