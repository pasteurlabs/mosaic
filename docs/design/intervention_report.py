# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize the intervention annotations of every tesseract (spike).

Reads the ``interventions:`` list from each tesseract's
``tesseract_config.yaml`` (under ``metadata.mosaic``) and the
``mosaic-intervention`` marker comments in its sources, then reports:

* interventions per tesseract, with level, consequence and confidence;
* lines of code covered per level, against the tesseract's total;
* inconsistencies between the YAML records and the code markers;
* hits of a heuristic L3 lint, and whether each hit sits inside an
  annotated L3 region.

See ``docs/design/interventions.md`` for the annotation format.

Usage::

    python docs/design/intervention_report.py [--out report.md]
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
TESSERACTS = ROOT / "mosaic" / "tesseracts"

LEVELS = ["L0", "L1", "L2", "L3"]
CONSEQUENCES = {
    "cannot-express",
    "crash",
    "non-finite",
    "inaccurate-forward",
    "no-gradient",
    "incorrect-gradient",
    "too-slow",
    "out-of-memory",
}
CONFIDENCES = ["verified", "stated", "inferred"]
SOURCE_SUFFIXES = {".py", ".jl", ".cpp", ".cc", ".h", ".hpp", ".C", ".H", ".sh"}
CONFIG_NAMES = {"tesseract_config.yaml"}

MARKER = re.compile(r"(?:#|//)\s*mosaic-intervention(-end)?:\s*([\w.-]+)")


@dataclass
class Region:
    """One begin/end marker pair."""

    ident: str
    path: Path
    start: int  # line number of the begin marker (1-based)
    end: int  # line number of the end marker


@dataclass
class Tesseract:
    """Annotations of one tesseract."""

    name: str
    path: Path
    records: list[dict] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _is_code_line(line: str) -> bool:
    s = line.strip()
    return bool(s) and not s.startswith(("#", "//"))


def _source_files(directory: Path) -> list[Path]:
    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file()
        and (p.suffix in SOURCE_SUFFIXES or p.name in CONFIG_NAMES)
        and "__pycache__" not in p.parts
    )


def _scan_markers(path: Path) -> tuple[list[Region], list[str]]:
    """Return the marker regions in ``path`` and any pairing problems."""
    regions, problems = [], []
    open_: dict[str, list[int]] = defaultdict(list)
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        m = MARKER.search(line)
        if not m:
            continue
        is_end, ident = bool(m.group(1)), m.group(2)
        if not is_end:
            open_[ident].append(i)
        elif open_[ident]:
            regions.append(Region(ident, path, open_[ident].pop(), i))
        else:
            problems.append(f"`{_rel(path)}:{i}`: end marker `{ident}` without begin")
    for ident, starts in open_.items():
        for s in starts:
            problems.append(f"`{_rel(path)}:{s}`: begin marker `{ident}` never closed")
    return regions, problems


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load() -> tuple[list[Tesseract], list[str]]:
    """Collect records and marker regions for every tesseract."""
    tesseracts: list[Tesseract] = []
    global_problems: list[str] = []
    shared_regions: list[Region] = []

    for cfg in sorted(TESSERACTS.glob("*/*/tesseract_config.yaml")):
        t = Tesseract(
            name=f"{cfg.parent.parent.name}/{cfg.parent.name}", path=cfg.parent
        )
        doc = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        meta = (doc.get("metadata") or {}).get("mosaic") or {}
        t.records = list(meta.get("interventions") or [])
        t.files = _source_files(cfg.parent)
        for f in t.files:
            regions, problems = _scan_markers(f)
            t.regions += regions
            t.problems += problems
        tesseracts.append(t)

    # Marker regions in shared code are attributed to the tesseracts whose
    # records list that file under ``location``.
    shared_root = ROOT / "mosaic" / "mosaic_shared"
    for f in _source_files(shared_root):
        regions, problems = _scan_markers(f)
        global_problems += problems
        shared_regions += regions
    for r in shared_regions:
        owners = [
            t
            for t in tesseracts
            if any(
                rec.get("id") == r.ident
                and any(
                    _rel(r.path).endswith(str(loc)) or str(loc) in _rel(r.path)
                    for loc in (rec.get("location") or [])
                )
                for rec in t.records
            )
        ]
        if not owners:
            global_problems.append(
                f"`{_rel(r.path)}:{r.start}`: shared marker `{r.ident}` not claimed "
                "by any tesseract record (`location:`)"
            )
        for t in owners:
            t.regions.append(r)
            if r.path not in t.files:
                t.files.append(r.path)

    for t in tesseracts:
        t.problems += _check_records(t)
    return tesseracts, global_problems


def _check_records(t: Tesseract) -> list[str]:
    problems = []
    ids = Counter(rec.get("id") for rec in t.records)
    for ident, n in ids.items():
        if n > 1:
            problems.append(f"duplicate id `{ident}`")
    marker_ids = {r.ident for r in t.regions}
    for rec in t.records:
        ident = rec.get("id", "?")
        if rec.get("level") not in LEVELS:
            problems.append(f"`{ident}`: level `{rec.get('level')}` not in {LEVELS}")
        why = rec.get("why") or {}
        if why.get("consequence") not in CONSEQUENCES:
            problems.append(
                f"`{ident}`: consequence `{why.get('consequence')}` unknown"
            )
        if why.get("confidence") not in CONFIDENCES:
            problems.append(f"`{ident}`: confidence `{why.get('confidence')}` unknown")
        if not str(why.get("symptom") or "").strip():
            problems.append(f"`{ident}`: no symptom")
        if ident not in marker_ids and not rec.get("location"):
            problems.append(f"`{ident}`: no code markers")
    for ident in sorted(marker_ids - set(ids)):
        problems.append(f"marker `{ident}` has no YAML record")
    return problems


def loc_by_level(t: Tesseract) -> tuple[dict[str, int], dict[str, int], int]:
    """Lines per level (each line counted once, at its highest level), lines
    per intervention id, and total code lines of the tesseract."""
    level_of = {rec.get("id"): rec.get("level") for rec in t.records}
    per_id: dict[str, int] = defaultdict(int)
    line_level: dict[tuple[Path, int], int] = {}
    cache: dict[Path, list[str]] = {}
    for r in t.regions:
        lines = cache.setdefault(
            r.path, r.path.read_text(encoding="utf-8").splitlines()
        )
        lvl = level_of.get(r.ident)
        rank = LEVELS.index(lvl) if lvl in LEVELS else -1
        for n in range(r.start + 1, r.end):
            if _is_code_line(lines[n - 1]):
                per_id[r.ident] += 1
                key = (r.path, n)
                line_level[key] = max(line_level.get(key, -1), rank)
    by_level = Counter(LEVELS[k] if k >= 0 else "?" for k in line_level.values())
    total = sum(
        sum(_is_code_line(x) for x in p.read_text(encoding="utf-8").splitlines())
        for p in t.files
        if p.suffix in SOURCE_SUFFIXES
    )
    return dict(by_level), dict(per_id), total


# ── Heuristic L3 lint ────────────────────────────────────────────────────────


@dataclass
class Hit:
    path: Path
    line: int
    rule: str
    text: str


def _imported_names(tree: ast.AST) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            for a in node.names:
                names.add(a.asname or a.name)
    return names


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def lint_python(path: Path) -> list[Hit]:
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    lines = src.splitlines()
    imported = _imported_names(tree)
    hits: list[Hit] = []

    def add(node: ast.AST, rule: str) -> None:
        hits.append(Hit(path, node.lineno, rule, lines[node.lineno - 1].strip()))

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for tgt in targets:
                if isinstance(tgt, ast.Attribute) and _root_name(tgt) in imported:
                    add(node, "assigns attribute of imported name")
        elif isinstance(node, ast.Call):
            f = node.func
            if (
                isinstance(f, ast.Name)
                and f.id == "setattr"
                and node.args
                and _root_name(node.args[0]) in imported
            ):
                add(node, "setattr on imported name")
        elif isinstance(node, ast.ImportFrom) and node.module:
            mod_private = any(
                p.startswith("_") and not p.startswith("__")
                for p in node.module.split(".")
            )
            name_private = any(
                a.name.startswith("_") and not a.name.startswith("__")
                for a in node.names
            )
            if mod_private or name_private:
                add(node, "imports private name")
        elif isinstance(node, ast.Import):
            if any(
                p.startswith("_") and not p.startswith("__")
                for a in node.names
                for p in a.name.split(".")[1:]
            ):
                add(node, "imports private module")
        elif isinstance(node, ast.Attribute):
            if (
                node.attr.startswith("_")
                and not node.attr.startswith("__")
                and _root_name(node) in imported
            ):
                add(node, "uses private attribute of imported name")
    return hits


JULIA_METHOD_EXT = re.compile(
    r"^\s*(?:function\s+[A-Z]\w*(?:\.[A-Za-z_]\w*)*\.\w+!?\s*\(|[A-Z]\w*(?:\.\w+)*\.\w+!?\(.*\)\s*=[^=])"
)
BUILD_PATCH = re.compile(r"\b(sed\s+-i|patch\s+-p|git\s+apply|perl\s+-pi)\b")


def lint_text(path: Path) -> list[Hit]:
    hits = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if path.suffix == ".jl" and JULIA_METHOD_EXT.search(line):
            hits.append(
                Hit(path, i, "extends a method of another module", line.strip())
            )
        if path.name in CONFIG_NAMES and BUILD_PATCH.search(line):
            hits.append(Hit(path, i, "patches files at build time", line.strip()[:120]))
    return hits


def lint(t: Tesseract) -> list[tuple[Hit, str]]:
    """Return lint hits with where they fall: L3 region, other region, or none."""
    level_of = {rec.get("id"): rec.get("level") for rec in t.records}
    out = []
    for f in t.files:
        hits = lint_python(f) if f.suffix == ".py" else lint_text(f)
        for h in hits:
            covering = [
                level_of.get(r.ident)
                for r in t.regions
                if r.path == h.path and r.start < h.line < r.end
            ]
            where = (
                "L3 region"
                if "L3" in covering
                else (
                    f"{max(covering, key=lambda x: str(x))} region"
                    if covering
                    else "unannotated"
                )
            )
            out.append((h, where))
    return out


# ── Report ───────────────────────────────────────────────────────────────────


def _one_line(text: object, n: int = 110) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def report(tesseracts: list[Tesseract], global_problems: list[str]) -> str:
    out = ["# Intervention annotations: spike report", ""]

    # Summary per tesseract.
    out += [
        "## Summary per tesseract",
        "",
        (
            "Lines are non-blank, non-comment lines inside marker regions; a line "
            "inside nested regions counts once, at the highest level."
        ),
        "",
        "| Tesseract | Records | L0 | L1 | L2 | L3 | Highest | Lines L2 | Lines L3 | Total lines |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    totals = Counter()
    for t in tesseracts:
        levels = Counter(rec.get("level") for rec in t.records)
        totals.update(levels)
        by_level, _, total = loc_by_level(t)
        highest = max(
            (lv for lv in levels if lv in LEVELS), key=LEVELS.index, default="—"
        )
        out.append(
            f"| {t.name} | {len(t.records)} | "
            + " | ".join(str(levels.get(lv, 0)) for lv in LEVELS)
            + f" | {highest} | {by_level.get('L2', 0)} | {by_level.get('L3', 0)} | {total} |"
        )
    out += [
        "",
        "Totals: " + ", ".join(f"{lv} {totals.get(lv, 0)}" for lv in LEVELS),
        "",
    ]

    # Distributions.
    records = [rec for t in tesseracts for rec in t.records]
    conf = Counter((rec.get("why") or {}).get("confidence") for rec in records)
    cons = Counter((rec.get("why") or {}).get("consequence") for rec in records)
    out += [
        '## How well the "why" is substantiated',
        "",
        "| Confidence | Records |",
        "|---|---:|",
        *[f"| {c} | {conf.get(c, 0)} |" for c in CONFIDENCES],
        "",
        "| Consequence without the intervention | Records |",
        "|---|---:|",
        *[f"| {c} | {n} |" for c, n in cons.most_common()],
        "",
    ]

    # All records.
    out += [
        "## All interventions",
        "",
        "| Tesseract | Id | Level | Consequence | Confidence | Lines | What |",
        "|---|---|---|---|---|---:|---|",
    ]
    for t in tesseracts:
        _, per_id, _ = loc_by_level(t)
        for rec in t.records:
            why = rec.get("why") or {}
            level = rec.get("level", "?") + (
                " (method)" if rec.get("method_change") else ""
            )
            out.append(
                f"| {t.name} | `{rec.get('id')}` | {level} | {why.get('consequence')} | "
                f"{why.get('confidence')} | {per_id.get(rec.get('id'), 0)} | "
                f"{_one_line(rec.get('what'))} |"
            )
    out.append("")

    # Classification notes.
    noted = [
        (t, rec)
        for t in tesseracts
        for rec in t.records
        if rec.get("classification_notes")
    ]
    if noted:
        out += ["## Classification notes", ""]
        for t, rec in noted:
            out.append(
                f"- **{t.name}** `{rec.get('id')}` ({rec.get('level')}): "
                f"{_one_line(rec.get('classification_notes'), 400)}"
            )
        out.append("")

    # Lint.
    out += [
        "## Heuristic L3 lint",
        "",
        "| Tesseract | Location | Rule | Falls in | Code |",
        "|---|---|---|---|---|",
    ]
    lint_summary = Counter()
    for t in tesseracts:
        for h, where in lint(t):
            lint_summary[where] += 1
            code = _one_line(h.text, 80).replace("|", "\\|")
            out.append(
                f"| {t.name} | `{_rel(h.path)}:{h.line}` | {h.rule} | {where} | `{code}` |"
            )
    l3_records = [
        (t, rec) for t in tesseracts for rec in t.records if rec.get("level") == "L3"
    ]
    caught = sum(
        1
        for t, rec in l3_records
        if any(
            where == "L3 region"
            and any(
                r.ident == rec.get("id")
                and r.path == h.path
                and r.start < h.line < r.end
                for r in t.regions
            )
            for h, where in lint(t)
        )
    )
    out += [
        "",
        "Hits by location: "
        + ", ".join(f"{k} {v}" for k, v in lint_summary.most_common()),
        "",
        f"L3 records with at least one lint hit inside their region: {caught} of {len(l3_records)}",
        "",
    ]

    # Consistency.
    problems = [(t.name, p) for t in tesseracts for p in t.problems]
    out += ["## Consistency problems", ""]
    if not problems and not global_problems:
        out.append("None.")
    out += [f"- **{name}**: {p}" for name, p in problems]
    out += [f"- {p}" for p in global_problems]
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path, help="write the report here instead of stdout"
    )
    args = parser.parse_args(argv)
    tesseracts, global_problems = load()
    text = report(tesseracts, global_problems)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
