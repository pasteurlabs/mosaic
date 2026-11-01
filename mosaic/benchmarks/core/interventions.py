# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Intervention records: what it took to run each solver, and why.

Each tesseract records, under ``metadata.mosaic`` in its
``tesseract_config.yaml``:

* ``abstraction``: the solver's abstraction level, which sets the baseline of
  ordinary use (``application``, ``framework`` or ``kernel-toolkit``);
* ``interventions``: everything beyond calling the solver as its baseline
  intends, each with a level (L0 documented use, L1 undocumented
  configuration, L2 user code on public primitives, L3 changing the solver),
  a necessity, and per setup what a user who skips it would see;
* ``deviations``: where the solver runs a substitute for the prescribed setup.

Example::

    metadata:
      mosaic:
        abstraction: framework
        interventions:
          - id: dense-pressure-solver
            level: L3
            necessity: required
            what: Replaces the FFT pressure solve with a cosine transform.
            why:
              ns-2d-periodic:
                consequences: [no-gradient]
                symptom: Enzyme has no rules for the FFTW calls.
                evidence: https://github.com/...
            upstream: https://github.com/...
            provenance: mosaic-team
        deviations: []

L2 and L3 interventions carry ``mosaic-intervention: <id>`` and
``mosaic-intervention-end: <id>`` comments around their code in the
tesseract's directory. :func:`check_tesseract` validates a record and its
markers; the test suite runs it on every tesseract.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ABSTRACTIONS = ("application", "framework", "kernel-toolkit")

# L4 (fixes that exist only in unreleased upstream code) is not accepted, so
# a record cannot use it.
LEVELS = ("L0", "L1", "L2", "L3")
LEVEL_NAMES = {
    "L0": "documented use",
    "L1": "undocumented configuration",
    "L2": "user code on public primitives",
    "L3": "changing the solver",
}
MARKED_LEVELS = frozenset({"L2", "L3"})

NECESSITIES = ("required", "choice", "unknown")

CONSEQUENCES = (
    "cannot-express",
    "crash",
    "non-finite",
    "inaccurate-forward",
    "no-gradient",
    "incorrect-gradient",
    "optimizer-stall",
    "too-slow",
    "out-of-memory",
    "non-reproducible",
)

PROVENANCES = (
    "mosaic-team",
    "contributor",
    "maintainer-authored",
    "maintainer-advised",
)


@dataclass(frozen=True)
class Setup:
    """One problem statement that results are reported for."""

    key: str
    problem: str  # Problem.name, e.g. "ns-grid"
    label: str


SETUPS: dict[str, Setup] = {
    s.key: s
    for s in (
        Setup("ns-2d-periodic", "ns-grid", "Periodic box (2D)"),
        Setup("ns-2d-cylinder", "ns-grid", "Channel flow past a cylinder"),
        Setup("ns-3d", "ns-3d-grid", "Triply periodic box (3D)"),
        Setup("structural", "structural-mesh", "Linear-elastic cantilever"),
        Setup("heat", "thermal-mesh", "Steady conduction in a slab"),
    )
}

# Setups each tesseract directory under mosaic/tesseracts/ can serve.
DOMAIN_SETUPS: dict[str, tuple[str, ...]] = {
    "navier-stokes-grid": ("ns-2d-periodic", "ns-2d-cylinder", "ns-3d"),
    "structural-mesh": ("structural",),
    "thermal-mesh": ("heat",),
}

_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_MARKER = re.compile(r"(?:#|//)\s*mosaic-intervention(-end)?:\s*(\S+)")
# Files scanned for markers: sources, build scripts and the tesseract config
# (build steps live there).
_MARKER_SUFFIXES = frozenset(
    {".py", ".jl", ".cpp", ".cc", ".c", ".h", ".hpp", ".sh", ".yaml", ".yml"}
)

_RECORD_KEYS = ("abstraction", "interventions", "deviations")
_INTERVENTION_KEYS = frozenset(
    {
        "id",
        "level",
        "necessity",
        "tuned",
        "active",
        "what",
        "why",
        "upstream",
        "provenance",
    }
)
_REASON_KEYS = frozenset({"consequences", "symptom", "evidence"})
_DEVIATION_KEYS = frozenset(
    {"id", "setups", "what", "consequences", "effect", "evidence"}
)


@dataclass(frozen=True)
class SetupReason:
    """Why an intervention exists on one setup."""

    consequences: tuple[str, ...]
    symptom: str
    evidence: str = ""


@dataclass(frozen=True)
class Intervention:
    """One thing done beyond the solver's baseline use."""

    id: str
    level: str
    necessity: str
    what: str
    why: dict[str, SetupReason]
    provenance: str
    tuned: bool = False
    active: bool = True
    upstream: str = ""

    @property
    def setups(self) -> tuple[str, ...]:
        """Setups this intervention has a reason on."""
        return tuple(self.why)

    @property
    def reason_unknown(self) -> bool:
        """Whether nobody knows why it is needed."""
        return self.necessity == "unknown"


@dataclass(frozen=True)
class Deviation:
    """A documented substitute for part of the prescribed setup."""

    id: str
    setups: tuple[str, ...]
    what: str
    consequences: tuple[str, ...]
    effect: str
    evidence: str = ""


@dataclass(frozen=True)
class InterventionRecord:
    """A solver's abstraction level, interventions and setup deviations."""

    abstraction: str
    interventions: tuple[Intervention, ...] = ()
    deviations: tuple[Deviation, ...] = ()

    def interventions_for(self, setup: str) -> list[Intervention]:
        """Active interventions with a reason on ``setup``."""
        return [i for i in self.interventions if i.active and setup in i.why]

    def deviations_for(self, setup: str) -> list[Deviation]:
        """Deviations that apply to ``setup``."""
        return [d for d in self.deviations if setup in d.setups]

    def effort(self, setup: str) -> str | None:
        """Highest level among the required interventions on ``setup``.

        ``None`` when nothing beyond baseline use is required there.
        """
        levels = [
            i.level for i in self.interventions_for(setup) if i.necessity == "required"
        ]
        return max(levels, key=LEVELS.index) if levels else None


@dataclass
class _Problems:
    """Collects validation messages with a location prefix."""

    where: str
    items: list[str] = field(default_factory=list)

    def add(self, msg: str) -> None:
        self.items.append(f"{self.where}: {msg}")


def _text(value: Any) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def _unknown_keys(obj: dict, allowed: frozenset[str], p: _Problems, what: str) -> None:
    for key in sorted(set(obj) - allowed):
        p.add(f"{what}: unknown key `{key}`")


def _parse_consequences(value: Any, p: _Problems, what: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        p.add(f"{what}: `consequences` must be a list")
        return ()
    for c in value:
        if c not in CONSEQUENCES:
            p.add(f"{what}: consequence `{c}` not in {list(CONSEQUENCES)}")
    return tuple(c for c in value if c in CONSEQUENCES)


def _parse_intervention(
    raw: Any, setups: tuple[str, ...] | None, p: _Problems
) -> Intervention | None:
    if not isinstance(raw, dict):
        p.add("every intervention must be a mapping")
        return None
    ident = raw.get("id")
    what = f"intervention `{ident}`"
    if not isinstance(ident, str) or not _ID.match(ident):
        p.add(f"{what}: `id` must be lower-case kebab-case")
    _unknown_keys(raw, _INTERVENTION_KEYS, p, what)

    level = raw.get("level")
    if level == "L4":
        p.add(f"{what}: L4 (unreleased upstream help) is not accepted")
    elif level not in LEVELS:
        p.add(f"{what}: level `{level}` not in {list(LEVELS)}")
    necessity = raw.get("necessity")
    if necessity not in NECESSITIES:
        p.add(f"{what}: necessity `{necessity}` not in {list(NECESSITIES)}")
    provenance = raw.get("provenance")
    if provenance not in PROVENANCES:
        p.add(f"{what}: provenance `{provenance}` not in {list(PROVENANCES)}")
    for flag in ("tuned", "active"):
        if flag in raw and not isinstance(raw[flag], bool):
            p.add(f"{what}: `{flag}` must be true or false")
    tuned = raw.get("tuned") is True
    if not _text(raw.get("what")):
        p.add(f"{what}: `what` is empty")
    upstream = raw.get("upstream")
    if upstream is not None and not _text(upstream):
        p.add(f"{what}: `upstream` must be a link or a reason, or be left out")

    why_raw = raw.get("why")
    why: dict[str, SetupReason] = {}
    if not isinstance(why_raw, dict) or not why_raw:
        p.add(f"{what}: `why` must map at least one setup to its reason")
        why_raw = {}
    for setup, reason in why_raw.items():
        where = f"{what}, setup `{setup}`"
        if setups is not None and setup not in setups:
            p.add(f"{where}: unknown setup (expected one of {list(setups)})")
        if not isinstance(reason, dict):
            p.add(f"{where}: reason must be a mapping")
            continue
        _unknown_keys(reason, _REASON_KEYS, p, where)
        consequences = _parse_consequences(reason.get("consequences"), p, where)
        # A choice may have no measurable consequence; the symptom says so.
        if not consequences and necessity == "required":
            p.add(f"{where}: `consequences` is empty but necessity is `required`")
        symptom = _text(reason.get("symptom"))
        if not symptom:
            p.add(f"{where}: `symptom` is empty")
        evidence = _text(reason.get("evidence"))
        # Above L0, a stated necessity needs evidence (rule 5); a tuned value
        # needs the measurement it was tuned on.
        needs_evidence = tuned or (level != "L0" and necessity != "unknown")
        if needs_evidence and not evidence:
            p.add(f"{where}: `evidence` is empty")
        why[str(setup)] = SetupReason(consequences, symptom, evidence)

    return Intervention(
        id=str(ident),
        level=str(level),
        necessity=str(necessity),
        what=_text(raw.get("what")),
        why=why,
        provenance=str(provenance),
        tuned=tuned,
        active=raw.get("active") is not False,
        upstream=_text(upstream),
    )


def _parse_deviation(
    raw: Any, setups: tuple[str, ...] | None, p: _Problems
) -> Deviation | None:
    if not isinstance(raw, dict):
        p.add("every deviation must be a mapping")
        return None
    ident = raw.get("id")
    what = f"deviation `{ident}`"
    if not isinstance(ident, str) or not _ID.match(ident):
        p.add(f"{what}: `id` must be lower-case kebab-case")
    _unknown_keys(raw, _DEVIATION_KEYS, p, what)
    dev_setups = raw.get("setups")
    if not isinstance(dev_setups, list) or not dev_setups:
        p.add(f"{what}: `setups` must list at least one setup")
        dev_setups = []
    for s in dev_setups:
        if setups is not None and s not in setups:
            p.add(f"{what}: unknown setup `{s}` (expected one of {list(setups)})")
    if not _text(raw.get("what")):
        p.add(f"{what}: `what` is empty")
    consequences = _parse_consequences(raw.get("consequences"), p, what)
    if not consequences:
        p.add(f"{what}: `consequences` is empty")
    if not _text(raw.get("effect")):
        p.add(f"{what}: `effect` is empty (say 'Not measured.' if unknown)")
    return Deviation(
        id=str(ident),
        setups=tuple(str(s) for s in dev_setups),
        what=_text(raw.get("what")),
        consequences=consequences,
        effect=_text(raw.get("effect")),
        evidence=_text(raw.get("evidence")),
    )


def parse_record(
    meta: dict, setups: tuple[str, ...] | None = None, where: str = "record"
) -> tuple[InterventionRecord | None, list[str]]:
    """Parse the intervention record from a ``metadata.mosaic`` block.

    Args:
        meta: The ``metadata.mosaic`` mapping.
        setups: Setup keys the reasons and deviations may name; ``None``
            skips that check.
        where: Prefix for the returned messages.

    Returns:
        ``(record, problems)``. ``record`` is ``None`` when the block has no
        record (no ``abstraction``, ``interventions`` or ``deviations`` key)
        or when ``problems`` is not empty.
    """
    p = _Problems(where)
    if not any(k in meta for k in _RECORD_KEYS):
        return None, []

    abstraction = meta.get("abstraction")
    if abstraction not in ABSTRACTIONS:
        p.add(f"abstraction `{abstraction}` not in {list(ABSTRACTIONS)}")

    interventions: list[Intervention] = []
    raw_list = meta.get("interventions", [])
    if not isinstance(raw_list, list):
        p.add("`interventions` must be a list")
        raw_list = []
    for raw in raw_list:
        parsed = _parse_intervention(raw, setups, p)
        if parsed is not None:
            interventions.append(parsed)

    deviations: list[Deviation] = []
    raw_list = meta.get("deviations", [])
    if not isinstance(raw_list, list):
        p.add("`deviations` must be a list")
        raw_list = []
    for raw in raw_list:
        parsed = _parse_deviation(raw, setups, p)
        if parsed is not None:
            deviations.append(parsed)

    counts: dict[str, int] = defaultdict(int)
    for item in (*interventions, *deviations):
        counts[item.id] += 1
    for ident, n in sorted(counts.items()):
        if n > 1:
            p.add(f"id `{ident}` used {n} times")

    if p.items:
        return None, p.items
    record = InterventionRecord(
        abstraction=str(abstraction),
        interventions=tuple(interventions),
        deviations=tuple(deviations),
    )
    return record, []


# ── Code markers ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MarkerRegion:
    """One ``mosaic-intervention`` begin/end pair."""

    id: str
    path: Path
    start: int  # 1-based line of the begin marker
    end: int  # 1-based line of the end marker


def scan_markers(directory: Path) -> tuple[list[MarkerRegion], list[str]]:
    """Find the marker regions under ``directory`` and any unpaired markers."""
    regions: list[MarkerRegion] = []
    problems: list[str] = []
    for path in sorted(directory.rglob("*")):
        if (
            not path.is_file()
            or path.suffix not in _MARKER_SUFFIXES
            or "__pycache__" in path.parts
        ):
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        open_: dict[str, list[int]] = defaultdict(list)
        for n, line in enumerate(lines, 1):
            m = _MARKER.search(line)
            if not m:
                continue
            is_end, ident = bool(m.group(1)), m.group(2)
            if not is_end:
                open_[ident].append(n)
            elif open_[ident]:
                regions.append(MarkerRegion(ident, path, open_[ident].pop(), n))
            else:
                problems.append(f"{path}:{n}: end marker `{ident}` without a begin")
        for ident, starts in open_.items():
            for n in starts:
                problems.append(f"{path}:{n}: marker `{ident}` is never closed")
    return regions, problems


def check_markers(
    record: InterventionRecord | None, directory: Path, where: str
) -> list[str]:
    """Check that markers and L2/L3 interventions match one to one."""
    regions, problems = scan_markers(directory)
    marked = {r.id for r in regions}
    known = {i.id for i in record.interventions} if record else set()
    for ident in sorted(marked - known):
        problems.append(f"{where}: marker `{ident}` has no intervention record")
    if record:
        for i in record.interventions:
            if i.level in MARKED_LEVELS and i.id not in marked:
                problems.append(
                    f"{where}: {i.level} intervention `{i.id}` has no "
                    "`mosaic-intervention` markers"
                )
    return problems


# ── Loading ──────────────────────────────────────────────────────────────────


def load_meta(tesseract_dir: Path) -> dict:
    """Return the ``metadata.mosaic`` block of a tesseract, or ``{}``."""
    import yaml

    doc = yaml.safe_load((tesseract_dir / "tesseract_config.yaml").read_text("utf-8"))
    meta = ((doc or {}).get("metadata") or {}).get("mosaic")
    return meta if isinstance(meta, dict) else {}


def check_tesseract(tesseract_dir: Path) -> tuple[InterventionRecord | None, list[str]]:
    """Validate one tesseract's record and markers.

    Returns:
        ``(record, problems)``; ``record`` is ``None`` when there is none or
        it is invalid.
    """
    where = f"{tesseract_dir.parent.name}/{tesseract_dir.name}"
    setups = DOMAIN_SETUPS.get(tesseract_dir.parent.name)
    record, problems = parse_record(load_meta(tesseract_dir), setups, where)
    if not problems:
        problems = check_markers(record, tesseract_dir, where)
    return record, problems
