# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Count bespoke lines of code per solver wrapper, for Table 3's effort columns.

Each Tesseract wrapper tags its code with ``mosaic:<category>`` comments:

* Python (``tesseract_api.py``): a tag on the signature of each top-level
  function or class, e.g. ``def build_mesh(...):  # mosaic:init``. The schemas,
  the Tesseract entry points, imports, and module-level statements are glue and
  are not counted.
* Julia and C/C++/CUDA files inside the wrapper: section markers
  (``# mosaic:<category>`` / ``// mosaic:<category>``) that apply to every
  following line until the next marker.

Categories map to the table's columns: ``io`` → IO, ``init`` → Conf., and
``physics``, ``grad`` and ``util`` → Num. Untagged code counts as ``unknown``
(also Num.).

Usage:
    uv run paper/count_loc.py
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

TESSERACTS = Path(__file__).resolve().parents[1] / "mosaic" / "tesseracts"

GLUE_CLASSES = {"InputSchema", "OutputSchema"}
GLUE_FUNCTIONS = {"apply", "vjp_jit", "jvp_jit", "abstract_eval"}
CATEGORIES = {"physics", "io", "init", "grad", "util"}
EXTERNAL_SUFFIXES = {".cpp", ".cu", ".c", ".h", ".hpp", ".jl"}
SKIP_DIRS = {"build", "__pycache__", ".git", "node_modules", ".tesseract_build"}
COLUMNS = {"io": "IO", "init": "Conf"}  # every other category counts as Num

_TAG = r"mosaic:(\w+)"


def _tag(lines: list[str], first: int, last: int) -> str:
    """Category tagged on lines first..last (1-based, inclusive), or 'unknown'."""
    for line in lines[first - 1 : last]:
        if (m := re.search(r"#\s*" + _TAG, line)) and m.group(1) in CATEGORIES:
            return m.group(1)
    return "unknown"


def count_python(path: Path) -> Counter:
    """Lines per category of the top-level functions and classes of a wrapper."""
    lines = path.read_text(encoding="utf-8").splitlines()
    counts: Counter = Counter()
    for node in ast.parse("\n".join(lines)).body:
        if isinstance(node, ast.ClassDef):
            glue = GLUE_CLASSES
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            glue = GLUE_FUNCTIONS
        else:
            continue
        if node.name in glue:
            continue
        # The tag may sit on any line of a multi-line signature.
        header_end = node.body[0].lineno - 1 if node.body else node.lineno
        span = node.end_lineno - node.lineno + 1
        counts[_tag(lines, node.lineno, header_end)] += span
    return counts


def count_external(path: Path) -> Counter:
    """Lines per category of a Julia or C/C++/CUDA source, by section markers."""
    marker = re.compile(("#" if path.suffix == ".jl" else "//") + r"\s*" + _TAG)
    text = path.read_text(encoding="utf-8", errors="replace")
    counts: Counter = Counter()
    current = "unknown"
    for line in text.splitlines():
        if m := marker.search(line):
            current = m.group(1) if m.group(1) in CATEGORIES else "unknown"
        counts[current] += 1
    return counts


def count_wrapper(wrapper: Path) -> dict[str, int]:
    """IO / Num / Conf line counts of one wrapper directory."""
    counts: Counter = Counter()
    if (api := wrapper / "tesseract_api.py").exists():
        counts += count_python(api)
    for path in sorted(wrapper.rglob("*")):
        if (
            path.is_file()
            and path.suffix in EXTERNAL_SUFFIXES
            and not SKIP_DIRS.intersection(path.parts)
        ):
            counts += count_external(path)
    columns = {"IO": 0, "Num": 0, "Conf": 0, "unknown": counts["unknown"]}
    for cat, n in counts.items():
        columns[COLUMNS.get(cat, "Num")] += n
    return columns


def line_counts() -> dict[tuple[str, str], dict[str, int]]:
    """Line counts keyed by (domain directory, wrapper directory)."""
    return {
        (domain.name, wrapper.name): count_wrapper(wrapper)
        for domain in sorted(p for p in TESSERACTS.iterdir() if p.is_dir())
        for wrapper in sorted(p for p in domain.iterdir() if p.is_dir())
    }


def main() -> None:
    """Print the line counts of every wrapper."""
    print(
        f"{'domain':20} {'wrapper':24} {'IO':>5} {'Num':>5} {'Conf':>5} {'unknown':>8}"
    )
    for (domain, wrapper), c in line_counts().items():
        print(
            f"{domain:20} {wrapper:24} {c['IO']:5} {c['Num']:5} {c['Conf']:5} {c['unknown']:8}"
        )


if __name__ == "__main__":
    main()
