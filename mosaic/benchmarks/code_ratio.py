"""Measures solver-specific vs. boilerplate code ratio in tesseract implementations.

Classifies top-level AST nodes in each tesseract_api.py:
  boilerplate — imports, InputSchema/OutputSchema, apply/vjp_jit/jvp_jit/abstract_eval,
                module-level assignments and constants
  solver code — all other functions, classes, and expressions

In-repo external source files (.cpp, .cu, .h, .hpp, .jl) are counted separately
and added to the solver bucket.  tesseract_shared Julia files referenced by a tesseract
are detected by scanning the source for known filenames and attributed to that
solver (labelled "shared:").

Tiers:
  1  pure Python, all physics in tesseract_api.py
  2  Python + in-repo C++/CUDA/Julia (local src/ or tesseract_shared)
  3  Python wrapper around an external dep (Julia package, OpenFOAM binary, …)
     — physics is fetched at image build time, not present in this repo

Primary metric: bp% = py_boilerplate / py_total
  "What fraction of this tesseract's own Python file is interface glue?"
  Independent of solver complexity.

Secondary metric: ratio = total_solver / py_boilerplate
  Lines of solver code per line of boilerplate.  Confounded by solver size.

Caveat: apply() and vjp_jit() bodies are counted as boilerplate because the AST
analysis cannot distinguish solver calls inside them from interface glue.
"""

from __future__ import annotations

import ast
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

INTERFACE_CLASSES = {"InputSchema", "OutputSchema"}
INTERFACE_FUNCTIONS = {"apply", "vjp_jit", "jvp_jit", "abstract_eval"}
EXTERNAL_EXTS = {".cpp", ".cu", ".c", ".h", ".hpp", ".jl"}
SKIP_DIRS = {"build", "__pycache__", ".git", "node_modules", ".tesseract_build"}

EXTERNAL_MARKERS = [
    "incompressiblenavierstokes",
    "julialang-s3",
    "openfoam",
    "su2",
]


VALID_CATEGORIES = {"physics", "io", "init", "grad", "util"}
VALID_GRAD_TYPES = {"autodiff", "adjoint", "analytic", "zero"}


@dataclass
class SolverStats:
    problem: str
    solver: str
    tier: int
    py_total: int = 0
    py_imports: int = 0  # import / from-import lines — declarations, not authored logic
    py_interface: int = 0  # tesseract-contract glue (InputSchema, apply, vjp_jit, …)
    py_solver: int = 0  # everything else — would exist without tesseracts
    ext_solver: int = 0  # in-repo C++/CUDA/Julia (local src/ or tesseract_shared)
    ext_files: list[str] = field(default_factory=list)
    solver_by_category: dict[str, int] = field(default_factory=dict)
    ext_by_category: dict[str, int] = field(default_factory=dict)
    # grad_by_variable: lines attributed to each differentiable input variable.
    # A function tagged  # mosaic:grad:v0,viscosity  contributes its line span to
    # both "v0" and "viscosity".  Functions tagged  # mosaic:grad  (no variable)
    # are counted in solver_by_category["grad"] but not here.
    grad_by_variable: dict[str, int] = field(default_factory=dict)
    # grad_variable_type: gradient implementation strategy per variable.
    # Populated from  # mosaic:grad:<vars>:<type>  tags where type ∈
    # {autodiff, adjoint, analytic, zero}.
    grad_variable_type: dict[str, str] = field(default_factory=dict)
    note: str = ""

    @property
    def py_written(self) -> int:
        """Lines of actual logic authored in the Python file (imports excluded)."""
        return self.py_interface + self.py_solver

    @property
    def total_solver(self) -> int:
        return self.py_solver + self.ext_solver

    @property
    def import_fraction(self) -> float | None:
        """Fraction of the Python file that is pure import declarations."""
        return self.py_imports / self.py_total if self.py_total else None

    @property
    def interface_fraction(self) -> float | None:
        """Fraction of written logic that is tesseract-contract glue."""
        return self.py_interface / self.py_written if self.py_written else None


def _node_span(node: ast.AST) -> int:
    return node.end_lineno - node.lineno + 1  # type: ignore[attr-defined]


def _extract_tag(
    lines: list[str], lineno: int, end_lineno: int | None = None
) -> tuple[str, list[str], str | None]:
    """Return (category, variables, grad_type) parsed from a tag.

    Tag format: # mosaic:<cat>[:<var,...>[:<type>]]

    Scans the closed range [lineno, end_lineno] (1-based) so the tag may live
    on any line of a multi-line ``def`` / ``class`` signature, not just the
    keyword line. ``end_lineno`` defaults to ``lineno`` (single-line case).

    Examples:
      # mosaic:grad                       → ("grad", [], None)
      # mosaic:grad:v0                    → ("grad", ["v0"], None)
      # mosaic:grad:rho,source            → ("grad", ["rho", "source"], None)
      # mosaic:grad:v0,viscosity:autodiff → ("grad", ["v0", "viscosity"], "autodiff")
    """
    end = end_lineno if end_lineno is not None else lineno
    for i in range(lineno - 1, end):
        m = re.search(r"#\s*mosaic:(\w+)(?::([^\s:]+)(?::(\w+))?)?", lines[i])
        if m:
            cat = m.group(1)
            if cat in VALID_CATEGORIES:
                vars_raw = m.group(2)
                type_raw = m.group(3)
                variables = [v for v in vars_raw.split(",") if v] if vars_raw else []
                grad_type = type_raw if type_raw in VALID_GRAD_TYPES else None
                return cat, variables, grad_type
    return "unknown", [], None


def _inline_grad_vars(
    lines: list[str], start: int, end: int
) -> tuple[dict[str, int], dict[str, str]]:
    """Scan lines[start:end] for standalone ``# mosaic:grad:<vars>[:<type>]`` markers.

    A standalone marker is a line where the only non-whitespace content is the
    comment (i.e. ``^\\s*#``), distinguishing it from a function-level tag which
    appears after code on the ``def`` line.

    Each marker starts a section; subsequent lines are attributed to its
    variable(s) until the next marker.  Lines before the first marker (shared
    setup code) are not attributed to any variable.

    Returns (var→lines, var→type) when at least one inline marker is found, or
    ({}, {}) to signal "no inline markers, use function-level tag".
    """
    pat = re.compile(r"^\s*#\s*mosaic:grad:([^\s:]+)(?::(\w+))?")
    counts: dict[str, int] = {}
    types: dict[str, str] = {}
    current_vars: list[str] = []
    has_markers = False

    for i in range(start, end):
        m = pat.match(lines[i])
        if m:
            has_markers = True
            current_vars = [v for v in m.group(1).split(",") if v]
            type_raw = m.group(2)
            if type_raw in VALID_GRAD_TYPES:
                for var in current_vars:
                    types[var] = type_raw
        if current_vars:
            for var in current_vars:
                counts[var] = counts.get(var, 0) + 1

    if not has_markers:
        return {}, {}
    return counts, types


def _record_grad_vars(
    grad_by_variable: dict[str, int],
    grad_variable_type: dict[str, str],
    variables: list[str],
    gtype: str | None,
    span: int,
) -> None:
    """Attribute ``span`` lines to each variable and record ``gtype`` if given."""
    for var in variables:
        grad_by_variable[var] = grad_by_variable.get(var, 0) + span
    if gtype:
        for var in variables:
            grad_variable_type.setdefault(var, gtype)


def _record_inline_grad(
    grad_by_variable: dict[str, int],
    grad_variable_type: dict[str, str],
    inline_counts: dict[str, int],
    inline_types: dict[str, str],
) -> None:
    """Merge per-variable inline counts and types into the running totals."""
    for var, cnt in inline_counts.items():
        grad_by_variable[var] = grad_by_variable.get(var, 0) + cnt
    for var, t in inline_types.items():
        grad_variable_type.setdefault(var, t)


def _classify_class(
    node: ast.ClassDef,
    lines: list[str],
    solver_by_category: dict[str, int],
    grad_by_variable: dict[str, int],
    grad_variable_type: dict[str, str],
) -> tuple[int, int]:
    """Classify a top-level class node.

    Returns (interface_delta, solver_delta) and mutates the dicts in place for
    any gradient attribution.
    """
    span = _node_span(node)
    if node.name in INTERFACE_CLASSES:
        return span, 0
    # The tag may live on any line of a multi-line class header;
    # bound the scan at the first body element.
    header_end = node.body[0].lineno - 1 if node.body else node.lineno
    cat, variables, gtype = _extract_tag(lines, node.lineno, header_end)
    solver_by_category[cat] = solver_by_category.get(cat, 0) + span
    if cat == "grad" and variables:
        _record_grad_vars(grad_by_variable, grad_variable_type, variables, gtype, span)
    return 0, span


def _classify_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    lines: list[str],
    solver_by_category: dict[str, int],
    grad_by_variable: dict[str, int],
    grad_variable_type: dict[str, str],
) -> tuple[int, int]:
    """Classify a top-level function node.

    Returns (interface_delta, solver_delta) and mutates the dicts in place for
    any gradient attribution.
    """
    span = _node_span(node)
    if node.name in INTERFACE_FUNCTIONS:
        return span, 0
    header_end = node.body[0].lineno - 1 if node.body else node.lineno
    cat, variables, gtype = _extract_tag(lines, node.lineno, header_end)
    solver_by_category[cat] = solver_by_category.get(cat, 0) + span
    if cat == "grad":
        # Prefer fine-grained inline section markers; fall back to
        # the function-level tag if none are present.
        inline_counts, inline_types = _inline_grad_vars(
            lines,
            node.lineno - 1,
            node.end_lineno,  # type: ignore[attr-defined]
        )
        if inline_counts:
            _record_inline_grad(
                grad_by_variable, grad_variable_type, inline_counts, inline_types
            )
        elif variables:
            _record_grad_vars(
                grad_by_variable, grad_variable_type, variables, gtype, span
            )
    return 0, span


def classify_python(
    path: Path,
) -> tuple[int, int, int, int, dict[str, int], dict[str, int], dict[str, str]]:
    """Return (total, imports, interface, solver, solver_by_category, grad_by_variable, grad_variable_type).

    imports            — import / from-import declarations
    interface          — tesseract-contract glue: InputSchema, OutputSchema, apply,
                         vjp_jit, jvp_jit, abstract_eval, module-level assignments
    solver             — everything else (would exist without tesseracts)
    solver_by_category — solver lines broken down by # mosaic:<category> tag
    grad_by_variable   — lines attributed to each differentiable input variable.
                         Populated from inline ``# mosaic:grad:<var>`` section
                         markers within function bodies when present; falls back to
                         the function-level tag otherwise.
    grad_variable_type — gradient implementation type per variable (autodiff/adjoint/analytic/zero).
    """
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    total = len(lines) if lines else 1
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return total, total, 0, 0, {}, {}, {}

    imports = 0
    interface = 0
    solver = 0
    solver_by_category: dict[str, int] = {}
    grad_by_variable: dict[str, int] = {}
    grad_variable_type: dict[str, str] = {}

    for node in ast.iter_child_nodes(tree):
        span = _node_span(node)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports += span
        elif isinstance(node, ast.ClassDef):
            i_delta, s_delta = _classify_class(
                node, lines, solver_by_category, grad_by_variable, grad_variable_type
            )
            interface += i_delta
            solver += s_delta
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            i_delta, s_delta = _classify_function(
                node, lines, solver_by_category, grad_by_variable, grad_variable_type
            )
            interface += i_delta
            solver += s_delta
        else:
            # Module-level assignments, constants, bare expressions
            interface += span

    # Blank lines and comments not attached to any node → interface overhead
    interface += total - (imports + interface + solver)
    return (
        total,
        imports,
        interface,
        solver,
        solver_by_category,
        grad_by_variable,
        grad_variable_type,
    )


def _ext_categories(
    text: str, suffix: str
) -> tuple[dict[str, int], dict[str, int], dict[str, str]]:
    """Count lines per category (and per grad variable) in an external source file.

    Recognises ``# mosaic:<cat>[:<vars>[:<type>]]`` (Julia / Python) and
    ``// mosaic:<cat>[:<vars>[:<type>]]`` (C / C++ / CUDA) section markers.
    A marker line sets the active category and optional variable list;
    subsequent lines are attributed to it until the next marker.  Lines before
    any marker → "unknown".

    Returns (cats_by_line, grad_vars_by_line, grad_variable_type).
    """
    if suffix in {".jl"}:
        _pat = re.compile(r"#\s*mosaic:(\w+)(?::([^\s:]+)(?::(\w+))?)?")
    else:  # C/C++/CUDA/header
        _pat = re.compile(r"//\s*mosaic:(\w+)(?::([^\s:]+)(?::(\w+))?)?")

    cats: dict[str, int] = {}
    grad_vars: dict[str, int] = {}
    grad_types: dict[str, str] = {}
    current_cat = "unknown"
    current_vars: list[str] = []
    current_type: str | None = None
    for line in text.splitlines():
        m = _pat.search(line)
        if m:
            cat = m.group(1)
            current_cat = cat if cat in VALID_CATEGORIES else "unknown"
            vars_raw = m.group(2)
            type_raw = m.group(3)
            current_vars = [v for v in vars_raw.split(",") if v] if vars_raw else []
            current_type = type_raw if type_raw in VALID_GRAD_TYPES else None
            if current_cat == "grad" and current_type:
                for var in current_vars:
                    grad_types.setdefault(var, current_type)
        cats[current_cat] = cats.get(current_cat, 0) + 1
        if current_cat == "grad" and current_vars:
            for var in current_vars:
                grad_vars[var] = grad_vars.get(var, 0) + 1
    return cats, grad_vars, grad_types


def count_external(
    tesseract_dir: Path,
) -> tuple[int, list[str], dict[str, int], dict[str, int], dict[str, str]]:
    """Count lines in in-repo external source files inside the tesseract dir.

    Returns (total_lines, file_labels, by_category, grad_by_variable, grad_variable_type).
    """
    total = 0
    files = []
    by_category: dict[str, int] = {}
    grad_by_variable: dict[str, int] = {}
    grad_variable_type: dict[str, str] = {}
    for path in sorted(tesseract_dir.rglob("*")):
        if not path.is_file() or path.suffix not in EXTERNAL_EXTS:
            continue
        if any(skip in path.parts for skip in SKIP_DIRS):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        n = text.count("\n") + 1
        total += n
        files.append(f"{path.relative_to(tesseract_dir)}({n})")
        cats, gvars, gtypes = _ext_categories(text, path.suffix)
        for cat, cnt in cats.items():
            by_category[cat] = by_category.get(cat, 0) + cnt
        for var, cnt in gvars.items():
            grad_by_variable[var] = grad_by_variable.get(var, 0) + cnt
        for var, t in gtypes.items():
            grad_variable_type.setdefault(var, t)
    return total, files, by_category, grad_by_variable, grad_variable_type


def find_tesseract_shared_jl(
    api_path: Path, tesseract_shared_root: Path
) -> list[tuple[Path, str]]:
    """Return [(path, label)] for tesseract_shared Julia files referenced by the tesseract.

    Scans the tesseract_api.py source for bare .jl filename strings (e.g.
    "ns_solver.jl") and resolves them against the tesseract_shared directory tree.
    Only files that actually exist in tesseract_shared are returned.
    """
    if not api_path.exists() or not tesseract_shared_root.exists():
        return []

    source = api_path.read_text(encoding="utf-8", errors="replace")
    # Only bother if this file imports tesseract_shared at all
    if "tesseract_shared" not in source:
        return []

    # Find all .jl filename literals in the source
    jl_names = set(re.findall(r'["\'](\w[\w_-]*\.jl)["\']', source))
    if not jl_names:
        return []

    results = []
    for name in sorted(jl_names):
        matches = list(tesseract_shared_root.rglob(name))
        for match in matches:
            label = f"shared:{match.relative_to(tesseract_shared_root)}"
            results.append((match, label))
    return results


def detect_tier(tesseract_dir: Path, has_ext: bool) -> tuple[int, str]:
    if has_ext:
        return 2, ""
    config = tesseract_dir / "tesseract_config.yaml"
    if config.exists():
        text = config.read_text(encoding="utf-8", errors="replace").lower()
        for marker in EXTERNAL_MARKERS:
            if marker in text:
                return 3, f"external:{marker}"
    return 1, ""


def analyse(
    problem: str,
    solver: str,
    tesseract_dir: Path,
    tesseract_shared_root: Path | None = None,
) -> SolverStats:
    ext_lines, ext_files, ext_by_cat, ext_grad_vars, ext_grad_types = count_external(
        tesseract_dir
    )

    # Attribute tesseract_shared Julia files referenced by this tesseract
    if tesseract_shared_root is not None:
        api = tesseract_dir / "tesseract_api.py"
        for jl_path, label in find_tesseract_shared_jl(api, tesseract_shared_root):
            text = jl_path.read_text(encoding="utf-8", errors="replace")
            n = text.count("\n") + 1
            ext_lines += n
            ext_files.append(f"{label}({n})")
            cats, gvars, gtypes = _ext_categories(text, jl_path.suffix)
            for cat, cnt in cats.items():
                ext_by_cat[cat] = ext_by_cat.get(cat, 0) + cnt
            for var, cnt in gvars.items():
                ext_grad_vars[var] = ext_grad_vars.get(var, 0) + cnt
            for var, t in gtypes.items():
                ext_grad_types.setdefault(var, t)

    tier, note = detect_tier(tesseract_dir, ext_lines > 0)
    stats = SolverStats(
        problem=problem,
        solver=solver,
        tier=tier,
        ext_solver=ext_lines,
        ext_files=ext_files,
        ext_by_category=ext_by_cat,
        note=note,
    )
    api = tesseract_dir / "tesseract_api.py"
    if api.exists():
        (
            stats.py_total,
            stats.py_imports,
            stats.py_interface,
            stats.py_solver,
            stats.solver_by_category,
            stats.grad_by_variable,
            stats.grad_variable_type,
        ) = classify_python(api)
    else:
        stats.note = "no tesseract_api.py"

    # Merge ext grad variable counts and types
    for var, cnt in ext_grad_vars.items():
        stats.grad_by_variable[var] = stats.grad_by_variable.get(var, 0) + cnt
    for var, t in ext_grad_types.items():
        stats.grad_variable_type.setdefault(var, t)

    return stats


def collect(
    tesseracts_root: Path,
    problem_filter: str | None = None,
    tesseract_shared_root: Path | None = None,
) -> list[SolverStats]:
    results: list[SolverStats] = []
    for problem_dir in sorted(tesseracts_root.iterdir()):
        if not problem_dir.is_dir():
            continue
        if problem_filter and problem_filter not in problem_dir.name:
            continue
        for solver_dir in sorted(problem_dir.iterdir()):
            if solver_dir.is_dir():
                results.append(
                    analyse(
                        problem_dir.name,
                        solver_dir.name,
                        solver_dir,
                        tesseract_shared_root,
                    )
                )
    return results


# ── Output formatters ──────────────────────────────────────────────────────


_CAT_ORDER = ["physics", "io", "init", "grad", "util", "unknown"]
_CAT_COLOR = {
    "physics": "red",
    "io": "cyan",
    "init": "yellow",
    "grad": "magenta",
    "util": "green",
    "unknown": "dim",
}


def _category_bar(
    solver_by_category: dict[str, int], max_bespoke: int, width: int = 28
) -> str:
    """Stacked absolute-scale bar. Bar length = bespoke/max_bespoke; segments = categories."""
    total_bespoke = sum(solver_by_category.values())
    if max_bespoke == 0 or total_bespoke == 0:
        return f"[dim]{'░' * width}[/dim]"

    filled_total = max(1, round(total_bespoke / max_bespoke * width))
    cats = [
        (c, solver_by_category.get(c, 0))
        for c in _CAT_ORDER
        if solver_by_category.get(c, 0) > 0
    ]

    segments = []
    allocated = 0
    for i, (cat, count) in enumerate(cats):
        if i == len(cats) - 1:
            n = filled_total - allocated
        else:
            n = max(1, round(count / total_bespoke * filled_total))
        n = min(n, filled_total - allocated)
        if n > 0:
            segments.append(f"[{_CAT_COLOR[cat]}]{'█' * n}[/{_CAT_COLOR[cat]}]")
            allocated += n

    empty = width - filled_total
    return "".join(segments) + f"[dim]{'░' * empty}[/dim]"


def print_rich(results: list[SolverStats]) -> None:
    from rich import box
    from rich.console import Console
    from rich.table import Table

    TIER_LABEL = {1: "py", 2: "py+ext", 3: "ext-dep"}

    console = Console(width=180)
    problems: dict[str, list[SolverStats]] = {}
    for s in results:
        problems.setdefault(s.problem, []).append(s)

    for problem, rows in problems.items():
        max_bespoke = max((s.py_solver + s.ext_solver for s in rows), default=1)

        table = Table(
            title=f"[bold]{problem}[/bold]",
            box=box.SIMPLE_HEAD,
            title_justify="left",
            header_style="bold white",
            show_edge=False,
            pad_edge=False,
        )
        table.add_column(
            "solver", style="white", no_wrap=True, min_width=20, max_width=28
        )
        table.add_column("tier", justify="center", no_wrap=True, min_width=8)
        table.add_column("total", justify="right", style="dim", min_width=6)
        table.add_column("glue", justify="right", style="dim", min_width=6)
        table.add_column("init", justify="right", style="yellow", min_width=5)
        table.add_column("io", justify="right", style="cyan", min_width=5)
        table.add_column("physics", justify="right", style="red", min_width=7)
        table.add_column("grad", justify="right", style="magenta", min_width=5)
        table.add_column("util", justify="right", style="green", min_width=5)
        table.add_column(f"bespoke  (max={max_bespoke})", no_wrap=True, min_width=30)

        for s in rows:
            cat = s.solver_by_category
            # Merge ext_by_category into the same colour buckets as Python code.
            # Unannotated ext lines land in "unknown" to preserve bar scale.
            ext_annotated = s.ext_by_category and set(s.ext_by_category) - {"unknown"}
            cat_with_ext = dict(cat)
            if s.ext_solver:
                if ext_annotated:
                    for k, v in s.ext_by_category.items():
                        cat_with_ext[k] = cat_with_ext.get(k, 0) + v
                else:
                    cat_with_ext["unknown"] = (
                        cat_with_ext.get("unknown", 0) + s.ext_solver
                    )
            bar = _category_bar(cat_with_ext, max_bespoke)

            def _cat(name: str) -> str:
                v = cat_with_ext.get(name, 0)
                return str(v) if v else "[dim]—[/dim]"

            table.add_row(
                s.solver,
                TIER_LABEL[s.tier],
                str(s.py_total + s.ext_solver),
                str(s.py_imports + s.py_interface),
                _cat("init"),
                _cat("io"),
                _cat("physics"),
                _cat("grad"),
                _cat("util"),
                bar,
            )
            for f in s.ext_files:
                table.add_row(f"[dim]  ↳ {f}[/dim]", "", "", "", "", "", "", "", "", "")
            if s.grad_by_variable:
                _TYPE_ABBR = {
                    "autodiff": "auto",
                    "adjoint": "adj",
                    "analytic": "ana",
                    "zero": "zero",
                }
                var_parts = "  ".join(
                    (
                        f"[magenta]{var}[/magenta]"
                        f"[dim]({cnt})[/dim]"
                        + (
                            f"[blue]:{_TYPE_ABBR.get(s.grad_variable_type[var], s.grad_variable_type[var])}[/blue]"
                            if var in s.grad_variable_type
                            else ""
                        )
                    )
                    for var, cnt in sorted(s.grad_by_variable.items())
                )
                table.add_row(
                    "[dim]  grad vars →[/dim]",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    f"[dim]{var_parts}[/dim]",
                )

        console.print(table)
        console.print()

    console.print(
        "[dim]bar:  [/dim]"
        "[red]█ physics[/red][dim]  [/dim]"
        "[cyan]█ io[/cyan][dim]  [/dim]"
        "[yellow]█ init[/yellow][dim]  [/dim]"
        "[magenta]█ grad[/magenta][dim]  [/dim]"
        "[green]█ util[/green][dim]  [/dim]"
        "[dim]█ unknown/unannotated  —  length = total bespoke lines (py + ext)[/dim]"
    )
    console.print(
        "[dim]glue: imports + tesseract contract (InputSchema/OutputSchema, apply/vjp_jit/…)[/dim]"
    )
    console.print(
        "[dim]tier: py=pure-Python  py+ext=in-repo C++/Julia  ext-dep=external binary[/dim]"
    )
    console.print(
        "[dim]grad vars: lines attributed per differentiable input variable.  "
        "Inline ``# mosaic:grad:<var>`` section markers within a function body "
        "are used when present (lines before first marker = shared setup, not counted); "
        "otherwise the function-level tag divides the full span equally across all "
        "listed variables.[/dim]"
    )


_TYPE_ABBR = {
    "autodiff": "auto",
    "adjoint": "adj",
    "analytic": "ana",
    "zero": "zero",
}

_TYPE_COLOR = {
    "autodiff": "green",
    "adjoint": "cyan",
    "analytic": "yellow",
    "zero": "dim",
}


def print_variable_table(results: list[SolverStats]) -> None:
    """Per-domain pivot table: variables (rows) × solvers (columns).

    Each cell shows the gradient implementation type and attributed line count
    for that variable in that solver, e.g. ``adj(17)``.  A dash marks solvers
    that have no grad attribution for the variable.
    """
    from rich import box
    from rich.console import Console
    from rich.table import Table

    console = Console(width=220)
    problems: dict[str, list[SolverStats]] = {}
    for s in results:
        problems.setdefault(s.problem, []).append(s)

    for problem, rows in problems.items():
        # Only render the table if at least one solver has grad attribution.
        all_vars = sorted({var for s in rows for var in s.grad_by_variable})
        if not all_vars:
            continue

        table = Table(
            title=f"[bold]{problem}[/bold]  — gradient implementation",
            box=box.SIMPLE_HEAD,
            title_justify="left",
            header_style="bold white",
            show_edge=False,
            pad_edge=False,
        )
        table.add_column("variable", style="white", no_wrap=True, min_width=16)
        for s in rows:
            table.add_column(s.solver, justify="center", no_wrap=True, min_width=12)

        for var in all_vars:
            cells = []
            for s in rows:
                lines = s.grad_by_variable.get(var, 0)
                gtype = s.grad_variable_type.get(var)
                if lines == 0:
                    cells.append("[dim]—[/dim]")
                elif gtype:
                    abbr = _TYPE_ABBR.get(gtype, gtype)
                    color = _TYPE_COLOR.get(gtype, "white")
                    cells.append(f"[{color}]{abbr}[/{color}][dim]({lines})[/dim]")
                else:
                    cells.append(f"[dim]({lines})[/dim]")
            table.add_row(var, *cells)

        console.print(table)
        console.print()

    console.print(
        "[dim]"
        "[green]auto[/green]=autodiff  "
        "[cyan]adj[/cyan]=adjoint  "
        "[yellow]ana[/yellow]=analytic  "
        "zero=zero/placeholder  "
        "lines = attributed implementation lines"
        "[/dim]"
    )


def print_effort_table(results: list[SolverStats]) -> None:
    """Per-domain effort table: solvers (rows, sorted by total grad lines) × variables (columns).

    Each cell shows the attributed line count for that variable in that solver, coloured by
    implementation type.  Rows are sorted by total grad lines descending so the most
    effort-intensive solvers appear first.
    """
    from rich import box
    from rich.console import Console
    from rich.table import Table

    console = Console(width=220)
    problems: dict[str, list[SolverStats]] = {}
    for s in results:
        problems.setdefault(s.problem, []).append(s)

    for problem, rows in problems.items():
        all_vars = sorted({var for s in rows for var in s.grad_by_variable})
        if not all_vars:
            continue

        def _real_lines(s: SolverStats) -> int:
            return sum(
                cnt
                for var, cnt in s.grad_by_variable.items()
                if s.grad_variable_type.get(var) != "zero"
            )

        sorted_rows = sorted(rows, key=_real_lines, reverse=True)

        table = Table(
            title=f"[bold]{problem}[/bold]  — gradient effort per variable",
            box=box.SIMPLE_HEAD,
            title_justify="left",
            header_style="bold white",
            show_edge=False,
            pad_edge=False,
        )
        table.add_column("solver", style="white", no_wrap=True, min_width=20)
        table.add_column("total", justify="right", style="bold", min_width=7)
        for var in all_vars:
            table.add_column(
                var, justify="right", no_wrap=True, min_width=max(7, len(var))
            )

        for s in sorted_rows:
            total = _real_lines(s)
            if total == 0:
                continue

            def _cell(var: str, s: SolverStats = s) -> str:
                cnt = s.grad_by_variable.get(var, 0)
                gtype = s.grad_variable_type.get(var)
                if cnt == 0 or gtype == "zero":
                    return "[dim]—[/dim]"
                color = _TYPE_COLOR.get(gtype, "white") if gtype else "white"
                return f"[{color}]{cnt}[/{color}]"

            table.add_row(s.solver, str(total), *[_cell(v) for v in all_vars])

        console.print(table)
        console.print()

    console.print(
        "[dim]"
        "[green]auto[/green]=autodiff  "
        "[cyan]adj[/cyan]=adjoint  "
        "[yellow]ana[/yellow]=analytic  "
        "dim=zero/placeholder  "
        "total = sum of attributed grad lines across all variables"
        "[/dim]"
    )


_SOLVER_DISPLAY: dict[str, str] = {
    # navier-stokes-grid
    "incompressible-navier-stokes-jl": r"ins\_jl~\cite{agdestein2024ins}",
    "warp-ns": r"Warp-NS~\cite{macklin2024warp}",
    "xlb": r"XLB~\cite{ataei2024xlb}",
    "exponax": r"Exponax~\cite{koehler2024apebench}",
    "su2": r"SU2~\cite{economon2015su2}",
    "pict": r"PICT~\cite{franz2025pict}",
    "jax-cfd": r"JAX-CFD~\cite{kochkov2021machine}",
    "phiflow": r"PhiFlow~\cite{holl2024phiflow}",
    "fenics-ns": r"FEniCS~\cite{farrell2013automated}",
    "fenics-ns-3d": r"FEniCS-3D~\cite{farrell2013automated}",
    "openfoam": r"OpenFOAM~\cite{weller1998tensorial}",
    # structural-mesh
    "topopt-jl": r"TopOpt.jl~\cite{huang2021topoptjl}",
    "dealii": r"deal.II~\cite{bangerth2007dealii}",
    "firedrake": r"Firedrake~\cite{rathgeber2016firedrake}",
    "fenics": r"FEniCS~\cite{farrell2013automated}",
    "jax-fem": r"JAX-FEM~\cite{xue2023jaxfem}",
    # thermal-mesh
    "dealii-heat": r"deal.II~\cite{bangerth2007dealii}",
    "fenics-heat": r"FEniCS~\cite{farrell2013automated}",
    "firedrake-heat": r"Firedrake~\cite{rathgeber2016firedrake}",
    "torch-fem": r"torch-fem",
}

_VAR_DISPLAY: dict[str, str] = {
    "v0": r"$v_0$",
    "viscosity": r"$\nu$",
    "dt": r"$\Delta t$",
    "inflow_profile": r"inflow",
    "rho": r"$\rho$",
    "source": r"src",
}

_TYPE_LATEX_SYMBOL: dict[str, str] = {
    "autodiff": r"$\bullet$",  # source-transformation / operator-overloading AD
    "adjoint": r"$\dagger$",  # tape-based or hand-derived discrete adjoint
    "analytic": r"$\star$",  # closed-form / analytic sensitivity
}

_DOMAIN_LABEL: dict[str, str] = {
    "navier-stokes-grid": r"Navier--Stokes grid (F2/F3)",
    "structural-mesh": r"Structural mechanics (S)",
    "thermal-mesh": r"Heat transfer (H)",
}

# Canonical variable order per domain for consistent column layout
_DOMAIN_VARS: dict[str, list[str]] = {
    "navier-stokes-grid": ["v0", "viscosity", "dt", "inflow_profile"],
    "structural-mesh": ["rho"],
    "thermal-mesh": ["rho", "source"],
}


def _effort_real_lines(s: SolverStats) -> int:
    """Total gradient lines for ``s`` excluding ``zero`` (stub) attributions."""
    return sum(
        cnt
        for var, cnt in s.grad_by_variable.items()
        if s.grad_variable_type.get(var) != "zero"
    )


def _effort_raw(s: SolverStats, var: str) -> int:
    """Raw line count for ``var`` on ``s`` (``zero``-typed entries count as 0)."""
    cnt = s.grad_by_variable.get(var, 0)
    gtype = s.grad_variable_type.get(var)
    return 0 if (cnt == 0 or gtype == "zero") else cnt


def _effort_cell_raw(s: SolverStats, var: str) -> str:
    """Render a LaTeX cell showing raw line counts with optional strategy symbol."""
    cnt = _effort_raw(s, var)
    if cnt == 0:
        return r"$-$"
    gtype = s.grad_variable_type.get(var)
    sym = _TYPE_LATEX_SYMBOL.get(gtype, "") if gtype else ""
    return rf"{sym}\,{cnt}" if sym else str(cnt)


def _effort_cell_scored(s: SolverStats, var: str, max_in_col: int) -> str:
    """Render a LaTeX cell showing a 1--10 score normalised by ``max_in_col``."""
    import math

    cnt = _effort_raw(s, var)
    if cnt == 0:
        return r"$-$"
    gtype = s.grad_variable_type.get(var)
    sym = _TYPE_LATEX_SYMBOL.get(gtype, "") if gtype else ""
    score = max(1, min(10, math.ceil(cnt / max_in_col * 10)))
    return rf"{sym}\,{score}" if sym else str(score)


def _effort_cell(
    s: SolverStats, var: str, col_maxes: dict[str, int], scored: bool
) -> str:
    """Dispatch to scored or raw cell rendering."""
    if scored:
        return _effort_cell_scored(s, var, col_maxes[var])
    return _effort_cell_raw(s, var)


def _effort_fwd_cell(s: SolverStats, fwd_max: int, scored: bool) -> str:
    """Render the Fwd column: total_solver as raw count or 1--10 score."""
    import math

    t = s.total_solver
    if scored:
        if t == 0:
            return "0"
        return str(max(1, min(10, math.ceil(t / fwd_max * 10))))
    return str(t)


def _effort_group_metrics(
    sorted_rows: list[SolverStats],
    vars_for_group: list[str],
) -> tuple[dict[str, int], int]:
    """Compute per-column max line counts and the Fwd-column max for a group."""
    col_maxes: dict[str, int] = {
        v: max((_effort_raw(s, v) for s in sorted_rows), default=1) or 1
        for v in vars_for_group
    }
    fwd_max = max((s.total_solver for s in sorted_rows), default=1) or 1
    return col_maxes, fwd_max


def _build_ns_subtable(
    sorted_rows: list[SolverStats],
    vars_for_group: list[str],
    col_spec: str,
    var_headers: str,
    col_maxes: dict[str, int],
    fwd_max: int,
    scored: bool,
) -> str:
    """Build the Navier-Stokes subtable LaTeX string."""
    header_label = _DOMAIN_LABEL["navier-stokes-grid"]
    lines: list[str] = [
        rf"  \textit{{{header_label}}}\\[2pt]",
        rf"  \begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        rf"    \textbf{{Solver}} & \textbf{{Fwd}} & {var_headers} \\",
        r"    \midrule",
    ]
    for s in sorted_rows:
        display = _SOLVER_DISPLAY.get(s.solver, s.solver)
        cells = " & ".join(
            _effort_cell(s, v, col_maxes, scored) for v in vars_for_group
        )
        lines.append(
            rf"    {display} & {_effort_fwd_cell(s, fwd_max, scored)} & {cells} \\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}"]
    return "\n".join(lines)


def _build_mesh_subtable(
    sorted_rows: list[SolverStats],
    domain_list: list[str],
    vars_for_group: list[str],
    col_spec: str,
    var_headers: str,
    ncols: int,
    col_maxes: dict[str, int],
    fwd_max: int,
    scored: bool,
) -> str:
    """Build the structural+thermal subtable LaTeX string."""
    lines: list[str] = [
        r"  \textit{Structural mechanics \& heat transfer}\\[2pt]",
        rf"  \begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        rf"    \textbf{{Solver}} & \textbf{{Fwd}} & {var_headers} \\",
    ]
    for problem in domain_list:
        domain_rows = [s for s in sorted_rows if s.problem == problem]
        if not domain_rows:
            continue
        label = _DOMAIN_LABEL.get(problem, problem)
        lines.append(r"    \midrule")
        lines.append(rf"    \multicolumn{{{ncols}}}{{@{{}}l}}{{\textit{{{label}}}}} \\")
        for s in domain_rows:
            display = _SOLVER_DISPLAY.get(s.solver, s.solver)
            cells = " & ".join(
                _effort_cell(s, v, col_maxes, scored) for v in vars_for_group
            )
            lines.append(
                rf"    {display} & {_effort_fwd_cell(s, fwd_max, scored)} & {cells} \\"
            )
    lines += [r"    \bottomrule", r"  \end{tabular}"]
    return "\n".join(lines)


def _build_effort_subtable(
    group_key: str,
    domain_list: list[str],
    problems: dict[str, list[SolverStats]],
    scored: bool,
) -> str | None:
    """Build one subtable for ``group_key``; returns None if no rows qualify."""
    all_rows: list[SolverStats] = []
    for problem in domain_list:
        all_rows.extend(problems.get(problem, []))

    if group_key == "navier-stokes-grid":
        vars_for_group = _DOMAIN_VARS["navier-stokes-grid"]
    else:
        # Structural uses [rho], thermal uses [rho, source] — union is [rho, source].
        vars_for_group = ["rho", "source"]

    sorted_rows = sorted(all_rows, key=_effort_real_lines, reverse=True)
    sorted_rows = [s for s in sorted_rows if _effort_real_lines(s) > 0]
    if not sorted_rows:
        return None

    ncols = 2 + len(vars_for_group)  # Solver + Fwd + vars
    col_spec = "@{}l" + "r" * (ncols - 1) + "@{}"

    var_headers = " & ".join(
        r"\textbf{" + _VAR_DISPLAY.get(v, v) + "}" for v in vars_for_group
    )

    col_maxes, fwd_max = _effort_group_metrics(sorted_rows, vars_for_group)

    if group_key == "navier-stokes-grid":
        return _build_ns_subtable(
            sorted_rows,
            vars_for_group,
            col_spec,
            var_headers,
            col_maxes,
            fwd_max,
            scored,
        )
    return _build_mesh_subtable(
        sorted_rows,
        domain_list,
        vars_for_group,
        col_spec,
        var_headers,
        ncols,
        col_maxes,
        fwd_max,
        scored,
    )


def _effort_caption_and_label(scored: bool) -> tuple[str, str]:
    """Return (caption, label) for the LaTeX table depending on ``scored``."""
    if scored:
        caption = (
            "Forward solver (Fwd) and per-variable gradient implementation effort,\n"
            "    scored 0--10 within each domain (10 = most lines, $-$ = not implemented).\n"
            "    Symbol indicates differentiation strategy:\n"
            "    $\\bullet$~autodiff, $\\dagger$~adjoint, $\\star$~analytic.\n"
            "    Full line counts are in \\cref{tab:grad_effort_full}."
        )
        return caption, "fig:grad_vs_gradfree"
    caption = (
        "Forward solver (Fwd) and per-variable gradient implementation effort\n"
        "    (raw attributed lines of code).\n"
        "    Symbol indicates differentiation strategy:\n"
        "    $\\bullet$~autodiff, $\\dagger$~adjoint, $\\star$~analytic.\n"
        "    Solvers sorted by total gradient lines within each domain."
    )
    return caption, "tab:grad_effort_full"


def generate_latex_effort_table(
    results: list[SolverStats],
    scored: bool = False,
) -> str:
    """Return LaTeX source for the gradient effort table (one subtable per domain).

    Rows = solvers sorted by real grad lines (zero-stubs excluded) descending.
    Columns = input variables for the domain.

    scored=False (default): cells show raw attributed line counts.
    scored=True: cells show 1--5 effort scores, normalised per column within each
        domain group (max lines in column → 5).  Dashes stay as dashes.
    """
    problems: dict[str, list[SolverStats]] = {}
    for s in results:
        problems.setdefault(s.problem, []).append(s)

    # NS grid gets its own subtable; structural + thermal share one (both use rho/source).
    _GROUPS: list[tuple[str, list[str]]] = [
        ("navier-stokes-grid", ["navier-stokes-grid"]),
        ("mesh-based", ["structural-mesh", "thermal-mesh"]),
    ]

    subtables: list[str] = []
    for group_key, domain_list in _GROUPS:
        sub = _build_effort_subtable(group_key, domain_list, problems, scored)
        if sub is not None:
            subtables.append(sub)

    assert len(subtables) == 2, "expected exactly NS and mesh-based subtables"
    ns_body, mesh_body = subtables

    caption, label = _effort_caption_and_label(scored)

    return (
        "\\begin{table}[t]\n"
        "  \\centering\\small\n"
        "  \\setlength{\\tabcolsep}{4pt}\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{{label}}}\n"
        "  \\begin{minipage}[t]{0.57\\linewidth}\n"
        + ns_body.replace("\n", "\n  ")
        + "\n  \\end{minipage}%\n"
        "  \\hspace{0.02\\linewidth}%\n"
        "  \\begin{minipage}[t]{0.40\\linewidth}\n"
        + mesh_body.replace("\n", "\n  ")
        + "\n  \\end{minipage}\n"
        "\\end{table}"
    )


_CSV_CATEGORIES = ["physics", "io", "init", "grad", "util", "unknown"]


def print_csv(results: list[SolverStats]) -> None:
    # Collect all variable names that appear across any solver (stable sort order)
    all_vars: list[str] = sorted({var for s in results for var in s.grad_by_variable})

    w = csv.writer(sys.stdout)
    w.writerow(
        [
            "problem",
            "solver",
            "tier",
            "py_total",
            "py_imports",
            "py_interface",
            "py_solver",
            "ext_solver",
            "import_frac",
            "interface_frac",
            *[f"cat_{c}" for c in _CSV_CATEGORIES],
            *[f"grad_var_{v}" for v in all_vars],
            *[f"grad_type_{v}" for v in all_vars],
            "note",
        ]
    )
    for s in results:
        w.writerow(
            [
                s.problem,
                s.solver,
                s.tier,
                s.py_total,
                s.py_imports,
                s.py_interface,
                s.py_solver,
                s.ext_solver,
                f"{s.import_fraction:.3f}" if s.import_fraction is not None else "",
                f"{s.interface_fraction:.3f}"
                if s.interface_fraction is not None
                else "",
                *[s.solver_by_category.get(c, 0) for c in _CSV_CATEGORIES],
                *[s.grad_by_variable.get(v, 0) for v in all_vars],
                *[s.grad_variable_type.get(v, "") for v in all_vars],
                s.note,
            ]
        )
