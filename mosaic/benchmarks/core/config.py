"""ProblemConfig and SolverSpec dataclasses — the only shared abstraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)


@dataclass
class IcSpec:
    """IC function wrapper: bundles the generator with its description and plot params.

    Callable — delegates to ``fn`` so existing code that does ``cfg.make_ic[name](...)``
    continues to work unchanged.
    """

    fn: Callable
    description: str = ""
    plot_params: dict = field(default_factory=dict)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


@dataclass
class SolverSpec:
    """Runtime descriptor for a single solver registered in a ProblemConfig.

    Most per-solver fields (name, scheme, backend, ad_strategy, differentiable,
    uses_gpu, internal_dtype, description, doc_url) are populated by
    :func:`discover_solvers` from the solver's ``tesseract_config.yaml`` —
    that YAML is the ground truth, problem configs only set per-(solver,
    problem) overrides (``input_overrides`` / ``exclusions`` /
    ``explained_anomalies``).

    Presentation-only fields — ``color`` / ``linestyle`` / ``marker`` — are
    populated by :func:`mosaic.benchmarks.plots.solver_styles.apply_styles`
    after discovery; they're attributes on the spec only so plot code can read
    them via ``spec.color`` etc.
    """

    dir: str  # subdir under tesseract_dir/ (matches the YAML's solver dir name)
    color: str  # hex; populated from solver_styles.SOLVER_STYLES, not YAML
    name: str  # display name for plots and tables
    scheme: str  # numerical scheme tag, e.g. "MAC FD + projection", "LBM BGK D2Q9"
    backend: str  # runtime / language: "jax", "pytorch", "julia", "cpp", "warp", ...
    family: str = (
        ""  # grouped-styling tag: "projection", "lbm", "spectral", "fd", "fv", "fem"
    )
    linestyle: str | tuple | None = None  # populated from solver_styles.SOLVER_STYLES
    marker: str | None = None  # populated from solver_styles.SOLVER_STYLES
    description: str = ""  # two-sentence summary from the top-level YAML description
    doc_url: str = ""  # canonical documentation / repository URL for the solver
    input_overrides: dict[str, Any] = field(default_factory=dict)
    # Per-problem default values merged into every call to ``cfg.make_inputs``
    # for this solver (e.g. material constants, solver-specific tunables).
    uses_gpu: bool = True  # False for CPU-only solvers
    image_tag: str | None = None
    # Set by discover_solvers to "<image_name>:latest". Used by ``--no-build``
    # mode and by ResourceSampler for GPU bookkeeping.
    normalize_output: Callable | None = None
    # Optional fn(arr) -> arr applied to forward-suite outputs to convert
    # solver units back to canonical IC units before comparison.
    differentiable: bool | None = None
    # Explicit VJP flag from YAML; None falls back to runtime detection via
    # _has_vjp (probes the container for a vector_jacobian_product endpoint).
    ad_strategy: str | None = None
    # How gradients are computed.  One of:
    #   "autodiff"  — native reverse-mode AD traces through the forward pass
    #                 (jax.vjp, torch.autograd, Zygote, wp.Tape)
    #   "adjoint"   — explicitly formulated adjoint equations
    #                 (dolfin-adjoint, pyadjoint, analytic self-adjoint SIMP sensitivity)
    #   "hybrid"    — analytic gradient rules combined with autodiff
    #                 (implicit-function theorem, custom VJP rules that call autodiff internally)
    #   None        — non-differentiable (no gradient support)
    internal_dtype: str = "float32"
    # Floating-point precision used internally by the solver. One of "float32"
    # or "float64". Solvers that compute in float64 but return float32 outputs
    # may show lower discretisation error at the same resolution — a precision
    # advantage, not a scheme advantage.
    exclusions: dict[str, dict | str] = field(default_factory=dict)
    # Maps a suite key ("gradient", "optimization", …) or a more specific
    # suite/experiment path ("forward/cylinder", "cost/vjp_cost") to a
    # categorical reason. Excluded solver–experiment pairs are skipped by the
    # runner and annotated as EXCLUDED in the status output. Values are usually
    # ``{"category": "categorical" | "infeasible", "reason": "..."}``; a plain
    # string is accepted as shorthand for the reason.
    explained_anomalies: dict[str, str | dict] = field(default_factory=dict)
    # Maps experiment keys (same format as exclusions) to documented reasons
    # why this solver is expected to produce anomalous results — the solver
    # CAN run and produces finite output, but underperforms peers for known,
    # method-intrinsic reasons (e.g. LBM O(Ma²) compressibility floor,
    # staggered MAC grid interpolation error). These appear as ANOMALY cells
    # (with the documented reason) rather than EXCLUDED, keeping the solver
    # in the score denominator so weaknesses remain visible.
    # Values: plain string reason, or dict with "reason" key.


def discover_solvers(tesseract_dir: Path) -> dict[str, SolverSpec]:
    """Auto-discover solvers from ``tesseract_config.yaml`` files.

    Scans *tesseract_dir* for subdirectories containing a
    ``tesseract_config.yaml`` with a ``metadata.mosaic:`` block and builds a
    :class:`SolverSpec` for each.  This allows new solver contributions to be
    picked up automatically without editing the problem config Python file.

    The block lives under ``metadata:`` because tesseract_core's
    ``TesseractConfig`` rejects unknown top-level keys (``extra="forbid"``).
    ``metadata`` is a free-form ``dict[str, Any]`` per upstream's schema.

    Supported keys (all optional except ``name``):

    .. code-block:: yaml

        # Top-level YAML: two-sentence summary used by Mosaic.
        description: >
          One- or two-sentence solver summary covering method and gradient strategy.

        metadata:
          mosaic:
            name: JAX-FEM                       # display name (required)
            backend: jax                        # runtime: jax, pytorch, julia, cpp, warp, fenics, firedrake
            family: fem                         # solver family for grouped styling
            scheme: "FEM HEX8"                  # numerical scheme tag
            discretization: FE                  # paper Table 2: FD | FV | FE | LBM | Spectral
            numerics: "Direct (UMFPACK)"        # paper Table 2 numerics column
            ad_strategy: autodiff               # autodiff | adjoint | hybrid | null
            differentiable: true                # explicit VJP flag
            uses_gpu: true
            internal_dtype: float32
            doc_url: "https://..."              # upstream docs link

    Note: ``color`` / ``linestyle`` / ``marker`` are *not* read from YAML —
    plot styling lives in :mod:`mosaic.benchmarks.plots.solver_styles` and is
    applied to each spec by ``apply_styles()`` after discovery.

    Returns a dict keyed by a normalised solver name (directory name with
    hyphens replaced by underscores). Problem configs can re-key (e.g.
    ``incompressible_navier_stokes_jl`` → ``ins_jl``) and apply per-(solver,
    problem) overrides (``input_overrides``, ``exclusions``,
    ``explained_anomalies``) before publishing the spec dict.
    """
    import yaml

    solvers: dict[str, SolverSpec] = {}
    if not tesseract_dir.is_dir():
        return solvers

    import warnings

    for solver_dir in sorted(tesseract_dir.iterdir()):
        config_path = solver_dir / "tesseract_config.yaml"
        if not config_path.exists():
            continue
        try:
            with open(config_path) as f:
                doc = yaml.safe_load(f)
        except Exception as exc:
            warnings.warn(
                f"Solver discovery: cannot parse {config_path}: {exc}",
                stacklevel=2,
            )
            continue
        if not isinstance(doc, dict):
            warnings.warn(
                f"Solver discovery: {config_path} is not a YAML mapping — skipping",
                stacklevel=2,
            )
            continue
        metadata = doc.get("metadata") or {}
        meta = metadata.get("mosaic") if isinstance(metadata, dict) else None
        if not isinstance(meta, dict):
            # No mosaic: block — not a Mosaic solver, skip silently
            continue
        if not meta.get("name"):
            warnings.warn(
                f"Solver discovery: {config_path} has a mosaic: block but no "
                f"mosaic.name — skipping. Add 'name: \"My Solver\"' to the mosaic: block.",
                stacklevel=2,
            )
            continue
        # Build the SolverSpec from the mosaic: block.
        dir_name = solver_dir.name
        solver_key = dir_name.replace("-", "_")
        image_name = doc.get("name", dir_name)

        backend = meta.get("backend", "")

        def _txt(key: str, default: str = "") -> str:
            # YAML block scalars (``description: >`` / ``narrative: >``) yield
            # a trailing newline that callers don't want. Strip it.
            val = meta.get(key, default)
            return val.strip() if isinstance(val, str) else val

        # Color / linestyle / marker are presentation-only; they live in
        # ``mosaic.benchmarks.plots.solver_styles`` and are applied by each
        # problem config via ``apply_styles()`` after re-keying. We leave them
        # at neutral defaults here so the SolverSpec is well-formed.
        solvers[solver_key] = SolverSpec(
            dir=dir_name,
            name=_txt("name", image_name),
            backend=backend,
            family=_txt("family"),
            scheme=_txt("scheme"),
            color="#999999",
            linestyle=None,
            marker=None,
            ad_strategy=meta.get("ad_strategy"),
            differentiable=meta.get("differentiable"),
            uses_gpu=meta.get("uses_gpu", True),
            internal_dtype=meta.get("internal_dtype", "float32"),
            # Per-solver description lives at the YAML's top level; it doubles as
            # the container description shown by ``tesseract info``.
            description=(doc.get("description") or "").strip(),
            doc_url=_txt("doc_url"),
            image_tag=f"{image_name}:latest",
        )
    log.info("Discovered %d solver(s) in %s", len(solvers), tesseract_dir)
    return solvers


@dataclass
class ProblemConfig:
    name: str  # "ns-grid", "n-body", "md"
    tesseract_dir: Path  # abs path to tesseracts/{problem}/
    output_key: str  # field to compare: "result", "density", …

    solvers: dict[str, SolverSpec]

    # IC and input construction
    make_ic: dict[str, IcSpec | Callable]  # ic_name → IcSpec (or bare callable)
    make_inputs: Callable  # (solver_name, ic, **physics) → dict

    # Comparison
    error_fn: Callable  # (pred, ref) → float
    diagnostics: dict[str, Callable]  # name → fn(array, **kw) → scalar | dict
    pairwise_diagnostics: dict[str, Callable] = field(default_factory=dict)
    analytic: Callable | None = None  # (ic, **physics, t) → array, or None

    ic_key: str = "ic"  # input dict key for the initial condition
    domain_extent: float = 1.0
    resolution_key: str = "N"  # param that controls IC size (e.g. "N", "mesh_level")
    n_to_cells: Callable | None = None  # N → total cell count for x-axis labelling

    # Field visualisation
    field_to_2d: Callable | None = None
    ic_to_2d: Callable | None = None
    field_cmap: str = "RdBu_r"
    field_symmetric: bool = True
    diagnostic_fields: bool = True

    # Optional spectral diagnostics for plot_agreement
    power_spectrum_fn: Callable | None = None

    # Problem-specific plot hooks
    extra_plots: dict[str, list[Callable]] = field(default_factory=dict)

    # Extra scalar outputs to capture alongside output_key
    extra_output_keys: list[str] = field(default_factory=list)

    # State keys threaded between stability chunks
    state_keys: list[str] = field(default_factory=list)

    # Agreement metric override
    agreement_transform: Callable | None = None
    agreement_xaxis: Callable | None = None
    agreement_xlabel: str = "x"
    agreement_ylabel: str = "value"

    # Pairwise diagnostic axis labels
    pairwise_xlabel: str = "k"
    pairwise_ylabels: dict[str, str] = field(default_factory=dict)

    # Units for axis labels
    units: dict[str, str] = field(default_factory=dict)

    # Per-suite experiment defaults
    forward_defaults: dict = field(default_factory=dict)
    cost_defaults: dict = field(default_factory=dict)
    gradient_defaults: dict = field(default_factory=dict)
    inverse_defaults: dict = field(default_factory=dict)

    # Legacy plot descriptions: (suite, experiment) → description string.
    # Prefer inline description/plot_description keys in each experiment def.
    plot_descriptions: dict[tuple[str, str], str] = field(default_factory=dict)

    # Per-problem thresholds consumed by `mosaic status` to flag solvers whose
    # results are finite but far from their peers. Keys are either a suite
    # name ("forward") or "suite/experiment" for experiment-specific overrides
    # ("gradient/fd_check"); the more specific key wins. Each value is a dict
    # of check name → threshold, e.g.
    #
    #   {"forward":           {"median_k": 3.0, "max_error": 0.5},
    #    "gradient/fd_check": {"min_cosine": 0.99},
    #    "optimization":          {"max_final_ratio": 0.5}}
    #
    # A check is skipped when its key is absent, so a new problem does not
    # start flagging anomalies until its author opts in.
    status_checks: dict[str, dict] = field(default_factory=dict)

    # Physics-focused problem description (no solver names).
    description: str = ""

    # Display label for the physics-domain section of the generated solver
    # reference page (docs/solvers.qmd). When multiple problems share the same
    # tesseract_dir (e.g. ns-grid and ns-3d-grid), the first non-empty value
    # encountered wins.
    category_label: str = ""

    # Boundary/domain condition description.
    bc_description: str = ""

    # Legacy IC descriptions (use IcSpec.description instead).
    ic_descriptions: dict[str, str] = field(default_factory=dict)

    # Legacy IC plot params (use IcSpec.plot_params instead).
    ic_plot_params: dict[str, dict] = field(default_factory=dict)

    # ── IC helpers ────────────────────────────────────────────────────────────

    def get_ic_description(self, ic_name: str) -> str:
        ic = self.make_ic.get(ic_name)
        if isinstance(ic, IcSpec):
            return ic.description
        return self.ic_descriptions.get(ic_name, "")

    def get_ic_plot_params(self, ic_name: str) -> dict:
        ic = self.make_ic.get(ic_name)
        if isinstance(ic, IcSpec):
            return ic.plot_params
        return self.ic_plot_params.get(ic_name, {})

    # ── Experiment description helpers ───────────────────────────────────────

    def _suite_defaults(self, suite: str) -> dict:
        if suite == "gradient":
            return self.gradient_defaults
        if suite == "optimization":
            return self.inverse_defaults
        if suite == "cost":
            return self.cost_defaults
        return self.forward_defaults

    def _exp_def(self, suite: str, experiment: str) -> dict:
        """Look up experiment definition, falling back to parent key for sub-experiments.

        Sub-experiments like "horizon_sweep/tgv3d" fall back to "horizon_sweep"
        if no exact key is found, allowing the parent experiment's description to
        be reused for sub-experiments that don't have their own key.
        """
        defaults = self._suite_defaults(suite)
        exp_def = defaults.get(experiment, {})
        if not exp_def and "/" in experiment:
            parent_key = experiment.split("/")[0]
            exp_def = defaults.get(parent_key, {})
        return exp_def

    def get_plot_description(self, suite: str, experiment: str) -> str:
        """Return the plot description for (suite, experiment).

        For the cost suite, experiment="" returns the suite-level description
        and a named experiment (e.g. "spatial_cost") returns its per-plot text.
        Falls back to the legacy plot_descriptions dict.
        """
        if suite == "cost":
            cost_def = self.cost_defaults.get("cost", self.cost_defaults)
            if isinstance(cost_def, dict):
                if not experiment:
                    return cost_def.get("description", "")
                return cost_def.get("plot_descriptions", {}).get(experiment, "")
        exp_def = self._exp_def(suite, experiment)
        if isinstance(exp_def, dict) and "plot_description" in exp_def:
            return exp_def["plot_description"]
        return self.plot_descriptions.get((suite, experiment), "")

    def get_experiment_description(self, suite: str, experiment: str) -> str:
        """Return the short experiment description (what it measures)."""
        if suite == "cost":
            cost_def = self.cost_defaults.get("cost", self.cost_defaults)
            if isinstance(cost_def, dict):
                return cost_def.get("description", "")
        exp_def = self._exp_def(suite, experiment)
        if isinstance(exp_def, dict):
            return exp_def.get("description", "")
        return ""

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self) -> None:
        """Check that the config is well-formed.  Raises on first error."""
        errors: list[str] = []
        if not self.name:
            errors.append("name is empty")
        if not self.solvers:
            errors.append("no solvers registered")
        for key, spec in self.solvers.items():
            for attr in ("name", "dir", "scheme", "backend", "color"):
                if not getattr(spec, attr, None):
                    errors.append(f"solver {key!r}: missing {attr}")
        if not self.make_ic:
            errors.append("make_ic is empty (no initial conditions)")
        for ic_name, ic in self.make_ic.items():
            if not callable(ic):
                errors.append(f"make_ic[{ic_name!r}] is not callable")
        if not callable(self.make_inputs):
            errors.append("make_inputs is not callable")
        if not callable(self.error_fn):
            errors.append("error_fn is not callable")
        if not self.tesseract_dir.is_dir():
            errors.append(f"tesseract_dir does not exist: {self.tesseract_dir}")

        # Validate suite defaults structure: each experiment should be a list of run dicts
        for suite_attr, suite_label in [
            ("forward_defaults", "forward_defaults"),
            ("gradient_defaults", "gradient_defaults"),
            ("inverse_defaults", "inverse_defaults (optimization)"),
        ]:
            defaults = getattr(self, suite_attr)
            for exp_name, exp_val in defaults.items():
                # Skip metadata keys (description, plot_description, etc.)
                if isinstance(exp_val, str):
                    continue
                if isinstance(exp_val, dict) and not any(
                    k in exp_val for k in ("ic", "physics", "sweep", "fd", "optim")
                ):
                    # Looks like a metadata dict (e.g. {"description": "...", "plot_description": "..."})
                    continue
                if isinstance(exp_val, dict):
                    errors.append(
                        f"{suite_label}[{exp_name!r}]: expected a list of run dicts, "
                        f"got a single dict. Wrap it in a list: [{exp_val!r}]"
                    )
                elif not isinstance(exp_val, list):
                    errors.append(
                        f"{suite_label}[{exp_name!r}]: expected a list of run dicts, "
                        f"got {type(exp_val).__name__}"
                    )

        valid_ad = {"autodiff", "adjoint", "hybrid", None}
        for key, spec in self.solvers.items():
            if spec.ad_strategy not in valid_ad:
                errors.append(
                    f"solver {key!r}: ad_strategy={spec.ad_strategy!r} "
                    f"not in {valid_ad}"
                )

        if errors:
            raise ValueError(
                f"ProblemConfig {self.name!r} validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )
