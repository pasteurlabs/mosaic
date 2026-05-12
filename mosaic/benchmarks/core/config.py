"""Problem and SolverSpec dataclasses — the only shared abstraction."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)


class ExclusionCategory(str, Enum):
    """Why a solver does not run for a given experiment.

    The ``(str, Enum)`` mixin means each member *is* a string (e.g.
    ``ExclusionCategory.CATEGORICAL == "categorical"``), so the on-disk
    serialisation of an :class:`Exclusion` lands as a plain string — no
    schema change vs. the legacy free-form strings.

    Members:
      * ``CATEGORICAL`` — method-intrinsic limitation (e.g. FFT-only solver
        on non-periodic BCs; non-differentiable C++ solver). Permanent;
        excluded from the campaign score denominator.
      * ``INFEASIBLE`` — would run but the result is not meaningful.
      * ``NOT_IMPLEMENTED`` — could in principle run but the support hasn't
        been wired yet. Counts in the score as "work to do".
      * ``UNSTABLE`` — runs but blows up; same as NOT_IMPLEMENTED in scoring.
      * ``UPSTREAM_BUG`` — failure attributable to a tracked upstream issue.
      * ``WIP`` — temporarily skipped while work is in progress.
      * ``UNSPECIFIED`` — fallback for legacy / un-categorised entries.
      * ``ANOMALY_EXPLAINED`` — the solver runs and produces output that's
        anomalous for documented method-intrinsic reasons; not a runtime
        skip, only a display annotation.
    """

    CATEGORICAL = "categorical"
    INFEASIBLE = "infeasible"
    NOT_IMPLEMENTED = "not_implemented"
    UNSTABLE = "unstable"
    UPSTREAM_BUG = "upstream_bug"
    WIP = "wip"
    UNSPECIFIED = "unspecified"
    ANOMALY_EXPLAINED = "anomaly_explained"


# Categories that are *permanent* — these stay out of the campaign score
# denominator. Everything else counts as "work to do" at the neutral weight.
EXCL_PERMANENT: frozenset[ExclusionCategory] = frozenset(
    {ExclusionCategory.CATEGORICAL}
)


# ── New experiment/plot abstractions ─────────────────────────────────────────
#
# These are the canonical types for the closure-style refactor. They live
# above ``Problem`` so the dataclass annotation below can reference
# them. See plan: `~/.claude/plans/i-want-to-make-encapsulated-token.md`.


class ExperimentFn(Protocol):
    """Callable that runs one experiment.

    Signature: ``(cfg, tags, **overrides) -> dict``. Experiments capture
    problem-specific state (``make_ic``, ``make_inputs``, ``error_fn``,
    per-experiment params, …) in their closures so the runner doesn't need
    to know about it. Returns a result dict saved by
    :func:`mosaic.benchmarks.core.io.save_experiment`.
    """

    def __call__(
        self, cfg: Problem, tags: dict[str, str], **overrides: Any
    ) -> dict: ...


class PlotFn(Protocol):
    """Callable that renders the plots for one experiment.

    Signature: ``(cfg, **kw) -> Any``. Like :class:`ExperimentFn`, plot
    functions close over problem-specific presentation state (colormap,
    axis labels, ``field_to_2d``, ``agreement_transform``, …) so the cfg
    surface stays small.
    """

    def __call__(self, cfg: Problem, **kw: Any) -> Any: ...


@dataclass
class Experiment:
    """An experiment is a closure plus the params it was built with.

    ``params`` is the introspection manifest — what configuration the closure
    captured. The runner only invokes ``fn``; ``params`` is read by status,
    documentation, and result-saving for provenance.
    """

    fn: ExperimentFn
    params: dict = field(default_factory=dict)


@dataclass
class Problem:
    """The single per-problem definition: closure deps + metadata +
    registries + builder methods.

    Each problem's ``experiments.py`` instantiates one ``Problem`` (with
    closure deps: ``make_ic``, ``error_fn``, ``output_key``, …), then
    :meth:`add` / :meth:`add_ic` / :meth:`add_extra_plot` register
    experiments + plot fns at ``"<suite>/<name>"`` keys. ``config.py``
    fills in solvers + metadata + exclusions and publishes the same
    ``Problem`` as ``CONFIG``. The runner / status / CLI consume
    ``Problem`` directly — there is no separate "config" wrapper.
    """

    # ── Metadata ─────────────────────────────────────────────────────────
    name: str = ""  # CLI slug: "ns-grid", "structural-mesh", …
    tesseract_dir: Path = Path()
    description: str = ""
    category_label: str = ""
    bc_description: str = ""

    # ── Runtime registry ─────────────────────────────────────────────────
    solvers: list[SolverSpec] = field(default_factory=list)
    make_inputs: Callable | None = None  # (solver_name, ic, **physics) → dict
    exclusions: dict[str, dict[str, Exclusion]] = field(default_factory=dict)
    status_checks: dict[str, dict] = field(default_factory=dict)

    # ── Experiment-time deps (threaded into Experiment.fn closures) ─────
    make_ic: dict = field(default_factory=dict)
    error_fn: Callable | None = None
    output_key: str = ""
    ic_key: str = ""
    domain_extent: float = 1.0
    resolution_key: str = "N"
    analytic: Callable | None = None
    diagnostics: dict = field(default_factory=dict)
    agreement_transform: Callable | None = None
    agreement_xaxis: Callable | None = None

    # ── Plot-time deps (used by per-problem plot lambdas) ────────────────
    field_to_2d: Callable | None = None
    ic_to_2d: Callable | None = None
    field_cmap: str = "RdBu_r"
    field_symmetric: bool = True
    diagnostic_fields: bool = True
    units: dict = field(default_factory=dict)
    power_spectrum_fn: Callable | None = None
    agreement_xlabel: str = "x"
    agreement_ylabel: str = "value"
    pairwise_xlabel: str = "k"
    pairwise_ylabels: dict = field(default_factory=dict)
    n_to_cells: Callable | None = None

    # ── Registries (filled by .add / .add_ic / .add_extra_plot) ─────────
    experiments: dict[str, Experiment] = field(default_factory=dict)
    plot_fns: dict[str, PlotFn] = field(default_factory=dict)

    # Optional per-key plot descriptions. When ``add(...)`` is called without
    # an explicit ``plot_description=`` kwarg, the value at
    # ``descriptions[key]`` is used instead. Lets each problem keep its
    # descriptions in one block at the top of experiments.py rather than
    # inline in every .add() call.
    descriptions: dict[str, str] = field(default_factory=dict)

    # ── Solver lookup ────────────────────────────────────────────────────

    def solver(self, name: str) -> SolverSpec:
        """Look up a solver by name. Raises ``KeyError`` if no match."""
        for s in self.solvers:
            if s.name == name:
                return s
        raise KeyError(name)

    @property
    def solver_names(self) -> list[str]:
        """List of ``SolverSpec.name`` strings, preserving insertion order."""
        return [s.name for s in self.solvers]

    # ── IC / experiment introspection ────────────────────────────────────

    def get_ic_description(self, ic_name: str) -> str:
        ic = self.make_ic.get(ic_name)
        return ic.description if isinstance(ic, IcSpec) else ""

    def get_ic_plot_params(self, ic_name: str) -> dict:
        ic = self.make_ic.get(ic_name)
        return ic.plot_params if isinstance(ic, IcSpec) else {}

    def _exp_params(self, suite: str, experiment: str) -> dict:
        """Look up an experiment's ``params`` payload via :attr:`experiments`.

        Sub-experiments like ``"horizon_sweep/tgv3d"`` fall back to
        ``"horizon_sweep"`` when no exact key is registered.
        """
        full = f"{suite}/{experiment}" if experiment else suite
        exp = self.experiments.get(full)
        if exp is None and experiment and "/" in experiment:
            parent = experiment.split("/")[0]
            exp = self.experiments.get(f"{suite}/{parent}")
        return exp.params if exp is not None else {}

    def get_plot_description(self, suite: str, experiment: str) -> str:
        """Return the plot description for (suite, experiment)."""
        params = self._exp_params(suite, experiment)
        if not params:
            return ""
        if suite == "cost" and "plot_descriptions" in params:
            if not experiment:
                return params.get("description", "")
            return params["plot_descriptions"].get(experiment, "")
        return params.get("plot_description", "")

    def get_experiment_description(self, suite: str, experiment: str) -> str:
        """Return the short experiment description."""
        params = self._exp_params(suite, experiment)
        return params.get("description", "") if isinstance(params, dict) else ""

    def _experiment_deps(self) -> dict:
        """Return the kwarg bag that runners may pull from."""
        return {
            "make_ic": self.make_ic,
            "error_fn": self.error_fn,
            "output_key": self.output_key,
            "ic_key": self.ic_key,
            "domain_extent": self.domain_extent,
            "resolution_key": self.resolution_key,
            "analytic": self.analytic,
            "diagnostics": self.diagnostics,
            "agreement_transform": self.agreement_transform,
            "agreement_xaxis": self.agreement_xaxis,
        }

    def add(
        self,
        key: str,
        runner: Callable,
        *,
        plot: Callable | None = None,
        plot_description: str | None = None,
        reduce: Callable | None = None,
        **config,
    ) -> None:
        """Register an experiment at ``key`` with optional sweep + plot + reduce.

        Any value inside ``**config`` (top-level or nested inside a dict
        kwarg) that is a list-of-primitives is treated as a **sweep axis**.
        Multiple sweep axes must have matching lengths; the framework
        parallel-zips them and registers one sub-experiment per zipped
        tuple, keyed at ``<key>/<auto-suffix>``. The suffix is derived
        from the differing key→value pairs (e.g. ``steps_10_nu_0p001``).

        If no sweep axes are present, a single experiment is registered
        at ``key`` exactly.

        ``plot`` (optional) is registered once at the **parent** ``key`` —
        it runs after all sub-experiments complete and receives either the
        list of per-sub-experiment result dicts (if ``reduce is None``) or
        ``reduce(results)`` otherwise. Plots that need ``exp_key=`` get it
        wrapped in automatically.

        ``plot_description`` is stored on every sub-experiment's params for
        introspection (e.g. ``mosaic status``).
        """
        # Fall back to the descriptions dict if no explicit kwarg given.
        if plot_description is None:
            plot_description = self.descriptions.get(key, "")

        sweep_axes = self._collect_sweep_axes(config)

        if not sweep_axes:
            # No sweep — a single leaf experiment at ``key``.
            self._register_one_experiment(
                key, runner, config, plot_description=plot_description
            )
            sub_keys = [key]
        else:
            n_subs = self._validate_axes(sweep_axes)
            sub_keys = []
            for i in range(n_subs):
                sub_config, suffix = self._materialize_one(config, sweep_axes, i)
                sub_key = f"{key}/{suffix}"
                sub_keys.append(sub_key)
                self._register_one_experiment(
                    sub_key, runner, sub_config, plot_description=plot_description
                )

        if plot is not None:
            self._register_plot(key, plot, sub_keys, reduce)

    def _collect_sweep_axes(self, config: dict) -> list:
        """Return ``[(path_tuple, list_values), ...]`` for every list-of-primitives
        nested one level inside a dict kwarg.

        Sweeps are detected **only inside dict kwargs** (e.g.
        ``physics={"nu": [...]}``), not at the top level — that lets the
        legacy ``runs=[...]`` kwarg pass through unchanged during the
        transition. Top-level lists fan out only when the user marks the
        kwarg explicitly via ``problem.add(..., ic=[{tgv}, {mm}])`` and we
        treat ``ic``'s value as the i-th element.
        """
        axes = []
        for k, v in config.items():
            if k == "runs":
                # Legacy: ``runs=[...]`` is the per-experiment run-list payload,
                # not a sweep axis. Skip.
                continue
            if isinstance(v, list) and self._is_sweep_list(v):
                axes.append(((k,), v))
            elif isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, list) and self._is_sweep_list(sub_v):
                        axes.append(((k, sub_k), sub_v))
        return axes

    @staticmethod
    def _is_sweep_list(value: list) -> bool:
        """A non-empty list whose elements are primitives or dicts is a sweep axis."""
        if not value:
            return False
        head = value[0]
        # Treat numbers / strings / dicts as sweep elements. Skip nested lists
        # (they're presumed to be already-list-valued single elements like
        # ``[16, 32, 64]`` for sweep keys).
        return isinstance(head, (int, float, str, bool, dict))

    @staticmethod
    def _validate_axes(axes: list) -> int:
        """Ensure all sweep axes have matching length. Returns that length."""
        lengths = {len(vals) for _, vals in axes}
        if len(lengths) > 1:
            sketch = ", ".join(f"{'.'.join(p)}={len(v)}" for p, v in axes)
            raise ValueError(
                f"Problem.add: sweep axes have mismatched lengths ({sketch}); "
                f"parallel-zip requires equal-length lists."
            )
        return lengths.pop()

    @staticmethod
    def _materialize_one(config: dict, axes: list, i: int) -> tuple[dict, str]:
        """Build the i-th sub-experiment's config + an auto-derived path suffix."""
        sub_config: dict = {}
        for k, v in config.items():
            if isinstance(v, dict):
                sub_config[k] = {**v}
            else:
                sub_config[k] = v
        parts: list[str] = []
        for path, vals in axes:
            v_i = vals[i]
            if len(path) == 1:
                sub_config[path[0]] = v_i
                name = path[0]
            else:
                k, sub_k = path
                sub_config[k][sub_k] = v_i
                name = sub_k
            parts.append(f"{name}_{Problem._fmt_val(v_i)}")
        suffix = "_".join(parts)
        return sub_config, suffix

    @staticmethod
    def _fmt_val(value) -> str:
        """Filesystem-safe formatting of a sweep value (no dots, no slashes)."""
        if isinstance(value, float):
            s = f"{value:g}"
            return s.replace(".", "p").replace("-", "m")
        if isinstance(value, dict):
            # Lists of dicts (e.g. ic) — use the dict's "name" if present.
            return str(value.get("name", "v"))
        return str(value).replace("/", "_").replace(".", "p")

    def _register_one_experiment(
        self,
        key: str,
        runner: Callable,
        config: dict,
        *,
        plot_description: str = "",
    ) -> None:
        """Build the Experiment lambda for a single (scalar) config and store it."""
        import inspect

        suite, _, exp_key_str = key.partition("/")
        all_deps = {k: v for k, v in self._experiment_deps().items() if v is not None}
        all_deps.update(config)

        sig = set(inspect.signature(runner).parameters)
        runner_kwargs = {k: v for k, v in all_deps.items() if k in sig}
        if "exp_key" in sig:
            runner_kwargs["exp_key"] = exp_key_str or suite

        def fn(cfg, tags, _runner=runner, _kw=runner_kwargs, **call_kw):
            return _runner(cfg, tags, make_inputs=cfg.make_inputs, **_kw, **call_kw)

        # Params is the **introspection manifest** — short and metadata-only.
        # The full config lives in the lambda's closure; we only surface what
        # status/docs actually read (the plot blurb).
        self.experiments[key] = Experiment(
            fn=fn,
            params={"plot_description": plot_description},
        )

    def add_ic(
        self,
        ic_name: str,
        plot_params: dict,
        *,
        plot: Callable | None = None,
    ) -> None:
        """Register an ``ics/<ic_name>`` entry.

        ``plot_params`` is the kwarg bag handed to the IC generator (e.g.
        ``{"N": 64, "U": 1.0}``). It is captured into the Experiment closure
        AND stored on ``Experiment.params`` for introspection.
        """
        from mosaic.benchmarks.shared.ics import run_ic

        make_ic = self.make_ic

        def fn(
            cfg,
            tags,
            _name=ic_name,
            _params=plot_params,
            _make_ic=make_ic,
            **_kw,
        ):
            return run_ic(cfg, _name, make_ic=_make_ic, params=_params)

        self.experiments[f"ics/{ic_name}"] = Experiment(fn=fn, params=dict(plot_params))
        if plot is not None:
            self.plot_fns[f"ics/{ic_name}"] = plot

    def add_extra_plot(self, key: str, plot: Callable) -> None:
        """Register a ``_extra/<suite>/<name>`` plot not tied to an experiment."""
        self.plot_fns[key] = plot

    def _register_plot(
        self,
        key: str,
        plot: Callable,
        sub_keys: list[str],
        reduce: Callable | None,
    ) -> None:
        """Store ``plot`` at ``key``.

        ``sub_keys`` is the list of one-or-more sub-experiment keys this plot
        spans. When ``len(sub_keys) == 1`` and the plot accepts ``exp_key=``,
        the plot is wrapped to pass the sub-key suffix (the legacy single-key
        plot pattern). When there is a real sweep (``len > 1``) the plot is
        wrapped to receive the list of sub-keys via ``sub_keys=`` (or, if a
        ``reduce`` callable is given, the reduced result).
        """
        import inspect

        plot_sig = set(inspect.signature(plot).parameters)
        if len(sub_keys) == 1 and reduce is None:
            single = sub_keys[0]
            _, _, exp_key_str = single.partition("/")
            if "exp_key" in plot_sig:
                self.plot_fns[key] = (
                    lambda cfg, _p=plot, _k=exp_key_str or single, **kw: _p(
                        cfg, exp_key=_k, **kw
                    )
                )
            else:
                self.plot_fns[key] = plot
            return

        # Sweep: the plot spans multiple sub-experiments. Pass them along.
        if "sub_keys" in plot_sig or "results" in plot_sig:
            if reduce is not None:
                self.plot_fns[key] = (
                    lambda cfg, _p=plot, _subs=sub_keys, _r=reduce, **kw: _p(
                        cfg, sub_keys=_subs, results=_r(_subs), **kw
                    )
                )
            else:
                self.plot_fns[key] = lambda cfg, _p=plot, _subs=sub_keys, **kw: _p(
                    cfg, sub_keys=_subs, **kw
                )
        else:
            # Plot doesn't take sub_keys — call it once per sub-key (independent
            # variants like jacobian_svd). The plot reads from its own dir.
            def _multi(cfg, _p=plot, _subs=sub_keys, **kw):
                for sub_key in _subs:
                    _, _, exp_key_str = sub_key.partition("/")
                    if "exp_key" in plot_sig:
                        _p(cfg, exp_key=exp_key_str, **kw)
                    else:
                        _p(cfg, **kw)

            self.plot_fns[key] = _multi

    # ── Validation ───────────────────────────────────────────────────────

    def validate(self) -> None:
        """Check that the Problem is well-formed.  Raises on first error."""
        errors: list[str] = []
        if not self.name:
            errors.append("name is empty")
        if not self.solvers:
            errors.append("no solvers registered")
        for spec in self.solvers:
            for attr in ("name", "dir", "scheme", "backend", "color"):
                if not getattr(spec, attr, None):
                    errors.append(f"solver {spec.name!r}: missing {attr}")
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

        valid_ad = {"autodiff", "adjoint", "hybrid", None}
        for spec in self.solvers:
            if spec.ad_strategy not in valid_ad:
                errors.append(
                    f"solver {spec.name!r}: ad_strategy={spec.ad_strategy!r} "
                    f"not in {valid_ad}"
                )

        if errors:
            raise ValueError(
                f"Problem {self.name!r} validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )


@dataclass(frozen=True)
class Exclusion:
    """Why a solver does not run for a given experiment.

    See :class:`ExclusionCategory` for the category taxonomy.
    """

    category: ExclusionCategory
    reason: str


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
    """Runtime descriptor for a single solver registered in a Problem.

    Most per-solver fields (name, scheme, backend, ad_strategy, differentiable,
    uses_gpu, internal_dtype, description, doc_url) are populated by
    :func:`discover_solvers` from the solver's ``tesseract_config.yaml`` —
    that YAML is the ground truth, problem configs only set per-(solver,
    problem) overrides (``input_overrides`` / ``exclusions`` /
    ``explained_anomalies``).

    Presentation-only fields — ``color`` / ``linestyle`` / ``marker`` — are
    populated by :func:`mosaic.benchmarks.shared.plots.solver_styles.apply_styles`
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
    # ``has_vjp`` (probes the container for a vector_jacobian_product endpoint).
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
    # NB: per-solver exclusions and explained-anomalies live on
    # :attr:`Problem.exclusions` — a problem-level dict keyed by solver
    # name. SolverSpec is solver-identity data only.


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
    plot styling lives in :mod:`mosaic.benchmarks.shared.plots.solver_styles` and is
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
        # ``mosaic.benchmarks.shared.plots.solver_styles`` and are applied by each
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


def has_vjp(spec: SolverSpec) -> bool:
    """Return True if the solver exposes a ``vector_jacobian_product`` endpoint.

    Respects the explicit ``spec.differentiable`` flag when set, avoiding a
    slow container probe for solvers whose differentiability is declared in
    YAML. Falls back to opening the Docker image and querying
    ``available_endpoints`` when the flag is ``None``.
    """
    explicit = getattr(spec, "differentiable", None)
    if explicit is not None:
        return bool(explicit)
    from tesseract_core import Tesseract

    tag = spec.image_tag
    if not tag:
        return False
    try:
        with Tesseract.from_image(tag) as t:
            return "vector_jacobian_product" in t.available_endpoints
    except Exception:
        return False
