"""Problem and SolverSpec dataclasses — the only shared abstraction."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)


# Filesystem root for tesseract solver directories. Derived from this file's
# location so per-problem configs don't have to repeat
# ``Path(__file__).parent.parent.parent.parent / "tesseracts" / <slug>``.
# Relative ``tesseract_dir`` values on :class:`Problem` are resolved against
# this in :meth:`Problem.__post_init__`.
TESSERACTS_DIR: Path = Path(__file__).resolve().parents[2] / "tesseracts"


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


class AdStrategy(str, Enum):
    """How a solver computes gradients.

    Same ``(str, Enum)`` mixin as :class:`ExclusionCategory` — members
    serialise to plain strings on disk (e.g. ``AdStrategy.AUTODIFF == "autodiff"``).

    Members:
      * ``AUTODIFF`` — native reverse-mode AD traces through the forward pass
        (``jax.vjp``, ``torch.autograd``, ``Zygote``, ``wp.Tape``).
      * ``ADJOINT``  — explicitly formulated adjoint equations
        (dolfin-adjoint, pyadjoint, analytic self-adjoint SIMP sensitivity).
      * ``HYBRID``   — analytic gradient rules combined with autodiff
        (implicit-function theorem, custom VJP rules that call autodiff internally).

    The ``None`` sentinel represents *non-differentiable*; it is not a
    member of the enum (so ``ad_strategy: AdStrategy | None`` carries the
    "no gradient support" case explicitly).
    """

    AUTODIFF = "autodiff"
    ADJOINT = "adjoint"
    HYBRID = "hybrid"


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

    Each problem's ``config.py`` instantiates one ``Problem`` (with closure
    deps: ``make_ic``, ``error_fn``, ``output_key``, …), then
    :meth:`add_experiment` / :meth:`add_ic` / :meth:`add_extra_plot`
    register experiments + ICs + plot fns at ``"<suite>/<name>"`` keys.
    The runner / status / CLI consume ``Problem`` directly — there is no
    separate "config" wrapper.
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
    status_checks: dict[str, list] = field(default_factory=dict)

    # ── Experiment-time deps (threaded into Experiment.fn closures) ─────
    make_ic: dict = field(default_factory=dict)
    error_fn: Callable | None = None
    output_key: str = ""
    ic_key: str = ""
    domain_extent: float = 1.0
    resolution_key: str = "N"
    # Analytic reference solution callable (e.g. TGV viscous decay). Threaded
    # into runner closures as the cfg-level fallback when per-run ``reference``
    # is a fine-grid spec dict rather than a callable. Lives at Problem level
    # because the analytic is tied to the problem (TGV analytic for ns-grid,
    # TGV3D for ns-3d-grid), not the experiment.
    reference: Callable | None = None

    # ── Registries (filled by .add_experiment / .add_ic / .add_extra_plot) ─────────
    experiments: dict[str, Experiment] = field(default_factory=dict)
    plot_fns: dict[str, PlotFn] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Setup that can't be expressed declaratively:

        - Resolve a relative :attr:`tesseract_dir` against
          :data:`TESSERACTS_DIR` (so configs can pass a slug string like
          ``"navier-stokes-grid"``).
        - Wrap :attr:`make_inputs` so it looks up the :class:`SolverSpec`
          from the solver name. The user-provided callable has signature
          ``(spec: SolverSpec, ic, **physics) -> dict``; the wrapped
          version exposed as ``cfg.make_inputs`` has signature
          ``(name: str, ic, **physics) -> dict``. This drops the
          per-problem ``build_make_inputs(solvers)`` closure factory.

        ICs are registered explicitly via :meth:`add_ic` after construction —
        no auto-registration from :attr:`make_ic` happens here.
        """
        if isinstance(self.tesseract_dir, str):
            self.tesseract_dir = Path(self.tesseract_dir)
        if self.tesseract_dir.parts and not self.tesseract_dir.is_absolute():
            self.tesseract_dir = TESSERACTS_DIR / self.tesseract_dir

        if self.make_inputs is not None:
            _user_fn = self.make_inputs
            _spec_by_name = {s.name: s for s in self.solvers}

            def _by_name(name: str, ic, **physics):
                return _user_fn(_spec_by_name[name], ic, **physics)

            self.make_inputs = _by_name

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
            "reference": self.reference,
        }

    def add_experiment(
        self,
        key: str,
        kernel: Callable,
        *,
        plot: Callable | None = None,
        plot_description: str = "",
        reduce: Callable | None = None,
        status_check: list | None = None,
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

        ``status_check`` (optional) is a list of check callables (built via
        the factories in :mod:`mosaic.benchmarks.core.status_checks` —
        ``median_k(k)``, ``max_error(t)``, ``min_cosine(t)``, …) attached to
        every sub-experiment's params. Merged on top of any suite-level
        defaults from :attr:`status_checks` when the status pipeline
        classifies a result.

        Per-experiment exclusions are attached via :meth:`exclude` — call
        ``problem.exclude(key, {solver_name: Exclusion, ...})`` immediately
        after this method to register them at the same key.
        """
        # Shorthand: ``run=dict`` wraps into ``runs=[dict]`` and list-valued
        # fields in the run get auto-converted into the runner's sweep dict
        # form (``sweep={"key": k, "values": [...]}`` + scalar override).
        config = self._normalize_run_shorthand(config)

        # Variant fan-out: when ``runs=[...]`` (from explicit kwarg or
        # shorthand list-of-dicts expansion) has >1 element, register
        # each variant as its own sub-experiment at ``<key>/<variant_name>``.
        # The runner is called with ``runs=[single_variant]`` so n_runs==1
        # always — subdir naming lives here at the sweep layer, not in
        # each runner.
        runs = config.get("runs")
        if isinstance(runs, list) and len(runs) > 1:
            sub_keys = []
            for variant in runs:
                variant_name = self._derive_variant_name(variant, key)
                sub_key = f"{key}/{variant_name}"
                sub_keys.append(sub_key)
                sub_config = {**config, "runs": [variant]}
                self._register_one_experiment(
                    sub_key,
                    kernel,
                    sub_config,
                    plot_description=plot_description,
                    status_check=status_check,
                )
        else:
            sweep_axes = self._collect_sweep_axes(config)
            if not sweep_axes:
                # No sweep — a single leaf experiment at ``key``.
                self._register_one_experiment(
                    key,
                    kernel,
                    config,
                    plot_description=plot_description,
                    status_check=status_check,
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
                        sub_key,
                        kernel,
                        sub_config,
                        plot_description=plot_description,
                        status_check=status_check,
                    )

        if plot is not None:
            self._register_plot(key, plot, sub_keys, reduce)

    @staticmethod
    def _derive_variant_name(variant: dict, parent_key: str) -> str:
        """Pick a sub-experiment name for a variant.

        Order: explicit ``variant["name"]`` → ``variant["ic"]["name"]`` →
        error. This matches the de-facto convention previously scattered
        across runners (``run_name = run.get("name", ic_name)``).
        """
        explicit = variant.get("name")
        if explicit:
            return str(explicit)
        ic_name = (
            variant.get("ic", {}).get("name")
            if isinstance(variant.get("ic"), dict)
            else None
        )
        if ic_name:
            return str(ic_name)
        raise ValueError(
            f".add_experiment({parent_key!r}): multi-variant runs= without a "
            f"derivable name (no 'name' field, no ic.name). Add a 'name' "
            f"field to each variant."
        )

    def _normalize_run_shorthand(self, config: dict) -> dict:
        """Convert ``.add_experiment()`` shorthand into the canonical runner payload.

        Two accepted input forms (in order of precedence):

        1. ``runs=[{...}, ...]`` (explicit multi-variant): pass through.
           Use this when variants have **structurally different** payloads
           — e.g. different swept keys per variant (``physical_laws``:
           ``vs_N``, ``vs_steps``, ``vs_nu``).
        2. Bare run-dict fields (``ic=``, ``physics=``, ``fd=``, ``optim=``,
           ``jacobian=``, ``reference=``, ``sweep=``) at the top level:
           collected into a single run dict, then wrapped into
           ``runs=[{...}]``. When any payload field is a **list of dicts**
           (e.g. ``ic=[{tgv}, {multimode}]``), it fans out: each element
           becomes one variant, with shared fields broadcast. Multiple
           list-of-dicts fields parallel-zip (same length required).

        After collection, each run dict is scanned for list-valued fields
        (nested one level inside ``physics``/``fd``/``optim``/etc.). The
        first list found becomes the sweep axis: ``sweep={"key": k, "values":
        [...]}`` is added and the field is replaced with its first value (a
        placeholder; the runner overwrites it per sweep iteration). When an
        explicit ``sweep=`` is supplied via shorthand, auto-detect is skipped.
        """
        # ── Step 1: collect run dict(s) ─────────────────────────────────
        if "runs" in config:
            runs = config["runs"]
            if not isinstance(runs, list):
                raise TypeError(
                    f".add_experiment(runs=...): expected a list, got {type(runs).__name__}"
                )
        else:
            # Collect known run-payload keys into a synthetic single run.
            payload_keys = {
                "ic",
                "physics",
                "fd",
                "optim",
                "jacobian",
                "reference",
                "cost",
                "ic_key",
                "output_key",
                "sweep",
            }
            run_dict = {k: config.pop(k) for k in list(config) if k in payload_keys}
            if not run_dict:
                # No run payload at all (e.g. an IC-registration call); leave
                # config untouched.
                return config
            # Detect list-of-dicts payload fields (e.g. ic=[{tgv}, {mm}])
            # — each one fans out to N variants. Multiple such fields
            # parallel-zip; their lengths must match.
            fanout_keys = [
                k
                for k, v in run_dict.items()
                if isinstance(v, list) and v and isinstance(v[0], dict)
            ]
            if fanout_keys:
                lengths = {len(run_dict[k]) for k in fanout_keys}
                if len(lengths) > 1:
                    sketch = ", ".join(
                        f"{k}=len {len(run_dict[k])}" for k in fanout_keys
                    )
                    raise ValueError(
                        f".add_experiment: fan-out fields have mismatched lengths "
                        f"({sketch}); parallel-zip requires equal-length lists."
                    )
                n = lengths.pop()
                # Deep-copy shared fields so the per-variant sweep auto-detect
                # (which mutates physics/fd/etc. in-place) doesn't leak across
                # variants.
                import copy

                runs = [
                    {
                        k: (v[i] if k in fanout_keys else copy.deepcopy(v))
                        for k, v in run_dict.items()
                    }
                    for i in range(n)
                ]
            else:
                runs = [run_dict]
            config["runs"] = runs

        # ── Step 2: auto-detect sweep in each run dict ─────────────────
        for run in runs:
            if not isinstance(run, dict):
                continue
            if "sweep" in run:
                continue
            sweep_axis = self._find_first_list_axis(run)
            if sweep_axis is None:
                continue
            parent_key, sub_key, values = sweep_axis
            run["sweep"] = {"key": sub_key, "values": values}
            # Replace the list with a placeholder scalar; the runner will
            # override this per sweep iteration. Use the first value as a
            # representative default.
            run[parent_key][sub_key] = values[0]
        return config

    @staticmethod
    def _find_first_list_axis(run: dict) -> tuple[str, str, list] | None:
        """Locate the first list-valued field nested one level inside a
        dict-valued kwarg of ``run``. Returns ``(parent_key, sub_key, values)``
        or ``None`` when no sweep axis is present."""
        for parent_key, parent_val in run.items():
            if not isinstance(parent_val, dict):
                continue
            for sub_key, sub_val in parent_val.items():
                if isinstance(sub_val, list) and Problem._is_sweep_list(sub_val):
                    return parent_key, sub_key, sub_val
        return None

    def _collect_sweep_axes(self, config: dict) -> list:
        """Return ``[(path_tuple, list_values), ...]`` for every list-of-primitives
        nested one level inside a dict kwarg.

        Sweeps are detected **only inside dict kwargs** (e.g.
        ``physics={"nu": [...]}``), not at the top level — ``runs=[...]``
        is the multi-variant payload (passed through to the runner) and
        must not be treated as a sweep axis. Top-level lists in other
        kwargs (e.g. ``ic=[{tgv}, {mm}]``) are treated as the i-th
        sub-experiment's value.

        Detection is one level deep only: ``optim={"adam": {"lr": [...]}}``
        is **not** auto-swept — restructure to ``optim={"lr": [...]}`` or
        provide an explicit ``sweep=`` dict.
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
            sketch = ", ".join(f"{'.'.join(p)}=len {len(v)}" for p, v in axes)
            raise ValueError(
                f"Problem.add_experiment: sweep axes have mismatched lengths ({sketch}); "
                f"parallel-zip requires equal-length lists. Found "
                f"{sorted(lengths)}."
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
        """Filesystem-safe formatting of a sweep value (no dots, no slashes).

        Floats outside the readable decimal range (``< 1e-3`` or ``>= 1e6``)
        render in scientific notation: ``1e-4`` → ``1em4``, ``5e-4`` → ``5em4``,
        ``2.5e7`` → ``2p5e7``. The previous ``:g``-only formatter produced
        ``0p0001`` and ``0p0005`` — uniquely identifying but unreadable.
        Negative values get an ``m`` prefix.
        """
        if isinstance(value, float):
            if value == 0:
                return "0"
            sign = "m" if value < 0 else ""
            abs_v = abs(value)
            if abs_v < 1e-3 or abs_v >= 1e6:
                mantissa_str, exp_str = f"{abs_v:e}".split("e")
                mantissa = mantissa_str.rstrip("0").rstrip(".") or "1"
                exp_int = int(exp_str)
                exp_part = f"m{-exp_int}" if exp_int < 0 else str(exp_int)
                return f"{sign}{mantissa.replace('.', 'p')}e{exp_part}"
            return f"{sign}{abs_v:g}".replace(".", "p")
        if isinstance(value, dict):
            # Lists of dicts (e.g. ic) — use the dict's "name" if present.
            return str(value.get("name", "v"))
        return str(value).replace("/", "_").replace(".", "p")

    def _register_one_experiment(
        self,
        key: str,
        kernel: Callable,
        config: dict,
        *,
        plot_description: str = "",
        status_check: list | None = None,
    ) -> None:
        """Build the Experiment lambda for a single (scalar) config and store it.

        ``kernel`` must be a function decorated with
        :func:`mosaic.benchmarks.core.experiment.kernel`. The framework
        drives the run via :func:`run_experiment` with the kernel's
        attached config plus the per-experiment overrides assembled here.
        """
        import inspect

        from mosaic.benchmarks.core.experiment import (
            get_kernel_config,
            is_kernel,
            run_experiment,
        )

        if not is_kernel(kernel):
            raise TypeError(
                f"add_experiment({key!r}): expected a kernel decorated with "
                f"@kernel(...), got {kernel!r}. Wrap the function with "
                f"`from mosaic.benchmarks.core.experiment import kernel` and "
                f"`@kernel(sweep_mode=..., ...)`."
            )

        suite, _, exp_key_str = key.partition("/")
        kcfg = get_kernel_config(kernel)
        deps = {k: v for k, v in self._experiment_deps().items() if v is not None}
        merged = {**deps, **config}
        run_sig = set(inspect.signature(run_experiment).parameters)
        run_kwargs = {k: v for k, v in merged.items() if k in run_sig}

        # Per-experiment kwargs not consumed by run_experiment or by the
        # problem-level deps (e.g. ``diagnostics`` for physical_laws,
        # ``optimizer`` for topopt) get folded into each run dict so kernels
        # and aggregates can read them off ``ctx.run``.
        extras = {k: v for k, v in config.items() if k not in run_sig and k != "runs"}
        if extras and "runs" in run_kwargs:
            run_kwargs = {
                **run_kwargs,
                "runs": [{**r, **extras} for r in run_kwargs["runs"]],
            }

        def fn(
            cfg,
            tags,
            _kernel=kernel,
            _kw=run_kwargs,
            _kcfg=kcfg,
            _suite=suite,
            _exp=exp_key_str or suite,
            **call_kw,
        ):
            return run_experiment(
                cfg,
                tags,
                _kernel,
                suite=_suite,
                exp_key=_exp,
                make_inputs=cfg.make_inputs,
                **_kcfg,
                **_kw,
                **call_kw,
            )

        # Params is the **introspection manifest** — short and metadata-only.
        # The full config lives in the lambda's closure; we only surface what
        # status/docs actually read (plot blurb + per-experiment thresholds).
        params: dict = {"plot_description": plot_description}
        if status_check:
            params["status_check"] = list(status_check)
        self.experiments[key] = Experiment(fn=fn, params=params)

    def add_ic(
        self,
        name: str,
        fn: Callable,
        *,
        description: str = "",
        plot_params: dict | None = None,
        plot: Callable | None = None,
    ) -> None:
        """Register an initial-condition generator.

        Side effects:
          1. Inserts an :class:`IcSpec` into :attr:`make_ic` at ``name``
             (the runtime registry that runners and runners read from).
          2. Registers an ``ics/<name>`` Experiment that, when run, calls
             the IC generator with ``plot_params`` and saves a snapshot
             plus a plot.
          3. If ``plot`` is given, registers it at ``ics/<name>`` in
             :attr:`plot_fns`.

        ``plot_params`` is the kwarg bag handed to ``fn`` at IC generation
        time (e.g. ``{"N": 64, "U": 1.0}``); the runner overrides ``L`` and
        ``seed`` itself.
        """
        from mosaic.benchmarks.problems.shared.ics import run_ic

        plot_params = dict(plot_params) if plot_params else {}
        self.make_ic[name] = IcSpec(
            fn=fn, description=description, plot_params=plot_params
        )

        make_ic = self.make_ic

        def _exp_fn(
            cfg,
            tags,
            _name=name,
            _params=plot_params,
            _make_ic=make_ic,
            **_kw,
        ):
            return run_ic(cfg, _name, make_ic=_make_ic, params=_params)

        self.experiments[f"ics/{name}"] = Experiment(
            fn=_exp_fn, params=dict(plot_params)
        )
        if plot is not None:
            self.plot_fns[f"ics/{name}"] = plot

    def add_extra_plot(self, key: str, plot: Callable) -> None:
        """Register a ``_extra/<suite>/<name>`` plot not tied to an experiment."""
        self.plot_fns[key] = plot

    def exclude(self, key: str, exclusions: dict[str, Exclusion]) -> None:
        """Attach exclusions at ``key``.

        This is the single entry point for registering exclusions. Use for:

        * Suite-level entries (e.g. ``key="gradient"`` to block a solver from
          every ``gradient/*`` experiment).
        * Per-experiment entries (call this immediately after the matching
          :meth:`add_experiment` with the same key).
        * Per-IC sub-keys (e.g. ``key="forward/agreement/tgv"`` to block a
          single IC variant of a multi-IC
          ``.add_experiment(runs=[...])`` call).

        :func:`exclusion_lookup` does longest-prefix matching against
        :attr:`exclusions`, so a single entry covers every sub-key the
        runner produces below it.
        """
        for solver_name, excl in exclusions.items():
            self.exclusions.setdefault(solver_name, {})[key] = excl

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
        the plot is wrapped to pass the sub-key suffix. When there is a real
        sweep (``len > 1``) the plot is wrapped to receive the list of
        sub-keys via ``sub_keys=`` (or, if a ``reduce`` callable is given,
        the reduced result).
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
            wants_exp_key = "exp_key" in plot_sig

            def _multi(cfg, _p=plot, _subs=sub_keys, _wants=wants_exp_key, **kw):
                for sub_key in _subs:
                    if _wants:
                        _, _, exp_key_str = sub_key.partition("/")
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

        valid_ad: set[AdStrategy | None] = {*AdStrategy, None}
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
    populated by :func:`mosaic.benchmarks.problems.shared.plots.solver_styles.apply_styles`
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
    ad_strategy: AdStrategy | None = None
    # How gradients are computed. See :class:`AdStrategy` for the taxonomy.
    # ``None`` is the non-differentiable sentinel (no gradient support).
    internal_dtype: str = "float32"
    # Floating-point precision used internally by the solver. One of "float32"
    # or "float64". Solvers that compute in float64 but return float32 outputs
    # may show lower discretisation error at the same resolution — a precision
    # advantage, not a scheme advantage.
    # NB: per-solver exclusions and explained-anomalies live on
    # :attr:`Problem.exclusions` — a problem-level dict keyed by solver
    # name. SolverSpec is solver-identity data only.

    def __post_init__(self) -> None:
        # Coerce raw strings from YAML / test fixtures into the AdStrategy enum.
        # ``AdStrategy(str, Enum)`` means the resulting member still compares
        # equal to its string form, so existing string equality checks keep
        # working.
        if isinstance(self.ad_strategy, str) and not isinstance(
            self.ad_strategy, AdStrategy
        ):
            self.ad_strategy = AdStrategy(self.ad_strategy)


def discover_solvers(tesseract_dir: str | Path) -> dict[str, SolverSpec]:
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
    plot styling lives in :mod:`mosaic.benchmarks.problems.shared.plots.solver_styles` and is
    applied to each spec by ``apply_styles()`` after discovery.

    Returns a dict keyed by a normalised solver name (directory name with
    hyphens replaced by underscores). Problem configs can apply per-(solver,
    problem) overrides (``input_overrides``, ``exclusions``,
    ``explained_anomalies``) before publishing the spec dict.

    ``tesseract_dir`` may be a slug string (resolved against
    :data:`TESSERACTS_DIR`) or an absolute :class:`~pathlib.Path`.
    """
    import yaml

    if isinstance(tesseract_dir, str):
        tesseract_dir = Path(tesseract_dir)
    if not tesseract_dir.is_absolute():
        tesseract_dir = TESSERACTS_DIR / tesseract_dir

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
        # ``mosaic.benchmarks.problems.shared.plots.solver_styles`` and are applied by each
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
