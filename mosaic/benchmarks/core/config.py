"""ProblemConfig and SolverSpec dataclasses — the only shared abstraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


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
    dir: str  # subdir under tesseract_dir/
    color: str  # hex, Tol bright palette
    name: str  # solver name for plots
    scheme: str  # numerical scheme, e.g. "LBM BGK", "spectral ETDRK"
    backend: str  # runtime/language, e.g. "jax", "julia", "pytorch"
    family: str = ""  # solver family for grouped styling: "lbm", "projection", "spectral", "fv", "fem", "ml"
    linestyle: str | tuple | None = (
        None  # explicit matplotlib linestyle; bypasses family-palette hue when set
    )
    marker: str | None = (
        None  # explicit matplotlib marker; bypasses family-palette hue when set
    )
    description: str = ""  # one-sentence solver description for reference tables
    doc_url: str = ""  # canonical documentation / repository URL for the solver
    input_overrides: dict[str, Any] = field(default_factory=dict)
    uses_gpu: bool = True  # False for CPU-only solvers (e.g. OpenFOAM, FEniCS)
    image_tag: str | None = None  # override tag for --no-build mode
    normalize_output: Callable | None = (
        None  # convert solver output back to canonical IC units
    )
    differentiable: bool | None = (
        None  # explicit VJP flag; None = runtime detection via _has_vjp
    )
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
    # Floating-point precision used internally by the solver.  One of "float32" or "float64".
    # Solvers that compute in float64 but return float32 outputs may show lower discretisation
    # error at the same resolution — a precision advantage, not a scheme advantage.
    exclusions: dict[str, str] = field(default_factory=dict)
    # Maps suite name (or "gradient", "optimization", "cost", "forward"), "cost", "forward") to a human-readable
    # reason string.  Excluded solvers are skipped by the runner and annotated in docs.
    # Example: {"gradient": "No IC gradient: SU2_CFD_AD only supports boundary DVs.",
    #           "optimization": "Same: IC sensitivity not available."}
    explained_anomalies: dict[str, str | dict] = field(default_factory=dict)
    # Maps experiment keys (same format as exclusions) to documented reasons why this
    # solver is expected to produce anomalous results — the solver CAN run and produces
    # finite output, but underperforms peers for known, method-intrinsic reasons (e.g.
    # LBM O(Ma²) compressibility floor, staggered MAC grid interpolation error).
    # These appear as ANOMALY cells (with the documented reason) rather than EXCLUDED,
    # keeping the solver in the score denominator so weaknesses remain visible.
    # Values: plain string reason, or dict with "reason" key.


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
