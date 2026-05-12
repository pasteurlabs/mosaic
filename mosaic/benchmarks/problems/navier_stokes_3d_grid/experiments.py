"""Assembled ``EXPERIMENTS`` registry for ns-3d-grid.

Every entry is a fully-explicit ``Experiment(fn=lambda ..., params=...)``
literal: the runner, the runs list, every closure-captured dependency, and
the introspection params are all visible at the call site. No helpers, no
dispatch tables — adding/changing an experiment is a local edit on the
entry itself.
"""

from __future__ import annotations

import math

from mosaic.benchmarks.core.config import Experiment
from mosaic.benchmarks.core.utils import l2_error_rel
from mosaic.benchmarks.shared.cost import (
    run_spatial_cost,
    run_temporal_cost,
    run_vjp_cost,
)
from mosaic.benchmarks.shared.forward import run_agreement, run_physical_laws
from mosaic.benchmarks.shared.gradient import (
    run_fd_check,
    run_horizon_sweep,
    run_horizon_sweep_limits,
    run_jacobian_svd,
)
from mosaic.benchmarks.shared.ics import run_ic
from mosaic.benchmarks.shared.optimization import (
    run_recovery_constant_ic,
    run_recovery_constant_ic_bfgs,
    run_recovery_constant_ic_bfgs_proj,
)

from .ics import MAKE_IC, _tgv3d_analytic
from .physics import DIAGNOSTICS

# ── Forward run-lists ────────────────────────────────────────────────────────

_BASELINE_RUNS = [
    {
        "ic": {"name": "tgv3d", "seed": 0},
        "physics": {"N": 16, "nu": 0.05, "dt": 0.01, "steps": 1},
        "sweep": {"key": "N", "values": [8, 16, 32]},
    }
]
_PHYSICAL_LAWS_RUNS = [
    {
        "name": "vs_N",
        "ic": {"name": "tgv3d", "seed": 0},
        "physics": {"nu": 0.05, "dt": 0.01, "steps": 20, "lbm_N_base": 16},
        "sweep": {"key": "N", "values": [8, 16, 32]},
    },
    {
        "name": "vs_steps",
        "ic": {"name": "tgv3d", "seed": 0},
        "physics": {"nu": 0.05, "dt": 0.01, "N": 16, "lbm_N_base": 16},
        "sweep": {"key": "steps", "values": [5, 10, 20, 50]},
    },
    {
        "name": "vs_nu",
        "ic": {"name": "tgv3d", "seed": 0},
        "physics": {"dt": 0.01, "steps": 20, "N": 16, "lbm_N_base": 16},
        "sweep": {"key": "nu", "values": [0.001, 0.01, 0.05, 0.1]},
    },
]
_AGREEMENT_RUNS = [
    {
        "ic": {"name": "tgv3d", "seed": 0},
        "physics": {"N": 16, "dt": 0.01, "steps": 50, "lbm_N_base": 16},
        "sweep": {"key": "nu", "values": [0.001, 0.01, 0.05]},
        # ins_jl removed from fine_set.
        # The ins_jl tesseract container crashes (ContainerDied) when
        # running the fine-grid reference (steps=250, dt=0.002) on a
        # 16³ grid — Julia OOM or resource exhaustion mid-computation.
        # Short runs (steps≤50) work fine; 3D is fully supported.
        # Using only exponax as the fine-grid reference avoids the
        # crash and provides a reliable single-solver consensus anchor.
        "fine": {"solvers": {"exponax"}, "dt": 0.002, "steps": 250},
    }
]

# ── Cost run-list (shared by spatial/temporal/vjp) ───────────────────────────

_COST_RUNS = [
    {
        "physics": {"nu": 0.01, "dt": 0.01, "lbm_N_base": 16},
        "cost": {
            "N_values": [16, 32, 48, 64],
            "steps_values": [10, 50, 100],
            "n_trials": 3,
        },
    }
]

# ── Gradient run-lists ───────────────────────────────────────────────────────

_FD_CHECK_RUNS = [
    {
        "ic": {"name": "tgv3d", "seed": 0},
        "physics": {"N": 16, "nu": 0.001, "dt": 0.05, "steps": 10},
        "fd": {
            "eps_values": [5e0, 1e0, 1e-1, 1e-2, 1e-3, 1e-4],
            "n_dirs": 10,
        },
    }
]
_HORIZON_SWEEP_RUNS = [
    {
        "ic": {"name": "tgv3d", "seed": 0},
        "physics": {"N": 16, "nu": 0.001, "dt": 0.05},
        "fd": {"eps_values": [1e0, 1e-1, 1e-2, 1e-3], "n_dirs": 8},
        "sweep": {"key": "steps", "values": [10, 20, 40, 80, 160]},
    }
]
_HORIZON_SWEEP_LIMITS_RUNS = [
    {
        "ic": {"name": "tgv3d", "seed": 0},
        "physics": {"N": 20, "nu": 0.001, "dt": 0.05},
        "sweep": {
            "key": "steps",
            "values": [40, 80, 160, 320, 640, 1280, 2560, 5120, 10240],
        },
    }
]
_JSVD_BASE_RUNS = [
    {
        "ic": {"name": "tgv3d", "seed": 0},
        "physics": {"N": 8, "nu": 0.001, "dt": 0.05, "steps": 10},
        "jacobian": {"n_alphas": 41, "alpha_range": 0.3},
    }
]
_JSVD_STEPS20_RUNS = [
    {
        "ic": {"name": "tgv3d", "seed": 0},
        "physics": {"N": 8, "nu": 0.001, "dt": 0.05, "steps": 20},
        "jacobian": {"n_alphas": 41, "alpha_range": 0.3},
    }
]
_JSVD_STEPS40_RUNS = [
    {
        "ic": {"name": "tgv3d", "seed": 0},
        "physics": {"N": 8, "nu": 0.001, "dt": 0.05, "steps": 40},
        "jacobian": {"n_alphas": 41, "alpha_range": 0.3},
    }
]
_JSVD_NU01_RUNS = [
    {
        "ic": {"name": "tgv3d", "seed": 0},
        "physics": {"N": 8, "nu": 0.01, "dt": 0.05, "steps": 10},
        "jacobian": {"n_alphas": 41, "alpha_range": 0.3},
    }
]

# ── Optimization run-lists ───────────────────────────────────────────────────

_RECOVERY_CONSTANT_IC_RUNS = [
    {
        "ic": {"name": "rand_div_free", "seed": 0},
        "physics": {"N": 16, "nu": 0.01, "dt": 0.02, "steps": 100},
        "sweep": {"key": "steps", "values": [100]},
        "optim": {
            "ic_init_type": "zeros",
            "lr": 1e-3,
            "max_iters": 500,
            "patience": 50,
            "failure_threshold": 2.0,
            "snap_interval": 20,
            "ic_seeds": [0, 1, 2],
            "record_diagnostics": True,
        },
    }
]
_RECOVERY_CONSTANT_IC_BFGS_RUNS = [
    {
        "ic": {"name": "rand_div_free", "seed": 0},
        "physics": {"N": 16, "nu": 0.01, "dt": 0.02, "steps": 100},
        "sweep": {"key": "steps", "values": [100]},
        "optim": {
            "ic_init_type": "zeros",
            "max_iters": 100,
            "patience": 20,
            "failure_threshold": 2.0,
            "snap_interval": 5,
            "ic_seeds": [0, 1, 2],
            "record_diagnostics": True,
        },
    }
]
_RECOVERY_CONSTANT_IC_BFGS_PROJ_RUNS = [
    {
        "ic": {"name": "rand_div_free", "seed": 0},
        "physics": {"N": 16, "nu": 0.01, "dt": 0.02, "steps": 100},
        "sweep": {"key": "steps", "values": [100]},
        "optim": {
            "ic_init_type": "zeros",
            "max_iters": 100,
            "patience": 20,
            "failure_threshold": 2.0,
            "snap_interval": 5,
            "ic_seeds": [0, 1, 2],
            "record_diagnostics": True,
        },
    }
]


# ── Assembled experiment registry ────────────────────────────────────────────

EXPERIMENTS = {
    # ─ Forward ─
    "forward/baseline": Experiment(
        fn=lambda cfg, tags, **kw: run_agreement(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            error_fn=l2_error_rel,
            output_key="result",
            domain_extent=2 * math.pi,
            analytic=_tgv3d_analytic,
            runs=_BASELINE_RUNS,
            exp_key="baseline",
            **kw,
        ),
        params={
            "runs": _BASELINE_RUNS,
            "plot_description": (
                "Relative error vs N at steps=1; validates single-step forward accuracy across 3D solvers. "
                "Results (ν=0.05, dt=0.01): all 5 solvers valid at all N. "
                "Error at N=32: phiflow 0.0027 (best), ins_jl 0.0033, xlb 0.0071, exponax 0.013. "
                "Error at N=16: phiflow 0.011, ins_jl 0.012, xlb 0.025, exponax 0.052. "
                "Spread (std of errors) at N=32: 0.0035. phiflow most accurate; exponax spectral resolution marginal at N≤16."
            ),
        },
    ),
    "forward/agreement": Experiment(
        fn=lambda cfg, tags, **kw: run_agreement(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            error_fn=l2_error_rel,
            output_key="result",
            domain_extent=2 * math.pi,
            analytic=_tgv3d_analytic,
            runs=_AGREEMENT_RUNS,
            exp_key="agreement",
            **kw,
        ),
        params={
            "runs": _AGREEMENT_RUNS,
            "plot_description": (
                "3D velocity magnitude fields and kinetic energy spectra for each solver, "
                "sweeping viscosity ν. Initial condition: 3D Taylor-Green vortex (N=16). "
                "Reference: exponax + ins_jl fine-grid consensus. "
                "Production results (N=16, steps=50, ν∈{0.001,0.01,0.05}, 6 solvers including phiflow after CG fix): "
                "5-solver reference (excluding phiflow): warp_ns/openfoam best (~0.014–0.016), exponax (~0.017–0.018), ins_jl (~0.025–0.034), xlb highest (~0.036–0.048). "
                "phiflow: error ~0.096–0.100 vs 5-solver reference (valid after CG fix but highest error — numerical dissipation from semi-Lagrangian + CG pressure solve). "
                "6-solver trimmed mean inflated ~12% due to phiflow outlier. "
                "Errors are ν-independent (consistent across all 3 viscosity values). "
                "warp_ns/openfoam cluster together (IPCS projection); xlb error consistent with LBM mass leakage (F-NS3D-1). "
                "F-NS3D-9 RESOLVED: phiflow explicit CG pressure solver (tol=1e-5) prevents 3D NaN at all tested steps/nu. "
                "F-NS3D-11: phiflow valid but least accurate at steps=50 (~10% vs 5-solver consensus); semi-Lagrangian dissipation increases with rollout length."
            ),
        },
    ),
    "forward/physical_laws": Experiment(
        fn=lambda cfg, tags, **kw: run_physical_laws(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            error_fn=l2_error_rel,
            output_key="result",
            domain_extent=2 * math.pi,
            analytic=_tgv3d_analytic,
            diagnostics=DIAGNOSTICS,
            runs=_PHYSICAL_LAWS_RUNS,
            **kw,
        ),
        params={
            "runs": _PHYSICAL_LAWS_RUNS,
            "plot_description": (
                "Divergence RMS and kinetic energy vs N / steps / ν for each solver; validates incompressibility and energy cascade in 3D. "
                "vs_N (steps=20, ν=0.05): xlb divergence_rms GROWS with N (4.7e-3→1.2e-2→2.0e-2, LBM mass leakage F-NS3D-1); "
                "exponax/ins_jl converge ~O(1/N²); phiflow div_rms=1.7e-3 (N=8) → 7.8e-4 (N=16) → 1.9e-4 (N=32), converging (CG pressure fix). "
                "vs_steps (N=16, ν=0.05): xlb div_rms persistent ~0.010–0.013 (LBM mass leakage, steps-independent); "
                "phiflow div_rms grows: 2.0e-4 (steps=5) → 3.9e-4 (steps=10) → 7.8e-4 (steps=20) → 2.4e-3 (steps=50) — semi-Lagrangian accumulation. "
                "exponax div_rms minimal at all steps. All phiflow points valid after CG fix (F-NS3D-9 RESOLVED). "
                "vs_nu (N=16, steps=20): all solvers ν-independent divergence (xlb ~0.01, exponax ~3e-4, phiflow ~8e-4 to 9e-4). "
                "KE decay is physical for all solvers. LBM N-growing divergence extends F1.1b to 3D. "
                "phiflow semi-Lagrangian div_rms ~8e-4 (N=16): higher than spectral/FD but lower than LBM; grows with rollout length."
            ),
        },
    ),
    # ─ Cost ─
    "cost/spatial_cost": Experiment(
        fn=lambda cfg, tags, **kw: run_spatial_cost(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            domain_extent=2 * math.pi,
            resolution_key="N",
            runs=_COST_RUNS,
            **kw,
        ),
        params={
            "runs": _COST_RUNS,
            "plot_description": (
                "Forward-pass wall-clock time vs N (steps=10, ν=0.01, 3 trials). "
                "Production results (6 solvers: exponax, xlb, ins_jl, warp_ns, openfoam, phiflow): "
                "exponax: 15ms (N=8), 16ms (N=16), 30ms (N=32) — nearly flat (FFT-accelerated, memory-bound). "
                "xlb: 9ms (N=8), 9ms (N=16), 26ms (N=32) — fastest GPU solver at all N. "
                "phiflow: 25ms (N=8), 49ms (N=16), 233ms (N=32) — semi-Lagrangian, ~3× slower than xlb. "
                "ins_jl: 100ms (N=8), 328ms (N=16), 2673ms (N=32) — CPU Julia FD, strong O(N³) scaling. "
                "openfoam: 713ms (N=8), 1069ms (N=16), 4470ms (N=32) — PISO, file I/O overhead. "
                "warp_ns: 3796ms (N=8), 3760ms (N=16), 3663ms (N=32) — constant time (fixed internal grid). "
                "Cost hierarchy (N=16): xlb < exponax < phiflow << ins_jl < openfoam < warp_ns."
            ),
        },
    ),
    "cost/temporal_cost": Experiment(
        fn=lambda cfg, tags, **kw: run_temporal_cost(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            domain_extent=2 * math.pi,
            resolution_key="N",
            runs=_COST_RUNS,
            **kw,
        ),
        params={
            "runs": _COST_RUNS,
            "plot_description": (
                "Forward-pass wall-clock time vs steps (N=16, ν=0.01, 3 trials). "
                "Production results (6 solvers): "
                "exponax: 9ms (steps=10) → 18ms (50) → 27ms (100) — linear in steps (scan unroll). "
                "xlb: 9ms → 14ms → 13ms — nearly constant (XLA scan + GPU amortization). "
                "phiflow: 32ms (steps=10) → 61ms (50) → 77ms (100) — sub-linear scaling (semi-Lagrangian overhead dominates). "
                "ins_jl: 97ms → 336ms → 649ms — approximately O(steps) CPU solver. "
                "openfoam: 740ms → 1094ms → 1465ms — mild scaling, I/O dominated. "
                "warp_ns: 799ms → 3598ms → 7012ms — strong O(steps) scaling (no scan/JIT overhead). "
                "phiflow faster than ins_jl/openfoam/warp_ns for all step counts at N=16."
            ),
        },
    ),
    "cost/vjp_cost": Experiment(
        fn=lambda cfg, tags, **kw: run_vjp_cost(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            domain_extent=2 * math.pi,
            resolution_key="N",
            output_key="result",
            ic_key="v0",
            runs=_COST_RUNS,
            **kw,
        ),
        params={
            "runs": _COST_RUNS,
            "plot_description": (
                "VJP wall-clock time vs N (steps=10, ν=0.01, 3 trials). "
                "Production results (5 differentiable solvers: exponax, xlb, ins_jl, warp_ns, phiflow): "
                "exponax: 123ms (N=8), 123ms (N=16), 145ms (N=32) — flat VJP cost (FFT, memory-bound; ratio fwd/VJP ~8×). "
                "phiflow: 145ms (N=8), 229ms (N=16), 998ms (N=32) — lowest VJP cost after exponax; 3× forward cost. "
                "xlb: 910ms (N=8), 920ms (N=16), 1322ms (N=32) — JIT-compiled XLA scan VJP (~100× forward). "
                "ins_jl: 825ms (N=8), 2564ms (N=16), 14259ms (N=32) — strong O(N³) VJP scaling (Zygote AD). "
                "warp_ns: 18463ms (N=8), 11631ms (N=16), 12066ms (N=32) — highest VJP cost (wp.Tape, no scan; N-flat). "
                "VJP hierarchy (N=16): exponax < phiflow << xlb ≈ ins_jl << warp_ns. "
                "phiflow VJP cost intermediate: 1.9× exponax but 4× cheaper than xlb; semi-Lagrangian backprop efficient."
            ),
        },
    ),
    # ─ Gradient ─
    "gradient/fd_check": Experiment(
        fn=lambda cfg, tags, **kw: run_fd_check(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            output_key="result",
            ic_key="v0",
            domain_extent=2 * math.pi,
            runs=_FD_CHECK_RUNS,
            exp_key="fd_check",
            **kw,
        ),
        params={
            "runs": _FD_CHECK_RUNS,
            "plot_description": (
                "FD gradient error U-curves and direction cosine for the 3D Taylor-Green vortex IC (N=16, shape 16³). "
                "Results (N=16, ν=0.001, dt=0.05, steps=10, 5 solvers): "
                "xlb OUTSTANDING — cosine~1.0 at all ε values including ε=0.0001 (cosine=0.9982); "
                "ins_jl EXCELLENT — cosine~1.0 at ε≥0.001 (cosine=0.9970 at ε=0.0001); "
                "exponax GOOD — cosine=0.9995 at ε=0.01, degrades at small ε due to FP noise (cosine=0.404 at ε=0.0001); "
                "warp_ns FAIL — flat cosine~0.973 across ALL ε (systematic gradient direction error, no U-curve). "
                "phiflow EXCELLENT — rel_err~6e-5 at ε=5.0, ~3e-5 at ε=1.0, cosine=1.0000; "
                "root cause of prior flat-plateau bias (rel_err~2e-2) was semi_lagrangian self-advection VJP missing "
                "cross-terms from backtracking position; fixed by replacing with explicit Euler differential advection "
                "(vel + dt * advect.differential(vel, vel)) in the periodic step function. "
                "3D finding: xlb VJP CORRECT in 3D (F-NS3D-2) — contrasts with broken 2D xlb VJP. "
                "warp_ns systematic scale error confirmed by flat cosine plateau (F-NS3D-3). "
                "phiflow gradient correct: rel_err < 1e-4, cosine=1.0000 at both ε values tested."
            ),
        },
    ),
    "gradient/horizon_sweep": Experiment(
        fn=lambda cfg, tags, **kw: run_horizon_sweep(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            output_key="result",
            ic_key="v0",
            domain_extent=2 * math.pi,
            runs=_HORIZON_SWEEP_RUNS,
            exp_key="horizon_sweep",
            **kw,
        ),
        params={
            "runs": _HORIZON_SWEEP_RUNS,
            "plot_description": (
                "Gradient norm, FD error, and direction cosine vs rollout horizon (T = steps × dt) for the 3D TGV. "
                "Production results (5 solvers: exponax, xlb, ins_jl, warp_ns, phiflow): "
                "exponax: cosine(ε=0.01)≥0.999 at T≤4s (steps≤80); chaos onset at T=8s (steps=160, cosine(ε=1.0)=0.566). Gradient norm grows 68→240 (vortex stretching). "
                "xlb: OUTSTANDING — cosine~1.0 at ALL horizons including steps=160 (T=8s). Gradient norm 57-210. Best 3D gradient solver. "
                "ins_jl: EXCELLENT — cosine~1.0 at all steps. Gradient norm 51-58 (stable). Second-best gradient solver. "
                "warp_ns: CATASTROPHIC — cosine~0.975 at steps=10 (F-NS3.3 scale error); degrades to 0.5 at steps=40; "
                "0.33 at steps=80 (grad_norm=34177, exploding); NaN at steps=160. "
                "phiflow: GOOD at short horizons — cosine~0.9998 at steps=10-40 (T≤2s); "
                "degrades at steps=80 (cosine~0.92 at ε=1.0 — chaos onset); "
                "FAILS at steps=160 (cosine=-0.33 at ε=1.0 — WRONG DIRECTION; semi-Lagrangian dissipation causes gradient reversal). "
                "Contrast: 2D TGV maintains cosine>0.96 at T=64s; 2D norm DECAYS monotonically. "
                "3D TGV chaos onset (for exponax) at T≈8s, ≥16× faster than 2D. "
                "xlb and ins_jl maintain gradient quality through chaos onset — their VJPs are more numerically stable than spectral/semi-Lagrangian methods in the chaotic regime. "
                "phiflow gradient fails at chaos onset (steps≥80, T≥4s) due to semi-Lagrangian numerical dissipation amplifying in the chaotic regime."
            ),
        },
    ),
    "gradient/horizon_sweep_limits": Experiment(
        fn=lambda cfg, tags, **kw: run_horizon_sweep_limits(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            output_key="result",
            ic_key="v0",
            domain_extent=2 * math.pi,
            runs=_HORIZON_SWEEP_LIMITS_RUNS,
            exp_key="horizon_sweep_limits",
            **kw,
        ),
        params={
            "runs": _HORIZON_SWEEP_LIMITS_RUNS,
            "plot_description": (
                "Per-solver rollout-limit table: step count at first failure, failure type, "
                "and wall time for each successful step. GPU solvers (xlb, phiflow, exponax, "
                "pict, warp_ns) expected to OOM or NaN; CPU solver ins_jl "
                "expected to show RAM growth with rollout length."
            ),
        },
    ),
    "gradient/jacobian_svd": Experiment(
        fn=lambda cfg, tags, **kw: run_jacobian_svd(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            output_key="result",
            ic_key="v0",
            domain_extent=2 * math.pi,
            runs=_JSVD_BASE_RUNS,
            exp_key="jacobian_svd",
            **kw,
        ),
        params={
            "runs": _JSVD_BASE_RUNS,
            "plot_description": (
                "Per-solver singular value spectra and cross-solver cosine similarity for the 3D TGV IC (N=8, ν=0.001, steps=10). "
                "Solvers: exponax, xlb, ins_jl (warp_ns INFEASIBLE: 1536 × 18s = 7.7h). "
                "Cross-solver cosine similarity: exponax–xlb=0.754, exponax–ins_jl=0.704, xlb–ins_jl=0.578. "
                "Condition numbers: exponax=1.84 (well-conditioned), xlb=228 (moderate), ins_jl=2.3e12 (nearly singular). "
                "Effective rank: exponax 1516/1536 (full-rank Jacobian), xlb 1196, ins_jl 785. "
                "Gradient norms: exponax=1.36, xlb=1.08, ins_jl=1.19 (similar magnitudes). "
                "Key finding (F-NS3D-8): exponax Jacobian is most informative (highest effective rank, lowest condition); "
                "ins_jl Jacobian nearly singular (condition 2.3e12) suggesting redundant gradient directions; "
                "LBM (xlb) and FD (ins_jl) gradient subspaces disagree most (cosine=0.578) — scheme-level structural divergence."
            ),
        },
    ),
    "gradient/jacobian_svd_steps20": Experiment(
        fn=lambda cfg, tags, **kw: run_jacobian_svd(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            output_key="result",
            ic_key="v0",
            domain_extent=2 * math.pi,
            runs=_JSVD_STEPS20_RUNS,
            exp_key="jacobian_svd_steps20",
            **kw,
        ),
        params={
            "runs": _JSVD_STEPS20_RUNS,
            "plot_description": (
                "Per-solver singular value spectra and cross-solver cosine similarity for the 3D TGV IC (N=8, nu=0.001, steps=20). "
                "Extended horizon vs base jacobian_svd (steps=10); probes gradient subspace alignment deeper into the chaotic regime."
            ),
        },
    ),
    "gradient/jacobian_svd_steps40": Experiment(
        fn=lambda cfg, tags, **kw: run_jacobian_svd(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            output_key="result",
            ic_key="v0",
            domain_extent=2 * math.pi,
            runs=_JSVD_STEPS40_RUNS,
            exp_key="jacobian_svd_steps40",
            **kw,
        ),
        params={
            "runs": _JSVD_STEPS40_RUNS,
            "plot_description": (
                "Per-solver singular value spectra and cross-solver cosine similarity for the 3D TGV IC (N=8, nu=0.001, steps=40). "
                "At T=2s the TGV is well into the chaotic regime; tests how gradient subspace structure evolves at chaos onset."
            ),
        },
    ),
    "gradient/jacobian_svd_nu01": Experiment(
        fn=lambda cfg, tags, **kw: run_jacobian_svd(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            output_key="result",
            ic_key="v0",
            domain_extent=2 * math.pi,
            runs=_JSVD_NU01_RUNS,
            exp_key="jacobian_svd_nu01",
            **kw,
        ),
        params={
            "runs": _JSVD_NU01_RUNS,
            "plot_description": (
                "Per-solver singular value spectra and cross-solver cosine similarity for the 3D TGV IC (N=8, nu=0.01, steps=10). "
                "10x more viscous than the base jacobian_svd (nu=0.001); expected to reduce condition number and raise effective rank."
            ),
        },
    ),
    # ─ Optimization ─
    "optimization/recovery_constant_ic": Experiment(
        fn=lambda cfg, tags, **kw: run_recovery_constant_ic(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            error_fn=l2_error_rel,
            output_key="result",
            domain_extent=2 * math.pi,
            runs=_RECOVERY_CONSTANT_IC_RUNS,
            **kw,
        ),
        params={
            "runs": _RECOVERY_CONSTANT_IC_RUNS,
            "plot_description": (
                "Final IC recovery error from zero-initialised optimisation "
                "(N=16, ν=0.01, dt=0.02, steps=100, rand_div_free seeds 0-2)."
            ),
        },
    ),
    "optimization/recovery_constant_ic_bfgs": Experiment(
        fn=lambda cfg, tags, **kw: run_recovery_constant_ic_bfgs(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            error_fn=l2_error_rel,
            output_key="result",
            domain_extent=2 * math.pi,
            runs=_RECOVERY_CONSTANT_IC_BFGS_RUNS,
            **kw,
        ),
        params={
            "runs": _RECOVERY_CONSTANT_IC_BFGS_RUNS,
            "plot_description": (
                "Final IC recovery error from zero-initialised L-BFGS optimisation "
                "(N=16, ν=0.01, dt=0.02, steps=100, rand_div_free seeds 0-2)."
            ),
        },
    ),
    "optimization/recovery_constant_ic_bfgs_proj": Experiment(
        fn=lambda cfg, tags, **kw: run_recovery_constant_ic_bfgs_proj(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            error_fn=l2_error_rel,
            output_key="result",
            domain_extent=2 * math.pi,
            runs=_RECOVERY_CONSTANT_IC_BFGS_PROJ_RUNS,
            **kw,
        ),
        params={
            "runs": _RECOVERY_CONSTANT_IC_BFGS_PROJ_RUNS,
            "plot_description": (
                "Final IC recovery error from zero-initialised L-BFGS+projection optimisation "
                "(N=16, ν=0.01, dt=0.02, steps=100, rand_div_free seeds 0-2)."
            ),
        },
    ),
    # ─ ICs ─
    "ics/tgv3d": Experiment(
        fn=lambda cfg, tags, **kw: run_ic(
            cfg,
            "tgv3d",
            make_ic=MAKE_IC,
            params={"N": 32},
        ),
        params={"N": 32},
    ),
    "ics/abc": Experiment(
        fn=lambda cfg, tags, **kw: run_ic(
            cfg,
            "abc",
            make_ic=MAKE_IC,
            params={"N": 32},
        ),
        params={"N": 32},
    ),
    "ics/rand_div_free": Experiment(
        fn=lambda cfg, tags, **kw: run_ic(
            cfg,
            "rand_div_free",
            make_ic=MAKE_IC,
            params={"N": 32},
        ),
        params={"N": 32},
    ),
}
