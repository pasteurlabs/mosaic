from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mosaic.benchmarks.core.config import (
    IcSpec,
    ProblemConfig,
    SolverSpec,
    discover_solvers,
)
from mosaic.benchmarks.core.utils import l2_error_rel
from mosaic.benchmarks.plots.solver_styles import apply_styles

_GYM_DIR = Path(__file__).parent.parent.parent
_TESSERACT_DIR = _GYM_DIR / "tesseracts" / "navier-stokes-grid"


# ── Solver registry ──────────────────────────────────────────────────────────
# Solvers and per-solver metadata come from each tesseract's YAML; styling is
# applied from mosaic.benchmarks.plots.solver_styles; only per-(solver, problem)
# overrides — exclusions, input_overrides, explained_anomalies, plus a few
# 3D-specific scheme/description tweaks — are set here.

_SOLVERS: dict[str, SolverSpec] = discover_solvers(_TESSERACT_DIR)

# JAX-CFD is a 2D-only solver (spectral pressure solve doesn't generalise to
# the 3D periodic-box benchmark configuration); drop it from the 3D suite.
_SOLVERS.pop("jax_cfd", None)

# Preserve historical solver key.
_SOLVERS["ins_jl"] = _SOLVERS.pop("incompressible_navier_stokes_jl")

apply_styles(_SOLVERS)

# ── 3D-specific scheme / description overrides ───────────────────────────────
# These differ from the 2D defaults shipped in the tesseract YAML because the
# same tesseract image runs a different lattice / pressure-solver / wrapper in
# the 3D periodic-box configuration.

_SOLVERS["phiflow"].scheme = "differential advection + projection"
_SOLVERS[
    "phiflow"
].description = "Explicit Euler differential advection with CG pressure projection."
_SOLVERS["phiflow"].ad_strategy = "autodiff"

_SOLVERS["xlb"].scheme = "LBM KBC/BGK D3Q27"
_SOLVERS["xlb"].description = (
    "JAX-accelerated LBM D3Q27 using XLB's KBC (entropic-stabilised, selected "
    "when omega > 1.8 or an obstacle is present) or BGK (default periodic), "
    "Stream, QuadraticEquilibrium, and Macroscopic operators (xlb 0.3.1) "
    "inside a jax.lax.scan time loop; incompressibility recovered in the "
    "low-Mach limit. VJP flows through the full scan unroll. V100 (CC 7.0) "
    "cuBLAS GEMM autotuning race fixed via --xla_gpu_enable_cublaslt=false "
    "--xla_gpu_autotune_level=0 in tesseract config."
)

_SOLVERS["ins_jl"].description = (
    "Julia finite-difference solver with pressure projection; CPU-only. "
    "Gradients via Zygote.jl reverse-mode AD."
)

_SOLVERS["warp_ns"].scheme = "IPCS (3D), wp.Tape VJP"
_SOLVERS[
    "warp_ns"
].description = "NVIDIA Warp NS solver: 3D IPCS projection. VJP via wp.Tape."


# ── Per-(solver, problem) overrides ──────────────────────────────────────────

_SOLVERS["exponax"].input_overrides = {
    "drag": jnp.array([0.0], dtype=jnp.float32),
    "order": 2,
    "kolmogorov_forcing": False,
    "injection_mode": 4,
    "injection_scale": jnp.array([1.0], dtype=jnp.float32),
}

_SOLVERS["phiflow"].exclusions = {
    "cost/temporal_cost": {
        "category": "infeasible",
        "reason": (
            "CUDA OOM during JAX CUDA graph profiling in 3D cost benchmark "
            "(allocate 16.09MiB failed); xla_gpu_autotune_level=0 fix deployed "
            "— pending re-run to confirm resolved"
        ),
    },
}

_SOLVERS["openfoam"].exclusions = {
    "gradient": {
        "category": "categorical",
        "reason": (
            "standard icoFoam is non-differentiable (C++, no AD path); "
            "DAFoam/OpenFOAM-AD exist but are not deployed in this tesseract"
        ),
    },
    "optimization": {
        "category": "categorical",
        "reason": "standard icoFoam is non-differentiable forward-only solver",
    },
    "cost/vjp_cost": {
        "category": "categorical",
        "reason": "standard icoFoam has no VJP to benchmark",
    },
}

# ── IC generators ─────────────────────────────────────────────────────────────


def _tgv3d(N: int, L: float = 2 * jnp.pi, seed: int = 0, **_) -> jax.Array:
    """3D Taylor-Green vortex: u=sin(x)cos(y)cos(z), v=-cos(x)sin(y)cos(z), w=0.

    Divergence-free initial condition for the canonical 3D TGV benchmark.
    Develops turbulent vortex structures at moderate Re; kinetic energy
    dissipation rate peaks around t≈9/ν.

    Returns shape (N, N, N, 3), float32.
    """
    x = jnp.linspace(0, L, N, endpoint=False)
    y = jnp.linspace(0, L, N, endpoint=False)
    z = jnp.linspace(0, L, N, endpoint=False)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    vx = jnp.sin(X) * jnp.cos(Y) * jnp.cos(Z)
    vy = -jnp.cos(X) * jnp.sin(Y) * jnp.cos(Z)
    vz = jnp.zeros_like(vx)
    return jnp.stack([vx, vy, vz], axis=-1).astype(jnp.float32)


def _abc_flow(
    N: int,
    L: float = 2 * jnp.pi,
    A: float = 1.0,
    B: float = 0.8165,  # sqrt(2/3)
    C: float = 0.5774,  # sqrt(1/3)
    seed: int = 0,
    **_,
) -> jax.Array:
    """Arnold-Beltrami-Childress (ABC) flow — a 3D Beltrami field.

    Exact steady solution to the Euler equations with helicity.  Particle
    trajectories are chaotic for A≈B≈C≈1, making this a demanding test for
    gradient signal retention at long horizons.

    u = A*sin(z) + C*cos(y)
    v = B*sin(x) + A*cos(z)
    w = C*sin(y) + B*cos(x)

    Returns shape (N, N, N, 3), float32, normalised to unit max speed.
    """
    x = jnp.linspace(0, L, N, endpoint=False)
    y = jnp.linspace(0, L, N, endpoint=False)
    z = jnp.linspace(0, L, N, endpoint=False)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    vx = A * jnp.sin(Z) + C * jnp.cos(Y)
    vy = B * jnp.sin(X) + A * jnp.cos(Z)
    vz = C * jnp.sin(Y) + B * jnp.cos(X)
    v = jnp.stack([vx, vy, vz], axis=-1)
    u_max = float(jnp.sqrt(jnp.sum(v**2, axis=-1)).max()) or 1.0
    return (v / u_max).astype(jnp.float32)


def _rand_div_free_3d(
    N: int,
    L: float = 2 * jnp.pi,
    seed: int = 0,
    k_peak: float = 2.0,
    k_width: float = 1.0,
    **_,
) -> jax.Array:
    """Random divergence-free 3D velocity field via curl of a spectral vector potential.

    Generates three random Fourier vector-potential components with energy
    concentrated in a shell at |k| = k_peak (width k_width), then computes
    u = curl(A) in Fourier space (exactly divergence-free: div(curl(A)) = 0).
    Normalised to unit max speed.
    """
    key = jax.random.PRNGKey(seed)
    kn = jnp.fft.fftfreq(N, d=1.0 / N)
    KX, KY, KZ = jnp.meshgrid(kn, kn, kn, indexing="ij")
    K_abs = jnp.sqrt(KX**2 + KY**2 + KZ**2)
    envelope = jnp.exp(-0.5 * ((K_abs - k_peak) / k_width) ** 2)
    keys = jax.random.split(key, 6)
    kfac = 2.0 * jnp.pi / L
    Ax = (
        jax.random.normal(keys[0], (N, N, N))
        + 1j * jax.random.normal(keys[1], (N, N, N))
    ) * envelope
    Ay = (
        jax.random.normal(keys[2], (N, N, N))
        + 1j * jax.random.normal(keys[3], (N, N, N))
    ) * envelope
    Az = (
        jax.random.normal(keys[4], (N, N, N))
        + 1j * jax.random.normal(keys[5], (N, N, N))
    ) * envelope
    ux_hat = 1j * kfac * (KY * Az - KZ * Ay)
    uy_hat = 1j * kfac * (KZ * Ax - KX * Az)
    uz_hat = 1j * kfac * (KX * Ay - KY * Ax)
    vx = jnp.real(jnp.fft.ifftn(ux_hat))
    vy = jnp.real(jnp.fft.ifftn(uy_hat))
    vz = jnp.real(jnp.fft.ifftn(uz_hat))
    u_max = float(jnp.sqrt(vx**2 + vy**2 + vz**2).max()) or 1.0
    return jnp.stack([vx / u_max, vy / u_max, vz / u_max], axis=-1).astype(jnp.float32)


# ── Analytic reference ────────────────────────────────────────────────────────


def _tgv3d_analytic(
    ic: jax.Array, nu: float, t: float, L: float = 2 * jnp.pi
) -> jax.Array:
    """Exact 3D TGV viscous decay: u(t) = u(0) * exp(-2*nu*t)."""
    N = ic.shape[0]
    x = jnp.linspace(0, L, N, endpoint=False)
    y = jnp.linspace(0, L, N, endpoint=False)
    z = jnp.linspace(0, L, N, endpoint=False)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    decay = jnp.exp(-2.0 * nu * t)
    vx = jnp.sin(X) * jnp.cos(Y) * jnp.cos(Z) * decay
    vy = -jnp.cos(X) * jnp.sin(Y) * jnp.cos(Z) * decay
    vz = jnp.zeros_like(vx)
    return jnp.stack([vx, vy, vz], axis=-1).astype(jnp.float32)


# ── Input factory ─────────────────────────────────────────────────────────────

_LBM_SOLVERS = {"xlb"}


def _make_inputs(
    solver_name: str,
    ic: jax.Array,
    *,
    nu: float,
    dt: float,
    steps: int,
    domain_extent: float = 2 * jnp.pi,
    lbm_N_base: int | None = None,
    **_,
) -> dict:
    """Build solver input dict, applying LBM dt-scaling when lbm_N_base is set.

    For standard 3-D periodic runs ic has shape (N, N, N, 3) and is passed
    directly as v0.
    """
    spec = _SOLVERS[solver_name]

    N = ic.shape[0]
    _dt, _steps = dt, steps

    if solver_name in _LBM_SOLVERS and lbm_N_base is not None:
        _dt = dt * (lbm_N_base / N)
        _steps = max(1, round(steps * (N / lbm_N_base)))

    base = {
        "v0": ic,
        "viscosity": jnp.array([nu], dtype=jnp.float32),
        "dt": jnp.array([_dt], dtype=jnp.float32),
        "steps": _steps,
        "domain_extent": float(domain_extent),
    }
    return {**base, **spec.input_overrides}


# ── Field visualisation helpers ───────────────────────────────────────────────


def _field_to_2d(v: np.ndarray) -> np.ndarray:
    """Extract a 2-D scalar from a 3-D velocity field (N,N,N,3).

    Returns the z-component of vorticity on the middle-z slice,
    shape (N, N).  Used as the primary visualisation slice for 3D field plots.
    """
    import numpy as np

    N = v.shape[0]
    zmid = N // 2
    vx = np.array(v[:, :, zmid, 0])
    vy = np.array(v[:, :, zmid, 1])
    dvydx = (np.roll(vy, -1, 0) - np.roll(vy, 1, 0)) * 0.5
    dvxdy = (np.roll(vx, -1, 1) - np.roll(vx, 1, 1)) * 0.5
    return (dvydx - dvxdy).astype(np.float32)


# ── Diagnostics ───────────────────────────────────────────────────────────────


def _divergence_rms(arr: jax.Array, domain_extent: float = 2 * jnp.pi, **_) -> float:
    """RMS divergence ∇·u for 3D (N,N,N,3) fields."""
    dx = domain_extent / arr.shape[0]
    div = sum(
        (jnp.roll(arr[..., i], -1, i) - jnp.roll(arr[..., i], 1, i)) for i in range(3)
    ) / (2 * dx)
    return float(jnp.sqrt(jnp.mean(div**2)))


def _kinetic_energy(arr: jax.Array, **_) -> float:
    """Mean kinetic energy ½〈|u|²〉 for 3D fields."""
    return float(0.5 * jnp.mean(jnp.sum(arr**2, axis=-1)))


def _energy_spectrum(arr: jax.Array, **_) -> dict:
    """Isotropic 1-D energy spectrum E(k) for 3D (N,N,N,3) fields."""
    N = arr.shape[0]
    kn = jnp.fft.fftfreq(N, d=1.0 / N)
    v_hat = [jnp.fft.fftn(arr[..., d]) / N**3 for d in range(3)]
    axes = jnp.meshgrid(kn, kn, kn, indexing="ij")
    E_hat = 0.5 * sum(jnp.abs(vh) ** 2 for vh in v_hat)
    K = jnp.sqrt(sum(ax**2 for ax in axes))
    k_bins = np.arange(1, N // 2)
    E_k = jnp.array(
        [float(E_hat[(k - 0.5 <= K) & (k + 0.5 > K)].sum()) for k in k_bins]
    )
    return {"k": k_bins.tolist(), "E_k": E_k.tolist()}


# ── Config instance ───────────────────────────────────────────────────────────

CONFIG = ProblemConfig(
    name="ns-3d-grid",
    category_label="Navier–Stokes (Grid)",
    n_to_cells=lambda n: n**3,
    description=(
        "3D incompressible Navier–Stokes on a triply-periodic domain with viscosity ν as "
        "the primary control parameter. The 3D extension admits helical structures, vortex "
        "stretching, and faster chaos onset than 2D: chaos horizon T* ≈ 8–16 s vs T* > 64 s "
        "in 2D (at ν=0.001, N=16). Gradient norms grow (vortex stretching) rather than "
        "decaying as in 2D."
    ),
    bc_description=(
        "Triply-periodic cubic domain [0, 2π]³; incompressibility enforced via "
        "pressure projection at each time step. No walls or inflow/outflow boundaries."
    ),
    tesseract_dir=_TESSERACT_DIR,
    output_key="result",
    ic_key="v0",
    field_to_2d=_field_to_2d,
    solvers=_SOLVERS,
    make_ic={
        "tgv3d": IcSpec(
            fn=_tgv3d,
            description=(
                "3D Taylor–Green vortex u=sin(x)cos(y)cos(z), v=−cos(x)sin(y)cos(z), w=0; "
                "divergence-free IC that develops turbulent vortex structures and a "
                "peak dissipation rate around t≈9/ν. Shape (N,N,N,3)."
            ),
            plot_params={"N": 32},
        ),
        "abc": IcSpec(
            fn=_abc_flow,
            description=(
                "Arnold–Beltrami–Childress flow — a 3D Beltrami field that is a steady "
                "Euler solution with non-zero helicity. Particle trajectories are chaotic "
                "for A≈B≈C≈1, making it a demanding test for gradient signal at long horizons. "
                "Shape (N,N,N,3)."
            ),
            plot_params={"N": 32},
        ),
        "rand_div_free": IcSpec(
            fn=_rand_div_free_3d,
            description=(
                "Random divergence-free 3D velocity field generated via curl of a spectral "
                "vector potential (energy ring at |k|=2, width 1). Seed-controlled, "
                "evolves non-trivially under NS dynamics — unlike ABC flow it has no "
                "near-steady-state structure. Shape (N,N,N,3)."
            ),
            plot_params={"N": 32},
        ),
    },
    make_inputs=_make_inputs,
    error_fn=l2_error_rel,
    analytic=_tgv3d_analytic,
    diagnostics={
        "divergence_rms": _divergence_rms,
        "kinetic_energy": _kinetic_energy,
        "energy_spectrum": _energy_spectrum,
    },
    domain_extent=2 * float(jnp.pi),
    units={"nu": "–"},
    forward_defaults={
        "baseline": {
            "description": "Single time-step agreement across grid resolution N at fixed ν for the 3D TGV IC.",
            "plot_description": (
                "Relative error vs N at steps=1; validates single-step forward accuracy across 3D solvers. "
                "Results (ν=0.05, dt=0.01): all 5 solvers valid at all N. "
                "Error at N=32: phiflow 0.0027 (best), ins_jl 0.0033, xlb 0.0071, exponax 0.013. "
                "Error at N=16: phiflow 0.011, ins_jl 0.012, xlb 0.025, exponax 0.052. "
                "Spread (std of errors) at N=32: 0.0035. phiflow most accurate; exponax spectral resolution marginal at N≤16."
            ),
            "runs": [
                {
                    "ic": {"name": "tgv3d", "seed": 0},
                    "physics": {"N": 16, "nu": 0.05, "dt": 0.01, "steps": 1},
                    "sweep": {"key": "N", "values": [8, 16, 32]},
                }
            ],
        },
        "physical_laws": {
            "description": "Physical laws sweeps: divergence RMS and kinetic energy vs grid resolution N, rollout length (steps), and viscosity ν.",
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
            "runs": [
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
            ],
        },
        "agreement": {
            "description": "3D forward accuracy sweep across ν for the 3D TGV IC at N=16. Solvers: exponax, phiflow, xlb, ins_jl, warp_ns, openfoam.",
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
            "runs": [
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
            ],
        },
    },
    cost_defaults={
        "description": "Wall-clock and memory profiling vs grid size N and step count for all 3D solvers.",
        "plot_descriptions": {
            "spatial_cost": (
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
            "temporal_cost": (
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
            "vjp_cost": (
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
        "runs": [
            {
                "physics": {"nu": 0.01, "dt": 0.01, "lbm_N_base": 16},
                "cost": {
                    "N_values": [16, 32, 48, 64],
                    "steps_values": [10, 50, 100],
                    "n_trials": 3,
                },
            }
        ],
    },
    gradient_defaults={
        "fd_check": {
            "description": "FD gradient check for the 3D TGV IC at N=16. Reveals whether VJPs are correctly wired for 3D domains.",
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
            "runs": [
                {
                    "ic": {"name": "tgv3d", "seed": 0},
                    "physics": {"N": 16, "nu": 0.001, "dt": 0.05, "steps": 10},
                    "fd": {
                        "eps_values": [5e0, 1e0, 1e-1, 1e-2, 1e-3, 1e-4],
                        "n_dirs": 10,
                    },
                }
            ],
        },
        "horizon_sweep": {
            "description": "Gradient quality vs rollout horizon for the 3D TGV IC at Re=6280 (N=16, ν=0.001, dt=0.05).",
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
            "runs": [
                {
                    "ic": {"name": "tgv3d", "seed": 0},
                    "physics": {"N": 16, "nu": 0.001, "dt": 0.05},
                    "fd": {"eps_values": [1e0, 1e-1, 1e-2, 1e-3], "n_dirs": 8},
                    "sweep": {"key": "steps", "values": [10, 20, 40, 80, 160]},
                }
            ],
        },
        "horizon_sweep_limits": {
            "description": (
                "Rollout-length limit sweep for 3D TGV at N=20: finds the VJP OOM/timeout/NaN "
                "failure boundary per solver. Each solver should run on its own dedicated GPU "
                "(pass --gpu-ids 0 1 2 3) so OOM precisely reflects a single 16 GB V100 budget. "
                "Per-step early stopping: once a solver fails at steps=S, all larger steps are "
                "skipped. Failure types recorded: OOM (GPU out-of-memory), timeout (>1200 s), "
                "container_died (Linux OOM-kill or crash), nan (non-finite VJP gradient), error. "
                "N=20 chosen so xlb D3Q27 state (~7 MB/step) hits 16 GB at ~steps=2300. "
                "fenics_ns_3d excluded (N=20 steps=40 ≈ 4000 s >> 1200 s timeout, measured). "
                "A warmup VJP at steps=40 is run before timing to exclude JIT compilation. "
                "XLB runs with XLA_PYTHON_CLIENT_PREALLOCATE=false to avoid VRAM pre-allocation."
            ),
            "plot_description": (
                "Per-solver rollout-limit table: step count at first failure, failure type, "
                "and wall time for each successful step. GPU solvers (xlb, phiflow, exponax, "
                "pict, warp_ns) expected to OOM or NaN; CPU solver ins_jl "
                "expected to show RAM growth with rollout length."
            ),
            "runs": [
                {
                    "ic": {"name": "tgv3d", "seed": 0},
                    "physics": {"N": 20, "nu": 0.001, "dt": 0.05},
                    "sweep": {
                        "key": "steps",
                        "values": [40, 80, 160, 320, 640, 1280, 2560, 5120, 10240],
                    },
                }
            ],
        },
        "jacobian_svd": {
            "description": "Full Jacobian SVD and inter-solver gradient subspace analysis for the 3D TGV IC at N=8.",
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
            "runs": [
                {
                    "ic": {"name": "tgv3d", "seed": 0},
                    "physics": {"N": 8, "nu": 0.001, "dt": 0.05, "steps": 10},
                    "jacobian": {"n_alphas": 41, "alpha_range": 0.3},
                }
            ],
        },
        "jacobian_svd_steps20": {
            "description": "Full Jacobian SVD for the 3D TGV IC at N=8, extended rollout steps=20 (T=1s).",
            "plot_description": (
                "Per-solver singular value spectra and cross-solver cosine similarity for the 3D TGV IC (N=8, nu=0.001, steps=20). "
                "Extended horizon vs base jacobian_svd (steps=10); probes gradient subspace alignment deeper into the chaotic regime."
            ),
            "runs": [
                {
                    "ic": {"name": "tgv3d", "seed": 0},
                    "physics": {"N": 8, "nu": 0.001, "dt": 0.05, "steps": 20},
                    "jacobian": {"n_alphas": 41, "alpha_range": 0.3},
                }
            ],
        },
        "jacobian_svd_steps40": {
            "description": "Full Jacobian SVD for the 3D TGV IC at N=8, extended rollout steps=40 (T=2s).",
            "plot_description": (
                "Per-solver singular value spectra and cross-solver cosine similarity for the 3D TGV IC (N=8, nu=0.001, steps=40). "
                "At T=2s the TGV is well into the chaotic regime; tests how gradient subspace structure evolves at chaos onset."
            ),
            "runs": [
                {
                    "ic": {"name": "tgv3d", "seed": 0},
                    "physics": {"N": 8, "nu": 0.001, "dt": 0.05, "steps": 40},
                    "jacobian": {"n_alphas": 41, "alpha_range": 0.3},
                }
            ],
        },
        "jacobian_svd_nu01": {
            "description": "Full Jacobian SVD for the 3D TGV IC at N=8, more viscous regime nu=0.01.",
            "plot_description": (
                "Per-solver singular value spectra and cross-solver cosine similarity for the 3D TGV IC (N=8, nu=0.01, steps=10). "
                "10x more viscous than the base jacobian_svd (nu=0.001); expected to reduce condition number and raise effective rank."
            ),
            "runs": [
                {
                    "ic": {"name": "tgv3d", "seed": 0},
                    "physics": {"N": 8, "nu": 0.01, "dt": 0.05, "steps": 10},
                    "jacobian": {"n_alphas": 41, "alpha_range": 0.3},
                }
            ],
        },
    },
    inverse_defaults={
        "recovery_constant_ic": {
            "description": (
                "IC recovery from a zero initial guess (cold start). "
                "Optimizer starts from u=0 rather than a perturbed IC, "
                "testing gradient signal without any warm initialisation. "
                "Fixed rollout steps=100, rand_div_free IC, 3 seeds."
            ),
            "plot_description": (
                "Final IC recovery error from zero-initialised optimisation "
                "(N=16, ν=0.01, dt=0.02, steps=100, rand_div_free seeds 0-2)."
            ),
            "runs": [
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
            ],
        },
        "recovery_constant_ic_bfgs": {
            "description": (
                "IC recovery with L-BFGS (cold start). Same setup as recovery_constant_ic "
                "but using L-BFGS with zoom line search instead of Adam."
            ),
            "plot_description": (
                "Final IC recovery error from zero-initialised L-BFGS optimisation "
                "(N=16, ν=0.01, dt=0.02, steps=100, rand_div_free seeds 0-2)."
            ),
            "runs": [
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
            ],
        },
        "recovery_constant_ic_bfgs_proj": {
            "description": (
                "IC recovery with L-BFGS + divergence-free gradient projection (cold start). "
                "Same as recovery_constant_ic_bfgs but gradients are projected onto the "
                "divergence-free subspace before each L-BFGS update."
            ),
            "plot_description": (
                "Final IC recovery error from zero-initialised L-BFGS+projection optimisation "
                "(N=16, ν=0.01, dt=0.02, steps=100, rand_div_free seeds 0-2)."
            ),
            "runs": [
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
            ],
        },
        "recovery_constant_ic_proj": {
            "description": (
                "IC recovery with Adam + divergence-free gradient projection (cold start). "
                "Same as recovery_constant_ic but gradients are projected onto the "
                "divergence-free subspace before each Adam update."
            ),
            "plot_description": (
                "Final IC recovery error from zero-initialised Adam+projection optimisation "
                "(N=16, ν=0.01, dt=0.02, steps=100, rand_div_free seeds 0-2)."
            ),
            "runs": [
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
            ],
        },
    },
    extra_plots={
        "gradient": [
            lambda cfg: __import__(
                "benchmarks.plots.gradient",
                fromlist=["plot_jacobian_svd_comparison"],
            ).plot_jacobian_svd_comparison(cfg),
        ],
    },
    status_checks={
        "gradient/fd_check": {
            "min_cosine": 0.99,
            # Best-ε median rel_error across FD directions. Catches the
            # warp_ns and phiflow 3D systematic backward-magnitude bias
            # (median rel_err ≈ 1.7e-2 / 1.6e-2) while leaving xlb/ins_jl/
            # pict/exponax (5e-6 to 1e-4) unflagged.
            "max_rel_err": 1e-3,
            # Peer-median outlier; ≥3 valid peers required.
            "rel_err_peer_k": 50.0,
        },
        "optimization": {"max_final_ratio": 0.5},
    },
)
