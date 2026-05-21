from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mosaic.benchmarks.core.config import IcSpec, ProblemConfig, SolverSpec
from mosaic.benchmarks.core.utils import l2_error_rel

_GYM_DIR = Path(__file__).parent.parent.parent
_TESSERACT_DIR = _GYM_DIR / "tesseracts" / "navier-stokes-grid"
_LBM_SOLVERS = {"xlb"}

_SOLVERS: dict[str, SolverSpec] = {
    "jax_cfd": SolverSpec(
        name="jax-cfd",
        backend="jax",
        family="projection",
        differentiable=True,
        ad_strategy="autodiff",
        uses_gpu=True,
        internal_dtype="float32",
        dir="jax-cfd",
        color="#4477AA",
        linestyle="-",
        marker="o",
        scheme="MAC FD + projection",
        description="Staggered MAC grid with pressure projection and finite-difference advection.",
        doc_url="https://github.com/google/jax-cfd",
        image_tag="jax_cfd_navier_stokes_grid:latest",
        input_overrides={
            "density": jnp.array([1.0], dtype=jnp.float32),
            "inner_steps": 1,
        },
        exclusions={
            "forward/cylinder": {
                "category": "categorical",
                "reason": "tesseract uses periodic FFT pressure solve + IBM volume penalization; channel-BC cylinder flow requires a non-periodic pressure solve that is not wired in this benchmark",
            },
            "optimization/drag_opt": {
                "category": "categorical",
                "reason": "periodic FFT pressure solve + IBM volume penalization is incompatible with cylinder obstacle channel BCs (same root cause as forward/cylinder)",
            },
        },
        explained_anomalies={
            "forward/baseline": {
                "reason": (
                    "staggered MAC grid double-interpolation: collocated TGV IC -> "
                    "staggered faces -> collocated output gives sin^2(pi/N) round-trip "
                    "error at all N; 35-40x above collocated peers"
                ),
            },
        },
    ),
    "phiflow": SolverSpec(
        name="PhiFlow",
        backend="jax",
        family="projection",
        differentiable=True,
        ad_strategy="hybrid",
        uses_gpu=True,
        internal_dtype="float32",
        dir="phiflow",
        color="#EE3333",
        linestyle="--",
        marker="s",
        scheme="semi-Lagrangian + projection",
        description="Semi-Lagrangian advection with pressure projection; unconditionally stable for large dt.",
        doc_url="https://tum-pbs.github.io/PhiFlow/",
        image_tag="phiflow_navier_stokes_grid:latest",
        exclusions={},
        explained_anomalies={
            "forward/agreement/tgv": {
                "reason": (
                    "phiflow's double CenteredGrid↔StaggeredGrid resampling gives 4.18% amplitude "
                    "damping (ratio=0.9582); cosine=0.9999924 (pattern correct); arithmetic-average "
                    "output conversion fix worsened error 9×; upstream library change required"
                ),
            },
        },
    ),
    "ins_jl": SolverSpec(
        name="INS.jl",
        backend="julia",
        family="projection",
        differentiable=True,
        ad_strategy="autodiff",
        uses_gpu=False,
        internal_dtype="float64",
        dir="incompressible-navier-stokes-jl",
        color="#228833",
        linestyle="-.",
        marker="^",
        scheme="FD + projection",
        description=(
            "Julia finite-difference solver with pressure projection; CPU-only. "
            "Gradients via Zygote.jl reverse-mode AD: v0, viscosity, dt, and the "
            "inflow_profile (channel mode) all have valid VJPs through the RK4 "
            "loop and spectral pressure projection. domain_extent always returns "
            "a zero gradient by design (structural grid parameter)."
        ),
        doc_url="https://agdestein.github.io/IncompressibleNavierStokes.jl/dev/",
        image_tag="ins_navier_stokes_grid:latest",
        exclusions={
            "forward/cylinder": {
                "category": "categorical",
                "reason": "no IBM or volume penalization — the cylinder obstacle cannot be represented in INS.jl; spectral/LU pressure projection is also periodic-only and incompatible with obstacle channel BCs",
            },
            "optimization/drag_opt": {
                "category": "categorical",
                "reason": "no IBM or volume penalization — the cylinder obstacle cannot be represented; inflow_profile VJP works only in periodic/channel mode without obstacles",
            },
        },
        explained_anomalies={
            "forward/baseline": {
                "reason": (
                    "staggered MAC grid double-interpolation: collocated TGV IC -> "
                    "staggered faces -> collocated output gives sin^2(pi/N) round-trip "
                    "error at all N; 35-40x above collocated peers"
                ),
            },
        },
    ),
    "openfoam": SolverSpec(
        name="OpenFOAM",
        backend="cpp",
        family="projection",
        differentiable=False,
        ad_strategy=None,
        uses_gpu=False,
        internal_dtype="float64",
        dir="openfoam",
        color="#DDAA33",
        linestyle=":",
        marker="D",
        scheme="icoFoam PISO",
        description="OpenFOAM icoFoam; forward-only non-differentiable reference baseline.",
        doc_url="https://www.openfoam.com/documentation/overview",
        image_tag="openfoam_navier_stokes_grid:latest",
        exclusions={
            "gradient": {
                "category": "categorical",
                "reason": "standard icoFoam is non-differentiable (C++, no AD path); DAFoam/OpenFOAM-AD exist but are not deployed in this tesseract",
            },
            "optimization": {
                "category": "categorical",
                "reason": "standard icoFoam is non-differentiable forward-only solver",
            },
            "cost/vjp_cost": {
                "category": "categorical",
                "reason": "standard icoFoam has no VJP to benchmark",
            },
        },
    ),
    "pict": SolverSpec(
        name="PICT",
        backend="pytorch",
        family="projection",
        differentiable=True,
        ad_strategy="autodiff",
        uses_gpu=True,
        internal_dtype="float32",
        dir="pict",
        color="#AA44AA",
        linestyle=(0, (5, 1)),
        marker="v",
        scheme="PISO (2nd-order)",
        description=(
            "PISOtorch PISO solver with CUDA kernels; differentiable via PyTorch autograd. "
            "VJP w.r.t. v0 is fully supported through the differentiable PISO time loop. "
            "Gradients w.r.t. viscosity and dt are not available: those parameters are "
            "consumed as Python scalars at domain construction / timestep time and have "
            "no autograd path; zero gradients are returned for them."
        ),
        doc_url="https://github.com/tum-pbs/PICT",
        image_tag="pict_navier_stokes_grid:latest",
        exclusions={},
        explained_anomalies={},
    ),
    "xlb": SolverSpec(
        name="XLB",
        backend="jax",
        family="lbm",
        differentiable=True,
        ad_strategy="autodiff",
        uses_gpu=True,
        internal_dtype="float64",
        dir="xlb",
        color="#66CCEE",
        linestyle=(0, (3, 1, 1, 1)),
        marker="P",
        scheme="LBM BGK/KBC D2Q9",
        description=(
            "JAX-accelerated LBM D2Q9 using XLB's BGK (default) / entropic-stabilised "
            "KBC (2-D obstacle flows), Stream, QuadraticEquilibrium, and Macroscopic "
            "operators (xlb 0.3.1) inside a jax.lax.scan time loop; incompressibility "
            "recovered in the low-Mach limit. "
            "VJP flows through the full scan unroll in float64 precision and is routed "
            "per diff-input key (v0, viscosity, dt, inflow_profile). "
            "Both apply() and vjp_jit() run in float64 internally (output cast to "
            "float32) to avoid float32 quantization noise at fine ε (omega≈2 at low "
            "viscosity). FD cosine ≥0.9999 at ε=1.0."
        ),
        doc_url="https://github.com/Autodesk/XLB",
        image_tag="xlb_navier_stokes_grid:latest",
        exclusions={},
        explained_anomalies={
            "forward/baseline": {
                "reason": (
                    "irreducible O(Ma²) LBM compressibility error floor: at fixed "
                    "dt=0.01, Ma=u·dt/dx grows with N; at N=128 Ma~0.2 giving ~0.007 "
                    "error floor (230× peers); anomalous at all N"
                ),
            },
            "forward/agreement/tgv": {
                "reason": (
                    "automatic k=9 sub-steps reduce Ma 0.88→0.098 (81× Ma² reduction); "
                    "errors drop from 0.216-0.278 → 0.026-0.031 (11-24× peers); "
                    "remaining floor is O(dx²) LBM spatial discretization at N=64, not reducible "
                    "by further sub-stepping (tested k=9..27); valid=True"
                ),
            },
            "forward/tgv_nu_sweep": {
                "reason": (
                    "same root cause as forward/agreement/tgv — automatic k=9 sub-stepping reduces Ma 0.88→0.098 "
                    "but residual O(dx²) LBM spatial discretization gives 11-24× peer errors "
                    "at all nu values (0.0001–0.05); 0.0309 at nu=0.05 is 12.0× peer median; "
                    "not reducible by further sub-stepping (tested k=9..27); valid=True"
                ),
            },
        },
    ),
    "warp_ns": SolverSpec(
        name="Warp-NS",
        backend="warp",
        family="fd",
        differentiable=True,
        ad_strategy="autodiff",
        uses_gpu=True,
        internal_dtype="float32",
        dir="warp-ns",
        color="#EE7733",
        linestyle=(0, (1, 1)),
        marker="X",
        scheme="IPCS (2D+3D), periodic spectral FFT Poisson, wp.Tape VJP",
        description=(
            "NVIDIA Warp periodic-only NS solver: IPCS primitive-variable projection "
            "for both 2D and 3D with spectral FFT pressure Poisson. VJP via wp.Tape; "
            "viscosity and dt gradients via per-step record_func callbacks accumulated "
            "analytically from the Laplacian / advection / divergence / pressure-correction "
            "terms."
        ),
        doc_url="https://github.com/NVIDIA/warp",
        image_tag="warp_ns_navier_stokes_grid:latest",
        exclusions={
            "forward/cylinder": {
                "category": "categorical",
                "reason": "warp-ns is periodic-only; obstacle flows are not supported",
            },
            "optimization/drag_opt": {
                "category": "categorical",
                "reason": "warp-ns is periodic-only; obstacle/inflow flows are not supported",
            },
        },
        explained_anomalies={},
    ),
}


# ── IC generators ─────────────────────────────────────────────────────────────

# ---- 2D ----


def _multimode(N: int, L: float = 2 * jnp.pi, seed: int = 42, **_) -> jax.Array:
    """Energy ring at k=2, σ=0.5, max speed normalised to 0.3."""
    key = jax.random.PRNGKey(seed)
    kn = jnp.fft.fftfreq(N, d=1.0 / N)
    KX, KY = jnp.meshgrid(kn, kn, indexing="ij")
    K_abs = jnp.sqrt(KX**2 + KY**2)
    envelope = jnp.exp(-0.5 * ((K_abs - 2.0) / 0.5) ** 2)
    phases = jax.random.uniform(key, (N, N), minval=0.0, maxval=2.0 * jnp.pi)
    psi_hat = envelope * jnp.exp(1j * phases)
    psi_hat = 0.5 * (psi_hat + jnp.conj(psi_hat[::-1, ::-1]))
    psi_hat = psi_hat.at[0, 0].set(0.0)
    kfac = 2.0 * jnp.pi / L
    vx = jnp.real(jnp.fft.ifft2(1j * KY * kfac * psi_hat))
    vy = jnp.real(jnp.fft.ifft2(-1j * KX * kfac * psi_hat))
    u_max = float(jnp.sqrt(vx**2 + vy**2).max()) or 1.0
    vx, vy = vx / u_max * 0.3, vy / u_max * 0.3
    return jnp.stack([vx, vy], axis=-1)[:, :, None, :].astype(jnp.float32)


def _tgv(N: int, L: float = 2 * jnp.pi, seed: int = 0, **_) -> jax.Array:
    """Taylor-Green vortex: u=sin(x)cos(y), v=-cos(x)sin(y)."""
    x = jnp.linspace(0, L, N, endpoint=False)
    y = jnp.linspace(0, L, N, endpoint=False)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    vx = jnp.sin(X) * jnp.cos(Y)
    vy = -jnp.cos(X) * jnp.sin(Y)
    return jnp.stack([vx, vy], axis=-1)[:, :, None, :].astype(jnp.float32)


def _uniform_flow(N: int, U: float = 1.0, **_) -> jax.Array:
    """Uniform rightward flow u=(U, 0) — canonical IC for cylinder-wake experiments."""
    vx = jnp.full((N, N), U, dtype=jnp.float32)
    vy = jnp.zeros((N, N), dtype=jnp.float32)
    return jnp.stack([vx, vy], axis=-1)[:, :, None, :]


def _flat_inflow(N: int = 64, U: float = 0.5, **_) -> jax.Array:
    """Flat inlet profile u_x(y) = U, shape (N,). Starting point for drag_opt."""
    return jnp.full((N,), U, dtype=jnp.float32)


# ── Analytic solution ─────────────────────────────────────────────────────────


def _tgv_analytic(
    ic: jax.Array, nu: float, t: float, L: float = 2 * jnp.pi
) -> jax.Array:
    """Exact TGV solution at time t: decays as exp(-2*nu*t)."""
    N = ic.shape[0]
    x = jnp.linspace(0, L, N, endpoint=False)
    y = jnp.linspace(0, L, N, endpoint=False)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    decay = jnp.exp(-2.0 * nu * t)
    vx = jnp.sin(X) * jnp.cos(Y) * decay
    vy = -jnp.cos(X) * jnp.sin(Y) * decay
    return jnp.stack([vx, vy], axis=-1)[:, :, None, :].astype(jnp.float32)


# ── Input factory ─────────────────────────────────────────────────────────────


def _make_inputs(
    solver_name: str,
    ic: jax.Array,
    *,
    nu: float,
    dt: float,
    steps: int,
    domain_extent: float = 2 * jnp.pi,
    lbm_N_base: int | None = None,
    obstacle: dict | None = None,
    U_mean: float = 0.5,
    **_,
) -> dict:
    """Build solver input dict, applying LBM dt-scaling when lbm_N_base is set.

    When ic is 1-D (shape (N,)) it is treated as an inflow profile for drag
    optimisation (v0 = uniform background at U_mean, ic → inflow_profile field).
    """
    spec = _SOLVERS[solver_name]

    if ic.ndim == 1:
        # Inflow-profile optimisation mode.
        N = ic.shape[0]
        _dt, _steps = dt, steps
        if solver_name in _LBM_SOLVERS and lbm_N_base is not None:
            # Only scale dt DOWN (when N > lbm_N_base) to reduce Ma_lattice at
            # high resolution. Clamp so dt never exceeds the base value — at
            # N < lbm_N_base the LBM is already stable and scaling dt UP would
            # increase T_phys and raise Ma_lattice (the opposite of the intent).
            _dt = dt * min(1.0, lbm_N_base / N)
            _steps = max(1, round(steps * max(1.0, N / lbm_N_base)))
        base = dict(
            v0=_uniform_flow(N, U=U_mean),
            inflow_profile=ic,
            viscosity=jnp.array([nu], dtype=jnp.float32),
            dt=jnp.array([_dt], dtype=jnp.float32),
            steps=_steps,
            domain_extent=float(domain_extent),
        )
        if obstacle is not None:
            base["obstacle"] = obstacle
            base["boundary_conditions"] = {
                "x_lo": {"type": "periodic"},
                "x_hi": {"type": "periodic"},
                "y_lo": {"type": "no_slip"},
                "y_hi": {"type": "no_slip"},
            }
        return {**base, **spec.input_overrides}

    N = ic.shape[0]
    _dt, _steps = dt, steps
    if solver_name in _LBM_SOLVERS and lbm_N_base is not None:
        # Only scale dt DOWN (when N > lbm_N_base) to reduce Ma_lattice at
        # high resolution. Clamp so dt never exceeds the base value — at
        # N < lbm_N_base the LBM is already stable and scaling dt UP would
        # increase T_phys and raise Ma_lattice (the opposite of the intent).
        _dt = dt * min(1.0, lbm_N_base / N)
        _steps = max(1, round(steps * max(1.0, N / lbm_N_base)))

    base = dict(
        v0=ic,
        viscosity=jnp.array([nu], dtype=jnp.float32),
        dt=jnp.array([_dt], dtype=jnp.float32),
        steps=_steps,
        domain_extent=float(domain_extent),
    )
    if obstacle is not None:
        base["obstacle"] = obstacle
        base["boundary_conditions"] = {
            "x_lo": {"type": "neumann"},
            "x_hi": {"type": "neumann"},
            "y_lo": {"type": "no_slip"},
            "y_hi": {"type": "no_slip"},
        }
    return {**base, **spec.input_overrides}


# ── Diagnostics ───────────────────────────────────────────────────────────────


def _divergence_rms(arr: jax.Array, domain_extent: float = 2 * jnp.pi, **_) -> float:
    """RMS divergence ∇·u for 2D fields (N,N,1,2)."""
    ndim = arr.shape[-1]  # 2 or 3
    dx = domain_extent / arr.shape[0]
    div = sum(
        (jnp.roll(arr[..., i], -1, i) - jnp.roll(arr[..., i], 1, i))
        for i in range(ndim)
    ) / (2 * dx)
    return float(jnp.sqrt(jnp.mean(div**2)))


def _kinetic_energy(arr: jax.Array, **_) -> float:
    """Mean kinetic energy ½〈|u|²〉."""
    return float(0.5 * jnp.mean(jnp.sum(arr**2, axis=-1)))


def _energy_spectrum(arr: jax.Array, **_) -> dict:
    """Isotropic 1-D energy spectrum E(k) for 2D fields."""
    ndim = arr.shape[-1]  # 2 or 3
    N = arr.shape[0]
    kn = jnp.fft.fftfreq(N, d=1.0 / N)

    if ndim == 2:
        v_hat = [jnp.fft.fft2(arr[:, :, 0, d]) / N**2 for d in range(2)]
        axes = jnp.meshgrid(kn, kn, indexing="ij")
    else:
        v_hat = [jnp.fft.fftn(arr[..., d]) / N**3 for d in range(3)]
        axes = jnp.meshgrid(kn, kn, kn, indexing="ij")

    E_hat = 0.5 * sum(jnp.abs(vh) ** 2 for vh in v_hat)
    K = jnp.sqrt(sum(ax**2 for ax in axes))
    k_bins = np.arange(1, N // 2)  # integer indices — numpy is fine here
    E_k = jnp.array(
        [float(E_hat[(K >= k - 0.5) & (K < k + 0.5)].sum()) for k in k_bins]
    )
    return {"k": k_bins.tolist(), "E_k": E_k.tolist()}


# ── Config instance ───────────────────────────────────────────────────────────

CONFIG = ProblemConfig(
    name="ns-grid",
    n_to_cells=lambda n: n**2,
    description=(
        "2D incompressible Navier–Stokes on a doubly-periodic domain with viscosity ν as "
        "the primary control parameter. The nonlinear advection term ∇·(u⊗u) transfers "
        "energy across scales; at low ν the flow develops turbulent cascades and the "
        "Lyapunov exponent grows, making long-horizon gradients exponentially sensitive "
        "to perturbations."
    ),
    bc_description=(
        "Doubly-periodic square domain [0, 2π]²; incompressibility enforced via "
        "pressure projection at each time step. No walls or inflow/outflow boundaries."
    ),
    tesseract_dir=_TESSERACT_DIR,
    output_key="result",
    ic_key="v0",
    solvers=_SOLVERS,
    make_ic={
        "multimode": IcSpec(
            fn=_multimode,
            description=(
                "Incompressible velocity field with energy concentrated in a ring at "
                "wavenumber k=2 (σ_k=0.5); supports multi-scale turbulent development."
            ),
            plot_params={"N": 64},
        ),
        "tgv": IcSpec(
            fn=_tgv,
            description=(
                "Taylor–Green vortex u=sin(x)cos(y), v=−cos(x)sin(y); has a closed-form "
                "analytic solution for viscous decay, enabling solver verification."
            ),
            plot_params={"N": 64},
        ),
        "uniform": IcSpec(
            fn=_uniform_flow,
            description=(
                "Uniform rightward flow u=(U, 0) — background flow for cylinder-wake "
                "(Kármán vortex street) experiments. Obstacle specified separately via "
                "the physics.obstacle field."
            ),
            plot_params={"N": 64, "U": 1.0},
        ),
        "flat_inflow": IcSpec(
            fn=_flat_inflow,
            description=(
                "Flat 1-D inlet velocity profile u_x(y) = U, shape (N,). "
                "Starting point for inflow-profile drag-optimisation experiments."
            ),
            plot_params={"N": 64, "U": 0.5},
        ),
    },
    make_inputs=_make_inputs,
    error_fn=l2_error_rel,
    diagnostics={
        "divergence_rms": _divergence_rms,
        "kinetic_energy": _kinetic_energy,
        "energy_spectrum": _energy_spectrum,
    },
    analytic=_tgv_analytic,
    domain_extent=2 * float(jnp.pi),
    units={"nu": "–"},
    forward_defaults={
        "baseline": dict(
            description="Single time-step agreement across grid resolution N at fixed ν.",
            plot_description="Relative error vs N at steps=1; validates single-step forward accuracy across solvers.",
            runs=[
                dict(
                    ic=dict(name="tgv", seed=0),
                    physics=dict(N=64, nu=0.05, dt=0.01, steps=1, lbm_N_base=64),
                    sweep=dict(key="N", values=[16, 32, 64, 128]),
                )
            ],
        ),
        "agreement": dict(
            description="Inter-solver agreement sweep across viscosity ν (TGV and multimode ICs).",
            plot_description="Relative error vs ν for each IC; vorticity fields comparing solver output to the fine-solver reference.",
            runs=[
                dict(
                    ic=dict(name="tgv", seed=42),
                    physics=dict(N=64, dt=0.05, steps=20),
                    sweep=dict(key="nu", values=[0.001, 0.005, 0.01, 0.02, 0.05]),
                    fine=dict(solvers={"jax_cfd"}, dt=0.01, steps=100),
                ),
                dict(
                    ic=dict(name="multimode", seed=42),
                    physics=dict(N=64, dt=0.05, steps=20),
                    sweep=dict(key="nu", values=[0.001, 0.005, 0.01, 0.02, 0.05]),
                    fine=dict(solvers={"jax_cfd"}, dt=0.01, steps=100),
                ),
            ],
        ),
        "tgv_nu_sweep": dict(
            description=(
                "Dense TGV ν-sweep probing xlb LBM BGK-relaxation ill-conditioning "
                "as ν→0. Covers ν ∈ {1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2} — three "
                "decades — so a monotonic xlb degradation curve can be distinguished "
                "from a single unlucky ν point. Uses the same N=64, dt=0.05, steps=20, "
                "jax_cfd fine reference as `agreement/tgv` for comparability."
            ),
            plot_description=(
                "Relative error vs ν for each solver at fixed TGV IC; "
                "xlb errors are flat 0.026-0.031 across all ν (KBC entropic operator removes "
                "BGK monotonic degradation at low ν); peer solvers stay ≤ 0.001 across the full range."
            ),
            runs=[
                dict(
                    ic=dict(name="tgv", seed=42),
                    physics=dict(N=64, dt=0.05, steps=20),
                    sweep=dict(
                        key="nu",
                        values=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
                    ),
                    fine=dict(solvers={"jax_cfd"}, dt=0.01, steps=100),
                ),
            ],
        ),
        "physical_laws": dict(
            description="Physical laws sweeps: diagnostics vs grid resolution N, rollout length (steps), and viscosity ν (TGV IC).",
            plot_description="Divergence RMS, kinetic energy, and analytic error vs N / steps / ν for each solver; validates incompressibility and accuracy across regimes.",
            runs=[
                dict(
                    name="vs_N",
                    ic=dict(name="tgv", seed=0),
                    physics=dict(nu=0.05, dt=0.01, steps=20),
                    sweep=dict(key="N", values=[16, 32, 64, 128]),
                ),
                dict(
                    name="vs_steps",
                    ic=dict(name="tgv", seed=0),
                    physics=dict(nu=0.05, dt=0.01, N=64),
                    sweep=dict(key="steps", values=[5, 10, 20, 50, 100]),
                ),
                dict(
                    name="vs_nu",
                    ic=dict(name="tgv", seed=0),
                    physics=dict(dt=0.01, steps=20, N=64),
                    sweep=dict(key="nu", values=[0.001, 0.005, 0.01, 0.05, 0.1]),
                ),
            ],
        ),
        "cylinder": dict(
            description=(
                "Cylinder-wake (Kármán vortex street) experiment: uniform inflow past a "
                "circular cylinder, sweeping Reynolds number Re=U·D/ν where D=2·radius·L."
            ),
            plot_description=(
                "Vorticity snapshots and kinetic-energy evolution vs time for each solver; "
                "phase transition from steady to vortex-shedding regime as Re increases."
            ),
            runs=[
                dict(
                    ic=dict(name="uniform", seed=0),
                    physics=dict(
                        N=64,
                        dt=0.01,
                        steps=500,
                        obstacle=dict(shape="cylinder", center=[0.5, 0.5], radius=0.1),
                    ),
                    sweep=dict(key="nu", values=[0.05, 0.02, 0.01, 0.005]),
                )
            ],
        ),
    },
    cost_defaults=dict(
        description="Wall-clock and memory profiling vs grid size N and step count for all solvers.",
        plot_descriptions={
            "spatial_cost": "Forward-pass wall-clock time vs N at fixed step count for all solvers.",
            "temporal_cost": "Forward-pass wall-clock time vs step count at fixed N for all solvers.",
            "vjp_cost": "VJP wall-clock time vs N and step count for differentiable solvers.",
        },
        runs=[
            dict(
                physics=dict(nu=0.01, dt=0.01),
                cost=dict(
                    N_values=[64, 128, 192, 256],
                    steps_values=[10, 50, 100, 500, 1000],
                    n_trials=3,
                ),
            )
        ],
    ),
    gradient_defaults={
        "fd_check": dict(
            description="FD gradient check vs analytic VJP at nominal physics (N=16, ν=0.001); multimode IC.",
            plot_description="U-curves (FD gradient error vs ε) and subspace cosine; validates VJP correctness. LBM solvers (XLB, Lettuce) require ε≥5 (abs_eps≥0.36) to exit their float32 noise floor.",
            runs=[
                dict(
                    ic=dict(name="multimode", seed=42),
                    physics=dict(N=16, nu=0.001, dt=0.05, steps=20),
                    fd=dict(eps_values=[5e0, 1e0, 1e-1, 1e-2, 1e-3, 1e-4], n_dirs=20),
                ),
            ],
        ),
        "param_sweep": dict(
            description="Gradient quality vs viscosity ν (2D multimode IC).",
            plot_description="Gradient norm, best-ε FD error, direction cosine, and U-curves vs the sweep parameter.",
            runs=[
                dict(
                    ic=dict(name="multimode", seed=42),
                    physics=dict(N=16, dt=0.05, steps=200),
                    fd=dict(eps_values=[5e0, 1e0, 1e-1, 1e-2, 1e-3], n_dirs=15),
                    sweep=dict(key="nu", values=[0.05, 0.01, 0.005, 0.001]),
                ),
            ],
        ),
        "horizon_sweep": dict(
            description="Gradient quality vs rollout horizon T=steps*dt for 2D multimode IC at Re=1000 (N=16, ν=0.001, dt=0.05). Establishes T* horizon limit for gradient-based optimization.",
            plot_description=(
                "Gradient norm, FD error, and direction cosine vs rollout horizon T=steps*dt for 2D multimode IC. "
                "Key metric: the step count T* at which cosine(ε=0.1) drops below 0.999 — the practical gradient quality limit."
            ),
            runs=[
                dict(
                    ic=dict(name="multimode", seed=42),
                    physics=dict(N=16, nu=0.001, dt=0.05),
                    fd=dict(eps_values=[1e0, 1e-1, 1e-2, 1e-3], n_dirs=8),
                    sweep=dict(key="steps", values=[5, 10, 20, 40, 80, 160, 320]),
                ),
            ],
        ),
        "jacobian_svd": dict(
            description="Full Jacobian SVD and inter-solver gradient subspace analysis at N=8, ν=0.001, T=0.5s (steps=10).",
            plot_description="Singular-value spectrum and pairwise cross-solver cosine similarity of gradient subspaces (2D multimode IC, N=8, ν=0.001, steps=10).",
            runs=[
                dict(
                    ic=dict(name="multimode", seed=42),
                    physics=dict(N=8, nu=0.001, dt=0.05, steps=10),
                    jacobian=dict(),
                )
            ],
        ),
        "jacobian_svd_steps20": dict(
            description="Full Jacobian SVD at N=8, extended rollout steps=20 (T=1s).",
            plot_description="Singular-value spectrum and cross-solver cosine similarity for 2D multimode IC (N=8, ν=0.001, steps=20). Extended horizon vs base (steps=10).",
            runs=[
                dict(
                    ic=dict(name="multimode", seed=42),
                    physics=dict(N=8, nu=0.001, dt=0.05, steps=20),
                    jacobian=dict(),
                )
            ],
        ),
        "jacobian_svd_steps40": dict(
            description="Full Jacobian SVD at N=8, extended rollout steps=40 (T=2s).",
            plot_description="Singular-value spectrum and cross-solver cosine similarity for 2D multimode IC (N=8, ν=0.001, steps=40). Probes deeper into the chaotic regime.",
            runs=[
                dict(
                    ic=dict(name="multimode", seed=42),
                    physics=dict(N=8, nu=0.001, dt=0.05, steps=40),
                    jacobian=dict(),
                )
            ],
        ),
        "jacobian_svd_nu01": dict(
            description="Full Jacobian SVD at N=8, more viscous regime ν=0.01 (T=0.5s).",
            plot_description="Singular-value spectrum and cross-solver cosine similarity for 2D multimode IC (N=8, ν=0.01, steps=10). 10× more viscous than base; expected to reduce condition number.",
            runs=[
                dict(
                    ic=dict(name="multimode", seed=42),
                    physics=dict(N=8, nu=0.01, dt=0.05, steps=10),
                    jacobian=dict(),
                )
            ],
        ),
    },
    inverse_defaults={
        # "drag_opt": dict(
        # description=(
        # "Inflow profile optimisation: minimise cylinder drag at Re=20 (steady Stokes-like regime). "
        # "Control variable is the 1-D inlet profile u_x(y); constraint is fixed mean flow rate "
        # "(flow-rate penalty). Geometry: cylinder at [0.5, 0.5], radius=0.05, domain [0,1]²."
        # ),
        # plot_description=(
        # "Drag convergence curves per solver at Re=20; "
        # "optimised vs initial inflow profiles; final drag coefficient comparison."
        # ),
        # runs=[
        # dict(
        # name="re20",
        # ic=dict(name="flat_inflow", seed=0),
        # physics=dict(
        # N=32,
        # # Re = U·D/ν = 0.5·0.1/0.0025 = 20
        # # N=32 used (not 64): IBM hard-masking in jax-cfd causes pressure
        # # projection divergence at N=64 (discontinuity at obstacle boundary
        # # overwhelms the periodic Poisson solve at step ~20).
        # nu=0.0025,
        # dt=0.02,
        # steps=400,
        # domain_extent=1.0,
        # U_mean=0.5,
        # obstacle=dict(shape="cylinder", center=[0.5, 0.5], radius=0.05),
        # ),
        # optim=dict(
        # lr=5e-4,
        # max_iters=500,
        # patience=100,
        # flow_penalty_weight=50.0,
        # snap_interval=20,
        # ),
        # ),
        # ],
        # ),
        # "drag_opt_bfgs": dict(
        # description=(
        # "Inflow profile optimisation with L-BFGS: minimise cylinder drag at Re=20. "
        # "Same setup as drag_opt but using L-BFGS with zoom line search."
        # ),
        # plot_description=(
        # "L-BFGS drag convergence curves per solver at Re=20; "
        # "optimised vs initial inflow profiles; final drag coefficient comparison."
        # ),
        # runs=[
        # dict(
        # name="re20",
        # ic=dict(name="flat_inflow", seed=0),
        # physics=dict(
        # N=32,
        # nu=0.0025,
        # dt=0.02,
        # steps=400,
        # domain_extent=1.0,
        # U_mean=0.5,
        # obstacle=dict(shape="cylinder", center=[0.5, 0.5], radius=0.05),
        # ),
        # optim=dict(
        # max_iters=50,
        # patience=15,
        # flow_penalty_weight=50.0,
        # snap_interval=5,
        # ),
        # ),
        # ],
        # ),
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
        "forward": {"median_k": 3.0, "max_error": 0.5},
        "gradient/fd_check": {
            "min_cosine": 0.99,
            "max_rel_err": 1e-3,
            "rel_err_peer_k": 50.0,
        },
        "cost": {"max_peer_k": 20.0},
        "forward/cylinder": {"median_k": 50.0, "max_error": 0.5},
        "forward/agreement/multimode": {"median_k": 3.0, "max_error": 1.5},
    },
)
