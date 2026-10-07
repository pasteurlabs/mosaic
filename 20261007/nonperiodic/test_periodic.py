"""Run inside the PhiFlow environment: pytest tests/test_periodic.py."""

import jax
import jax.numpy as jnp
import pytest
import tesseract_api as api


@pytest.mark.parametrize("shape", [(24, 20), (16, 20, 24)])
def test_discrete_projection_and_energy(shape: tuple[int, ...]) -> None:
    """Projection is orthogonal and skew transport cannot create kinetic energy."""
    ndim = len(shape)
    faces = jax.random.normal(
        jax.random.PRNGKey(ndim), (ndim, *shape), dtype=jnp.float32
    )
    projected = api._project_periodic_faces(faces)
    divergence = sum(
        (jnp.roll(projected[a], -1, axis=a) - projected[a]) * shape[a]
        for a in range(ndim)
    )
    assert float(jnp.linalg.norm(divergence) / jnp.linalg.norm(faces)) < 2e-5
    assert (
        float(
            jnp.linalg.norm(api._project_periodic_faces(projected) - projected)
            / jnp.linalg.norm(projected)
        )
        < 1e-5
    )
    other = jax.random.normal(
        jax.random.PRNGKey(ndim + 10), faces.shape, dtype=jnp.float32
    )
    assert (
        float(
            abs(
                jnp.mean(projected * other)
                - jnp.mean(faces * api._project_periodic_faces(other))
            )
        )
        < 1e-6
    )
    advection = api._periodic_skew_advection(projected, 2 * jnp.pi)
    normalized_power = jnp.abs(jnp.sum(projected * advection)) / jnp.maximum(
        jnp.linalg.norm(projected) * jnp.linalg.norm(advection), 1e-12
    )
    assert float(normalized_power) < 1e-6
    assert (
        float(
            jnp.max(
                jnp.abs(api._periodic_skew_advection(jnp.ones_like(faces), 2 * jnp.pi))
            )
        )
        == 0
    )


@pytest.mark.parametrize("ndim", [2, 3])
def test_periodic_shear_and_velocity_vjp(ndim: int) -> None:
    """Check analytic transport/diffusion and the actual multistep derivative."""
    n = 16
    shape = (n, n, 1) if ndim == 2 else (n, n, n)
    y = jnp.arange(n, dtype=jnp.float32) * 2 * jnp.pi / n
    initial = (
        jnp.zeros((*shape, ndim), dtype=jnp.float32)
        .at[..., 0]
        .set(0.2 * jnp.sin(y)[None, :, None])
    )
    initial = initial.at[..., 1].set(0.15)
    bc = {a + "_" + s: {"type": "periodic"} for a in "xyz" for s in ("lo", "hi")}

    @jax.jit
    def forward(v: jax.Array) -> jax.Array:
        return api.phiflow_fwd(v, 0.01, 0.01, 8, 2 * jnp.pi, bc, return_state=True)[0]

    reference = forward(initial)
    exact = initial.at[..., 0].set(
        0.2 * jnp.exp(-0.01 * 0.08) * jnp.sin(y - 0.15 * 0.08)[None, :, None]
    )
    assert float(jnp.linalg.norm(reference - exact) / jnp.linalg.norm(exact)) < 0.002
    direction = jnp.zeros_like(initial).at[..., 0].set(jnp.cos(2 * y)[None, :, None])

    def loss(v: jax.Array) -> jax.Array:
        return jnp.mean((forward(v) - reference) * (direction + 0.3 * initial))

    ad = jnp.sum(jax.grad(loss)(initial) * direction)
    fd = (loss(initial + 0.001 * direction) - loss(initial - 0.001 * direction)) / 0.002
    assert float(jnp.abs(ad - fd) / jnp.maximum(jnp.abs(ad), 1e-12)) < 0.05


@pytest.mark.parametrize("case", ["walls", "inflow", "obstacle"])
def test_other_paths_do_not_use_periodic_integrator(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute legacy routes while making the new periodic operators unavailable."""

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Nonperiodic/obstacle route called periodic operator")

    monkeypatch.setattr(api, "_project_periodic_faces", forbidden)
    monkeypatch.setattr(api, "_periodic_skew_advection", forbidden)
    bc = {a + "_" + side: {"type": "periodic"} for a in "xyz" for side in ("lo", "hi")}
    if case == "walls":
        bc = {key: {"type": "no_slip"} for key in bc}
    initial = jnp.zeros((8, 8, 1, 2), dtype=jnp.float32)
    obstacle = (
        {"shape": "cylinder", "center": [0.5, 0.5], "radius": 0.15}
        if case == "obstacle"
        else None
    )
    inflow = jnp.zeros((8,), dtype=jnp.float32) if case == "inflow" else None
    result, _, state = api.phiflow_fwd(
        initial,
        0.01,
        0.001,
        1,
        1.0,
        bc,
        obstacle=obstacle,
        inflow_profile=inflow,
        return_state=True,
    )
    assert result.shape == initial.shape
    assert bool(jnp.isfinite(result).all())
    assert bool(jnp.isfinite(state).all())


@pytest.mark.parametrize("ndim", [2, 3])
def test_wall_forward_and_vjp(ndim: int) -> None:
    """The native-wall RK path remains finite and differentiates its actual map."""
    n = 8
    shape = (n, n, 1) if ndim == 2 else (n, n, n)
    x = (jnp.arange(n, dtype=jnp.float32) + 0.5) / n
    xx, yy = jnp.meshgrid(x, x, indexing="ij")
    ux = 0.03 * jnp.sin(jnp.pi * xx) ** 2 * jnp.sin(2 * jnp.pi * yy)
    uy = -0.03 * jnp.sin(2 * jnp.pi * xx) * jnp.sin(jnp.pi * yy) ** 2
    initial = (
        jnp.zeros((*shape, ndim), dtype=jnp.float32)
        .at[..., 0]
        .set(ux[:, :, None])
        .at[..., 1]
        .set(uy[:, :, None])
    )
    bc = {a + "_" + s: {"type": "no_slip"} for a in "xyz" for s in ("lo", "hi")}

    @jax.jit
    def forward(v: jax.Array) -> jax.Array:
        return api.phiflow_fwd(v, 0.01, 0.001, 8, 1.0, bc, return_state=True)[0]

    reference = forward(initial)
    assert bool(jnp.isfinite(reference).all())
    direction = initial / 0.03

    def loss(v: jax.Array) -> jax.Array:
        return jnp.mean((forward(v) - reference) * direction)

    ad = jnp.sum(jax.grad(loss)(initial) * direction)
    fd = (loss(initial + 0.001 * direction) - loss(initial - 0.001 * direction)) / 0.002
    assert float(jnp.abs(ad - fd) / jnp.maximum(jnp.abs(ad), 1e-12)) < 0.05
