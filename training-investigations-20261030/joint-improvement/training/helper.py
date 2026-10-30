"""External research corrector: independent temperature-input and heat constraints."""
import jax
import jax.numpy as jnp

def effective_params(params, recipe):
    """Apply fixed output-head reparameterization once to weights and bias."""
    count = 2 if recipe["joint"] else 1
    factors = jnp.asarray([1.0, recipe.get("temperature_head_scale", 1.0)][:count], jnp.float32)
    return (*params[:-1], {key: value * factors for key, value in params[-1].items()})

def apply_variant(params, state, scales, recipe, return_diagnostics=False):
    """Keep physical identity path; constrain thermal residual before adding it."""
    state = jnp.asarray(state, dtype=jnp.float32)
    scales = jnp.asarray(scales, dtype=state.dtype)
    nn_state = state
    if recipe.get("center_temperature", False):
        nn_state = state.at[..., 2].add(-jnp.mean(state[..., 2]))
    x = (nn_state / scales)[None]
    pp = effective_params(params, recipe)
    for i, layer in enumerate(pp):
        x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
        x = jax.lax.conv_general_dilated(
            x, layer["w"], (1, 1), "VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        ) + layer["b"]
        if i < len(pp) - 1:
            x = jax.nn.tanh(x)
    psi = jnp.pad(x[0, :-1, :-1, 0], ((1, 1), (1, 1))) * scales[0]
    velocity_delta = jnp.stack(
        (jnp.diff(psi, axis=1)[1:], -jnp.diff(psi, axis=0)[:, 1:]), axis=-1,
    )
    corrected = state.at[..., :2].add(velocity_delta)
    thermal_delta = jnp.zeros_like(state[..., 2])
    if recipe["joint"]:
        # Form the residual directly, before adding the physical identity path.
        thermal_delta = x[0, ..., 1] * scales[2]
        if recipe.get("zero_mean_temperature", False):
            mean_delta = jnp.mean(thermal_delta.astype(jnp.float64)).astype(thermal_delta.dtype)
            thermal_delta = thermal_delta - mean_delta
        corrected = corrected.at[..., 2].add(thermal_delta)
    if return_diagnostics:
        return corrected, jnp.stack([
            jnp.mean(thermal_delta.astype(jnp.float64)),
            jnp.max(jnp.abs(thermal_delta)).astype(jnp.float64),
            jnp.mean((corrected[..., 2] - state[..., 2]).astype(jnp.float64)),
        ])
    return corrected
