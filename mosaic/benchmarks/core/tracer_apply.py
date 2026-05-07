"""Thin wrapper around ``tesseract_jax.apply_tesseract``.

Deadline enforcement is done by the HTTP timeout monkey-patch in
``runner.py`` (see ``_install_tesseract_http_timeout``).  This module
exists so gradient/optimization suites can import a single
``apply_tesseract`` that handles JAX tracer detection — when inputs
contain a tracer (during ``jax.grad`` tracing), the call must run on
the caller's thread so primitive binding participates in the active
JAX trace.  At that stage no HTTP call happens (the primitive is
merely recorded), so timeout enforcement is irrelevant.
"""

from __future__ import annotations

from typing import Any

__all__ = ["apply_tesseract"]


def _inputs_contain_tracer(inputs: Any) -> bool:
    """Return True if *inputs* contains any JAX tracer.

    When a tracer is present, ``apply_tesseract`` must run on the caller's
    thread so ``tesseract_dispatch_p.bind`` participates in the active JAX
    trace.  Running it on a worker thread would cause
    ``UnexpectedTracerError``.
    """
    try:
        import jax
        import jax.core as _jc
    except ImportError:
        return False
    try:
        flat, _ = jax.tree_util.tree_flatten(inputs)
    except Exception:
        if isinstance(inputs, dict):
            for v in inputs.values():
                if isinstance(v, _jc.Tracer):
                    return True
        return False
    return any(isinstance(leaf, _jc.Tracer) for leaf in flat)


def apply_tesseract(
    t: Any,
    inputs: dict,
    *,
    _apply_fn: Any = None,
) -> Any:
    """Call ``tesseract_jax.apply_tesseract(t, inputs)``."""
    if _apply_fn is None:
        from tesseract_jax import apply_tesseract as _apply_fn  # type: ignore

    return _apply_fn(t, inputs)
