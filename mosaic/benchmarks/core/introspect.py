"""Container introspection for the tesseract watchdog.

The watchdog needs two capabilities:

1. Given a :class:`tesseract_core.Tesseract` instance, discover the docker
   container name/id that backs it. See :func:`get_container_id`.
2. Given a container id, report whether it is still running. See
   :func:`container_running`.

Both are implemented as narrow shims over ``docker inspect`` / the
``tesseract_core`` SDK so the watchdog's core logic (``watchdog.py``) stays
pure Python with no docker dependency at import time.

Attribute discovery
-------------------
In the currently installed ``tesseract_core`` (see
``/anaconda/envs/gym/lib/python3.13/site-packages/tesseract_core/sdk/tesseract.py``)
a :class:`Tesseract` created via ``from_image`` stores its container handle in::

    t._serve_context = {
        "container_name": "<name>",
        "port": ...,
        "network": ...,
        "network_alias": ...,
    }

We read that attribute directly. The private-underscore name is a stable
interface for our purposes — tesseract_core is vendored in the conda env and
the campaign pins it, so any rename would show up as a hard failure in our
unit test (``test_watchdog.py::test_container_id_from_serve_context``) before
it ever reaches a daemon.

If the attribute is missing (remote Tesseract, direct ``from_url`` path, or a
future tesseract_core rename) ``get_container_id`` returns ``None`` and the
watchdog silently falls back to wall-clock-only mode. That is strictly better
than the prior ARCH-6/ARCH-7 behaviour (wedged forever with no deadline
enforcement at all).
"""

from __future__ import annotations

import subprocess
import time
from typing import Any

# How long to wait for a `docker inspect` call before giving up. Docker is
# normally a few-millisecond local socket; anything over a few seconds means
# the docker daemon itself is unhealthy, and we should not add to the pile
# of stuck watchdog calls.
_DOCKER_INSPECT_TIMEOUT: float = 5.0

# Negative-result cache: if `docker inspect` reports the container is gone,
# we cache that for this many seconds before asking docker again. Prevents
# the watchdog loop from hammering `docker inspect` at every poll iteration
# once the container has already been declared dead. Positive results are
# never cached (we want to notice death quickly).
_NEGATIVE_CACHE_TTL: float = 2.0
_negative_cache: dict[str, float] = {}


def get_container_id(t: Any) -> str | None:
    """Return the docker container name (or id) backing Tesseract ``t``.

    Returns ``None`` if we cannot identify a container — the watchdog then
    falls back to wall-clock-only mode.

    Resolution order:

    1. ``t._serve_context["container_name"]`` — canonical location for a
       Tesseract opened via ``from_image`` (the normal harness path).
    2. ``t._container_name`` / ``t._container_id`` — tolerant to future
       tesseract_core renames. Tried as plain attribute lookups.
    3. Otherwise ``None``.
    """
    if t is None:
        return None

    serve_context = getattr(t, "_serve_context", None)
    if isinstance(serve_context, dict):
        name = serve_context.get("container_name")
        if name:
            return str(name)

    for attr in ("_container_name", "_container_id", "_container"):
        val = getattr(t, attr, None)
        if isinstance(val, str) and val:
            return val
        # ``_container`` may be an object with ``.id`` / ``.name``
        if val is not None:
            name = getattr(val, "name", None) or getattr(val, "id", None)
            if isinstance(name, str) and name:
                return name

    return None


def container_running(container_id: str) -> bool:
    """Return ``True`` if the named container is in the ``running`` state.

    Any other state (``exited``, ``dead``, ``created``, ``removing``, or
    ``not found``) → ``False``. A docker subprocess error (daemon down,
    permission denied, etc.) is also treated as ``False`` on the conservative
    principle that if we cannot verify the container is alive, the watchdog
    should err on the side of aborting the hung call rather than waiting
    indefinitely.

    Negative results are cached for :data:`_NEGATIVE_CACHE_TTL` seconds to
    avoid spamming ``docker inspect`` on every poll iteration after the
    container is already gone.
    """
    if not container_id:
        return False

    now = time.monotonic()
    cached_deadline = _negative_cache.get(container_id)
    if cached_deadline is not None and now < cached_deadline:
        return False

    try:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{.State.Status}}",
                container_id,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=_DOCKER_INSPECT_TIMEOUT,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        # Docker daemon wedged or docker CLI missing — treat as dead so the
        # watchdog does not itself hang.
        _negative_cache[container_id] = now + _NEGATIVE_CACHE_TTL
        return False

    if result.returncode != 0:
        # Typical cause: "Error: No such container". Cache negative.
        _negative_cache[container_id] = now + _NEGATIVE_CACHE_TTL
        return False

    status = (result.stdout or "").strip()
    if status == "running":
        # Positive result — drop any stale negative cache entry so death is
        # noticed at the next poll.
        _negative_cache.pop(container_id, None)
        return True

    _negative_cache[container_id] = now + _NEGATIVE_CACHE_TTL
    return False


def _reset_caches() -> None:
    """Test helper: drop the negative-result cache."""
    _negative_cache.clear()
