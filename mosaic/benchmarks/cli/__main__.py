"""Allow ``python -m mosaic.benchmarks.cli`` to invoke the Typer app."""

from __future__ import annotations

from mosaic.benchmarks.cli import app

if __name__ == "__main__":
    app()
