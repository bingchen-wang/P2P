"""Generate or serve the interactive run inspector."""

from __future__ import annotations

import os

from p2p.diagnostics import (
    DiagnosticDashboard,
    build_dashboard,
    serve_dashboard,
)


def run(
    _config=None,
    *,
    run_dir: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
    advanced: bool = False,
    serve: bool = False,
    host: str = "127.0.0.1",
    port: int = 8050,
    refresh_seconds: float = 2.0,
    verbose: bool = False,
) -> DiagnosticDashboard | None:
    """Build a static dashboard or serve a refreshing local view."""

    if serve:
        serve_dashboard(
            run_dir,
            output_path=output_path,
            advanced=advanced,
            host=host,
            port=port,
            refresh_seconds=refresh_seconds,
        )
        return None
    result = build_dashboard(
        run_dir,
        output_path=output_path,
        advanced=advanced,
    )
    print(f"[inspect] wrote {result.output_path}")
    return result


__all__ = ["run"]
