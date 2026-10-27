"""Forecast and serve token/cost budgets without calling a backend."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
from typing import Any, Sequence

from p2p.budget_dashboard import (
    BudgetDashboard,
    build_budget_dashboard,
)
from p2p.budgeting import (
    BudgetArtifacts,
    BudgetForecast,
    forecast_budget,
    write_budget_artifacts,
)
from p2p.config import Config


@dataclass(frozen=True, slots=True)
class BudgetCommandResult:
    """Artifacts emitted by one static budget build."""

    forecast: BudgetForecast
    artifacts: BudgetArtifacts
    dashboard: BudgetDashboard


def _history_roots(
    values: Sequence[str | os.PathLike[str]] | None,
) -> tuple[Path, ...]:
    if values:
        return tuple(Path(value).resolve() for value in values)
    # Every run recorded in this workspace.
    default = (Path.cwd() / "outputs").resolve()
    return (default,)


def _build(
    config: Config,
    *,
    output_dir: Path,
    history: Sequence[str | os.PathLike[str]] | None,
    target_command: str,
    comparison_models: Sequence[str],
    pricing_path: str | os.PathLike[str] | None,
    pilot_run: str | os.PathLike[str] | None,
    confidence: float,
    budget_usd: float | None,
    refresh_seconds: float | None = None,
) -> BudgetCommandResult:
    forecast = forecast_budget(
        config,
        history_roots=_history_roots(history),
        target_command=target_command,
        comparison_models=comparison_models,
        pricing_path=pricing_path,
        pilot_run=pilot_run,
        confidence=confidence,
        budget_usd=budget_usd,
    )
    artifacts = write_budget_artifacts(forecast, output_dir)
    dashboard = build_budget_dashboard(
        forecast,
        output_dir / "budget.html",
        refresh_seconds=refresh_seconds,
    )
    return BudgetCommandResult(
        forecast=forecast,
        artifacts=artifacts,
        dashboard=dashboard,
    )


def _serve(
    config: Config,
    *,
    output_dir: Path,
    history: Sequence[str | os.PathLike[str]] | None,
    target_command: str,
    comparison_models: Sequence[str],
    pricing_path: str | os.PathLike[str] | None,
    pilot_run: str | os.PathLike[str] | None,
    confidence: float,
    budget_usd: float | None,
    host: str,
    port: int,
    refresh_seconds: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = (output_dir / "budget.html").resolve()

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib hook
            if self.path in {"/", "/budget.html"}:
                _build(
                    config,
                    output_dir=output_dir,
                    history=history,
                    target_command=target_command,
                    comparison_models=comparison_models,
                    pricing_path=pricing_path,
                    pilot_run=pilot_run,
                    confidence=confidence,
                    budget_usd=budget_usd,
                    refresh_seconds=refresh_seconds,
                )
                self.path = "/budget.html"
            super().do_GET()

        def log_message(self, format: str, *args: Any) -> None:
            return

    try:
        server = ThreadingHTTPServer(
            (host, port),
            partial(Handler, directory=str(output_dir)),
        )
        print(f"[budget] serving http://{host}:{port}/")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if "server" in locals():
            server.server_close()


def run(
    config: Config,
    *,
    output_dir: str | os.PathLike[str],
    history: Sequence[str | os.PathLike[str]] | None = None,
    target_command: str = "pipeline",
    comparison_models: Sequence[str] = (),
    pricing_path: str | os.PathLike[str] | None = None,
    pilot_run: str | os.PathLike[str] | None = None,
    confidence: float = 0.8,
    budget_usd: float | None = None,
    serve: bool = False,
    host: str = "127.0.0.1",
    port: int = 8060,
    refresh_seconds: float = 2.0,
    verbose: bool = False,
) -> BudgetCommandResult | None:
    """Build or serve a pre-run budget forecast."""

    if not isinstance(config, Config):
        raise TypeError("config must be Config")
    output = Path(output_dir).resolve()
    if serve:
        _serve(
            config,
            output_dir=output,
            history=history,
            target_command=target_command,
            comparison_models=comparison_models,
            pricing_path=pricing_path,
            pilot_run=pilot_run,
            confidence=confidence,
            budget_usd=budget_usd,
            host=host,
            port=port,
            refresh_seconds=refresh_seconds,
        )
        return None

    result = _build(
        config,
        output_dir=output,
        history=history,
        target_command=target_command,
        comparison_models=comparison_models,
        pricing_path=pricing_path,
        pilot_run=pilot_run,
        confidence=confidence,
        budget_usd=budget_usd,
    )
    configured = result.forecast.scenarios[0]
    tokens = configured.total_tokens
    cost = configured.total_cost_usd
    cost_text = (
        "unpriced"
        if cost is None
        else (
            f"${cost.point:.4f} "
            f"[${cost.low:.4f}, ${cost.high:.4f}]"
        )
    )
    print(
        "[budget] "
        f"{tokens.point:,.0f} tokens "
        f"[{tokens.low:,.0f}, {tokens.high:,.0f}]; "
        f"{cost_text}; wrote {result.dashboard.output_path}"
    )
    if verbose:
        print(
            "[budget] trained from "
            f"{result.forecast.historical_runs} runs and "
            f"{result.forecast.historical_records:,} usage records"
        )
    return result


__all__ = [
    "BudgetCommandResult",
    "run",
]
