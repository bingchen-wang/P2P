"""Command-line interface for P2P."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import logging
from pathlib import Path
import sys

import p2p.commands  # Registers implemented command handlers.
from p2p.models.constrained_model import DEFAULT_SOLVER
from p2p.config import Config
from p2p.registry import COMMANDS


def _add_config_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the central YAML manifest.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose stage output.",
    )


def _add_solver_argument(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--solver",
        default=DEFAULT_SOLVER,
        help=(
            f"CVXPY solver name (default: {DEFAULT_SOLVER}). Use OSQP to "
            "reproduce fits made with it."
        ),
    )


def _add_output_argument(
    parser: argparse.ArgumentParser,
    *,
    help_text: str,
) -> None:
    parser.add_argument(
        "--output",
        dest="run_dir",
        required=True,
        type=Path,
        help=help_text,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the single command-line parser."""

    parser = argparse.ArgumentParser(
        prog="p2p",
        description="Prompts to Proxies",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    regress_parser = subparsers.add_parser(
        "regress",
        help="Run Stage 2 regression and evaluation.",
    )
    _add_config_arguments(regress_parser)
    _add_solver_argument(regress_parser)
    _add_output_argument(
        regress_parser,
        help_text="Directory for Stage-2 result artifacts.",
    )

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run generation, elicitation, and Stage 2.",
    )
    _add_config_arguments(pipeline_parser)
    _add_solver_argument(pipeline_parser)
    _add_output_argument(
        pipeline_parser,
        help_text=(
            "Directory for pipeline basis and result artifacts."
        ),
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate existing prediction or fitted outputs.",
    )
    _add_config_arguments(evaluate_parser)
    _add_output_argument(
        evaluate_parser,
        help_text="Directory for canonical evaluation artifacts.",
    )

    aggregate_parser = subparsers.add_parser(
        "aggregate",
        help="Validate and summarize canonical run artifacts.",
    )
    aggregate_source = (
        aggregate_parser.add_mutually_exclusive_group(
            required=True
        )
    )
    aggregate_source.add_argument(
        "--config",
        type=Path,
        help="Path to an aggregation manifest.",
    )
    aggregate_source.add_argument(
        "--runs",
        nargs="+",
        type=Path,
        help="Explicit canonical run directories.",
    )
    aggregate_parser.add_argument(
        "--group-by",
        nargs="+",
        default=("model", "split"),
        help="Quick-mode grouping fields.",
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Build or serve interactive diagnostics for a run.",
    )
    inspect_parser.add_argument(
        "--run",
        dest="inspect_run_dir",
        required=True,
        type=Path,
        help="Completed or running output directory.",
    )
    inspect_parser.add_argument(
        "--output",
        dest="inspect_output",
        type=Path,
        help="Dashboard HTML path (default: RUN/diagnostics.html).",
    )
    inspect_parser.add_argument(
        "--advanced",
        action="store_true",
        help="Include MCA and singular-value views.",
    )
    inspect_parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve and refresh the dashboard while a run progresses.",
    )
    inspect_parser.add_argument("--host", default="127.0.0.1")
    inspect_parser.add_argument("--port", type=int, default=8050)
    inspect_parser.add_argument(
        "--refresh",
        dest="refresh_seconds",
        type=float,
        default=2.0,
        help="Live page refresh interval in seconds.",
    )
    inspect_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose command output.",
    )

    budget_parser = subparsers.add_parser(
        "budget",
        help="Forecast tokens and cost before an API-backed run.",
    )
    _add_config_arguments(budget_parser)
    budget_parser.add_argument(
        "--history",
        nargs="+",
        type=Path,
        help="Run directories to learn token usage from (default: outputs/).",
    )
    budget_parser.add_argument(
        "--for-command",
        dest="target_command",
        choices=("generate", "pipeline"),
        default="pipeline",
        help="API workflow being budgeted (default: pipeline).",
    )
    budget_parser.add_argument(
        "--compare-model",
        dest="comparison_models",
        action="append",
        default=[],
        help="Add a same-workload model comparison scenario.",
    )
    budget_parser.add_argument(
        "--pricing",
        dest="pricing_path",
        type=Path,
        help="Optional current per-million-token pricing YAML.",
    )
    budget_parser.add_argument(
        "--pilot-run",
        type=Path,
        help="Optional small or in-progress run used for calibration.",
    )
    budget_parser.add_argument(
        "--confidence",
        type=float,
        default=0.8,
        help="Central forecast interval in [0.5, 1) (default: 0.8).",
    )
    budget_parser.add_argument(
        "--budget-usd",
        type=float,
        help="Optional USD ceiling shown against the forecast interval.",
    )
    budget_parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve and refresh forecasts as pilot usage appears.",
    )
    budget_parser.add_argument("--host", default="127.0.0.1")
    budget_parser.add_argument("--port", type=int, default=8060)
    budget_parser.add_argument(
        "--refresh",
        dest="refresh_seconds",
        type=float,
        default=2.0,
        help="Live page refresh interval in seconds.",
    )
    budget_parser.add_argument(
        "--output",
        dest="output_dir",
        required=True,
        type=Path,
        help="Directory for budget JSON, CSV, provenance, and HTML.",
    )

    ui_parser = subparsers.add_parser(
        "ui",
        help="Launch the local P2P Experiment Studio.",
    )
    ui_parser.add_argument("--host", default="127.0.0.1")
    ui_parser.add_argument("--port", type=int, default=5173)
    ui_parser.add_argument(
        "--no-open-browser",
        action="store_true",
        help="Do not open the Studio in the default browser.",
    )
    ui_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose command output.",
    )
    aggregate_parser.add_argument(
        "--metrics",
        nargs="+",
        help="Quick-mode requested metrics.",
    )
    aggregate_parser.add_argument(
        "--output",
        dest="output_dir",
        required=True,
        type=Path,
        help="Directory for aggregation artifacts.",
    )
    aggregate_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose command output.",
    )

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate endowments from the configured strategy.",
    )
    _add_config_arguments(generate_parser)
    _add_output_argument(
        generate_parser,
        help_text="Directory for generation artifacts.",
    )

    elicit_parser = subparsers.add_parser(
        "elicit",
        help="Elicit responses from existing endowments.",
    )
    _add_config_arguments(elicit_parser)
    elicit_parser.add_argument(
        "--endowments",
        dest="endowments_path",
        required=True,
        type=Path,
        help="Path to an existing endowments.csv artifact.",
    )
    _add_output_argument(
        elicit_parser,
        help_text="Directory for elicitation artifacts.",
    )

    for name, help_text in (
        (
            "simulate",
            "Execute a configured simulation sweep.",
        ),
        (
            "ablate",
            "Execute a configured ablation sweep.",
        ),
        (
            "panel",
            "Execute a configured panel sweep.",
        ),
    ):
        sweep_parser = subparsers.add_parser(
            name,
            help=help_text,
        )
        _add_config_arguments(sweep_parser)
        _add_solver_argument(sweep_parser)
        _add_output_argument(
            sweep_parser,
            help_text="Directory for sweep child artifacts.",
        )

    cluster_parser = subparsers.add_parser(
        "cluster",
        help="Cluster existing endowment text offline.",
    )
    _add_config_arguments(cluster_parser)
    _add_output_argument(
        cluster_parser,
        help_text="Directory for clustering artifacts.",
    )

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a central manifest for one command.",
    )
    _add_config_arguments(validate_parser)
    validate_parser.add_argument(
        "--target-verb",
        "--verb",
        dest="target_verb",
        required=True,
        help="Command contract to validate.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch through the command registry."""

    parser = build_parser()
    arguments = parser.parse_args(argv)

    if arguments.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            stream=sys.stdout,
        )
        # Keep P2P progress and diagnostics readable. The provider clients
        # otherwise emit one INFO line per HTTP 200, interleaving thousands
        # of transport messages with tqdm progress bars.
        for logger_name in ("httpx", "httpcore", "openai"):
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    if arguments.command == "ui":
        from p2p.studio import run_studio

        run_studio(
            root=Path.cwd(),
            host=arguments.host,
            port=arguments.port,
            open_browser=not arguments.no_open_browser,
        )
        return 0

    handler = COMMANDS.get(arguments.command)

    if arguments.command == "aggregate":
        config = (
            Config.load(arguments.config)
            if arguments.config is not None
            else None
        )
        handler(
            config,
            output_dir=arguments.output_dir,
            runs=arguments.runs,
            group_by=arguments.group_by,
            metrics=arguments.metrics,
            verbose=arguments.verbose,
        )
        return 0

    if arguments.command == "inspect":
        handler(
            None,
            run_dir=arguments.inspect_run_dir,
            output_path=arguments.inspect_output,
            advanced=arguments.advanced,
            serve=arguments.serve,
            host=arguments.host,
            port=arguments.port,
            refresh_seconds=arguments.refresh_seconds,
            verbose=arguments.verbose,
        )
        return 0

    if arguments.command == "budget":
        config = Config.load(arguments.config)
        handler(
            config,
            output_dir=arguments.output_dir,
            history=arguments.history,
            target_command=arguments.target_command,
            comparison_models=arguments.comparison_models,
            pricing_path=arguments.pricing_path,
            pilot_run=arguments.pilot_run,
            confidence=arguments.confidence,
            budget_usd=arguments.budget_usd,
            serve=arguments.serve,
            host=arguments.host,
            port=arguments.port,
            refresh_seconds=arguments.refresh_seconds,
            verbose=arguments.verbose,
        )
        return 0

    config = Config.load(arguments.config)

    command_kwargs = {
        "verbose": arguments.verbose,
    }
    if hasattr(arguments, "solver"):
        command_kwargs["solver"] = arguments.solver
    if hasattr(arguments, "run_dir"):
        command_kwargs["run_dir"] = arguments.run_dir
    if hasattr(arguments, "endowments_path"):
        command_kwargs["endowments_path"] = (
            arguments.endowments_path
        )
    if hasattr(arguments, "target_verb"):
        command_kwargs["target_verb"] = arguments.target_verb

    handler(
        config,
        **command_kwargs,
    )
    return 0


__all__ = [
    "build_parser",
    "main",
]
