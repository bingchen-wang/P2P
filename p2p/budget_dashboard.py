"""Self-contained interactive presentation for budget forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html

from p2p.budgeting import BudgetForecast, IntervalEstimate


_COLORS = {
    "attribute_learner": "#a78bfa",
    "endowment_model": "#71d99c",
    "endowment_generator": "#71d99c",
    "survey_conductor": "#4ea1ff",
}


@dataclass(frozen=True, slots=True)
class BudgetDashboard:
    """One generated interactive budget report."""

    output_path: Path
    sections: tuple[str, ...]


def _plot(
    figure: go.Figure,
    *,
    div_id: str,
    include_js: bool,
) -> str:
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#121927",
        plot_bgcolor="#121927",
        font={"color": "#dce7f7"},
        margin={"l": 60, "r": 25, "t": 64, "b": 60},
        legend={"orientation": "h"},
    )
    return to_html(
        figure,
        full_html=False,
        include_plotlyjs="inline" if include_js else False,
        div_id=div_id,
        config={
            "responsive": True,
            "displaylogo": False,
            "scrollZoom": True,
        },
    )


def _error(value: IntervalEstimate) -> dict[str, Any]:
    return {
        "type": "data",
        "symmetric": False,
        "array": [max(value.high - value.point, 0.0)],
        "arrayminus": [max(value.point - value.low, 0.0)],
        "color": "#dce7f7",
    }


def _scenario_tokens(forecast: BudgetForecast) -> go.Figure:
    figure = go.Figure()
    for scenario in forecast.scenarios:
        figure.add_trace(
            go.Bar(
                name=scenario.name,
                x=[scenario.name],
                y=[scenario.total_tokens.point],
                error_y=_error(scenario.total_tokens),
                marker_color="#4ea1ff",
                text=[f"{scenario.total_tokens.point:,.0f}"],
                textposition="auto",
                hovertemplate=(
                    "%{x}<br>tokens=%{y:,.0f}"
                    "<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title=(
            f"Total token forecast · "
            f"{forecast.confidence:.0%} central interval"
        ),
        yaxis_title="input + output tokens",
        showlegend=False,
    )
    return figure


def _module_tokens(forecast: BudgetForecast) -> go.Figure:
    figure = go.Figure()
    for scenario in forecast.scenarios:
        for module in scenario.modules:
            label = f"{scenario.name} · {module.module_name}"
            figure.add_trace(
                go.Bar(
                    name=module.module_name,
                    x=[label],
                    y=[module.input_tokens.point],
                    marker_color=_COLORS.get(
                        module.module_name,
                        "#4ea1ff",
                    ),
                    hovertemplate=(
                        "%{x}<br>input=%{y:,.0f}"
                        "<extra></extra>"
                    ),
                )
            )
            figure.add_trace(
                go.Bar(
                    name=f"{module.module_name} output",
                    x=[label],
                    y=[module.output_tokens.point],
                    marker_color="#ffd166",
                    hovertemplate=(
                        "%{x}<br>output=%{y:,.0f}"
                        "<extra></extra>"
                    ),
                )
            )
    figure.update_layout(
        title="Expected tokens by module",
        barmode="stack",
        yaxis_title="tokens",
        showlegend=False,
    )
    return figure


def _scenario_cost(forecast: BudgetForecast) -> go.Figure | None:
    priced = [
        scenario
        for scenario in forecast.scenarios
        if scenario.total_cost_usd is not None
    ]
    if not priced:
        return None
    figure = go.Figure()
    for scenario in priced:
        cost = scenario.total_cost_usd
        assert cost is not None
        figure.add_trace(
            go.Bar(
                x=[scenario.name],
                y=[cost.point],
                error_y=_error(cost),
                marker_color="#71d99c",
                text=[f"${cost.point:,.4f}"],
                textposition="auto",
                hovertemplate=(
                    "%{x}<br>cost=$%{y:,.6f}"
                    "<extra></extra>"
                ),
            )
        )
        if scenario.budget_usd is not None:
            figure.add_hline(
                y=scenario.budget_usd,
                line_dash="dash",
                line_color="#ff6b7a",
                annotation_text="budget ceiling",
            )
    figure.update_layout(
        title="Estimated API cost",
        yaxis_title="USD",
        showlegend=False,
    )
    return figure


def _history_scatter(forecast: BudgetForecast) -> go.Figure | None:
    if not forecast.history_rows:
        return None
    history = pd.DataFrame(forecast.history_rows)
    if history.empty:
        return None
    grouped = (
        history.groupby(
            ["run_id", "run_dir", "model_name", "target_n"],
            as_index=False,
        )[["input_tokens", "output_tokens"]]
        .sum()
    )
    grouped["total_tokens"] = (
        grouped["input_tokens"] + grouped["output_tokens"]
    )
    figure = go.Figure()
    for model_name, group in grouped.groupby("model_name", sort=True):
        figure.add_trace(
            go.Scatter(
                x=group["target_n"],
                y=group["total_tokens"],
                mode="markers",
                name=str(model_name),
                text=group["run_id"],
                marker={"size": 8, "opacity": 0.65},
                hovertemplate=(
                    "%{text}<br>target_n=%{x}<br>"
                    "tokens=%{y:,.0f}<extra></extra>"
                ),
            )
        )
    configured = forecast.scenarios[0]
    figure.add_trace(
        go.Scatter(
            x=[forecast.features.target_n],
            y=[configured.total_tokens.point],
            mode="markers",
            name="forecast",
            marker={
                "size": 16,
                "symbol": "star",
                "color": "#ffd166",
                "line": {"width": 1, "color": "#fff"},
            },
        )
    )
    figure.update_layout(
        title="Historical runs and configured forecast",
        xaxis_title="target personas",
        yaxis_title="total tracked tokens",
    )
    return figure


def _pilot_trajectory(forecast: BudgetForecast) -> go.Figure | None:
    if forecast.pilot_run is None:
        return None
    run_dir = Path(forecast.pilot_run)
    path = next(
        (
            run_dir / filename
            for filename in ("usage.csv", "token_usage_summary.csv")
            if (run_dir / filename).is_file()
        ),
        None,
    )
    if path is None:
        return None
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    required = {"input_tokens", "output_tokens"}
    if frame.empty or not required <= set(frame.columns):
        return None
    input_tokens = pd.to_numeric(
        frame["input_tokens"],
        errors="coerce",
    ).fillna(0.0)
    output_tokens = pd.to_numeric(
        frame["output_tokens"],
        errors="coerce",
    ).fillna(0.0)
    cumulative = (input_tokens + output_tokens).cumsum()
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=list(range(1, len(cumulative) + 1)),
            y=cumulative,
            mode="lines+markers",
            marker_color="#ffd166",
            name="observed pilot",
            hovertemplate=(
                "record=%{x}<br>cumulative tokens=%{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )
    configured_modules = {
        (module.module_name, module.model_name): module
        for module in forecast.scenarios[0].modules
    }
    aliases = {
        "attributelearner": "attribute_learner",
        "attribute_learner": "attribute_learner",
        "endowmentgenerator": "endowment_generator",
        "endowment_generator": "endowment_generator",
        "endowmentmodel": "endowment_model",
        "endowment_model": "endowment_model",
        "surveyconductor": "survey_conductor",
        "survey_conductor": "survey_conductor",
    }
    record_costs: list[float] = []
    all_priced = True
    for index, row in frame.iterrows():
        module_name = aliases.get(
            str(row.get("module_name", "")).strip().lower()
        )
        model_name = str(row.get("model_name", ""))
        module = configured_modules.get((module_name, model_name))
        if module is None or not module.pricing.available:
            all_priced = False
            break
        input_value = float(input_tokens.iloc[index])
        output_value = float(output_tokens.iloc[index])
        cached_value = min(
            float(
                pd.to_numeric(
                    pd.Series([row.get("cached_input_tokens", 0)]),
                    errors="coerce",
                ).fillna(0.0).iloc[0]
            ),
            input_value,
        )
        record_costs.append(
            (
                (input_value - cached_value)
                * float(module.pricing.input_per_million_usd)
                + output_value
                * float(module.pricing.output_per_million_usd)
                + cached_value
                * float(module.pricing.cached_input_per_million_usd)
            )
            / 1_000_000
        )
    if all_priced and record_costs:
        figure.add_trace(
            go.Scatter(
                x=list(range(1, len(record_costs) + 1)),
                y=pd.Series(record_costs).cumsum(),
                mode="lines+markers",
                marker_color="#71d99c",
                name="observed pilot cost",
                yaxis="y2",
                hovertemplate=(
                    "record=%{x}<br>cumulative cost=$%{y:,.6f}"
                    "<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title="Live pilot usage trajectory",
        xaxis_title="usage records observed",
        yaxis_title="cumulative input + output tokens",
        yaxis2={
            "title": "cumulative USD",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
    )
    return figure


def _format_interval(
    value: IntervalEstimate | None,
    *,
    money: bool = False,
) -> str:
    if value is None:
        return "unpriced"
    if money:
        return (
            f"${value.point:,.4f} "
            f"[${value.low:,.4f}, ${value.high:,.4f}]"
        )
    return (
        f"{value.point:,.0f} "
        f"[{value.low:,.0f}, {value.high:,.0f}]"
    )


def _cards(forecast: BudgetForecast) -> str:
    configured = forecast.scenarios[0]
    values = [
        ("history", f"{forecast.historical_runs} runs"),
        ("tracked records", f"{forecast.historical_records:,}"),
        ("target personas", f"{forecast.features.target_n:,}"),
        ("questions", f"{forecast.features.question_count:,}"),
        ("tokens", _format_interval(configured.total_tokens)),
        (
            "cost",
            _format_interval(configured.total_cost_usd, money=True),
        ),
        ("budget", configured.budget_status),
        ("pilot records", f"{forecast.pilot_records:,}"),
    ]
    return "".join(
        "<div class='card'><span>"
        + escape(label)
        + "</span><strong>"
        + escape(value)
        + "</strong></div>"
        for label, value in values
    )


def _module_table(forecast: BudgetForecast) -> str:
    rows = []
    for scenario in forecast.scenarios:
        for module in scenario.modules:
            rows.append(
                "<tr><td>"
                + escape(scenario.name)
                + "</td><td>"
                + escape(module.module_name)
                + "</td><td>"
                + escape(module.model_name)
                + "</td><td>"
                + escape(_format_interval(module.calls))
                + "</td><td>"
                + escape(_format_interval(module.input_tokens))
                + "</td><td>"
                + escape(_format_interval(module.output_tokens))
                + "</td><td>"
                + escape(_format_interval(module.cost_usd, money=True))
                + "</td><td>"
                + escape(module.coverage)
                + "</td><td>"
                + escape(module.pricing.source)
                + "</td></tr>"
            )
    return (
        "<table><thead><tr><th>scenario</th><th>module</th>"
        "<th>model</th><th>calls</th><th>input tokens</th>"
        "<th>output tokens</th><th>cost</th><th>coverage</th>"
        "<th>pricing</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _feature_table(forecast: BudgetForecast) -> str:
    rows = "".join(
        "<tr><td>"
        + escape(name)
        + "</td><td>"
        + escape(str(value))
        + "</td></tr>"
        for name, value in {
            "config": forecast.config_source,
            "config digest": forecast.config_digest,
            "target command": forecast.target_command,
            "confidence": f"{forecast.confidence:.0%}",
            "history digest": forecast.history_digest,
            **{
                name: getattr(forecast.features, name)
                for name in forecast.features.__dataclass_fields__
            },
        }.items()
    )
    return (
        "<table><thead><tr><th>input</th><th>resolved value</th>"
        "</tr></thead><tbody>"
        + rows
        + "</tbody></table>"
    )


def build_budget_dashboard(
    forecast: BudgetForecast,
    output_path: str | os.PathLike[str],
    *,
    refresh_seconds: float | None = None,
) -> BudgetDashboard:
    """Write one self-contained interactive budget dashboard."""

    if not isinstance(forecast, BudgetForecast):
        raise TypeError("forecast must be BudgetForecast")
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if refresh_seconds is not None and (
        not isinstance(refresh_seconds, (int, float))
        or isinstance(refresh_seconds, bool)
        or not math.isfinite(float(refresh_seconds))
        or refresh_seconds <= 0
    ):
        raise ValueError("refresh_seconds must be positive")

    figures: list[tuple[str, go.Figure]] = [
        ("scenario-tokens", _scenario_tokens(forecast)),
        ("module-tokens", _module_tokens(forecast)),
    ]
    for name, figure in (
        ("scenario-cost", _scenario_cost(forecast)),
        ("history", _history_scatter(forecast)),
        ("pilot", _pilot_trajectory(forecast)),
    ):
        if figure is not None:
            figures.append((name, figure))
    fragments = []
    for index, (name, figure) in enumerate(figures):
        fragments.append(
            (
                name,
                _plot(
                    figure,
                    div_id=f"p2p-budget-{name}",
                    include_js=index == 0,
                ),
            )
        )
    sections = "".join(
        f"<section id='{escape(name)}'>{fragment}</section>"
        for name, fragment in fragments
    )
    refresh = (
        f"<meta http-equiv='refresh' content='{refresh_seconds:g}'>"
        if refresh_seconds is not None
        else ""
    )
    notes = "".join(
        f"<li>{escape(note)}</li>" for note in forecast.notes
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
{refresh}
<title>P2P budget forecast</title>
<style>
:root {{ color-scheme:dark; font-family:Inter,ui-sans-serif,system-ui; }}
body {{ margin:0; background:#0b101a; color:#dce7f7; }}
header {{ padding:32px max(24px,5vw) 20px; background:linear-gradient(135deg,#182943,#0b101a); }}
h1 {{ margin:0 0 8px; font-size:clamp(30px,4vw,46px); }}
.muted {{ color:#8ea1bb; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(165px,1fr)); gap:12px; margin-top:24px; }}
.card {{ min-width:0; background:#182235; border:1px solid #283851; border-radius:12px; padding:14px; }}
.card span {{ display:block; color:#8ea1bb; font-size:12px; text-transform:uppercase; letter-spacing:.08em; }}
.card strong {{ display:block; margin-top:7px; overflow-wrap:anywhere; }}
main {{ padding:20px max(16px,4vw) 48px; display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }}
section,.panel,details {{ min-width:0; background:#121927; border:1px solid #25334a; border-radius:14px; overflow:hidden; }}
.wide,details {{ grid-column:1/-1; }}
.panel,details {{ padding:20px; overflow:auto; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th,td {{ text-align:left; padding:9px; border-bottom:1px solid #25334a; vertical-align:top; }}
th {{ color:#8ea1bb; }}
footer {{ padding:0 max(24px,5vw) 32px; color:#71839d; }}
@media (max-width:900px) {{ main {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
  <div class="muted">PROMPTS TO PROXIES · PRE-RUN BUDGET PLANNER</div>
  <h1>Token and cost forecast</h1>
  <div class="muted">{escape(forecast.config_source)}</div>
  <div class="cards">{_cards(forecast)}</div>
</header>
<main>
  <div class="panel wide"><h2>Forecast by module</h2>{_module_table(forecast)}</div>
  {sections}
  <details><summary>Assumptions and model card</summary><ul>{notes}</ul>{_feature_table(forecast)}</details>
</main>
<footer>Forecasts are planning aids, not billing guarantees. No API call is made by this command.</footer>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")
    return BudgetDashboard(
        output_path=output,
        sections=tuple(name for name, _ in fragments),
    )


__all__ = [
    "BudgetDashboard",
    "build_budget_dashboard",
]
