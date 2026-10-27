"""Self-contained interactive diagnostics for completed or running P2P runs."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from functools import partial
from html import escape
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html
import yaml

from p2p.earlier_runs import load_earlier_run
from p2p.responses import Responses
from p2p.responses.converter import ResponseUtils
from p2p.survey import Survey


_COLORS = {
    "train": "#4ea1ff",
    "valid": "#71d99c",
    "test": "#ffd166",
    "unknown": "#a78bfa",
}


@dataclass(frozen=True, slots=True)
class DiagnosticDashboard:
    """One generated dashboard and the evidence it could load."""

    run_dir: Path
    output_path: Path
    sections: tuple[str, ...]
    warnings: tuple[str, ...]


def _load_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_yaml(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None


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
        margin={"l": 55, "r": 25, "t": 58, "b": 55},
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


def _source_path(run: Any, name: str) -> Path | None:
    if not isinstance(run, dict):
        return None
    source = run.get("source_artifacts", {}).get(name)
    if not isinstance(source, dict):
        return None
    value = source.get("path")
    if not isinstance(value, str) or not value:
        return None
    return Path(value)


def _config_data(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    data = config.get("config", config)
    return data if isinstance(data, dict) else {}


def _load_responses(
    run_dir: Path,
    run: Any,
    config: Any,
) -> tuple[Survey | None, Responses | None, str | None]:
    config_data = _config_data(config)
    survey_config = config_data.get("data", {}).get("survey", {})
    survey_path = _source_path(run, "survey")
    schema_path = _source_path(run, "survey_schema")
    if survey_path is None and isinstance(survey_config, dict):
        value = survey_config.get("source")
        if isinstance(value, str):
            survey_path = Path(value)
    if schema_path is None and isinstance(survey_config, dict):
        value = survey_config.get("schema")
        if isinstance(value, str):
            schema_path = Path(value)

    responses_path = run_dir / "responses.csv"
    if not responses_path.is_file():
        responses_path = _source_path(run, "responses") or responses_path
    if (
        survey_path is None
        or schema_path is None
        or not survey_path.is_file()
        or not schema_path.is_file()
        or not responses_path.is_file()
    ):
        return None, None, (
            "survey/schema/responses were not all available; "
            "response-level views are hidden"
        )
    try:
        survey = Survey(survey_path, schema_path)
        responses = Responses(
            str(responses_path),
            survey,
            output_format="answer",
        )
    except Exception as error:
        return None, None, f"could not load response basis: {error}"
    return survey, responses, None


def _entropy_data(
    survey: Survey,
    responses: Responses,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for question in survey.questions:
        qid = question["id"]
        values = list(responses.get_question_vector(qid).values())
        order = list(question.get("code_to_answer", {}).values())
        counts = {label: values.count(label) for label in order}
        for value in values:
            if value not in counts:
                counts[value] = values.count(value)
        entropy = ResponseUtils.categorical_entropy(list(counts.values()))
        rows.append(
            {
                "qid": qid,
                "split": question.get("split") or "unknown",
                "entropy": float(entropy),
                "n": len(values),
            }
        )
        details[qid] = {
            "qid": qid,
            "question": question.get("question", ""),
            "labels": list(counts),
            "counts": list(counts.values()),
        }
    return rows, details


def _metric_figure(metrics: Any) -> go.Figure | None:
    if not isinstance(metrics, dict):
        return None
    rows = metrics.get("metrics")
    if not isinstance(rows, list) or not rows:
        return None
    labels = [
        (
            f"{row.get('model')} · {row.get('metric')} · "
            f"{row.get('split')}"
        )
        for row in rows
    ]
    values = [row.get("value") for row in rows]
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color="#4ea1ff",
            text=[f"{float(value):.4g}" for value in values],
            textposition="auto",
        )
    )
    figure.update_layout(title="Run metrics", yaxis_title="value")
    return figure


def _entropy_figure(rows: list[dict[str, Any]]) -> go.Figure:
    ordered = sorted(rows, key=lambda row: row["entropy"])
    figure = go.Figure(
        go.Bar(
            x=[row["qid"] for row in ordered],
            y=[row["entropy"] for row in ordered],
            marker_color=[
                _COLORS.get(row["split"], _COLORS["unknown"])
                for row in ordered
            ],
            customdata=[row["qid"] for row in ordered],
            hovertemplate=(
                "%{x}<br>entropy=%{y:.3f}<br>"
                "click to inspect responses<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="Question entropy (current basis)",
        xaxis_title="question",
        yaxis_title="normalized entropy",
        yaxis_range=[0, 1],
    )
    return figure


def _prediction_figures(
    diagnostics: Any,
) -> tuple[go.Figure, go.Figure, dict[str, Any]] | None:
    if not isinstance(diagnostics, dict):
        return None
    questions = diagnostics.get("questions")
    if not isinstance(questions, list) or not questions:
        return None
    x: list[float] = []
    y: list[float] = []
    text: list[str] = []
    qids: list[str] = []
    details: dict[str, Any] = {}
    for question in questions:
        qid = question.get("qid")
        target = question.get("target", {})
        prediction = question.get("prediction", {})
        order = list(question.get("option_order", ()))
        for label in target:
            if label not in order:
                order.append(label)
        for label in prediction:
            if label not in order:
                order.append(label)
        details[qid] = {
            **question,
            "option_order": order,
        }
        for label in order:
            x.append(float(target.get(label, 0.0)))
            y.append(float(prediction.get(label, 0.0)))
            text.append(label)
            qids.append(qid)

    lower = min([0.0, *x, *y])
    upper = max([1.0, *x, *y])
    scatter = go.Figure()
    scatter.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            text=text,
            customdata=qids,
            marker={
                "size": 10,
                "color": "#71d99c",
                "line": {"width": 1, "color": "#dce7f7"},
            },
            hovertemplate=(
                "%{customdata}<br>%{text}<br>"
                "human=%{x:.3f}<br>prediction=%{y:.3f}"
                "<extra></extra>"
            ),
        )
    )
    scatter.add_trace(
        go.Scatter(
            x=[lower, upper],
            y=[lower, upper],
            mode="lines",
            line={"dash": "dash", "color": "#7f8da5"},
            hoverinfo="skip",
        )
    )
    scatter.update_layout(
        title="Prediction vs human target",
        xaxis_title="human probability",
        yaxis_title="predicted probability",
        showlegend=False,
    )
    first = questions[0]["qid"]
    distribution = _distribution_figure(details[first])
    return scatter, distribution, details


def _distribution_figure(detail: dict[str, Any]) -> go.Figure:
    order = detail.get("option_order", [])
    target = detail.get("target", {})
    prediction = detail.get("prediction", {})
    figure = go.Figure(
        [
            go.Bar(
                name="human",
                x=order,
                y=[target.get(label, 0.0) for label in order],
                marker_color="#4ea1ff",
            ),
            go.Bar(
                name="prediction",
                x=order,
                y=[prediction.get(label, 0.0) for label in order],
                marker_color="#71d99c",
            ),
        ]
    )
    figure.update_layout(
        title=f"{detail.get('qid')} · {detail.get('question', '')}",
        barmode="group",
        yaxis_title="probability",
    )
    return figure


def _weights_figure(diagnostics: Any) -> go.Figure | None:
    if not isinstance(diagnostics, dict):
        return None
    coefficients = diagnostics.get("model", {}).get("coefficients")
    if not isinstance(coefficients, dict) or not coefficients:
        return None
    nonzero = [
        item
        for item in coefficients.items()
        if abs(float(item[1])) > 1e-15
    ]
    ordered = sorted(
        nonzero,
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )
    if not ordered:
        return None
    figure = go.Figure(
        go.Bar(
            x=[item[0] for item in ordered],
            y=[item[1] for item in ordered],
            marker_color=[
                "#71d99c" if float(item[1]) >= 0 else "#ff6b7a"
                for item in ordered
            ],
        )
    )
    figure.update_layout(
        title="Proxy coefficients",
        xaxis_title="agent",
        yaxis_title="coefficient",
    )
    return figure


def _trajectory_figures(run_dir: Path) -> list[tuple[str, go.Figure]]:
    results: list[tuple[str, go.Figure]] = []
    for filename, title in (
        ("aggregate_entropy_trajectory.csv", "Aggregate entropy trajectory"),
        ("question_entropy_trajectories.csv", "Question entropy trajectories"),
        ("persistent_mode_entropy.csv", "Mode entropy trajectories"),
    ):
        path = run_dir / filename
        if not path.is_file():
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if "step" not in frame or len(frame.columns) < 2:
            continue
        figure = go.Figure()
        for column in frame.columns:
            if column == "step":
                continue
            figure.add_trace(
                go.Scatter(
                    x=frame["step"],
                    y=frame[column],
                    mode="lines+markers",
                    name=column,
                )
            )
        figure.update_layout(
            title=title,
            xaxis_title="generation step",
            yaxis_title="entropy",
        )
        results.append((filename.removesuffix(".csv"), figure))
    return results


def _usage_figure(run_dir: Path) -> go.Figure | None:
    path = next(
        (
            candidate
            for candidate in (
                run_dir / "usage.csv",
                run_dir / "token_usage_summary.csv",
            )
            if candidate.is_file()
        ),
        None,
    )
    if path is None:
        return None
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    required = {"module_name", "input_tokens", "output_tokens"}
    if not required <= set(frame.columns):
        return None
    grouped = frame.groupby("module_name", sort=True)[
        ["input_tokens", "output_tokens"]
    ].sum()
    figure = go.Figure(
        [
            go.Bar(
                name="input",
                x=grouped.index,
                y=grouped["input_tokens"],
                marker_color="#4ea1ff",
            ),
            go.Bar(
                name="output",
                x=grouped.index,
                y=grouped["output_tokens"],
                marker_color="#71d99c",
            ),
        ]
    )
    figure.update_layout(
        title="Token usage by module",
        barmode="stack",
        yaxis_title="tokens",
    )
    return figure


def _matrix_figures(
    survey: Survey,
    responses: Responses,
) -> list[tuple[str, go.Figure]]:
    qids = [question["id"] for question in survey.questions]
    eids = sorted(
        {
            eid
            for answers in responses.responses.values()
            for eid in answers
        }
    )
    if len(qids) < 2 or len(eids) < 2:
        return []

    coded = np.full((len(eids), len(qids)), np.nan)
    categorical_columns: dict[str, list[str]] = {}
    for column, question in enumerate(survey.questions):
        qid = question["id"]
        answers = responses.responses.get(qid, {})
        labels = list(question.get("code_to_answer", {}).values())
        label_to_code = {label: index for index, label in enumerate(labels)}
        categorical_columns[qid] = [
            answers.get(eid, "<missing>")
            for eid in eids
        ]
        for row, eid in enumerate(eids):
            answer = answers.get(eid)
            if answer is not None:
                coded[row, column] = label_to_code.get(answer, np.nan)
    categorical = pd.DataFrame(
        categorical_columns,
        index=eids,
    )
    column_means = np.nanmean(coded, axis=0)
    column_means = np.where(np.isfinite(column_means), column_means, 0.0)
    missing = np.where(np.isnan(coded))
    coded[missing] = column_means[missing[1]]
    centered = coded - coded.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(
        centered,
        full_matrices=False,
        compute_uv=False,
    )
    spectrum = go.Figure(
        go.Scatter(
            x=list(range(1, len(singular_values) + 1)),
            y=singular_values,
            mode="lines+markers",
            marker_color="#4ea1ff",
        )
    )
    spectrum.update_layout(
        title="Response-matrix singular values",
        xaxis_title="component",
        yaxis_title="singular value",
    )

    indicator = pd.get_dummies(categorical, prefix_sep="=").to_numpy(
        dtype=float
    )
    total = indicator.sum()
    if total <= 0:
        return [("singular_values", spectrum)]
    probability = indicator / total
    row_mass = probability.sum(axis=1)
    column_mass = probability.sum(axis=0)
    expected = np.outer(row_mass, column_mass)
    denominator = np.sqrt(expected)
    standardized = np.divide(
        probability - expected,
        denominator,
        out=np.zeros_like(probability),
        where=denominator > 0,
    )
    u, singular, _ = np.linalg.svd(standardized, full_matrices=False)
    dimensions = min(2, u.shape[1])
    coordinates = np.divide(
        u[:, :dimensions],
        np.sqrt(row_mass)[:, None],
        out=np.zeros((len(eids), dimensions)),
        where=row_mass[:, None] > 0,
    ) * singular[:dimensions]
    if dimensions == 1:
        coordinates = np.column_stack(
            [coordinates[:, 0], np.zeros(len(eids))]
        )
    mca = go.Figure(
        go.Scatter(
            x=coordinates[:, 0],
            y=coordinates[:, 1],
            mode="markers",
            text=eids,
            marker={
                "size": 9,
                "color": "#a78bfa",
                "opacity": 0.8,
            },
            hovertemplate="%{text}<extra></extra>",
        )
    )
    mca.update_layout(
        title="Multiple correspondence map (agents)",
        xaxis_title="dimension 1",
        yaxis_title="dimension 2",
    )
    return [
        ("singular_values", spectrum),
        ("mca_agents", mca),
    ]


def _cards(
    run_dir: Path,
    run: Any,
    metrics: Any,
    diagnostics: Any,
) -> str:
    run_id = run.get("run_id", run_dir.name) if isinstance(run, dict) else run_dir.name
    metric_count = (
        len(metrics.get("metrics", []))
        if isinstance(metrics, dict)
        else 0
    )
    response_path = run_dir / "responses.csv"
    endowment_path = run_dir / "endowments.csv"
    response_source = _source_path(run, "responses")
    endowment_source = _source_path(run, "endowments")
    has_responses = response_path.is_file() or (
        response_source is not None and response_source.is_file()
    )
    has_endowments = endowment_path.is_file() or (
        endowment_source is not None and endowment_source.is_file()
    )
    artifact_type = (
        diagnostics.get("artifact_type")
        if isinstance(diagnostics, dict)
        else None
    )
    if isinstance(run, dict):
        status = "reported"
    elif artifact_type == "legacy_regression_diagnostics":
        status = "legacy parsed"
    else:
        status = "in progress / legacy"
    model = (
        diagnostics.get("model", {}).get("method")
        if isinstance(diagnostics, dict)
        else None
    )
    values = [
        ("run", run_id),
        ("status", status),
        ("model", model or "unknown"),
        ("metrics", metric_count),
        ("responses", "yes" if has_responses else "no"),
        ("endowments", "yes" if has_endowments else "no"),
    ]
    return "".join(
        "<div class='card'><span>"
        + escape(str(label))
        + "</span><strong>"
        + escape(str(value))
        + "</strong></div>"
        for label, value in values
    )


def _legacy_image_gallery(paths: tuple[Path, ...]) -> str:
    if not paths:
        return ""
    cards = []
    for path in paths:
        try:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        except OSError:
            continue
        cards.append(
            "<figure><figcaption>"
            + escape(path.stem.replace("_", " "))
            + "</figcaption><img loading='lazy' alt='"
            + escape(path.stem)
            + "' src='data:image/png;base64,"
            + encoded
            + "'></figure>"
        )
    if not cards:
        return ""
    return (
        "<details class='wide legacy-gallery'><summary>"
        "Archived legacy regression plots</summary><div>"
        + "".join(cards)
        + "</div></details>"
    )


def _artifact_table(run_dir: Path) -> str:
    rows = []
    for path in sorted(run_dir.iterdir()):
        if not path.is_file():
            continue
        rows.append(
            "<tr><td>"
            + escape(path.name)
            + "</td><td>"
            + f"{path.stat().st_size / 1024:.1f} KiB"
            + "</td><td>"
            + escape(
                str(
                    pd.Timestamp(
                        path.stat().st_mtime,
                        unit="s",
                    )
                )
            )
            + "</td></tr>"
        )
    return (
        "<table><thead><tr><th>artifact</th><th>size</th>"
        "<th>modified</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def build_dashboard(
    run_dir: str | os.PathLike[str],
    *,
    output_path: str | os.PathLike[str] | None = None,
    advanced: bool = False,
    refresh_seconds: float | None = None,
) -> DiagnosticDashboard:
    """Build a standalone Plotly dashboard from a run directory."""

    run_path = Path(run_dir).resolve()
    if not run_path.is_dir():
        raise ValueError(f"run directory does not exist: {run_path}")
    output = (
        Path(output_path).resolve()
        if output_path is not None
        else (run_path / "diagnostics.html").resolve()
    )
    if output.parent != run_path and not output.parent.exists():
        output.parent.mkdir(parents=True, exist_ok=True)

    run = _load_json(run_path / "run.json")
    metrics = _load_json(run_path / "metrics.json")
    diagnostics = _load_json(run_path / "diagnostics.json")
    config = _load_yaml(run_path / "config.resolved.yaml")
    survey, responses, response_warning = _load_responses(
        run_path,
        run,
        config,
    )
    legacy = load_earlier_run(run_path)
    if metrics is None and legacy.metrics is not None:
        metrics = legacy.metrics
    if diagnostics is None and legacy.diagnostics is not None:
        diagnostics = legacy.diagnostics
    if survey is None and legacy.survey is not None:
        survey = legacy.survey
        responses = legacy.responses
        response_warning = None
    warnings: list[str] = []
    if response_warning is not None:
        warnings.append(response_warning)
    warnings.extend(legacy.notes)

    plots: list[tuple[str, str]] = []
    include_js = True

    def add_plot(name: str, figure: go.Figure) -> None:
        nonlocal include_js
        plots.append(
            (
                name,
                _plot(
                    figure,
                    div_id=f"p2p-{name}",
                    include_js=include_js,
                ),
            )
        )
        include_js = False

    metric_figure = _metric_figure(metrics)
    if metric_figure is not None:
        add_plot("metrics", metric_figure)

    entropy_rows: list[dict[str, Any]] = []
    response_details: dict[str, Any] = {}
    if survey is not None and responses is not None:
        entropy_rows, response_details = _entropy_data(survey, responses)
        if entropy_rows:
            add_plot("question-entropy", _entropy_figure(entropy_rows))
            first = response_details[entropy_rows[0]["qid"]]
            response_figure = go.Figure(
                go.Bar(
                    x=first["labels"],
                    y=first["counts"],
                    marker_color="#ffd166",
                )
            )
            response_figure.update_layout(
                title=f"{first['qid']} · {first['question']}",
                yaxis_title="responses",
            )
            add_plot("response-distribution", response_figure)

    prediction_bundle = _prediction_figures(diagnostics)
    prediction_details: dict[str, Any] = {}
    if prediction_bundle is not None:
        scatter, distribution, prediction_details = prediction_bundle
        add_plot("prediction-scatter", scatter)
        add_plot("prediction-distribution", distribution)
    elif (run_path / "run.json").is_file():
        warnings.append(
            "no diagnostics.json prediction payload; regenerate the "
            "cached regression with the current code to enable drill-down"
        )
    elif legacy.detected and legacy.archived_images:
        warnings.append(
            "interactive legacy prediction data could not be reconstructed; "
            "the archived static plots are available below"
        )

    weights = _weights_figure(diagnostics)
    if weights is not None:
        add_plot("weights", weights)
    for name, figure in _trajectory_figures(run_path):
        add_plot(name, figure)
    usage = _usage_figure(run_path)
    if usage is not None:
        add_plot("usage", usage)
    if advanced and survey is not None and responses is not None:
        for name, figure in _matrix_figures(survey, responses):
            add_plot(name, figure)

    if not plots:
        placeholder = go.Figure()
        placeholder.add_annotation(
            text="Artifacts are not available yet. Refresh as the run progresses.",
            showarrow=False,
        )
        placeholder.update_layout(title="Waiting for run artifacts")
        add_plot("waiting", placeholder)

    refresh = ""
    if refresh_seconds is not None:
        if not math.isfinite(refresh_seconds) or refresh_seconds <= 0:
            raise ValueError("refresh_seconds must be positive")
        refresh = (
            f"<meta http-equiv='refresh' content='{refresh_seconds:g}'>"
        )
    warning_html = "".join(
        f"<li>{escape(warning)}</li>"
        for warning in warnings
    )
    legacy_gallery = _legacy_image_gallery(legacy.archived_images)
    sections = "".join(
        f"<section id='{escape(name)}'>{fragment}</section>"
        for name, fragment in plots
    )
    log_tail = ""
    log_path = run_path / "pipeline.log"
    if log_path.is_file():
        lines = log_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()[-80:]
        log_tail = (
            "<details><summary>Pipeline log (last 80 lines)</summary>"
            f"<pre>{escape(chr(10).join(lines))}</pre></details>"
        )

    scripts = []
    if response_details:
        encoded = json.dumps(response_details).replace("</", "<\\/")
        scripts.append(
            f"""
const responseDetails = {encoded};
document.getElementById('p2p-question-entropy').on('plotly_click', event => {{
  const qid = event.points[0].customdata;
  const item = responseDetails[qid];
  Plotly.react('p2p-response-distribution', [{{
    type: 'bar', x: item.labels, y: item.counts,
    marker: {{color: '#ffd166'}}
  }}], {{
    template: 'plotly_dark', paper_bgcolor: '#121927',
    plot_bgcolor: '#121927', font: {{color: '#dce7f7'}},
    margin: {{l: 55, r: 25, t: 58, b: 55}},
    title: {{text: item.qid + ' · ' + item.question}},
    yaxis: {{title: {{text: 'responses'}}}}
  }});
}});
"""
        )
    if prediction_details:
        encoded = json.dumps(prediction_details).replace("</", "<\\/")
        scripts.append(
            f"""
const predictionDetails = {encoded};
document.getElementById('p2p-prediction-scatter').on('plotly_click', event => {{
  const qid = event.points[0].customdata;
  const item = predictionDetails[qid];
  const labels = item.option_order;
  Plotly.react('p2p-prediction-distribution', [
    {{type: 'bar', name: 'human', x: labels,
      y: labels.map(label => item.target[label] || 0),
      marker: {{color: '#4ea1ff'}}}},
    {{type: 'bar', name: 'prediction', x: labels,
      y: labels.map(label => item.prediction[label] || 0),
      marker: {{color: '#71d99c'}}}}
  ], {{
    template: 'plotly_dark', paper_bgcolor: '#121927',
    plot_bgcolor: '#121927', font: {{color: '#dce7f7'}},
    margin: {{l: 55, r: 25, t: 58, b: 55}},
    legend: {{orientation: 'h'}},
    title: {{text: item.qid + ' · ' + item.question}},
    barmode: 'group', yaxis: {{title: {{text: 'probability'}}}}
  }});
}});
"""
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
{refresh}
<title>P2P diagnostics · {escape(run_path.name)}</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui; }}
body {{ margin:0; background:#0b101a; color:#dce7f7; }}
header {{ padding:32px max(24px,5vw) 20px; background:linear-gradient(135deg,#152137,#0b101a); }}
h1 {{ margin:0 0 8px; font-size:clamp(28px,4vw,44px); }}
.muted {{ color:#8ea1bb; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin-top:24px; }}
.card {{ background:#182235; border:1px solid #283851; border-radius:12px; padding:14px; }}
.card span {{ display:block; color:#8ea1bb; font-size:12px; text-transform:uppercase; letter-spacing:.08em; }}
.card strong {{ display:block; margin-top:7px; overflow-wrap:anywhere; }}
main {{ padding:20px max(16px,4vw) 48px; display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }}
section, .panel {{ min-width:0; background:#121927; border:1px solid #25334a; border-radius:14px; overflow:hidden; }}
.wide {{ grid-column:1/-1; }}
.panel {{ padding:20px; }}
.warnings {{ color:#ffd166; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th,td {{ text-align:left; padding:8px; border-bottom:1px solid #25334a; }}
details {{ grid-column:1/-1; background:#121927; border:1px solid #25334a; border-radius:14px; padding:16px; }}
pre {{ overflow:auto; white-space:pre-wrap; color:#aebed3; }}
.legacy-gallery > div {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:16px; margin-top:16px; }}
.legacy-gallery figure {{ margin:0; min-width:0; }}
.legacy-gallery figcaption {{ color:#aebed3; margin-bottom:8px; text-transform:capitalize; }}
.legacy-gallery img {{ display:block; width:100%; height:auto; border-radius:8px; background:#fff; }}
footer {{ padding:0 max(24px,5vw) 32px; color:#71839d; }}
@media (max-width:900px) {{ main {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
  <div class="muted">PROMPTS TO PROXIES · LIVE RUN INSPECTOR</div>
  <h1>{escape(run_path.name)}</h1>
  <div class="muted">{escape(str(run_path))}</div>
  <div class="cards">{_cards(run_path, run, metrics, diagnostics)}</div>
</header>
<main>
  {"<div class='panel wide warnings'><strong>Notes</strong><ul>" + warning_html + "</ul></div>" if warnings else ""}
  {sections}
  {legacy_gallery}
  <div class="panel wide"><h2>Artifacts</h2>{_artifact_table(run_path)}</div>
  {log_tail}
</main>
<footer>Generated from local artifacts. No backend or network request is made.</footer>
<script>{"".join(scripts)}</script>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")
    return DiagnosticDashboard(
        run_dir=run_path,
        output_path=output,
        sections=tuple(name for name, _ in plots)
        + (("legacy-images",) if legacy_gallery else ()),
        warnings=tuple(warnings),
    )


def serve_dashboard(
    run_dir: str | os.PathLike[str],
    *,
    output_path: str | os.PathLike[str] | None = None,
    advanced: bool = False,
    host: str = "127.0.0.1",
    port: int = 8050,
    refresh_seconds: float = 2.0,
) -> None:
    """Serve a dashboard that rebuilds whenever the browser requests it."""

    run_path = Path(run_dir).resolve()
    output = (
        Path(output_path).resolve()
        if output_path is not None
        else (run_path / "diagnostics.html").resolve()
    )

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib hook
            if self.path in {"/", "/diagnostics.html"}:
                build_dashboard(
                    run_path,
                    output_path=output,
                    advanced=advanced,
                    refresh_seconds=refresh_seconds,
                )
                self.path = "/" + output.name
            super().do_GET()

        def log_message(self, format: str, *args: Any) -> None:
            return

    try:
        server = ThreadingHTTPServer(
            (host, port),
            partial(Handler, directory=str(output.parent)),
        )
        print(f"[inspect] serving http://{host}:{port}/")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if "server" in locals():
            server.server_close()


__all__ = [
    "DiagnosticDashboard",
    "build_dashboard",
    "serve_dashboard",
]
