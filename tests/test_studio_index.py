"""The Studio run index: persistent, incremental, compact, with a fast first page.

A workspace can hold tens of thousands of runs. Summaries are cached under
outputs/.studio/run_index.json.gz, keyed by the timestamps and sizes of the
files they are read from, so only new or changed runs are read again.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

import p2p.studio as studio
from p2p.studio import StudioState


def _write_run(root: Path, name: str, *, mse: float, mtime: float | None = None) -> Path:
    run_dir = root / "outputs" / "batch" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps({"run_id": name, "seed": 1}), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({"metrics": [{
        "metric": "mse", "value": mse, "split": "test", "qids_requested": 2, "qids_used": 2,
        "diagnostics": {"negative_entry_count": 0},
    }]}), encoding="utf-8")
    if mtime is not None:
        for path in run_dir.iterdir():
            os.utime(path, (mtime, mtime))
    return run_dir


@pytest.fixture()
def counted(monkeypatch):
    calls: list[str] = []
    original = studio._run_summary

    def counting(run_file, root):
        calls.append(run_file.parent.name)
        return original(run_file, root)

    monkeypatch.setattr(studio, "_run_summary", counting)
    return calls


def test_index_persists_and_reads_only_new_or_changed_runs(tmp_path: Path, counted) -> None:
    for index in range(5):
        _write_run(tmp_path, f"run_{index}", mse=index / 10)

    first = StudioState(tmp_path)._run_summaries()
    assert len(first) == 5 and len(counted) == 5
    assert (tmp_path / "outputs" / ".studio" / "run_index.json.gz").is_file()

    counted.clear()
    assert len(StudioState(tmp_path)._run_summaries()) == 5
    assert counted == []  # served entirely from the stored index

    _write_run(tmp_path, "run_new", mse=0.9)
    time.sleep(0.01)
    _write_run(tmp_path, "run_2", mse=0.05)  # rewritten: new size and time
    (tmp_path / "outputs" / "batch" / "run_4" / "run.json").unlink()
    counted.clear()
    runs = {item["runId"]: item for item in StudioState(tmp_path)._run_summaries()}

    assert sorted(counted) == ["run_2", "run_new"]
    assert set(runs) == {"run_0", "run_1", "run_2", "run_3", "run_new"}
    assert runs["run_2"]["metrics"][0]["value"] == 0.05


def test_sorting_and_search_use_the_index(tmp_path: Path) -> None:
    for index, mse in enumerate((0.3, 0.1, 0.2)):
        _write_run(tmp_path, f"run_{index}", mse=mse)
    state = StudioState(tmp_path)

    best = state.list_runs(sort="mse")
    found = state.list_runs(query="run_2")

    assert [item["runId"] for item in best["items"]] == ["run_1", "run_2", "run_0"]
    assert [item["runId"] for item in found["items"]] == ["run_2"]
    assert "indexing" not in best


def test_first_newest_page_does_not_wait_for_the_index(tmp_path: Path, counted) -> None:
    now = time.time()
    for index in range(6):
        _write_run(tmp_path, f"run_{index}", mse=index, mtime=now - 100 + index)
    state = StudioState(tmp_path)

    page = state.list_runs(limit=2)

    assert page["indexing"] is True
    assert page["total"] == 6
    assert [item["runId"] for item in page["items"]] == ["run_5", "run_4"]
    assert sorted(counted) == ["run_4", "run_5"]  # only the page was read


def test_index_keeps_what_the_library_shows_and_drops_detail(tmp_path: Path) -> None:
    summary = {
        "runId": "r", "path": "outputs/r", "modifiedAt": "2026-10-04T00:00:00+00:00",
        "metrics": [{"metric": "mse", "value": 0.1, "split": "test", "diagnostics": {"x": 1}}],
        "population": {
            "generated": 10, "selected": 4, "antiTypes": 1, "sourcePath": "outputs/r",
            "modes": [
                {"mode": "core", "family": "Core", "generated": 6, "selected": 3, "antiTypes": 1},
                {"mode": "core+a", "family": "Thematic", "generated": 3, "selected": 1, "antiTypes": 0},
                {"mode": "core+b", "family": "Thematic", "generated": 1, "selected": 0, "antiTypes": 0},
            ],
        },
    }

    compact = studio._library_summary(summary)

    assert compact["metrics"] == [{"metric": "mse", "value": 0.1, "split": "test"}]
    assert {mode["family"]: mode["generated"] for mode in compact["population"]["modes"]} == {
        "Core": 6, "Thematic": 4,
    }
    assert compact["population"]["selected"] == 4
    assert summary["metrics"][0]["diagnostics"] == {"x": 1}  # the source is not modified


def test_workspace_counts_manifests_without_parsing_them(tmp_path: Path, monkeypatch) -> None:
    experiments = tmp_path / "config" / "experiments" / "batch"
    experiments.mkdir(parents=True)
    for index in range(60):
        (experiments / f"run_{index}.yaml").write_text("run:\n  name: x\n", encoding="utf-8")
    parsed: list[Path] = []
    original = studio._manifest_summary
    monkeypatch.setattr(studio, "_manifest_summary", lambda path, root: (parsed.append(path), original(path, root))[1])

    state = StudioState(tmp_path)
    assert state.workspace()["counts"]["manifests"] == 60
    assert parsed == []
    assert len(state.bootstrap()["manifests"]) == 40
