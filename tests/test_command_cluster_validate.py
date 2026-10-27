"""Offline cluster and verb-validation command contracts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from p2p.commands import cluster, validate
from p2p.config import Config, ConfigValidationError


def _write_endowments(path: Path) -> Path:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "eid",
                "endow_text",
                "role",
                "weight",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "eid": "z_farm",
                    "endow_text": (
                        "farmer crops fields rural harvest"
                    ),
                    "role": "proxy",
                    "weight": "",
                },
                {
                    "eid": "a_code",
                    "endow_text": (
                        "software developer code computers"
                    ),
                    "role": "proxy",
                    "weight": "",
                },
                {
                    "eid": "m_rural",
                    "endow_text": (
                        "rural agriculture crops farm"
                    ),
                    "role": "proxy",
                    "weight": "",
                },
                {
                    "eid": "b_tech",
                    "endow_text": (
                        "technology programmer software code"
                    ),
                    "role": "proxy",
                    "weight": "",
                },
            ]
        )
    return path


def _cluster_config(
    tmp_path: Path,
    *,
    n_clusters: int = 2,
) -> Config:
    endowments_path = _write_endowments(
        tmp_path / "endowments.csv"
    )
    return Config.from_mapping(
        {
            "run": {
                "name": "offline-cluster",
                "seed": 23,
            },
            "data": {
                "basis": {
                    "endowments": endowments_path,
                },
            },
            "cluster": {
                "algorithm": "tfidf_kmeans",
                "n_clusters": n_clusters,
            },
        },
        declared_in=tmp_path / "cluster.yaml",
    )


def test_cluster_is_deterministic_and_writes_named_artifacts(
    tmp_path: Path,
) -> None:
    config = _cluster_config(tmp_path)

    first = cluster.run(
        config,
        run_dir=tmp_path / "first",
    )
    second = cluster.run(
        config,
        run_dir=tmp_path / "second",
    )

    assert first.assignments == second.assignments
    assert [
        assignment.eid
        for assignment in first.assignments
    ] == [
        "a_code",
        "b_tech",
        "m_rural",
        "z_farm",
    ]
    assert {
        assignment.cluster
        for assignment in first.assignments
    } == {
        0,
        1,
    }

    with first.assignments_path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))
    assert rows == [
        {
            "eid": assignment.eid,
            "cluster": str(assignment.cluster),
        }
        for assignment in first.assignments
    ]

    metadata = json.loads(
        first.metadata_path.read_text(
            encoding="utf-8"
        )
    )
    assert metadata == {
        "algorithm": "tfidf_kmeans",
        "assignments": "clusters.csv",
        "cluster_sizes": {
            "0": 2,
            "1": 2,
        },
        "config_digest": config.digest,
        "n_clusters": 2,
        "n_endowments": 4,
        "schema_version": 1,
        "seed": 23,
        "source": {
            "path": str(
                (
                    tmp_path / "endowments.csv"
                ).resolve()
            ),
            "sha256": metadata["source"]["sha256"],
        },
    }
    assert len(metadata["source"]["sha256"]) == 64
    assert first.assignments_path.name == "clusters.csv"
    assert first.metadata_path.name == "clustering.json"
    assert first.config_path.name == "config.resolved.yaml"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda manifest: manifest.pop("cluster"),
            "cluster",
        ),
        (
            lambda manifest: manifest["cluster"].update(
                {"n_clusters": 0}
            ),
            "n_clusters",
        ),
        (
            lambda manifest: manifest["cluster"].update(
                {"n_clusters": 5}
            ),
            "cannot exceed",
        ),
        (
            lambda manifest: manifest["cluster"].update(
                {"algorithm": "remote_embeddings"}
            ),
            "algorithm",
        ),
    ],
)
def test_cluster_rejects_invalid_local_config(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    endowments_path = _write_endowments(
        tmp_path / "endowments.csv"
    )
    manifest = {
        "data": {
            "basis": {
                "endowments": endowments_path,
            },
        },
        "cluster": {
            "n_clusters": 2,
        },
    }
    mutate(manifest)
    config = Config.from_mapping(
        manifest,
        declared_in=tmp_path / "cluster.yaml",
    )

    with pytest.raises(
        ConfigValidationError,
        match=message,
    ):
        cluster.run(
            config,
            run_dir=tmp_path / "run",
        )


def test_cluster_rejects_duplicate_endowment_ids(
    tmp_path: Path,
) -> None:
    config = _cluster_config(tmp_path)
    path = tmp_path / "endowments.csv"
    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "a_code",
                "duplicate",
                "proxy",
                "",
            ]
        )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        cluster.run(
            config,
            run_dir=tmp_path / "run",
        )


def test_validate_is_a_thin_verb_scoped_wrapper(
    tmp_path: Path,
) -> None:
    config = Config.from_mapping(
        {
            "simulation": {
                "kind": "fraction",
                "invokes_generation": False,
            },
        },
        declared_in=tmp_path / "simulate.yaml",
    )

    result = validate.run(
        config,
        target_verb="simulate",
        verbose=True,
    )

    assert result == validate.ValidationResult(
        target_verb="simulate",
        config_digest=config.digest,
        source_path=config.source_path,
    )


def test_validate_propagates_verb_validation_errors(
    tmp_path: Path,
) -> None:
    config = Config.from_mapping(
        {},
        declared_in=tmp_path / "invalid.yaml",
    )

    with pytest.raises(
        ConfigValidationError,
        match="simulation",
    ):
        validate.run(
            config,
            target_verb="simulate",
        )
