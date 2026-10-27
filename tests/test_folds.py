"""Contracts for cross-validation fold assignment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from p2p.config import Config
from p2p.models.folds import (
    FOLD_RECORD_FILENAME,
    FoldAssignment,
    assign_folds,
    count_question_sizes,
    read_fold_record,
    resolve_fold_assignment,
    write_fold_record,
)
from p2p.stages import fit
from tests.test_stage_fit import make_regression_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGRESS_FIXTURE = PROJECT_ROOT / "config" / "central" / "regress_fixture.yaml"
ATP_W42_FOLDS = (
    PROJECT_ROOT / "tests" / "fixtures" / "commands" / "parity" / "atp_w42_cv_folds.json"
)

TIES = {q: 2 for q in "abcde"}
MIXED = {"q1": 5, "q2": 5, "q3": 4, "q4": 4, "q5": 3, "q6": 3, "q7": 3, "q8": 2}


def _labels(sizes: dict[str, int]) -> list[str]:
    return [question for question, size in sizes.items() for _ in range(size)]


def _reference_hashed(sizes: dict[str, int], n_splits: int, seed: int) -> dict[str, int]:
    """The ``hashed`` rule restated from the design note, independently of the code."""

    order = sorted(
        sizes,
        key=lambda q: (-sizes[q], hashlib.sha256(f"{seed}:{q}".encode()).hexdigest()),
    )
    loads, assignment = [0] * n_splits, {}
    for question in order:
        fold = min(range(n_splits), key=lambda index: (loads[index], index))
        assignment[question] = fold
        loads[fold] += sizes[question]
    return assignment


def _partition(assignment) -> set[frozenset[str]]:
    folds: dict[int, set[str]] = {}
    for question, fold in assignment.items():
        folds.setdefault(fold, set()).add(question)
    return {frozenset(members) for members in folds.values()}


# --- hashed --------------------------------------------------------------------


def test_hashed_golden_assignment() -> None:
    """Pinned values: any change to the rule, or to its default seed, fails here."""

    assert dict(assign_folds(TIES, n_splits=3).assignment) == {
        "a": 1, "b": 1, "c": 0, "d": 0, "e": 2,
    }
    assert dict(assign_folds(MIXED, n_splits=3, seed=101).assignment) == {
        "q1": 1, "q2": 0, "q3": 2, "q4": 2, "q5": 0, "q6": 0, "q7": 1, "q8": 1,
    }


@pytest.mark.parametrize("seed", [101, 102, 7, 2026])
@pytest.mark.parametrize("sizes", [TIES, MIXED], ids=["ties", "mixed"])
def test_hashed_matches_the_rule_as_specified(sizes, seed) -> None:
    result = assign_folds(sizes, n_splits=3, seed=seed)
    assert dict(result.assignment) == _reference_hashed(sizes, 3, seed)
    assert result.method == "hashed"
    assert result.seed == seed


def test_hashed_is_repeatable_for_a_seed_and_varies_across_seeds() -> None:
    first = assign_folds(TIES, n_splits=3, seed=101)
    again = assign_folds(TIES, n_splits=3, seed=101)
    other = assign_folds(TIES, n_splits=3, seed=102)

    assert first.assignment == again.assignment
    assert first.partition_digest == again.partition_digest
    assert _partition(first.assignment) != _partition(other.assignment)


def test_hashed_balances_option_rows() -> None:
    result = assign_folds(MIXED, n_splits=3, seed=101)
    loads = [
        sum(MIXED[q] for q, fold in result.assignment.items() if fold == index)
        for index in range(3)
    ]
    assert max(loads) - min(loads) <= max(MIXED.values())


def test_hashed_ignores_row_and_question_order() -> None:
    labels = _labels(MIXED)
    shuffled = list(np.random.default_rng(0).permutation(labels))
    reversed_questions = dict(reversed(list(MIXED.items())))

    base = resolve_fold_assignment(labels, n_splits=3, seed=101)
    assert resolve_fold_assignment(shuffled, n_splits=3, seed=101).assignment == base.assignment
    assert assign_folds(reversed_questions, n_splits=3, seed=101).assignment == base.assignment


def test_hashed_rejects_a_non_integer_seed() -> None:
    with pytest.raises(ValueError, match="seed must be an integer"):
        assign_folds(TIES, n_splits=3, seed="101")
    with pytest.raises(ValueError, match="seed must be an integer"):
        assign_folds(TIES, n_splits=3, seed=True)


# --- sklearn -------------------------------------------------------------------


def test_sklearn_method_is_the_installed_group_kfold() -> None:
    import sklearn
    from sklearn.model_selection import GroupKFold

    labels = np.asarray(_labels(MIXED))
    expected = {}
    for fold, (_, validation) in enumerate(
        GroupKFold(n_splits=3).split(np.zeros(labels.size), groups=labels)
    ):
        for question in set(labels[validation].tolist()):
            expected[question] = fold

    result = assign_folds(MIXED, n_splits=3, method="sklearn")
    assert dict(result.assignment) == expected
    assert result.seed is None
    assert result.producer["scikit-learn"] == sklearn.__version__
    assert result.producer["numpy"] == np.__version__


# --- records and replay -------------------------------------------------------------


def test_record_round_trip_reproduces_the_folds(tmp_path: Path) -> None:
    labels = _labels(MIXED)
    original = resolve_fold_assignment(labels, n_splits=3, seed=101)
    path = write_fold_record(tmp_path / FOLD_RECORD_FILENAME, original)

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["schema_version"] == 2
    assert record["method"] == "hashed" and record["seed"] == 101
    assert record["question_count"] == len(MIXED)
    assert record["option_row_count"] == sum(MIXED.values())
    assert record["question_sizes"] == dict(sorted(MIXED.items()))
    assert sum(entry["option_rows"] for entry in record["folds"]) == sum(MIXED.values())

    replayed = read_fold_record(path)
    assert replayed.method == "replay"
    assert replayed.origin["method"] == "hashed" and replayed.origin["seed"] == 101
    assert replayed.partition_digest == original.partition_digest
    for (train_a, valid_a), (train_b, valid_b) in zip(
        original.indices(labels), replayed.indices(labels)
    ):
        assert np.array_equal(train_a, train_b) and np.array_equal(valid_a, valid_b)


def test_schema_1_parity_fixture_loads() -> None:
    record = read_fold_record(ATP_W42_FOLDS)
    assert record.n_splits == 5
    assert len(record.assignment) == 109
    assert record.question_sizes is None
    assert record.fold_row_counts is not None and len(record.fold_row_counts) == 5
    assert record.origin["schema_version"] == 1


def _replayable(tmp_path: Path) -> tuple[list[str], Path]:
    labels = _labels(MIXED)
    path = write_fold_record(
        tmp_path / FOLD_RECORD_FILENAME,
        resolve_fold_assignment(labels, n_splits=3, seed=101),
    )
    return labels, path


def test_replay_rejects_missing_and_unexpected_questions(tmp_path: Path) -> None:
    labels, path = _replayable(tmp_path)
    record = read_fold_record(path)

    with pytest.raises(ValueError, match="must cover exactly"):
        record.indices([label for label in labels if label != "q8"])
    with pytest.raises(ValueError, match="must cover exactly"):
        record.indices(labels + ["q9", "q9"])


def test_replay_rejects_a_differently_cleaned_design(tmp_path: Path) -> None:
    labels, path = _replayable(tmp_path)
    record = read_fold_record(path)
    fewer_rows = labels.copy()
    fewer_rows.remove("q1")                         # q1 now has 4 option rows, not 5

    with pytest.raises(ValueError, match="differently cleaned design"):
        record.indices(fewer_rows)


def test_replay_of_schema_1_checks_per_fold_row_totals() -> None:
    record = read_fold_record(ATP_W42_FOLDS)
    labels = [question for question in record.assignment for _ in range(4)]
    with pytest.raises(ValueError, match="differently cleaned design"):
        record.indices(labels)


def _write_record(path: Path, **changes) -> Path:
    base = {
        "schema_version": 2, "n_splits": 2,
        "folds": [{"fold": 0, "questions": ["q1"]}, {"fold": 1, "questions": ["q2"]}],
    }
    base.update(changes)
    path.write_text(json.dumps(base), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"folds": [{"fold": 0, "questions": ["q1"]}, {"fold": 1, "questions": ["q1"]}]},
         "more than once"),
        ({"folds": [{"fold": 0, "questions": ["q1"]}, {"fold": 1, "questions": []}]},
         "nonempty sequence"),
        ({"folds": [{"fold": 0, "questions": ["q1", "q2"]}]}, "exactly 2 entries"),
        ({"schema_version": 3}, "schema_version"),
        ({"partition_digest": "sha256:0"}, "does not match its folds"),
        ({"question_sizes": {"q1": 2}}, "exactly its assigned questions"),
    ],
)
def test_read_fold_record_rejects_malformed_records(tmp_path: Path, changes, match) -> None:
    with pytest.raises(ValueError, match=match):
        read_fold_record(_write_record(tmp_path / "folds.json", **changes))


def test_replay_rejects_a_fold_count_mismatch(tmp_path: Path) -> None:
    labels, path = _replayable(tmp_path)
    with pytest.raises(ValueError, match="3 folds but 2 were requested"):
        resolve_fold_assignment(labels, n_splits=2, assignments=read_fold_record(path))


# --- digest --------------------------------------------------------------------


def test_digest_ignores_fold_numbering_but_not_membership() -> None:
    base = FoldAssignment.from_mapping({"a": 0, "b": 0, "c": 1, "d": 2}, n_splits=3)
    renumbered = FoldAssignment.from_mapping({"a": 2, "b": 2, "c": 0, "d": 1}, n_splits=3)
    moved = FoldAssignment.from_mapping({"a": 0, "b": 1, "c": 1, "d": 2}, n_splits=3)

    assert base.partition_digest == renumbered.partition_digest
    assert base.partition_digest != moved.partition_digest
    assert base.partition_digest.startswith("sha256:")


def test_count_question_sizes() -> None:
    assert count_question_sizes(["b", "a", "b"]) == {"a": 1, "b": 2}
    with pytest.raises(ValueError, match="nonempty strings"):
        count_question_sizes(["a", ""])


# --- fit_model -----------------------------------------------------------------


def _config(**folds) -> dict:
    validation = {"strategy": "cv", "cv_folds": 2}
    if folds:
        validation["folds"] = folds
    return {"method": "lasso", "alpha": (0.0001, 0.001, 0.01, 0.1), "validation": validation}


def test_fit_model_seeds_the_folds_and_returns_them(cvxpy_solver: str) -> None:
    data = make_regression_data()
    sizes = {"Q1": 2, "Q2": 2, "Q3": 2, "Q4": 2}
    # 101 and 103 give different partitions of this fixture (101 and 102 do
    # not), so the test would fail if the seed never reached the folds.
    seeds = (101, 103)
    assert (
        assign_folds(sizes, n_splits=2, seed=seeds[0]).partition_digest
        != assign_folds(sizes, n_splits=2, seed=seeds[1]).partition_digest
    )

    digests = []
    for seed in seeds:
        result = fit.fit_model(data, _config(), solver=cvxpy_solver, seed=seed)
        assert isinstance(result.folds, FoldAssignment)
        assert result.folds.method == "hashed" and result.folds.seed == seed
        expected = assign_folds(sizes, n_splits=2, seed=seed)
        assert result.folds.partition_digest == expected.partition_digest
        assert result.diagnostics["fold_partition_digest"] == expected.partition_digest
        assert result.diagnostics["fold_seed"] == seed
        digests.append(result.folds.partition_digest)
    assert digests[0] != digests[1]


def test_fit_model_generate_then_replay_is_identical(cvxpy_solver: str, tmp_path: Path) -> None:
    data = make_regression_data()
    generated = fit.fit_model(data, _config(), solver=cvxpy_solver, seed=101)
    path = write_fold_record(tmp_path / FOLD_RECORD_FILENAME, generated.folds)

    by_config = fit.fit_model(
        data, _config(method="replay", file=str(path)), solver=cvxpy_solver, seed=999,
    )
    by_argument = fit.fit_model(
        data, _config(), solver=cvxpy_solver, seed=999, fold_assignments=read_fold_record(path),
    )

    for replayed in (by_config, by_argument):
        assert replayed.folds.method == "replay"
        assert replayed.folds.partition_digest == generated.folds.partition_digest
        assert replayed.best_alpha == generated.best_alpha
        assert np.array_equal(replayed.model.beta_, generated.model.beta_)


@pytest.mark.parametrize(
    ("folds", "match"),
    [
        ({"metod": "hashed"}, "did you mean 'method'"),
        ({"method": "random"}, "must be one of"),
        ({"method": "replay"}, "needs model.validation.folds.file"),
        ({"method": "hashed", "file": "folds.json"}, "only used with method: replay"),
    ],
)
def test_fit_model_validates_the_folds_block(cvxpy_solver: str, folds, match) -> None:
    with pytest.raises(ValueError, match=match):
        fit.fit_model(make_regression_data(), _config(**folds), solver=cvxpy_solver, seed=101)


def test_fit_model_rejects_folds_given_twice(cvxpy_solver: str, tmp_path: Path) -> None:
    data = make_regression_data()
    generated = fit.fit_model(data, _config(), solver=cvxpy_solver, seed=101)
    path = write_fold_record(tmp_path / FOLD_RECORD_FILENAME, generated.folds)

    with pytest.raises(ValueError, match="given twice"):
        fit.fit_model(
            data, _config(method="replay", file=str(path)), solver=cvxpy_solver, seed=101,
            fold_assignments=generated.folds,
        )


def test_fit_model_rejects_a_record_with_the_wrong_fold_count(
    cvxpy_solver: str, tmp_path: Path,
) -> None:
    record = FoldAssignment.from_mapping({"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 2}, n_splits=3)
    path = write_fold_record(tmp_path / FOLD_RECORD_FILENAME, record)

    with pytest.raises(ValueError, match="3 folds but model.validation.cv_folds is 2"):
        fit.fit_model(
            make_regression_data(), _config(method="replay", file=str(path)),
            solver=cvxpy_solver, seed=101,
        )


# --- config and command -------------------------------------------------------------


def _fixture_config(tmp_path: Path, folds: dict | None = None) -> Config:
    data = yaml.safe_load(REGRESS_FIXTURE.read_text(encoding="utf-8"))
    data["model"]["alpha"] = [0.001, 0.01, 0.1]
    if folds is not None:
        data["model"]["validation"]["folds"] = folds
    return Config.from_mapping(data, declared_in=REGRESS_FIXTURE.resolve())


def test_config_resolves_the_replay_file_like_any_other_path(tmp_path: Path) -> None:
    from p2p.config import ConfigValidationError

    record = write_fold_record(
        tmp_path / FOLD_RECORD_FILENAME,
        FoldAssignment.from_mapping({"q1": 0, "q2": 1}, n_splits=2),
    )
    config = _fixture_config(tmp_path, {"method": "replay", "file": str(record)})
    assert config.data["model"]["validation"]["folds"]["file"] == str(record.resolve())

    with pytest.raises(ConfigValidationError, match="does not exist"):
        _fixture_config(tmp_path, {"method": "replay", "file": str(tmp_path / "absent.json")})


def test_regress_writes_folds_json_and_replays_it(tmp_path: Path, cvxpy_solver: str) -> None:
    from p2p.commands import regress

    first = regress.run(_fixture_config(tmp_path), solver=cvxpy_solver, run_dir=tmp_path / "first")
    folds_json = first.report.folds_json
    assert folds_json is not None and folds_json.name == FOLD_RECORD_FILENAME

    record = json.loads(folds_json.read_text(encoding="utf-8"))
    diagnostics = json.loads(first.report.diagnostics_json.read_text(encoding="utf-8"))
    assert record["method"] == "hashed" and record["seed"] == 101
    assert diagnostics["model"]["folds"] == {
        "method": "hashed",
        "seed": 101,
        "n_splits": record["n_splits"],
        "partition_digest": record["partition_digest"],
        "record": FOLD_RECORD_FILENAME,
    }

    second = regress.run(
        _fixture_config(tmp_path, {"method": "replay", "file": str(folds_json)}),
        solver=cvxpy_solver,
        run_dir=tmp_path / "second",
    )
    assert second.fit.folds.partition_digest == record["partition_digest"]
    assert second.fit.best_alpha == first.fit.best_alpha
    assert np.array_equal(second.fit.model.beta_, first.fit.model.beta_)
    replayed = json.loads(second.report.folds_json.read_text(encoding="utf-8"))
    assert replayed["method"] == "replay"
    assert replayed["source"] == str(folds_json.resolve())
    assert replayed["origin"]["method"] == "hashed"
