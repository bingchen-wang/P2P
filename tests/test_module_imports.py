"""Import-smoke coverage for the survey, response, model and generator modules."""

from __future__ import annotations

import importlib

import pytest


CORE_MODULES = [
    "p2p.survey",
    "p2p.survey.converter",
    "p2p.responses",
    "p2p.responses.cleaner",
    "p2p.responses.converter",
    "p2p.endowments",
    "p2p.endowments.manager",
    "p2p.models",
    "p2p.models.base",
    "p2p.models.constrained_model",
    "p2p.models.constrained_lasso",
    "p2p.models.constrained_elasticnet",
    "p2p.models.cross_validation",
    "p2p.models.constrained_cv",
    "p2p.models.weighted_mixin",
    "p2p.models.weighted_constrained_lasso",
    "p2p.models.weighted_constrained_elasticnet",
    "p2p.models.functions",
    "p2p.models.plotting",
    "p2p.generators",
    "p2p.generators.utils",
    "p2p.generators.theme_variability_tracker",
    "p2p.generators.active_endowment_generator",
    "p2p.generators.vanilla_endowment_generator",
]


@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_module_imports_without_project_cwd(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """These modules must not depend on the process working directory."""
    monkeypatch.chdir(tmp_path)

    imported = importlib.import_module(module_name)

    assert imported.__name__ == module_name