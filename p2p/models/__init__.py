"""Constrained regression models and fitting helpers."""

from p2p.models.base import Model
from p2p.models.constrained_cv import ConstrainedModelCV
from p2p.models.constrained_elasticnet import ConstrainedElasticNet
from p2p.models.constrained_lasso import ConstrainedLasso
from p2p.models.constrained_model import (
    DEFAULT_SOLVER,
    DEFAULT_SOLVER_FALLBACKS,
    ConstrainedModel,
)
from p2p.models.cross_validation import ModelCV
from p2p.models.functions import fit_elastic_net_model, fit_lasso_model
from p2p.models.simple_average import SimpleAverage
from p2p.models.weighted_constrained_elasticnet import (
    WeightedConstrainedElasticNet,
)
from p2p.models.weighted_constrained_lasso import WeightedConstrainedLasso
from p2p.models.weighted_mixin import WeightedConstrainedModelMixin
from p2p.registry import MODELS


def register_builtin_models() -> None:
    """Register built-in Stage-2 model factories idempotently."""

    factories = {
        "elasticnet": ConstrainedElasticNet,
        "lasso": ConstrainedLasso,
        "simple_average": SimpleAverage,
        "weighted-elasticnet": WeightedConstrainedElasticNet,
        "weighted-lasso": WeightedConstrainedLasso,
    }

    registered = set(MODELS.names())
    for name, factory in factories.items():
        if name not in registered:
            MODELS.register(name, factory)


register_builtin_models()


__all__ = [
    "DEFAULT_SOLVER",
    "DEFAULT_SOLVER_FALLBACKS",
    "Model",
    "ConstrainedModel",
    "ConstrainedLasso",
    "ModelCV",
    "ConstrainedModelCV",
    "ConstrainedElasticNet",
    "SimpleAverage",
    "WeightedConstrainedLasso",
    "WeightedConstrainedElasticNet",
    "WeightedConstrainedModelMixin",
    "fit_lasso_model",
    "fit_elastic_net_model",
    "register_builtin_models",
]