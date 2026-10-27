"""Constrained Lasso model."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from p2p.models.constrained_model import ConstrainedModel


class ConstrainedLasso(ConstrainedModel):
    """Lasso regression with optional constraints and explicit solver choice."""

    def __init__(
        self,
        alpha: float = 1.0,
        zero_threshold: float = 1e-5,
        solver: str | None = None,
        **kwargs,
    ):
        """
        Initialize constrained Lasso regression.

        Args:
            alpha: L1 regularization strength.
            zero_threshold: Coefficients below this absolute value are set to
                zero after fitting.
            solver: Optional CVXPY solver name. When omitted, CVXPY selects its
                default solver.
            **kwargs: Arguments passed to ``ConstrainedModel``, including
                constraint groups, nonnegativity, sum-to-one, and verbosity.
        """
        super().__init__(**kwargs)
        if solver is not None and (
            not isinstance(solver, str) or not solver.strip()
        ):
            raise ValueError("solver must be a nonempty string or None")

        self.alpha = alpha
        self.zero_threshold = zero_threshold
        self.solver = solver

    def fit(self, A: pd.DataFrame, y: np.ndarray):
        X = self.safe_df_to_numpy(A)
        n, p = X.shape
        beta = cp.Variable(p)
        intercept = self._intercept_variable()

        loss = cp.sum_squares(self._linear_predictor(X, beta, intercept) - y) / (2 * n)
        penalty_l1 = self.alpha * cp.norm1(beta)  # the intercept is not penalised

        constraints, penalty_soft = self._build_constraints(X, A, beta, intercept)

        objective = cp.Minimize(loss + penalty_l1 + penalty_soft)
        problem = cp.Problem(objective, constraints)

        self.solve_with_fallback(problem, beta, "constrained Lasso")
        self._store_intercept(intercept)

        self.beta_raw_ = beta.value
        self.beta_ = beta.value.copy()
        self.beta_[np.abs(self.beta_) < self.zero_threshold] = 0.0
        self.status_ = problem.status
        self.objective_value_ = problem.value
        self.feature_names_ = A.columns.tolist()

        self.X_ = X
        self.A_ = A
        self.y_ = y
        self.n_obs_, self.n_features_ = X.shape

        self.log_fit_summary()
        return self

    def count_nonzero_weights(self) -> int:
        return int(np.sum(self.beta_ != 0))

    def get_support(self) -> np.ndarray:
        return self.beta_ != 0