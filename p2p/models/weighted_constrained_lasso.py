import cvxpy as cp
import numpy as np
import pandas as pd

from p2p.models.constrained_lasso import ConstrainedLasso
from p2p.models.weighted_mixin import WeightedConstrainedModelMixin


class WeightedConstrainedLasso(WeightedConstrainedModelMixin, ConstrainedLasso):
    """
    Weighted Constrained Lasso.
    Overrides only the loss term. Constraints, penalties,
    coefficient thresholding, and metadata storage unchanged.
    """

    def fit(self, A: pd.DataFrame, y: np.ndarray):
        X = self.safe_df_to_numpy(A)
        n, p = X.shape
        beta = cp.Variable(p)
        intercept = self._intercept_variable()

        # Weighted loss instead of unweighted loss
        weighted_loss = self._weighted_loss_term(X, y, beta, A, intercept) / (2 * n)

        penalty_l1 = self.alpha * cp.norm1(beta)  # the intercept is not penalised

        constraints, penalty_soft = self._build_constraints(X, A, beta, intercept)

        objective = cp.Minimize(weighted_loss + penalty_l1 + penalty_soft)
        problem = cp.Problem(objective, constraints)
        # The configured solver, with fallback and a record of what answered,
        # exactly as the unweighted models.
        self.solve_with_fallback(problem, beta, "weighted constrained Lasso")
        self._store_intercept(intercept)

        # Store outputs exactly like ConstrainedLasso
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