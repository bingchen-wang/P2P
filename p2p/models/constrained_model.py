from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd

from p2p.models.base import Model


# P2P's default solver. Clarabel is an interior-point solver whose solutions do
# not depend on how the problem is written down; OSQP's first-order solutions
# can be loose enough to move the selected penalty. To reproduce fits made with
# OSQP, pass solver="OSQP" explicitly.
DEFAULT_SOLVER = "CLARABEL"

# Tried in order after a genuine failure of the configured solver, skipping it.
DEFAULT_SOLVER_FALLBACKS = ("CLARABEL", "OSQP", "SCS")


class ConstrainedModel(Model):
    def __init__(
        self,
        constraint_groups: Optional[dict] = None,
        group_total: float = 1.0,
        nonneg: bool = False,
        sum_to_one: bool = False,
        verbose: bool = False,
        soft_group_sum: bool = True,
        group_penalty_strength: float = 100.0,  # Large enough to approximately enforce
        solver_fallbacks: tuple[str, ...] = DEFAULT_SOLVER_FALLBACKS,
        intercept: bool = False,
    ):
        super().__init__(verbose=verbose)
        self.constraint_groups = constraint_groups
        self.group_total = group_total
        self.nonneg = nonneg
        self.sum_to_one = sum_to_one
        self.soft_group_sum = soft_group_sum
        self.group_penalty_strength = group_penalty_strength
        self.solver_fallbacks = tuple(solver_fallbacks or ())
        # An unpenalised constant added to every option row (scikit-learn's
        # fit_intercept). Unlike the reference respondent, which adds 1/k to each
        # option of a k-option question, it adds k * intercept to that question's
        # mass. Off by default.
        self.intercept = bool(intercept)

    def _intercept_variable(self) -> Optional[cp.Variable]:
        """The intercept's optimisation variable, or None when it is off."""
        return cp.Variable() if self.intercept else None

    @staticmethod
    def _linear_predictor(X: np.ndarray, beta: cp.Variable, intercept: Optional[cp.Variable]):
        """Predictions as a CVXPY expression: X @ beta, plus the intercept if any."""
        predictions = X @ beta
        return predictions if intercept is None else predictions + intercept

    def _store_intercept(self, intercept: Optional[cp.Variable]) -> None:
        self.intercept_ = 0.0 if intercept is None else float(intercept.value)

    def _convert_index_groups_to_numeric(self, A: pd.DataFrame, drop_empty=True) -> dict[str, list[int]]:
        # Constraint_groups might contain information for the whole survey (but we need only information on the relevant questions)
        row_index_map = {label: i for i, label in enumerate(A.index)}
        result = {}
        for group_name, members in self.constraint_groups.items():
            idx_list = [row_index_map[label] for label in members if label in row_index_map]
            if not idx_list and not drop_empty:
                raise ValueError(f"Group '{group_name}' has no members in A.index")
            if idx_list:
                result[group_name] = idx_list
        return result

    def _build_constraints(
        self,
        X: np.ndarray,
        A: pd.DataFrame,
        beta: cp.Variable,
        intercept: Optional[cp.Variable] = None,
    ) -> tuple[list[cp.Constraint], cp.Expression]:
        """
        Returns:
            constraints (list): Hard constraints (e.g., nonnegativity, sum-to-one).
            penalty_term (cp.Expression): Group soft penalty term to be added to loss.

        A group's mass is the sum of its predictions, so with an intercept it
        includes len(group) * intercept. Sum-to-one concerns the weights only.
        """
        constraints = []
        penalty_term = 0

        if self.constraint_groups:
            G_idx = self._convert_index_groups_to_numeric(A)

            for group_name, g in G_idx.items():
                group_sum_expr = cp.sum(X[g] @ beta)
                if intercept is not None:
                    group_sum_expr = group_sum_expr + len(g) * intercept
                if self.soft_group_sum:
                    penalty_term += cp.square(group_sum_expr - self.group_total)
                else:
                    constraints.append(group_sum_expr == self.group_total)

        if self.nonneg:
            constraints.append(beta >= 0)
        if self.sum_to_one:
            constraints.append(cp.sum(beta) == 1)

        return constraints, penalty_term * self.group_penalty_strength
    

    def solve_with_fallback(self, problem: cp.Problem, beta: cp.Variable, label: str):
        """Solve, falling back to another solver if the configured one fails.

        A first-order solver can fail on an individual cross-validation fold
        while succeeding on every neighbouring one -- the stiff group-mass
        penalty scales with the number of questions, and some folds land badly
        conditioned. Without a fallback, one such fold aborts an entire run.

        Falling back only happens after a genuine failure, so a solve that
        already succeeds is untouched. ``solver_used_`` records what answered.
        """

        attempts = [self.solver] + [
            name
            for name in self.solver_fallbacks
            if name != self.solver and name in cp.installed_solvers()
        ]
        errors = []
        for name in attempts:
            kwargs = {"verbose": self.verbose}
            if name is not None:
                kwargs["solver"] = name
            try:
                problem.solve(**kwargs)
            except Exception as error:  # noqa: BLE001 - solver-specific types
                errors.append(f"{name}: {type(error).__name__}")
                continue
            if beta.value is not None:
                self.solver_used_ = name
                self.solver_fallback_ = name != attempts[0]
                return
            errors.append(f"{name}: status {problem.status!r}")

        raise RuntimeError(
            f"{label} solve failed with every solver tried ({'; '.join(errors)})"
        )

    def compute_group_constraint_violations(self, A: pd.DataFrame = None) -> dict[str, float]:
        """
        Compute the absolute ``sum(A[g] @ beta) - group_total`` per group.

        Returns:
            Dict of group name -> absolute constraint violation.
        """
        if A is None:
            A = self.A_
        if self.constraint_groups is None:
            raise ValueError("constraint_groups must be defined.")
        if not hasattr(self, "beta_"):
            raise ValueError("Model must be fit before computing constraint violations.")

        X = self.safe_df_to_numpy(A)
        beta = self.beta_

        violations = {}
        G_idx = self._convert_index_groups_to_numeric(A)  # drop_empty=True by default

        intercept = getattr(self, "intercept_", 0.0)
        for group_name, row_indices in G_idx.items():
            group_sum = np.sum(X[row_indices] @ beta) + len(row_indices) * intercept
            violations[group_name] = abs(group_sum - self.group_total)

        return violations
    
    @property
    def mean_group_constraint_violation(self):
        v = self.compute_group_constraint_violations()
        return np.mean(list(v.values()))
