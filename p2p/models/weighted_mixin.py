import numpy as np
import pandas as pd
import cvxpy as cp

class WeightedConstrainedModelMixin:
    """
    Mixin class that adds per-row weights for the squared loss.
    Must be used together with ConstrainedModel subclasses.

    Provides:
        - _row_weights_from_question_weights()
        - _weighted_loss_term()
    """

    def __init__(self, question_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.question_weights = question_weights or {}

    def _row_weights_from_question_weights(self, A: pd.DataFrame):
        """
        Convert question-level weights into row-level weights.
        Ignores binary_ids that are not present in A.index,
        exactly like the constraint constructor does.
        """
        n = A.shape[0]
        row_weights = np.ones(n)

        if not self.question_weights:
            return row_weights

        index = A.index

        for orig_qid, weight in self.question_weights.items():
            if orig_qid not in self.constraint_groups:
                continue

            for binary_id in self.constraint_groups[orig_qid]:

                # === NEW FIX: skip missing rows, never KeyError ===
                if binary_id not in index:
                    continue

                idx = index.get_loc(binary_id)
                row_weights[idx] = weight

        return row_weights

    def _weighted_loss_term(self, X, y, beta, A, intercept=None):
        """
        Construct the weighted residual sum of squares term.
        """
        predictions = X @ beta
        residuals = (predictions if intercept is None else predictions + intercept) - y
        row_weights = self._row_weights_from_question_weights(A)
        return cp.sum(cp.multiply(row_weights, cp.square(residuals)))