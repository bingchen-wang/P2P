"""Stage 2 without regression: the uniform average of the proxy panel's answers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from p2p.models.base import Model


class SimpleAverage(Model):
    """Give every agent in the design the same weight, 1/n.

    Nothing is fitted and the targets are not used: ``fit`` only reads the
    agent columns. On P2P's one-hot design, where every retained agent answered
    every question (column-drop cleaning), a question's predicted option masses
    are the shares of agents who chose each option, so they sum to one. This is
    the no-regression baseline: the same agents and the same evaluation as a
    fitted model, without the weights.
    """

    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)

    def fit(self, A: pd.DataFrame, y: np.ndarray | None = None) -> "SimpleAverage":
        if not isinstance(A, pd.DataFrame):
            raise TypeError("A must be a DataFrame with one column per agent")
        if A.shape[1] == 0:
            raise ValueError("a simple average needs at least one agent")
        self.feature_names_ = A.columns.tolist()
        self.beta_ = np.full(A.shape[1], 1.0 / A.shape[1])
        self.status_ = "uniform"
        self.objective_value_ = None
        return self


__all__ = ["SimpleAverage"]
