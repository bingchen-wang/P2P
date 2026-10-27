from typing import Sequence, Union, Tuple, Type, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from p2p.models.base import Model
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class ModelCV:
    def __init__(
        self,
        model_cls: Type[Model],
        alphas: Sequence[float],
        l1_ratios: Optional[Sequence[float]] = None,
        k: int = 5,
        model_kwargs: dict = {},
        random_state: int = 101,
        verbose: bool = True,
    ):
        """
        Cross-validation interface for constrained models.

        Args:
            model_cls: Subclass of Model to be cross-validated.
            alphas: List of alpha values.
            l1_ratios: List of l1_ratios (optional, for ElasticNet-style models).
            k: Number of folds.
            model_kwargs: Additional arguments to pass to model.
            random_state: Random seed for reproducibility.
        """
        self.model_cls = model_cls
        self.alphas = alphas
        self.l1_ratios = l1_ratios
        self.k = k
        self.model_kwargs = model_kwargs
        self.random_state = random_state
        self.verbose = verbose

        self.cv_mean_mse_ = {}
        self.cv_std_mse_ = {}
        self.diagnostics_ = None
        self.best_params_ = None
        self.best_score_ = float("inf")

    def fit(self, A: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        self.A_ = A
        self.y_ = y

        y = y.values if isinstance(y, pd.Series) else y
        kf = KFold(n_splits=self.k, shuffle=True, random_state=self.random_state)

        total_configs = len(self.alphas) * (len(self.l1_ratios) if self.l1_ratios else 1)
        config_count = 0

        for alpha in self.alphas:
            ratios = self.l1_ratios if self.l1_ratios is not None else [None]
            for l1_ratio in ratios:
                config_count += 1
                if self.verbose:
                    logger.info(f"Evaluating config {config_count}/{total_configs}: alpha={alpha}, l1_ratio={l1_ratio}")
                errors = []
                for train_idx, val_idx in kf.split(A):
                    A_train, A_val = A.iloc[train_idx], A.iloc[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]

                    kwargs = self.model_kwargs.copy()
                    kwargs.update({"alpha": alpha})
                    if l1_ratio is not None:
                        kwargs.update({"l1_ratio": l1_ratio})

                    model = self.model_cls(**kwargs)
                    model.fit(A_train, y_train)
                    error = model.score(A_val, y_val)
                    errors.append(error)

                avg_error = np.mean(errors)
                std_error = np.std(errors)
                self.cv_mean_mse_[(alpha, l1_ratio)] = avg_error
                self.cv_std_mse_[(alpha, l1_ratio)] = std_error

                if self.verbose:
                    logger.info(f"Mean MSE: {avg_error:.4f} ± {std_error:.4f}")

                if avg_error < self.best_score_:
                    self.best_score_ = avg_error
                    self.best_params_ = (alpha, l1_ratio)

        if self.l1_ratios is None:
            mean_mse_arr = np.array([self.cv_mean_mse_[(a, None)] for a in self.alphas])
            std_mse_arr = np.array([self.cv_std_mse_[(a, None)] for a in self.alphas])
        else:
            mean_mse_arr = np.array([
                [self.cv_mean_mse_[(a, l1)] for a in self.alphas] for l1 in self.l1_ratios
            ])
            std_mse_arr = np.array([
                [self.cv_std_mse_[(a, l1)] for a in self.alphas] for l1 in self.l1_ratios
            ])            
        
        self.diagnostics_ = {
            "alphas": self.alphas,
            "l1_ratios": self.l1_ratios,
            "mean_mse": mean_mse_arr,
            "std_mse": std_mse_arr,
            "best_alpha": self.best_params_[0],
            "best_l1_ratio": self.best_params_[1],
        }

        return self

    def get_best_model(self, A: pd.DataFrame = None, y: Union[pd.Series, np.ndarray] = None) -> Model:
        """Refit the best model on full data using best cross-validated parameters."""

        if A is None or y is None:
            A = self.A_
            y = self.y_
        alpha, l1_ratio = self.best_params_
        kwargs = self.model_kwargs.copy()
        kwargs.update({"alpha": alpha})
        if l1_ratio is not None:
            kwargs.update({"l1_ratio": l1_ratio})

        model = self.model_cls(**kwargs)
        model.fit(A, y)
        return model

    def results_as_dataframe(self) -> pd.DataFrame:
        """Return CV results as a sorted DataFrame."""
        rows = [
            {
                "alpha": k[0],
                "l1_ratio": k[1],
                "mean_error": self.cv_mean_mse_[k],
                "std_error": self.cv_std_mse_[k],
            }
            for k in self.cv_mean_mse_.keys()
        ]
        return pd.DataFrame(rows).sort_values("mean_error")
    
    def score(self, A: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> float:
        """Return MSE of best refitted model on provided data."""
        return self.get_best_model(A, y).score(A, y)
    
    @property
    def is_fitted(self) -> bool:
        return self.best_params_ is not None
    
    def plot_diagnostics(self, style="2D"):
        from p2p.models.plotting import (
            plot_elasticnet_diagnostics,
            plot_lasso_diagnostics,
        )
        if self.l1_ratios is None:
            return plot_lasso_diagnostics(self.diagnostics_, self.best_params_[0])
        return plot_elasticnet_diagnostics(self.diagnostics_, *self.best_params_, style=style)
