import numpy as np
import cvxpy as cp
import pandas as pd
from typing import Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Default level, can be overridden

# Optional: add handler if not already configured by user
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class Model(ABC):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.beta_ = None
        self.status_ = None
        self.feature_names_ = None
        self.objective_value_ = None
        # A constant added to every prediction; 0.0 unless a model fits one.
        self.intercept_ = 0.0

    @abstractmethod
    def fit(self, A: pd.DataFrame, y: np.ndarray):
        self.feature_names_ = A.columns.tolist()
        pass

    def predict(self, A: pd.DataFrame) -> np.ndarray:
        if self.beta_ is None:
            raise ValueError("Model has not been fit yet.")
        predictions = Model.safe_df_to_numpy(A) @ self.beta_
        intercept = getattr(self, "intercept_", 0.0)
        return predictions + intercept if intercept else predictions

    def score(self, A: pd.DataFrame, y: np.ndarray) -> float:
        """Return mean squared error."""
        y_pred = self.predict(A)
        return np.mean((y - y_pred) ** 2)
    
    def r2(self, A: pd.DataFrame, y: np.ndarray) -> float:
        y_pred = self.predict(A)
        return 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
    
    def get_params(self) -> dict:
        return {"verbose": self.verbose}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
    
    @property
    def coef_(self) -> np.ndarray:
        if self.beta_ is None:
            raise ValueError("Model has not been fit yet.")
        return self.beta_
    
    @property
    def is_fitted(self) -> bool:
        return self.beta_ is not None
    
    @property
    def coef_dict_(self) -> dict:
        if not self.is_fitted or self.feature_names_ is None:
            raise ValueError("Model must be fit before accessing coef_dict_.")
        return dict(zip(self.feature_names_, self.beta_))
    
    @property
    def num_nonzero_(self) -> int:
        return int(np.sum(self.coef_ != 0))
    
    @staticmethod
    def safe_df_to_numpy(df: pd.DataFrame) -> np.ndarray:
        """
        Safely converts a DataFrame to a float64 NumPy array after ensuring all values are numeric.

        Args:
            df: Pandas DataFrame to convert.

        Returns:
            A NumPy ndarray of float64 values.

        Raises:
            ValueError if any value cannot be converted or if NaNs are found.
        """
        df_clean = df.apply(pd.to_numeric, errors='raise')  # will raise if non-numeric
        arr = df_clean.to_numpy(dtype=np.float64)

        if np.isnan(arr).any():
            raise ValueError("DataFrame contains NaNs after conversion.")

        return arr
    
    def log_fit_summary(self):
        if self.verbose:
            logger.info(f"Fit status: {self.status_}")
            logger.info(f"Objective value: {self.objective_value_:.4f}")
            logger.info(f"Num nonzero coefficients: {np.sum(self.beta_ != 0)}")

    def to_dict(self):
        if not self.is_fitted:
            raise ValueError("Model has not been fit yet.")
        return {
            "beta": self.beta_.tolist(),
            "feature_names": self.feature_names_,
            "coef_dict": self.coef_dict_,
            "status": self.status_,
            "objective": self.objective_value_
        }