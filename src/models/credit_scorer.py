"""Credit limit estimation models with regulatory constraints."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


@dataclass(frozen=True)
class RegressionTuningResult:
    """Container for hyperparameter tuning results."""

    best_params: Dict[str, Any]
    best_cv_rmse: float
    validation_mae: float
    validation_rmse: float
    validation_r2: float


class CreditScorer:
    """Unified credit limit regression model with hard regulatory constraints."""

    SUPPORTED_MODELS = {
        "linear_regression",
        "ridge",
        "lasso",
        "lightgbm",
    }

    def __init__(self, model_type: str = "lightgbm") -> None:
        """Initialize the scorer with the requested regression model."""

        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Supported models: {sorted(self.SUPPORTED_MODELS)}"
            )

        self.model_type = model_type
        self.model = self._build_model()
        self.best_model_ = None
        self.best_params_: Dict[str, Any] = {}
        self.best_cv_rmse_: float | None = None
        self.validation_metrics_: Dict[str, float] = {}
        self.training_feature_names_: list[str] | None = None
        self.feature_name_map_: dict[str, str] = {}
        self.is_fitted_ = False

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
    ) -> "CreditScorer":
        """Train the configured regression model."""

        self.training_feature_names_ = (
            X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else None
        )
        self._initialize_feature_name_map()
        X_train_array = self._ensure_2d_array(X_train)
        y_train_array = np.asarray(y_train, dtype=float)

        self.model = self._build_model()
        self.model.fit(X_train_array, y_train_array)
        self.best_model_ = self.model
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return raw predicted credit limits."""

        estimator = self._get_fitted_estimator()
        predictions = estimator.predict(self._ensure_2d_array(X))
        return np.asarray(predictions, dtype=float)

    def predict_with_constraints(
        self,
        X: pd.DataFrame | np.ndarray,
        risk_labels: pd.Series | np.ndarray,
    ) -> np.ndarray:
        """Apply hard regulatory constraints to raw credit predictions.

        Constraint rules
        ----------------
        - Risk grades 3/4: limit = 0
        - Risk grade 2: limit <= 3 * monthly disposable income
        - Risk grade 1: limit <= 6 * monthly disposable income
        - Risk grade 0: limit <= 12 * monthly disposable income

        The method expects ``annual_inc`` and ``dti`` to be available when a
        DataFrame is provided. Monthly disposable income is estimated as:

        ``annual_inc * (1 - min(dti / 100, 1.0)) / 12``
        """

        raw_predictions = np.maximum(self.predict(X), 0.0)
        risk_array = np.asarray(risk_labels, dtype=int)
        monthly_income = self._estimate_monthly_disposable_income(X)

        constrained_predictions = raw_predictions.copy()
        constrained_predictions[np.isin(risk_array, [3, 4])] = 0.0
        constrained_predictions[risk_array == 2] = np.minimum(
            constrained_predictions[risk_array == 2],
            monthly_income[risk_array == 2] * 3.0,
        )
        constrained_predictions[risk_array == 1] = np.minimum(
            constrained_predictions[risk_array == 1],
            monthly_income[risk_array == 1] * 6.0,
        )
        constrained_predictions[risk_array == 0] = np.minimum(
            constrained_predictions[risk_array == 0],
            monthly_income[risk_array == 0] * 12.0,
        )

        constrained_predictions = np.maximum(constrained_predictions, 0.0)
        return np.round(constrained_predictions, 2)

    def hyperparameter_tune(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray,
        y_val: pd.Series | np.ndarray,
    ) -> "CreditScorer":
        """Tune model hyperparameters with 5-fold time-series CV."""

        self.training_feature_names_ = (
            X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else None
        )
        self._initialize_feature_name_map()
        X_train_array = self._ensure_2d_array(X_train)
        y_train_array = np.asarray(y_train, dtype=float)
        X_val_array = self._ensure_2d_array(X_val)
        y_val_array = np.asarray(y_val, dtype=float)

        base_model = self._build_model()
        param_grid = self._get_param_grid()
        time_series_cv = TimeSeriesSplit(n_splits=5)

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=time_series_cv,
            n_jobs=1,
            refit=True,
            verbose=0,
        )
        grid_search.fit(X_train_array, y_train_array)

        tuned_estimator = grid_search.best_estimator_
        val_pred = tuned_estimator.predict(X_val_array)

        self.best_model_ = tuned_estimator
        self.model = tuned_estimator
        self.best_params_ = grid_search.best_params_
        self.best_cv_rmse_ = float(-grid_search.best_score_)
        self.validation_metrics_ = {
            "mae": float(mean_absolute_error(y_val_array, val_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_val_array, val_pred))),
            "r2": float(r2_score(y_val_array, val_pred)),
        }
        self.is_fitted_ = True
        return self

    def get_tuning_summary(self) -> RegressionTuningResult | None:
        """Return the latest tuning summary."""

        if self.best_cv_rmse_ is None:
            return None
        return RegressionTuningResult(
            best_params=self.best_params_,
            best_cv_rmse=self.best_cv_rmse_,
            validation_mae=self.validation_metrics_["mae"],
            validation_rmse=self.validation_metrics_["rmse"],
            validation_r2=self.validation_metrics_["r2"],
        )

    def _get_fitted_estimator(self):
        """Return the fitted estimator."""

        if not self.is_fitted_ or self.best_model_ is None:
            raise RuntimeError("CreditScorer must be fitted before inference.")
        return self.best_model_

    def _ensure_2d_array(self, X: object) -> pd.DataFrame | np.ndarray:
        """Convert supported feature containers to a model-ready matrix."""

        if isinstance(X, pd.DataFrame):
            if self.training_feature_names_ is not None:
                missing_columns = sorted(
                    set(self.training_feature_names_) - set(X.columns)
                )
                if missing_columns:
                    raise ValueError(
                        "Input DataFrame is missing training feature columns: "
                        f"{missing_columns}"
                    )
                frame = X[self.training_feature_names_].copy()
            else:
                frame = X.copy()

            if self.model_type == "lightgbm":
                return frame.rename(columns=self.feature_name_map_).copy()
            return frame
        if isinstance(X, np.ndarray):
            return X
        raise TypeError("Input features must be a pandas DataFrame or numpy array.")

    def _initialize_feature_name_map(self) -> None:
        """Build a LightGBM-safe feature name mapping when needed."""

        if self.training_feature_names_ is None:
            self.feature_name_map_ = {}
            return

        if self.model_type != "lightgbm":
            self.feature_name_map_ = {
                feature_name: feature_name for feature_name in self.training_feature_names_
            }
            return

        sanitized_names: list[str] = []
        used_names: set[str] = set()
        for index, feature_name in enumerate(self.training_feature_names_):
            sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", feature_name).strip("_")
            if not sanitized:
                sanitized = f"feature_{index:04d}"
            candidate = sanitized
            duplicate_index = 1
            while candidate in used_names:
                candidate = f"{sanitized}_{duplicate_index}"
                duplicate_index += 1
            used_names.add(candidate)
            sanitized_names.append(candidate)

        self.feature_name_map_ = dict(zip(self.training_feature_names_, sanitized_names))

    def _estimate_monthly_disposable_income(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """Estimate monthly disposable income from annual income and DTI."""

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "predict_with_constraints requires a pandas DataFrame containing "
                "'annual_inc' and 'dti' columns."
            )
        required_columns = {"annual_inc", "dti"}
        missing_columns = required_columns - set(X.columns)
        if missing_columns:
            raise ValueError(
                "predict_with_constraints requires original features with "
                f"columns: {sorted(required_columns)}. Missing: {sorted(missing_columns)}"
            )

        annual_income = pd.to_numeric(X["annual_inc"], errors="coerce").fillna(0.0)
        dti_ratio = (
            pd.to_numeric(X["dti"], errors="coerce").fillna(0.0) / 100.0
        ).clip(lower=0.0, upper=1.0)
        monthly_disposable_income = annual_income * (1.0 - dti_ratio) / 12.0
        return np.maximum(monthly_disposable_income.to_numpy(dtype=float), 0.0)

    def _build_model(self):
        """Instantiate the requested regression model."""

        if self.model_type == "linear_regression":
            return LinearRegression()

        if self.model_type == "ridge":
            return Ridge(alpha=1.0, random_state=42)

        if self.model_type == "lasso":
            return Lasso(alpha=0.1, random_state=42, max_iter=5000)

        if self.model_type == "lightgbm":
            return LGBMRegressor(
                objective="regression",
                n_estimators=200,
                max_depth=7,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                n_jobs=1,
                verbosity=-1,
            )

        raise ValueError(f"Unsupported model_type '{self.model_type}'.")

    def _get_param_grid(self) -> Dict[str, list]:
        """Return model-specific tuning grids."""

        grids: Dict[str, Dict[str, list]] = {
            "linear_regression": {
                "fit_intercept": [True, False],
            },
            "ridge": {
                "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
                "fit_intercept": [True, False],
            },
            "lasso": {
                "alpha": [0.001, 0.01, 0.1, 1.0],
                "fit_intercept": [True, False],
            },
            "lightgbm": {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [3, 5, 7, 9],
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves": [15, 31, 63],
            },
        }
        return grids[self.model_type]
