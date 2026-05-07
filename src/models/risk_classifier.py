"""Risk classification models for credit risk grading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


@dataclass(frozen=True)
class TuningResult:
    """Container for hyperparameter tuning outputs."""

    best_params: Dict[str, Any]
    best_cv_score: float
    validation_accuracy: float
    validation_f1_weighted: float


class CreditRiskClassifier:
    """Unified wrapper for multiple credit risk classification models."""

    SUPPORTED_MODELS = {
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
        "lightgbm",
    }

    def __init__(self, model_type: str = "lightgbm") -> None:
        """Initialize the classifier with the requested model family."""

        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Supported models: {sorted(self.SUPPORTED_MODELS)}"
            )

        self.model_type = model_type
        self.model = self._build_model()
        self.best_model_ = None
        self.best_params_: Dict[str, Any] = {}
        self.best_cv_score_: float | None = None
        self.validation_metrics_: Dict[str, float] = {}
        self.classes_: np.ndarray | None = None
        self.label_encoder_ = LabelEncoder()
        self.is_fitted_ = False

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
    ) -> "CreditRiskClassifier":
        """Train the configured model."""

        X_train_array = self._ensure_2d_array(X_train)
        y_train_array = np.asarray(y_train)
        y_train_encoded = self.label_encoder_.fit_transform(y_train_array)

        self.classes_ = self.label_encoder_.classes_
        self.model = self._build_model(num_classes=len(self.classes_))
        self.model.fit(X_train_array, y_train_encoded)
        self.best_model_ = self.model
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class labels."""

        estimator = self._get_fitted_estimator()
        predicted_encoded = estimator.predict(self._ensure_2d_array(X))
        return self.label_encoder_.inverse_transform(predicted_encoded.astype(int))

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probabilities."""

        estimator = self._get_fitted_estimator()
        if not hasattr(estimator, "predict_proba"):
            raise AttributeError(
                f"Model '{self.model_type}' does not support probability prediction."
            )
        return estimator.predict_proba(self._ensure_2d_array(X))

    def hyperparameter_tune(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray,
        y_val: pd.Series | np.ndarray,
    ) -> "CreditRiskClassifier":
        """Tune hyperparameters using 5-fold time-series CV and grid search."""

        X_train_array = self._ensure_2d_array(X_train)
        y_train_array = np.asarray(y_train)
        X_val_array = self._ensure_2d_array(X_val)
        y_val_array = np.asarray(y_val)

        y_train_encoded = self.label_encoder_.fit_transform(y_train_array)
        self.classes_ = self.label_encoder_.classes_
        base_model = self._build_model(num_classes=len(self.classes_))
        param_grid = self._get_param_grid()
        time_series_cv = TimeSeriesSplit(n_splits=5)

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="f1_weighted",
            cv=time_series_cv,
            n_jobs=1,
            refit=True,
            verbose=0,
        )
        grid_search.fit(X_train_array, y_train_encoded)

        tuned_estimator = grid_search.best_estimator_
        val_pred_encoded = tuned_estimator.predict(X_val_array)
        val_pred = self.label_encoder_.inverse_transform(val_pred_encoded.astype(int))

        self.best_model_ = tuned_estimator
        self.model = tuned_estimator
        self.best_params_ = grid_search.best_params_
        self.best_cv_score_ = float(grid_search.best_score_)
        self.validation_metrics_ = {
            "accuracy": float(accuracy_score(y_val_array, val_pred)),
            "f1_weighted": float(
                f1_score(y_val_array, val_pred, average="weighted")
            ),
        }
        self.is_fitted_ = True
        return self

    def get_tuning_summary(self) -> TuningResult | None:
        """Return a structured summary of the latest tuning run."""

        if self.best_cv_score_ is None:
            return None
        return TuningResult(
            best_params=self.best_params_,
            best_cv_score=self.best_cv_score_,
            validation_accuracy=self.validation_metrics_["accuracy"],
            validation_f1_weighted=self.validation_metrics_["f1_weighted"],
        )

    def _get_fitted_estimator(self):
        """Return the trained estimator."""

        if not self.is_fitted_ or self.best_model_ is None:
            raise RuntimeError("CreditRiskClassifier must be fitted before inference.")
        return self.best_model_

    def _ensure_2d_array(self, X: object) -> pd.DataFrame | np.ndarray:
        """Convert supported input formats to a model-ready matrix."""

        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, np.ndarray):
            return X
        raise TypeError("Input features must be a pandas DataFrame or numpy array.")

    def _build_model(self, num_classes: int | None = None):
        """Instantiate the requested model family."""

        if self.model_type == "logistic_regression":
            return LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )

        if self.model_type == "decision_tree":
            return DecisionTreeClassifier(
                random_state=42,
                class_weight="balanced",
            )

        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample",
            )

        if self.model_type == "xgboost":
            return XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                num_class=num_classes,
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=1,
            )

        if self.model_type == "lightgbm":
            return LGBMClassifier(
                objective="multiclass",
                n_estimators=200,
                max_depth=7,
                learning_rate=0.05,
                num_leaves=31,
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
                verbosity=-1,
            )

        raise ValueError(f"Unsupported model_type '{self.model_type}'.")

    def _get_param_grid(self) -> Dict[str, list]:
        """Return the model-specific hyperparameter grid."""

        grids: Dict[str, Dict[str, list]] = {
            "logistic_regression": {
                "C": [0.1, 1.0, 10.0],
                "solver": ["lbfgs"],
            },
            "decision_tree": {
                "max_depth": [3, 5, 7, 9, None],
                "min_samples_split": [2, 10, 30],
                "class_weight": ["balanced"],
            },
            "random_forest": {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 7, 9, None],
                "min_samples_leaf": [1, 5, 10],
                "class_weight": ["balanced_subsample"],
            },
            "xgboost": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
            "lightgbm": {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [3, 5, 7, 9],
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves": [15, 31, 63],
                "class_weight": ["balanced"],
            },
        }
        return grids[self.model_type]
