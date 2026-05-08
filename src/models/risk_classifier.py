"""Risk classification models for credit risk grading."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
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

    def __init__(
        self,
        model_type: str = "lightgbm",
        class_weight_mode: str = "none",
        custom_class_weight: dict[Any, float] | None = None,
        tuning_class_weight_modes: list[str] | None = None,
    ) -> None:
        """Initialize the classifier with the requested model family.

        Parameters
        ----------
        model_type:
            Model family to use.
        class_weight_mode:
            Weighting strategy for imbalanced classification.
            - ``none``: no class weighting, suitable as a baseline.
            - ``auto``: compute balanced weights from the training label
              distribution, suitable for a first-pass imbalance correction.
            - ``manual``: use a user-provided class weight dictionary when the
              business side wants stronger penalties on high-risk classes.
        custom_class_weight:
            Manual class weight dictionary keyed by original class labels.
        tuning_class_weight_modes:
            Optional list of weighting modes to search during
            ``hyperparameter_tune``. If omitted, the classifier searches
            ``none`` and ``auto``, and additionally includes ``manual`` when a
            manual weight dictionary is provided.
        """

        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Supported models: {sorted(self.SUPPORTED_MODELS)}"
            )
        if class_weight_mode not in {"none", "auto", "manual"}:
            raise ValueError(
                "class_weight_mode must be one of: 'none', 'auto', 'manual'."
            )
        if class_weight_mode == "manual" and not custom_class_weight:
            raise ValueError(
                "custom_class_weight must be provided when class_weight_mode='manual'."
            )

        self.model_type = model_type
        self.class_weight_mode = class_weight_mode
        self.custom_class_weight = dict(custom_class_weight or {})
        self.tuning_class_weight_modes = list(tuning_class_weight_modes or [])
        self.model = self._build_model()
        self.best_model_ = None
        self.best_params_: Dict[str, Any] = {}
        self.best_cv_score_: float | None = None
        self.validation_metrics_: Dict[str, float] = {}
        self.classes_: np.ndarray | None = None
        self.label_encoder_ = LabelEncoder()
        self.active_class_weight_config_: Dict[Any, float] | None = None
        self.active_encoded_class_weight_config_: Dict[int, float] | None = None
        self.active_tuned_weight_mode_: str = class_weight_mode
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
        resolved_weights = self._resolve_class_weight_config(
            y_train_array,
            self.class_weight_mode,
        )
        encoded_weights = self._encode_class_weight_dict(resolved_weights)

        self.model = self._build_model(
            num_classes=len(self.classes_),
            encoded_class_weight=encoded_weights,
        )
        fit_kwargs = self._build_fit_kwargs(y_train_encoded, encoded_weights)
        self.model.fit(X_train_array, y_train_encoded, **fit_kwargs)

        self.active_class_weight_config_ = resolved_weights
        self.active_encoded_class_weight_config_ = encoded_weights
        self.active_tuned_weight_mode_ = self.class_weight_mode
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
        """Tune hyperparameters using time-series CV and weight search.

        The method searches both model hyperparameters and class-weight
        strategies. For XGBoost, class weights are passed as ``sample_weight``.
        For RandomForest, LightGBM, LogisticRegression, and DecisionTree, the
        resolved weight dictionary is injected through the native
        ``class_weight`` argument.
        """

        X_train_array = self._ensure_2d_array(X_train)
        y_train_array = np.asarray(y_train)
        X_val_array = self._ensure_2d_array(X_val)
        y_val_array = np.asarray(y_val)

        y_train_encoded = self.label_encoder_.fit_transform(y_train_array)
        self.classes_ = self.label_encoder_.classes_
        param_grid = self._get_param_grid()
        param_candidates = self._expand_param_grid(param_grid)
        weight_modes = self._get_tuning_weight_modes()
        time_series_cv = TimeSeriesSplit(n_splits=5)

        best_score = float("-inf")
        best_params: Dict[str, Any] = {}
        best_weight_mode = self.class_weight_mode

        for params in param_candidates:
            for weight_mode in weight_modes:
                cv_scores: list[float] = []

                for fold_train_index, fold_valid_index in time_series_cv.split(
                    X_train_array
                ):
                    X_fold_train = self._slice_rows(X_train_array, fold_train_index)
                    X_fold_valid = self._slice_rows(X_train_array, fold_valid_index)

                    y_fold_train = y_train_array[fold_train_index]
                    y_fold_valid = y_train_array[fold_valid_index]
                    y_fold_train_encoded = self.label_encoder_.transform(y_fold_train)

                    fold_weights = self._resolve_class_weight_config(
                        y_fold_train,
                        weight_mode,
                    )
                    fold_encoded_weights = self._encode_class_weight_dict(fold_weights)

                    candidate_model = self._build_model(
                        num_classes=len(self.classes_),
                        encoded_class_weight=fold_encoded_weights,
                        override_params=params,
                    )
                    fit_kwargs = self._build_fit_kwargs(
                        y_fold_train_encoded,
                        fold_encoded_weights,
                    )
                    candidate_model.fit(
                        X_fold_train,
                        y_fold_train_encoded,
                        **fit_kwargs,
                    )

                    fold_pred_encoded = candidate_model.predict(X_fold_valid)
                    fold_pred = self.label_encoder_.inverse_transform(
                        fold_pred_encoded.astype(int)
                    )
                    fold_score = f1_score(
                        y_fold_valid,
                        fold_pred,
                        average="weighted",
                        zero_division=0,
                    )
                    cv_scores.append(float(fold_score))

                mean_cv_score = float(np.mean(cv_scores)) if cv_scores else float("-inf")
                if mean_cv_score > best_score:
                    best_score = mean_cv_score
                    best_params = dict(params)
                    best_weight_mode = weight_mode

        tuned_weights = self._resolve_class_weight_config(
            y_train_array,
            best_weight_mode,
        )
        tuned_encoded_weights = self._encode_class_weight_dict(tuned_weights)
        tuned_estimator = self._build_model(
            num_classes=len(self.classes_),
            encoded_class_weight=tuned_encoded_weights,
            override_params=best_params,
        )
        tuned_fit_kwargs = self._build_fit_kwargs(y_train_encoded, tuned_encoded_weights)
        tuned_estimator.fit(X_train_array, y_train_encoded, **tuned_fit_kwargs)
        val_pred_encoded = tuned_estimator.predict(X_val_array)
        val_pred = self.label_encoder_.inverse_transform(val_pred_encoded.astype(int))

        self.best_model_ = tuned_estimator
        self.model = tuned_estimator
        self.best_params_ = {
            **best_params,
            "class_weight_mode": best_weight_mode,
            "class_weight": tuned_weights,
        }
        self.best_cv_score_ = best_score
        self.validation_metrics_ = {
            "accuracy": float(accuracy_score(y_val_array, val_pred)),
            "f1_weighted": float(
                f1_score(y_val_array, val_pred, average="weighted", zero_division=0)
            ),
        }
        self.active_class_weight_config_ = tuned_weights
        self.active_encoded_class_weight_config_ = tuned_encoded_weights
        self.active_tuned_weight_mode_ = best_weight_mode
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

    def calculate_auto_class_weights(
        self,
        y_train: pd.Series | np.ndarray,
    ) -> Dict[Any, float]:
        """Calculate balanced weights from the observed training labels."""

        y_train_array = np.asarray(y_train)
        unique_classes = np.unique(y_train_array)
        computed_weights = compute_class_weight(
            class_weight="balanced",
            classes=unique_classes,
            y=y_train_array,
        )
        return {
            class_label: float(class_weight)
            for class_label, class_weight in zip(unique_classes, computed_weights)
        }

    def get_class_weight_distribution(self) -> Dict[str, Any]:
        """Return the current weight configuration for debugging and reports."""

        return {
            "model_type": self.model_type,
            "configured_mode": self.class_weight_mode,
            "active_mode": self.active_tuned_weight_mode_,
            "custom_class_weight": dict(self.custom_class_weight),
            "resolved_class_weight": (
                dict(self.active_class_weight_config_)
                if self.active_class_weight_config_ is not None
                else None
            ),
            "encoded_class_weight": (
                dict(self.active_encoded_class_weight_config_)
                if self.active_encoded_class_weight_config_ is not None
                else None
            ),
        }

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

    def _build_model(
        self,
        num_classes: int | None = None,
        encoded_class_weight: dict[int, float] | None = None,
        override_params: dict[str, Any] | None = None,
    ):
        """Instantiate the requested model family."""

        params = dict(override_params or {})
        resolved_random_state = 42
        is_multiclass = num_classes is not None and num_classes > 2

        if self.model_type == "logistic_regression":
            base_params = {
                "max_iter": 1000,
                "class_weight": encoded_class_weight,
                "random_state": resolved_random_state,
            }
            base_params.update(params)
            return LogisticRegression(**base_params)

        if self.model_type == "decision_tree":
            base_params = {
                "random_state": resolved_random_state,
                "class_weight": encoded_class_weight,
            }
            base_params.update(params)
            return DecisionTreeClassifier(**base_params)

        if self.model_type == "random_forest":
            base_params = {
                "n_estimators": 300,
                "max_depth": None,
                "random_state": resolved_random_state,
                "n_jobs": -1,
                "class_weight": encoded_class_weight,
            }
            base_params.update(params)
            return RandomForestClassifier(**base_params)

        if self.model_type == "xgboost":
            objective = "multi:softprob" if is_multiclass else "binary:logistic"
            base_params = {
                "objective": objective,
                "eval_metric": "mlogloss",
                "num_class": num_classes,
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": resolved_random_state,
                "n_jobs": 1,
            }
            base_params.update(params)
            return XGBClassifier(**base_params)

        if self.model_type == "lightgbm":
            objective = "multiclass" if is_multiclass else "binary"
            base_params = {
                "objective": objective,
                "n_estimators": 200,
                "max_depth": 7,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "class_weight": encoded_class_weight,
                "random_state": resolved_random_state,
                "n_jobs": 1,
                "verbosity": -1,
            }
            base_params.update(params)
            return LGBMClassifier(**base_params)

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
            },
            "random_forest": {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 7, 9, None],
                "min_samples_leaf": [1, 5, 10],
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
            },
        }
        return grids[self.model_type]

    def _resolve_class_weight_config(
        self,
        y_train: np.ndarray,
        weight_mode: str,
    ) -> Dict[Any, float] | None:
        """Resolve the effective class-weight dictionary for one training run."""

        if weight_mode == "none":
            return None
        if weight_mode == "auto":
            return self.calculate_auto_class_weights(y_train)
        if weight_mode == "manual":
            return self._filter_manual_class_weights(y_train)
        raise ValueError(f"Unsupported weight mode '{weight_mode}'.")

    def _filter_manual_class_weights(
        self,
        y_train: np.ndarray,
    ) -> Dict[Any, float]:
        """Keep only manual weights for classes that exist in the training fold."""

        if not self.custom_class_weight:
            raise ValueError(
                "Manual class weighting requires a non-empty custom_class_weight."
            )

        present_classes = set(np.unique(y_train).tolist())
        filtered_weights = {
            class_label: float(class_weight)
            for class_label, class_weight in self.custom_class_weight.items()
            if class_label in present_classes
        }
        if not filtered_weights:
            raise ValueError(
                "No manual class weights match the classes present in the training data."
            )

        for class_label in present_classes:
            filtered_weights.setdefault(class_label, 1.0)
        return filtered_weights

    def _encode_class_weight_dict(
        self,
        class_weight_dict: Dict[Any, float] | None,
    ) -> Dict[int, float] | None:
        """Map original labels to encoded labels for estimator consumption."""

        if class_weight_dict is None:
            return None

        encoded_weights: Dict[int, float] = {}
        for class_label, class_weight in class_weight_dict.items():
            encoded_label = int(self.label_encoder_.transform([class_label])[0])
            encoded_weights[encoded_label] = float(class_weight)
        return encoded_weights

    def _build_fit_kwargs(
        self,
        y_train_encoded: np.ndarray,
        encoded_class_weight: Dict[int, float] | None,
    ) -> Dict[str, Any]:
        """Build model-specific fit keyword arguments.

        XGBoost does not accept a multiclass ``class_weight`` parameter, so the
        weighting effect is injected through per-sample ``sample_weight``.
        For LightGBM multiclass problems, ``class_weight`` is the primary
        weighting interface; ``scale_pos_weight`` is binary-only and therefore
        not used here.
        """

        if encoded_class_weight is None:
            return {}

        if self.model_type == "xgboost":
            sample_weight = np.asarray(
                [encoded_class_weight.get(int(label), 1.0) for label in y_train_encoded],
                dtype=float,
            )
            return {"sample_weight": sample_weight}

        return {}

    def _expand_param_grid(
        self,
        param_grid: Dict[str, list],
    ) -> list[Dict[str, Any]]:
        """Expand a parameter grid into a list of concrete combinations."""

        if not param_grid:
            return [{}]

        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        return [
            dict(zip(param_names, combination))
            for combination in product(*param_values)
        ]

    def _get_tuning_weight_modes(self) -> list[str]:
        """Return the list of weight modes to search during tuning."""

        if self.tuning_class_weight_modes:
            return list(self.tuning_class_weight_modes)

        modes = ["none", "auto"]
        if self.custom_class_weight:
            modes.append("manual")
        if self.class_weight_mode == "manual" and "manual" not in modes:
            modes.append("manual")
        return modes

    def _slice_rows(
        self,
        X: pd.DataFrame | np.ndarray,
        indices: np.ndarray,
    ) -> pd.DataFrame | np.ndarray:
        """Slice rows while preserving the original matrix type."""

        if isinstance(X, pd.DataFrame):
            return X.iloc[indices].copy()
        return X[indices]
