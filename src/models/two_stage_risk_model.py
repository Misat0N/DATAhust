"""Two-stage credit risk model for merged four-class grading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from .model_evaluator import CreditModelEvaluator
from .risk_classifier import CreditRiskClassifier
from .threshold_adjuster import RiskThresholdAdjuster


@dataclass(frozen=True)
class TwoStageEvaluationResult:
    """Container for structured two-stage evaluation outputs."""

    overall_metrics: dict[str, Any]
    first_stage_metrics: dict[str, Any]
    second_stage_metrics: dict[str, Any] | None


class TwoStageCreditRiskModel:
    """Train and infer a two-stage four-class credit risk architecture.

    Stage 1 solves a binary problem:
    - class 0: normal applicants
    - class 1: all risky applicants merged from labels 1..3

    Stage 2 only focuses on applicants routed to the risky branch and predicts
    the fine-grained labels 1..3.
    """

    def __init__(
        self,
        first_stage_model_type: str = "lightgbm",
        second_stage_model_type: str = "lightgbm",
        first_stage_class_weight_mode: str = "auto",
        second_stage_class_weight_mode: str = "auto",
        first_stage_custom_class_weight: dict[int, float] | None = None,
        second_stage_custom_class_weight: dict[int, float] | None = None,
        first_stage_min_normal_precision: float = 0.8,
        random_state: int = 42,
    ) -> None:
        """Initialize the two-stage risk model."""

        self.first_stage_model_type = first_stage_model_type
        self.second_stage_model_type = second_stage_model_type
        self.first_stage_class_weight_mode = first_stage_class_weight_mode
        self.second_stage_class_weight_mode = second_stage_class_weight_mode
        self.first_stage_custom_class_weight = dict(
            first_stage_custom_class_weight or {0: 1.0, 1: 3.0}
        )
        self.second_stage_custom_class_weight = dict(
            second_stage_custom_class_weight or {1: 5.0, 2: 20.0, 3: 35.0}
        )
        self.first_stage_min_normal_precision = first_stage_min_normal_precision
        self.random_state = random_state

        self.first_stage_model = CreditRiskClassifier(
            model_type=self.first_stage_model_type,
            class_weight_mode=self.first_stage_class_weight_mode,
            custom_class_weight=self.first_stage_custom_class_weight,
            tuning_class_weight_modes=["none", "auto", "manual"],
        )
        self.second_stage_model = CreditRiskClassifier(
            model_type=self.second_stage_model_type,
            class_weight_mode=self.second_stage_class_weight_mode,
            custom_class_weight=self.second_stage_custom_class_weight,
            tuning_class_weight_modes=["none", "auto", "manual"],
        )
        self.threshold_adjuster = RiskThresholdAdjuster(normal_class_threshold=0.9)

        self.classes_ = np.array([0, 1, 2, 3], dtype=int)
        self.second_stage_classes_: np.ndarray | None = None
        self.second_stage_constant_label_: int | None = None
        self.is_fitted_ = False

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
    ) -> "TwoStageCreditRiskModel":
        """Fit the two-stage model with automatic threshold optimization."""

        y_train_array = np.asarray(y_train, dtype=int)
        binary_target = self._build_first_stage_labels(y_train_array)
        self.first_stage_model.fit(X_train, binary_target)

        first_stage_proba = self.first_stage_model.predict_proba(X_train)
        best_threshold, _ = self.threshold_adjuster.find_optimal_threshold(
            y_val_proba=first_stage_proba,
            y_val=binary_target,
            min_normal_precision=self.first_stage_min_normal_precision,
        )
        self.threshold_adjuster = RiskThresholdAdjuster(best_threshold)

        risk_mask = y_train_array != 0
        if not np.any(risk_mask):
            raise ValueError("Second-stage training requires at least one risky sample.")

        X_risk = self._slice_rows(X_train, risk_mask)
        y_risk = y_train_array[risk_mask]
        self.second_stage_classes_ = np.unique(y_risk)

        if len(self.second_stage_classes_) == 1:
            self.second_stage_constant_label_ = int(self.second_stage_classes_[0])
        else:
            self.second_stage_constant_label_ = None
            self.second_stage_model.fit(X_risk, y_risk)

        self.is_fitted_ = True
        return self

    def hyperparameter_tune(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray,
        y_val: pd.Series | np.ndarray,
    ) -> "TwoStageCreditRiskModel":
        """Tune both stages with validation data, then optimize the threshold."""

        y_train_array = np.asarray(y_train, dtype=int)
        y_val_array = np.asarray(y_val, dtype=int)

        binary_train = self._build_first_stage_labels(y_train_array)
        binary_val = self._build_first_stage_labels(y_val_array)
        self.first_stage_model.hyperparameter_tune(X_train, binary_train, X_val, binary_val)

        first_stage_val_proba = self.first_stage_model.predict_proba(X_val)
        best_threshold, _ = self.threshold_adjuster.find_optimal_threshold(
            y_val_proba=first_stage_val_proba,
            y_val=binary_val,
            min_normal_precision=self.first_stage_min_normal_precision,
        )
        self.threshold_adjuster = RiskThresholdAdjuster(best_threshold)

        train_risk_mask = y_train_array != 0
        val_risk_mask = y_val_array != 0
        X_train_risk = self._slice_rows(X_train, train_risk_mask)
        y_train_risk = y_train_array[train_risk_mask]
        X_val_risk = self._slice_rows(X_val, val_risk_mask)
        y_val_risk = y_val_array[val_risk_mask]

        self.second_stage_classes_ = np.unique(y_train_risk)
        if len(self.second_stage_classes_) == 1:
            self.second_stage_constant_label_ = int(self.second_stage_classes_[0])
        else:
            self.second_stage_constant_label_ = None
            self.second_stage_model.hyperparameter_tune(
                X_train_risk,
                y_train_risk,
                X_val_risk,
                y_val_risk,
            )

        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict the final four-class labels end to end."""

        self._ensure_fitted()
        first_stage_proba = self.first_stage_model.predict_proba(X)
        first_stage_binary_pred = self.threshold_adjuster.adjust_prediction(first_stage_proba)
        final_predictions = np.zeros(len(first_stage_binary_pred), dtype=int)

        risk_indices = np.where(first_stage_binary_pred == 1)[0]
        if len(risk_indices) == 0:
            return final_predictions

        if self.second_stage_constant_label_ is not None:
            final_predictions[risk_indices] = self.second_stage_constant_label_
            return final_predictions

        X_risk = self._slice_rows(X, risk_indices)
        second_stage_pred = self.second_stage_model.predict(X_risk)
        final_predictions[risk_indices] = np.asarray(second_stage_pred, dtype=int)
        return final_predictions

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return the four-class probability matrix."""

        self._ensure_fitted()

        first_stage_proba = self.first_stage_model.predict_proba(X)
        normal_index = self._get_class_index(self.first_stage_model.classes_, 0)
        risk_index = self._get_class_index(self.first_stage_model.classes_, 1)

        p_normal = first_stage_proba[:, normal_index]
        p_risk = first_stage_proba[:, risk_index]

        probability_matrix = np.zeros((len(p_normal), 4), dtype=float)
        probability_matrix[:, 0] = p_normal

        if self.second_stage_constant_label_ is not None:
            probability_matrix[:, self.second_stage_constant_label_] = p_risk
            return probability_matrix

        second_stage_proba = self.second_stage_model.predict_proba(X)
        second_stage_classes = np.asarray(self.second_stage_model.classes_, dtype=int)

        for class_offset, class_label in enumerate(second_stage_classes):
            probability_matrix[:, int(class_label)] = p_risk * second_stage_proba[:, class_offset]

        row_sums = probability_matrix.sum(axis=1, keepdims=True)
        row_sums = np.clip(row_sums, 1e-12, None)
        probability_matrix = probability_matrix / row_sums
        return probability_matrix

    def predict_with_threshold_and_rule(
        self,
        X: pd.DataFrame | np.ndarray,
        user_data: pd.DataFrame | list[dict[str, Any]] | dict[str, Any] | None = None,
        normal_threshold: float = 0.9,
    ) -> np.ndarray:
        """Predict with a custom threshold and optional business-rule overrides."""

        probability_matrix = self.predict_proba(X)
        local_adjuster = RiskThresholdAdjuster(normal_class_threshold=normal_threshold)
        adjusted_predictions = local_adjuster.adjust_prediction(probability_matrix)

        if user_data is None:
            return adjusted_predictions

        user_rows = self._normalize_user_data(user_data)
        if len(user_rows) != len(adjusted_predictions):
            raise ValueError(
                "user_data row count must match the number of prediction samples."
            )

        final_predictions = []
        for user_row, model_prediction in zip(user_rows, adjusted_predictions):
            final_predictions.append(
                local_adjuster.business_rule_override(user_row, int(model_prediction))
            )
        return np.asarray(final_predictions, dtype=int)

    def evaluate_two_stage_model(
        self,
        X_test: pd.DataFrame | np.ndarray,
        y_test: pd.Series | np.ndarray,
    ) -> dict[str, Any]:
        """Evaluate both the overall model and its two internal stages."""

        self._ensure_fitted()
        y_test_array = np.asarray(y_test, dtype=int)
        overall_metrics = CreditModelEvaluator(
            model=self,
            X_test=X_test,
            y_test=y_test_array,
            class_names=["正常类", "关注类", "次级/可疑类", "损失类"],
        ).evaluate_imbalanced_multiclass()

        first_stage_binary_true = self._build_first_stage_labels(y_test_array)
        first_stage_metrics = CreditModelEvaluator(
            model=self.first_stage_model,
            X_test=X_test,
            y_test=first_stage_binary_true,
            class_names=["正常类", "风险类"],
        ).evaluate_imbalanced_multiclass()

        second_stage_metrics: dict[str, Any] | None = None
        risk_mask = y_test_array != 0
        if np.any(risk_mask):
            X_risk = self._slice_rows(X_test, risk_mask)
            y_risk = y_test_array[risk_mask]
            if self.second_stage_constant_label_ is not None:
                constant_pred = np.full(len(y_risk), self.second_stage_constant_label_, dtype=int)
                constant_proba = np.zeros((len(y_risk), 4), dtype=float)
                constant_proba[:, self.second_stage_constant_label_] = 1.0
                second_stage_metrics = self._evaluate_fixed_predictions(
                    y_true=y_risk,
                    y_pred=constant_pred,
                    y_proba=constant_proba,
                )
            else:
                second_stage_pred = np.asarray(
                    self.second_stage_model.predict(X_risk),
                    dtype=int,
                )
                second_stage_proba = np.asarray(
                    self.second_stage_model.predict_proba(X_risk),
                    dtype=float,
                )
                second_stage_metrics = self._evaluate_relabeled_predictions(
                    y_true=y_risk,
                    y_pred=second_stage_pred,
                    y_proba=second_stage_proba,
                )

        return {
            "overall": overall_metrics,
            "first_stage": first_stage_metrics,
            "second_stage": second_stage_metrics,
            "normal_threshold": self.threshold_adjuster.normal_class_threshold,
        }

    def get_threshold(self) -> float:
        """Return the optimized normal-class threshold."""

        return float(self.threshold_adjuster.normal_class_threshold)

    def _build_first_stage_labels(self, y: np.ndarray) -> np.ndarray:
        """Convert the original four-class labels into normal-vs-risk binary labels."""

        return np.where(np.asarray(y, dtype=int) == 0, 0, 1).astype(int)

    def _slice_rows(
        self,
        X: pd.DataFrame | np.ndarray,
        row_selector: np.ndarray,
    ) -> pd.DataFrame | np.ndarray:
        """Slice rows while preserving the original feature container type."""

        if isinstance(X, pd.DataFrame):
            if row_selector.dtype == bool:
                return X.loc[row_selector].copy()
            return X.iloc[row_selector].copy()
        return np.asarray(X)[row_selector]

    def _normalize_user_data(
        self,
        user_data: Any,
    ) -> list[dict[str, Any]]:
        """Normalize user data into a list of dictionaries."""

        if isinstance(user_data, pd.DataFrame):
            return user_data.to_dict(orient="records")
        if isinstance(user_data, dict):
            return [dict(user_data)]
        if isinstance(user_data, list):
            return [dict(item) for item in user_data]
        raise TypeError(
            "user_data must be a DataFrame, a dictionary, or a list of dictionaries."
        )

    def _ensure_fitted(self) -> None:
        """Validate that the two-stage model has been trained."""

        if not self.is_fitted_:
            raise RuntimeError("TwoStageCreditRiskModel must be fitted before inference.")

    def _get_class_index(self, classes: np.ndarray, target_class: int) -> int:
        """Find the probability column index for a target class."""

        matching_indices = np.where(np.asarray(classes, dtype=int) == int(target_class))[0]
        if len(matching_indices) == 0:
            raise ValueError(f"Target class {target_class} is missing from model classes.")
        return int(matching_indices[0])

    def _evaluate_fixed_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> dict[str, Any]:
        """Evaluate constant or externally generated predictions."""

        labels = np.unique(np.concatenate([y_true, y_pred]))
        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        predicted_distribution = {
            int(label): float(np.mean(y_pred == label)) for label in labels
        }
        return {
            "classification_report": report,
            "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
            "weighted_f1": float(
                report.get("weighted avg", {}).get("f1-score", 0.0)
            ),
            "predicted_class_distribution": predicted_distribution,
            "probability_shape": list(np.asarray(y_proba).shape),
        }

    def _evaluate_relabeled_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> dict[str, Any]:
        """Evaluate predictions on sparse original labels without AUC dependency."""

        observed_labels = np.unique(
            np.concatenate([np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)])
        )
        report = classification_report(
            y_true,
            y_pred,
            labels=observed_labels,
            output_dict=True,
            zero_division=0,
        )
        per_class_report = {str(int(label)): report.get(str(int(label)), {}) for label in observed_labels}
        predicted_distribution = {
            int(class_label): float(np.mean(y_pred == int(class_label)))
            for class_label in observed_labels
        }
        return {
            "classification_report": report,
            "classification_report_by_original_label": per_class_report,
            "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
            "weighted_f1": float(
                report.get("weighted avg", {}).get("f1-score", 0.0)
            ),
            "predicted_class_distribution": predicted_distribution,
            "probability_shape": list(np.asarray(y_proba).shape),
            "class_labels": [int(class_label) for class_label in observed_labels],
        }
