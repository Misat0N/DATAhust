"""Threshold adjustment and business-rule overrides for credit risk outputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


class RiskThresholdAdjuster:
    """Adjust raw model predictions with thresholds and business rules.

    The class is designed for credit-risk models where the normal class
    (label 0) often dominates. The threshold layer makes class 0 harder to
    obtain, while the business-rule layer provides deterministic overrides for
    clearly risky applicants.
    """

    def __init__(self, normal_class_threshold: float = 0.9) -> None:
        """Initialize the adjuster with the normal-class probability threshold."""

        self.normal_class_threshold = self._clip_threshold(normal_class_threshold)

    def adjust_prediction(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """Adjust predictions based on the normal-class confidence threshold.

        Business intention
        ------------------
        - Predict class 0 only when the model is sufficiently confident.
        - Otherwise, force the prediction into the highest-probability risk
          class among labels 1..K-1.

        Parameters
        ----------
        y_pred_proba:
            Probability matrix with shape ``(n_samples, n_classes)``. The
            intended full setup is a four-class probability matrix, but the
            method remains compatible with the current smaller-class artifacts.
        """

        probability_array = np.asarray(y_pred_proba, dtype=float)
        if probability_array.ndim != 2:
            raise ValueError("y_pred_proba must be a 2D probability matrix.")
        if probability_array.shape[1] < 2:
            raise ValueError(
                "Threshold adjustment requires at least two class probabilities."
            )

        adjusted_predictions = []
        for sample_probabilities in probability_array:
            normal_probability = float(sample_probabilities[0])
            if normal_probability >= self.normal_class_threshold:
                adjusted_predictions.append(0)
                continue

            risk_probabilities = sample_probabilities[1:]
            risk_label = int(np.argmax(risk_probabilities)) + 1
            adjusted_predictions.append(risk_label)

        return np.asarray(adjusted_predictions, dtype=int)

    def find_optimal_threshold(
        self,
        y_val_proba: np.ndarray,
        y_val: pd.Series | np.ndarray,
        min_normal_precision: float = 0.8,
    ) -> tuple[float, dict[str, Any]]:
        """Search the best threshold on validation data.

        Optimization target
        -------------------
        Under the constraint that class-0 precision is at least
        ``min_normal_precision``, maximize the average recall of all risk
        classes. If no threshold satisfies the precision floor, the search
        returns the threshold with the highest normal-class precision and uses
        risk recall as a tie-breaker.
        """

        probability_array = np.asarray(y_val_proba, dtype=float)
        y_true = np.asarray(y_val, dtype=int)
        if probability_array.ndim != 2:
            raise ValueError("y_val_proba must be a 2D probability matrix.")
        if len(probability_array) != len(y_true):
            raise ValueError(
                "y_val_proba and y_val must contain the same number of samples."
            )

        candidate_thresholds = np.round(np.arange(0.50, 0.991, 0.01), 2)
        best_threshold = self.normal_class_threshold
        best_metrics: dict[str, Any] | None = None
        fallback_metrics: dict[str, Any] | None = None

        for threshold in candidate_thresholds:
            self.normal_class_threshold = float(threshold)
            adjusted_predictions = self.adjust_prediction(probability_array)
            metrics = self._evaluate_threshold_candidate(y_true, adjusted_predictions)
            metrics["threshold"] = float(threshold)

            if fallback_metrics is None:
                fallback_metrics = metrics
            else:
                fallback_precision = float(fallback_metrics["normal_precision"])
                current_precision = float(metrics["normal_precision"])
                if (
                    current_precision > fallback_precision
                    or (
                        np.isclose(current_precision, fallback_precision)
                        and float(metrics["risk_recall_mean"])
                        > float(fallback_metrics["risk_recall_mean"])
                    )
                ):
                    fallback_metrics = metrics

            if float(metrics["normal_precision"]) < float(min_normal_precision):
                continue

            if best_metrics is None:
                best_threshold = float(threshold)
                best_metrics = metrics
                continue

            if (
                float(metrics["risk_recall_mean"]) > float(best_metrics["risk_recall_mean"])
                or (
                    np.isclose(
                        float(metrics["risk_recall_mean"]),
                        float(best_metrics["risk_recall_mean"]),
                    )
                    and float(metrics["macro_f1"]) > float(best_metrics["macro_f1"])
                )
            ):
                best_threshold = float(threshold)
                best_metrics = metrics

        if best_metrics is None:
            if fallback_metrics is None:
                raise ValueError("Unable to evaluate any threshold candidate.")
            best_threshold = float(fallback_metrics["threshold"])
            best_metrics = fallback_metrics

        self.normal_class_threshold = self._clip_threshold(best_threshold)
        return best_threshold, best_metrics

    def business_rule_override(
        self,
        user_data: pd.Series | dict[str, Any],
        model_pred: int,
    ) -> int:
        """Apply deterministic risk overrides based on Lending Club fields.

        Rules
        -----
        1. Historical max overdue days > 180 -> label 3
        2. Historical max overdue days in [91, 180] -> label 2
        3. Historical max overdue days in [31, 90] -> label 2
        4. dti > 60 and delinq_2yrs > 0 -> label 2
        5. inq_last_6mths > 10 and dti > 50 -> label 1
        """

        row = self._normalize_user_data(user_data)
        overdue_days = self._estimate_historical_max_overdue_days(row)
        dti = self._safe_float(row.get("dti", 0.0))
        delinq_2yrs = self._safe_float(row.get("delinq_2yrs", 0.0))
        inq_last_6mths = self._safe_float(row.get("inq_last_6mths", 0.0))

        if overdue_days > 180:
            return 3
        if 91 <= overdue_days <= 180:
            return 2
        if 31 <= overdue_days <= 90:
            return 2
        if dti > 60.0 and delinq_2yrs > 0:
            return 2
        if inq_last_6mths > 10 and dti > 50.0:
            return 1
        return int(model_pred)

    def _evaluate_threshold_candidate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, Any]:
        """Compute the threshold-search metrics for one candidate threshold."""

        observed_labels = np.unique(np.concatenate([y_true, y_pred]))
        report = classification_report(
            y_true,
            y_pred,
            labels=observed_labels,
            output_dict=True,
            zero_division=0,
        )

        normal_precision = float(report.get("0", {}).get("precision", 0.0))
        risk_recalls = [
            float(report.get(str(label), {}).get("recall", 0.0))
            for label in observed_labels
            if int(label) != 0
        ]
        risk_recall_mean = float(np.mean(risk_recalls)) if risk_recalls else 0.0

        return {
            "normal_precision": normal_precision,
            "risk_recall_mean": risk_recall_mean,
            "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
            "weighted_f1": float(
                report.get("weighted avg", {}).get("f1-score", 0.0)
            ),
            "predicted_normal_ratio": float(np.mean(y_pred == 0)),
        }

    def _normalize_user_data(
        self,
        user_data: Any,
    ) -> dict[str, Any]:
        """Normalize user data into a plain dictionary."""

        if isinstance(user_data, pd.Series):
            return user_data.to_dict()
        if isinstance(user_data, dict):
            return dict(user_data)
        raise TypeError("user_data must be a pandas Series or a dictionary.")

    def _estimate_historical_max_overdue_days(self, row: dict[str, Any]) -> float:
        """Estimate the maximum overdue-days signal from available raw fields."""

        explicit_candidates = [
            row.get("historical_max_overdue_days"),
            row.get("max_overdue_days"),
            row.get("estimated_overdue_days"),
            row.get("hardship_dpd"),
        ]
        numeric_candidates = [
            self._safe_float(candidate, default=np.nan)
            for candidate in explicit_candidates
        ]
        numeric_candidates = [
            candidate for candidate in numeric_candidates if np.isfinite(candidate)
        ]

        status = str(row.get("loan_status", "")).strip().lower()
        delinq_amnt = max(self._safe_float(row.get("delinq_amnt", 0.0)), 0.0)
        installment = self._safe_float(row.get("installment", np.nan), default=np.nan)

        inferred_from_amount = 0.0
        if np.isfinite(installment) and installment > 0:
            inferred_from_amount = float(np.ceil(delinq_amnt / installment) * 30.0)

        if status in {"charged off", "default"}:
            numeric_candidates.append(181.0)
        elif status == "does not meet the credit policy. status:charged off":
            numeric_candidates.append(181.0)
        elif status == "late (31-120 days)":
            numeric_candidates.append(
                float(np.clip(inferred_from_amount or 60.0, 31.0, 120.0))
            )
        elif status == "late (16-30 days)":
            numeric_candidates.append(23.0)
        elif status == "in grace period":
            numeric_candidates.append(15.0)

        if delinq_amnt > 0 and inferred_from_amount > 0:
            numeric_candidates.append(inferred_from_amount)

        if not numeric_candidates:
            return 0.0
        return float(max(numeric_candidates))

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Convert arbitrary input to float safely."""

        numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric_value):
            return float(default)
        return float(numeric_value)

    def _clip_threshold(self, threshold: float) -> float:
        """Clip a user-provided threshold into a valid probability range."""

        return float(np.clip(float(threshold), 0.0, 1.0))
