"""Model evaluation and explainability utilities for credit risk models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


class CreditModelEvaluator:
    """Evaluate and explain a trained credit risk classification model."""

    def __init__(
        self,
        model,
        X_test: pd.DataFrame | np.ndarray,
        y_test: pd.Series | np.ndarray,
        bad_label: int | None = None,
        top_capture_ratio: float = 0.2,
    ) -> None:
        """Initialize the evaluator."""

        self.model_wrapper = model
        self.estimator = self._resolve_estimator(model)
        self.X_test = self._ensure_dataframe(X_test)
        self.y_test = np.asarray(y_test)
        self.classes_ = self._resolve_classes(model, self.y_test)
        self.bad_label = int(bad_label) if bad_label is not None else int(np.max(self.classes_))
        self.top_capture_ratio = top_capture_ratio
        self.evaluation_results_: Dict[str, Any] | None = None

    def evaluate_classification(self) -> Dict[str, Any]:
        """Compute academic and risk-control evaluation metrics."""

        y_pred = self.estimator.predict(self.X_test)
        y_proba = self.estimator.predict_proba(self.X_test)

        confusion = confusion_matrix(
            self.y_test,
            y_pred,
            labels=self.classes_,
        )

        y_test_bin = label_binarize(self.y_test, classes=self.classes_)
        if y_test_bin.shape[1] == 1:
            y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])

        per_class_auc = {}
        for class_index, class_label in enumerate(self.classes_):
            if len(np.unique(y_test_bin[:, class_index])) < 2:
                per_class_auc[str(class_label)] = np.nan
                continue
            per_class_auc[str(class_label)] = float(
                roc_auc_score(y_test_bin[:, class_index], y_proba[:, class_index])
            )

        ks_value = self._compute_ks(y_proba)
        bad_rate_capture = self._compute_bad_rate_capture(y_proba)

        results = {
            "accuracy": float(accuracy_score(self.y_test, y_pred)),
            "precision_weighted": float(
                precision_score(
                    self.y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "recall_weighted": float(
                recall_score(
                    self.y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "f1_weighted": float(
                f1_score(
                    self.y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "confusion_matrix": confusion.tolist(),
            "classification_report": classification_report(
                self.y_test,
                y_pred,
                output_dict=True,
                zero_division=0,
            ),
            "auc_roc_per_class": per_class_auc,
            "ks_value": float(ks_value),
            "bad_rate_capture": float(bad_rate_capture),
        }
        self.evaluation_results_ = results
        return results

    def evaluate_regression(self) -> Dict[str, Any]:
        """Compute regression metrics and save regression diagnostic plots."""

        y_pred = np.asarray(self.estimator.predict(self.X_test), dtype=float)
        y_true = np.asarray(self.y_test, dtype=float)

        non_zero_mask = np.abs(y_true) > 1e-8
        if np.any(non_zero_mask):
            mape = np.mean(
                np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
            )
        else:
            mape = 0.0

        scatter_path = FIGURE_DIR / "credit_scoring_prediction_vs_actual.png"
        error_hist_path = FIGURE_DIR / "credit_scoring_error_distribution.png"
        self._plot_regression_scatter(y_true, y_pred, scatter_path)
        self._plot_regression_error_distribution(y_true, y_pred, error_hist_path)

        results = {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mape": float(mape),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "prediction_vs_actual_plot": str(scatter_path),
            "error_distribution_plot": str(error_hist_path),
        }
        self.evaluation_results_ = results
        return results

    def plot_evaluation_results(self) -> Dict[str, str]:
        """Plot confusion matrix, ROC curves, and PR curves."""

        y_pred = self.estimator.predict(self.X_test)
        y_proba = self.estimator.predict_proba(self.X_test)
        y_test_bin = label_binarize(self.y_test, classes=self.classes_)
        if y_test_bin.shape[1] == 1:
            y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])

        confusion_path = FIGURE_DIR / "risk_model_confusion_matrix.png"
        roc_path = FIGURE_DIR / "risk_model_roc_curves.png"
        pr_path = FIGURE_DIR / "risk_model_pr_curves.png"

        self._plot_confusion_matrix(y_pred, confusion_path)
        self._plot_roc_curves(y_test_bin, y_proba, roc_path)
        self._plot_pr_curves(y_test_bin, y_proba, pr_path)

        return {
            "confusion_matrix": str(confusion_path),
            "roc_curves": str(roc_path),
            "pr_curves": str(pr_path),
        }

    def explain_model(self, X_test: pd.DataFrame | np.ndarray) -> Dict[str, Any]:
        """Run SHAP explainability and save explainability artifacts."""

        feature_frame = self._ensure_dataframe(X_test)
        sample_size = min(500, len(feature_frame))
        feature_sample = feature_frame.iloc[:sample_size].copy()

        explainer = shap.Explainer(self.estimator, feature_sample)
        shap_values = explainer(feature_sample)

        target_class_index = self._get_class_index(self.bad_label)
        shap_matrix = self._extract_class_shap_matrix(shap_values, target_class_index)
        base_value = self._extract_base_value(shap_values, target_class_index)

        mean_abs_importance = np.abs(shap_matrix).mean(axis=0)
        importance_frame = pd.DataFrame(
            {
                "feature": feature_sample.columns,
                "mean_abs_shap": mean_abs_importance,
            }
        ).sort_values("mean_abs_shap", ascending=False)

        importance_csv_path = FIGURE_DIR / "shap_feature_importance.csv"
        summary_plot_path = FIGURE_DIR / "shap_summary_plot.png"
        force_plot_path = FIGURE_DIR / "shap_force_plot.html"

        importance_frame.to_csv(importance_csv_path, index=False)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_matrix,
            feature_sample,
            show=False,
            max_display=20,
        )
        plt.tight_layout()
        plt.savefig(summary_plot_path, dpi=200, bbox_inches="tight")
        plt.close()

        force_plot = shap.force_plot(
            base_value=base_value,
            shap_values=shap_matrix[0],
            features=feature_sample.iloc[0],
            matplotlib=False,
        )
        shap.save_html(str(force_plot_path), force_plot)

        return {
            "feature_importance_ranking": importance_frame.head(20).to_dict(
                orient="records"
            ),
            "feature_importance_csv": str(importance_csv_path),
            "summary_plot": str(summary_plot_path),
            "force_plot": str(force_plot_path),
        }

    def _resolve_estimator(self, model):
        """Resolve the actual fitted estimator from a wrapper or raw model."""

        if hasattr(model, "best_model_") and model.best_model_ is not None:
            return model.best_model_
        if hasattr(model, "model") and model.model is not None:
            return model.model
        return model

    def _resolve_classes(self, model, y_test: np.ndarray) -> np.ndarray:
        """Resolve model classes from the wrapper or the target array."""

        if hasattr(model, "classes_") and model.classes_ is not None:
            return np.asarray(model.classes_)
        if hasattr(self.estimator, "classes_"):
            return np.asarray(self.estimator.classes_)
        return np.unique(y_test)

    def _ensure_dataframe(
        self,
        X: object,
    ) -> pd.DataFrame:
        """Convert the input to a pandas DataFrame if needed."""

        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, np.ndarray):
            columns = [f"feature_{index:04d}" for index in range(X.shape[1])]
            return pd.DataFrame(X, columns=columns)
        raise TypeError("Input features must be a pandas DataFrame or numpy array.")

    def _get_class_index(self, class_label: int) -> int:
        """Map a class label to its probability column index."""

        matching_indices = np.where(self.classes_ == class_label)[0]
        if len(matching_indices) == 0:
            raise ValueError(f"Class label {class_label} not found in model classes.")
        return int(matching_indices[0])

    def _compute_ks(self, y_proba: np.ndarray) -> float:
        """Compute KS using the bad-class one-vs-rest score."""

        bad_index = self._get_class_index(self.bad_label)
        bad_scores = y_proba[:, bad_index]
        y_binary = (self.y_test == self.bad_label).astype(int)

        if len(np.unique(y_binary)) < 2:
            return 0.0

        fpr, tpr, _ = roc_curve(y_binary, bad_scores)
        return float(np.max(tpr - fpr))

    def _compute_bad_rate_capture(self, y_proba: np.ndarray) -> float:
        """Compute bad-rate capture in the top score bucket."""

        bad_index = self._get_class_index(self.bad_label)
        bad_scores = y_proba[:, bad_index]
        y_binary = (self.y_test == self.bad_label).astype(int)

        total_bad = int(y_binary.sum())
        if total_bad == 0:
            return 0.0

        ranking_frame = pd.DataFrame(
            {
                "bad_score": bad_scores,
                "is_bad": y_binary,
            }
        ).sort_values("bad_score", ascending=False)

        top_n = max(1, int(len(ranking_frame) * self.top_capture_ratio))
        captured_bad = int(ranking_frame.head(top_n)["is_bad"].sum())
        return float(captured_bad / total_bad)

    def _plot_confusion_matrix(self, y_pred: np.ndarray, output_path: Path) -> None:
        """Save a confusion matrix heatmap."""

        matrix = confusion_matrix(self.y_test, y_pred, labels=self.classes_)
        fig, ax = plt.subplots(figsize=(8, 6))
        image = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(image, ax=ax)
        ax.set_xticks(np.arange(len(self.classes_)))
        ax.set_yticks(np.arange(len(self.classes_)))
        ax.set_xticklabels(self.classes_)
        ax.set_yticklabels(self.classes_)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")

        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                ax.text(
                    col_index,
                    row_index,
                    matrix[row_index, col_index],
                    ha="center",
                    va="center",
                    color="black",
                )

        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _plot_roc_curves(
        self,
        y_test_bin: np.ndarray,
        y_proba: np.ndarray,
        output_path: Path,
    ) -> None:
        """Save one-vs-rest ROC curves for each class."""

        fig, ax = plt.subplots(figsize=(9, 7))
        for class_index, class_label in enumerate(self.classes_):
            if len(np.unique(y_test_bin[:, class_index])) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_test_bin[:, class_index], y_proba[:, class_index])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"Class {class_label} (AUC={roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("One-vs-Rest ROC Curves")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _plot_pr_curves(
        self,
        y_test_bin: np.ndarray,
        y_proba: np.ndarray,
        output_path: Path,
    ) -> None:
        """Save one-vs-rest precision-recall curves for each class."""

        fig, ax = plt.subplots(figsize=(9, 7))
        for class_index, class_label in enumerate(self.classes_):
            if len(np.unique(y_test_bin[:, class_index])) < 2:
                continue
            precision, recall, _ = precision_recall_curve(
                y_test_bin[:, class_index],
                y_proba[:, class_index],
            )
            pr_auc = auc(recall, precision)
            ax.plot(recall, precision, label=f"Class {class_label} (AUC={pr_auc:.3f})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("One-vs-Rest PR Curves")
        ax.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _extract_class_shap_matrix(
        self,
        shap_values,
        class_index: int,
    ) -> np.ndarray:
        """Extract a 2D SHAP matrix for the selected class."""

        values = shap_values.values
        if isinstance(values, list):
            return np.asarray(values[class_index])
        if values.ndim == 3:
            return np.asarray(values[:, :, class_index])
        return np.asarray(values)

    def _extract_base_value(self, shap_values, class_index: int) -> float:
        """Extract the base value for the selected class."""

        base_values = shap_values.base_values
        if isinstance(base_values, list):
            return float(np.asarray(base_values[class_index]).mean())
        base_array = np.asarray(base_values)
        if base_array.ndim == 2:
            return float(base_array[:, class_index].mean())
        if base_array.ndim == 1:
            return float(base_array.mean())
        return float(base_array)

    def _plot_regression_scatter(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: Path,
    ) -> None:
        """Save a prediction-vs-actual scatter plot."""

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_true, y_pred, alpha=0.5, color="#4C72B0")
        min_value = float(min(np.min(y_true), np.min(y_pred)))
        max_value = float(max(np.max(y_true), np.max(y_pred)))
        ax.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="gray")
        ax.set_xlabel("Actual Credit Limit")
        ax.set_ylabel("Predicted Credit Limit")
        ax.set_title("Predicted vs Actual Credit Limits")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _plot_regression_error_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: Path,
    ) -> None:
        """Save a prediction error distribution plot."""

        residuals = y_pred - y_true
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(residuals, bins=30, color="#55A868", alpha=0.8)
        ax.axvline(0.0, color="gray", linestyle="--")
        ax.set_xlabel("Prediction Error")
        ax.set_ylabel("Frequency")
        ax.set_title("Prediction Error Distribution")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
