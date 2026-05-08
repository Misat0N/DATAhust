"""Model evaluation and explainability utilities for credit risk models."""

from __future__ import annotations

from itertools import combinations
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

DEFAULT_CLASS_NAMES = ["正常类", "关注类", "次级/可疑类", "损失类"]

plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


class CreditModelEvaluator:
    """Evaluate classification and regression models in the credit domain.

    The class is backward compatible with the earlier project interface while
    exposing a new imbalanced-multiclass diagnosis workflow required by the
    current model review stage.
    """

    def __init__(
        self,
        model,
        X_test: pd.DataFrame | np.ndarray,
        y_test: pd.Series | np.ndarray,
        class_names: list[str] | None = None,
    ) -> None:
        """Initialize the evaluator.

        Parameters
        ----------
        model:
            A fitted estimator or a project wrapper that exposes `best_model_`
            or `model`.
        X_test:
            Test feature matrix.
        y_test:
            Test labels or regression targets.
        class_names:
            Optional display names for classes. If omitted, the evaluator uses
            the default five-grade credit-risk names and automatically truncates
            or pads them to match the fitted model classes.
        """

        self.model_wrapper = model
        self.estimator = self._resolve_estimator(model)
        self.X_test = self._ensure_dataframe(X_test)
        self.y_test = np.asarray(y_test)
        self.top_capture_ratio = 0.2
        self.evaluation_results_: Dict[str, Any] | None = None
        self.estimator_classes_ = self._resolve_estimator_classes(model)
        self.classes_ = self._resolve_classes(self.estimator_classes_, self.y_test)
        self.class_names_ = self._resolve_class_names(class_names, self.classes_)

    def evaluate_imbalanced_multiclass(self) -> Dict[str, Any]:
        """Evaluate a classification model for imbalanced multiclass tasks.

        The output prioritizes per-class precision/recall/F1, macro-F1,
        weighted-F1, multiclass OVO AUC, KS, bad capture rate, and the
        confusion matrix. Accuracy is kept only as an auxiliary compatibility
        field for existing project code.
        """

        self._ensure_classifier_supports_proba()

        y_pred = np.asarray(self.estimator.predict(self.X_test))
        y_proba = self._align_predicted_probabilities(
            np.asarray(self.estimator.predict_proba(self.X_test), dtype=float)
        )
        y_test_bin = label_binarize(self.y_test, classes=self.classes_)

        if y_test_bin.shape[1] == 1:
            y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])

        confusion = confusion_matrix(
            self.y_test,
            y_pred,
            labels=self.classes_,
        )

        report_by_name = classification_report(
            self.y_test,
            y_pred,
            labels=self.classes_,
            target_names=self.class_names_,
            output_dict=True,
            zero_division=0,
        )
        report_by_label = classification_report(
            self.y_test,
            y_pred,
            labels=self.classes_,
            output_dict=True,
            zero_division=0,
        )

        auc_per_class = self._compute_ovr_auc_per_class(y_test_bin, y_proba)
        ks_per_class = self._compute_ks_per_class(y_test_bin, y_proba)
        capture_per_class = self._compute_bad_capture_per_class(y_test_bin, y_proba)

        macro_f1 = float(
            f1_score(self.y_test, y_pred, average="macro", zero_division=0)
        )
        weighted_f1 = float(
            f1_score(self.y_test, y_pred, average="weighted", zero_division=0)
        )
        precision_weighted = float(
            precision_score(self.y_test, y_pred, average="weighted", zero_division=0)
        )
        recall_weighted = float(
            recall_score(self.y_test, y_pred, average="weighted", zero_division=0)
        )

        dominant_prediction_ratio = float(
            pd.Series(y_pred).value_counts(normalize=True).max()
        )

        results = {
            "accuracy": float(accuracy_score(self.y_test, y_pred)),
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": weighted_f1,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "f1_gap_weighted_minus_macro": float(weighted_f1 - macro_f1),
            "classification_report": report_by_label,
            "classification_report_named": report_by_name,
            "confusion_matrix": confusion.tolist(),
            "auc_roc_per_class": auc_per_class,
            "auc_roc_ovo_macro": self._compute_multiclass_auc(y_proba, average="macro"),
            "auc_roc_ovo_weighted": self._compute_multiclass_auc(
                y_proba,
                average="weighted",
            ),
            "ks_per_class": ks_per_class,
            "ks_value": float(np.nanmean(list(ks_per_class.values()))),
            "bad_rate_capture_per_class": capture_per_class,
            "bad_rate_capture": float(np.nanmean(list(capture_per_class.values()))),
            "predicted_class_distribution": self._compute_prediction_distribution(y_pred),
            "true_class_distribution": self._compute_prediction_distribution(self.y_test),
            "dominant_prediction_ratio": dominant_prediction_ratio,
            "class_names": self.class_names_,
            "class_labels": self.classes_.tolist(),
            "plot_paths": {
                "confusion_matrix": "results/figures/confusion_matrix_diagnosis.png",
                "class_metrics": "results/figures/class_metrics_comparison.png",
                "multiclass_roc_curve": "results/figures/multiclass_roc_curve.png",
            },
        }
        self.evaluation_results_ = results
        return results

    def evaluate_classification(self) -> Dict[str, Any]:
        """Backward-compatible wrapper for the classification evaluator."""

        return self.evaluate_imbalanced_multiclass()

    def plot_confusion_matrix(self) -> str:
        """Plot the confusion matrix diagnosis heatmap.

        The figure is saved to `results/figures/confusion_matrix_diagnosis.png`
        with raw count annotations in every cell.
        """

        results = self._ensure_classification_results()
        matrix = np.asarray(results["confusion_matrix"], dtype=int)
        output_path = FIGURE_DIR / "confusion_matrix_diagnosis.png"

        fig, ax = plt.subplots(figsize=(8, 6))
        image = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(image, ax=ax)
        ax.set_xticks(np.arange(len(self.class_names_)))
        ax.set_yticks(np.arange(len(self.class_names_)))
        ax.set_xticklabels(self.class_names_, rotation=30, ha="right")
        ax.set_yticklabels(self.class_names_)
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")
        ax.set_title("风险分类混淆矩阵诊断")

        max_value = max(matrix.max(), 1)
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                text_color = "white" if matrix[row_index, col_index] > max_value / 2 else "black"
                ax.text(
                    col_index,
                    row_index,
                    f"{matrix[row_index, col_index]:,}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10,
                )

        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return "results/figures/confusion_matrix_diagnosis.png"

    def plot_classification_metrics(self) -> str:
        """Plot per-class precision, recall, and F1 comparison bars."""

        results = self._ensure_classification_results()
        report = results["classification_report_named"]
        output_path = FIGURE_DIR / "class_metrics_comparison.png"

        precision_values = []
        recall_values = []
        f1_values = []
        for class_name in self.class_names_:
            class_result = report.get(class_name, {})
            precision_values.append(float(class_result.get("precision", 0.0)))
            recall_values.append(float(class_result.get("recall", 0.0)))
            f1_values.append(float(class_result.get("f1-score", 0.0)))

        positions = np.arange(len(self.class_names_))
        bar_width = 0.25

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(positions - bar_width, precision_values, width=bar_width, label="Precision")
        ax.bar(positions, recall_values, width=bar_width, label="Recall")
        ax.bar(positions + bar_width, f1_values, width=bar_width, label="F1-Score")

        ax.set_xticks(positions)
        ax.set_xticklabels(self.class_names_, rotation=25, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Score")
        ax.set_title("各类别 Precision / Recall / F1 对比")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return "results/figures/class_metrics_comparison.png"

    def plot_roc_curve_multiclass(self) -> str:
        """Plot pairwise ROC curves for multiclass diagnosis.

        The core numeric ROC metric remains multiclass OVO AUC, while the plot
        visualizes every valid class-pair ROC curve to show which boundaries are
        actually separable.
        """

        self._ensure_classifier_supports_proba()
        y_proba = self._align_predicted_probabilities(
            np.asarray(self.estimator.predict_proba(self.X_test), dtype=float)
        )
        output_path = FIGURE_DIR / "multiclass_roc_curve.png"

        fig, ax = plt.subplots(figsize=(10, 7))
        plotted_curve_count = 0

        for left_index, right_index in combinations(range(len(self.classes_)), 2):
            left_label = self.classes_[left_index]
            right_label = self.classes_[right_index]
            pair_mask = np.isin(self.y_test, [left_label, right_label])

            if pair_mask.sum() < 2:
                continue

            pair_y = self.y_test[pair_mask]
            binary_y = (pair_y == right_label).astype(int)
            if len(np.unique(binary_y)) < 2:
                continue

            left_scores = y_proba[pair_mask, left_index]
            right_scores = y_proba[pair_mask, right_index]
            pair_scores = right_scores / np.clip(left_scores + right_scores, 1e-12, None)

            fpr, tpr, _ = roc_curve(binary_y, pair_scores)
            pair_auc = auc(fpr, tpr)
            ax.plot(
                fpr,
                tpr,
                label=(
                    f"{self.class_names_[left_index]} vs "
                    f"{self.class_names_[right_index]} (AUC={pair_auc:.3f})"
                ),
            )
            plotted_curve_count += 1

        if plotted_curve_count == 0:
            raise ValueError("ROC curves cannot be plotted because valid class pairs are missing.")

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("多分类一对一 ROC 曲线")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3, linestyle="--")

        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return "results/figures/multiclass_roc_curve.png"

    def plot_evaluation_results(self) -> Dict[str, str]:
        """Backward-compatible plot wrapper for the main diagnosis figures."""

        return {
            "confusion_matrix": self.plot_confusion_matrix(),
            "class_metrics": self.plot_classification_metrics(),
            "roc_curves": self.plot_roc_curve_multiclass(),
        }

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
            "prediction_vs_actual_plot": "results/figures/credit_scoring_prediction_vs_actual.png",
            "error_distribution_plot": "results/figures/credit_scoring_error_distribution.png",
        }
        self.evaluation_results_ = results
        return results

    def explain_model(self, X_test: pd.DataFrame | np.ndarray) -> Dict[str, Any]:
        """Run SHAP explainability and save explainability artifacts."""

        if not hasattr(self.estimator, "predict"):
            raise TypeError("The provided model does not support SHAP explanation.")

        feature_frame = self._ensure_dataframe(X_test)
        sample_size = min(500, len(feature_frame))
        feature_sample = feature_frame.iloc[:sample_size].copy()

        explainer = shap.Explainer(self.estimator, feature_sample)
        shap_values = explainer(feature_sample)

        target_class_index = self._get_target_class_index()
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
            "feature_importance_csv": "results/figures/shap_feature_importance.csv",
            "summary_plot": "results/figures/shap_summary_plot.png",
            "force_plot": "results/figures/shap_force_plot.html",
        }

    def _ensure_classifier_supports_proba(self) -> None:
        """Validate that the estimator is suitable for classification diagnosis."""

        if not hasattr(self.estimator, "predict") or not hasattr(self.estimator, "predict_proba"):
            raise TypeError(
                "CreditModelEvaluator classification diagnosis requires a fitted "
                "classifier that supports both predict() and predict_proba()."
            )

    def _ensure_classification_results(self) -> Dict[str, Any]:
        """Return cached classification results or compute them on demand."""

        if self.evaluation_results_ is None or "classification_report" not in self.evaluation_results_:
            return self.evaluate_imbalanced_multiclass()
        return self.evaluation_results_

    def _resolve_estimator(self, model):
        """Resolve the actual fitted estimator from a wrapper or raw model."""

        if hasattr(model, "best_model_") and model.best_model_ is not None:
            return model.best_model_
        if hasattr(model, "model") and model.model is not None:
            return model.model
        return model

    def _resolve_estimator_classes(self, model) -> np.ndarray:
        """Resolve the class labels exposed by the fitted estimator."""

        if hasattr(model, "classes_") and model.classes_ is not None:
            return np.asarray(model.classes_)
        if hasattr(self.estimator, "classes_"):
            return np.asarray(self.estimator.classes_)
        return np.array([], dtype=np.asarray(self.y_test).dtype)

    def _resolve_classes(
        self,
        estimator_classes: np.ndarray,
        y_test: np.ndarray,
    ) -> np.ndarray:
        """Use the union of fitted and observed labels for robust evaluation."""

        observed_classes = np.unique(y_test)
        if estimator_classes.size == 0:
            return observed_classes
        return np.unique(np.concatenate([np.asarray(estimator_classes), observed_classes]))

    def _align_predicted_probabilities(self, y_proba: np.ndarray) -> np.ndarray:
        """Align probability columns to the full evaluation class order.

        If the fitted estimator has not seen every label that still appears in
        `y_test`, scikit-learn metrics can fail when `labels=` includes those
        unseen classes. Filling the missing class columns with zeros keeps the
        evaluation stable while still reflecting that the model cannot predict
        those labels.
        """

        if y_proba.ndim != 2:
            raise ValueError("predict_proba output must be a 2D array.")
        if len(self.estimator_classes_) == 0:
            if y_proba.shape[1] != len(self.classes_):
                raise ValueError(
                    "predict_proba column count does not match resolved class count."
                )
            return y_proba
        if np.array_equal(self.estimator_classes_, self.classes_):
            return y_proba

        aligned = np.zeros((y_proba.shape[0], len(self.classes_)), dtype=float)
        class_to_index = {
            class_label: index for index, class_label in enumerate(self.classes_)
        }
        for source_index, class_label in enumerate(self.estimator_classes_):
            target_index = class_to_index.get(class_label)
            if target_index is not None:
                aligned[:, target_index] = y_proba[:, source_index]
        return aligned

    def _resolve_class_names(
        self,
        class_names: list[str] | None,
        classes: np.ndarray,
    ) -> list[str]:
        """Align class display names with the actual model classes."""

        base_names = list(class_names or DEFAULT_CLASS_NAMES)
        resolved_names = []
        for class_label in classes:
            if isinstance(class_label, (int, np.integer)) and 0 <= int(class_label) < len(base_names):
                resolved_names.append(base_names[int(class_label)])
            else:
                resolved_names.append(f"类别{class_label}")

        if len(resolved_names) < len(classes):
            for class_label in classes[len(resolved_names):]:
                resolved_names.append(f"类别{class_label}")
        return resolved_names

    def _compute_ovr_auc_per_class(
        self,
        y_test_bin: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, float]:
        """Compute one-vs-rest AUC for each class."""

        auc_per_class: Dict[str, float] = {}
        for class_index, class_name in enumerate(self.class_names_):
            if len(np.unique(y_test_bin[:, class_index])) < 2:
                auc_per_class[class_name] = float("nan")
                continue
            auc_per_class[class_name] = float(
                roc_auc_score(y_test_bin[:, class_index], y_proba[:, class_index])
            )
        return auc_per_class

    def _compute_multiclass_auc(
        self,
        y_proba: np.ndarray,
        average: str,
    ) -> float:
        """Compute macro or weighted multiclass OVO AUC."""

        if len(self.classes_) < 2:
            return float("nan")
        if len(self.classes_) == 2:
            positive_scores = y_proba[:, 1]
            return float(roc_auc_score(self.y_test, positive_scores))
        return float(
            roc_auc_score(
                self.y_test,
                y_proba,
                labels=self.classes_,
                multi_class="ovo",
                average=average,
            )
        )

    def _compute_ks_per_class(
        self,
        y_test_bin: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, float]:
        """Compute one-vs-rest KS for each class."""

        ks_per_class: Dict[str, float] = {}
        for class_index, class_name in enumerate(self.class_names_):
            if len(np.unique(y_test_bin[:, class_index])) < 2:
                ks_per_class[class_name] = float("nan")
                continue
            fpr, tpr, _ = roc_curve(y_test_bin[:, class_index], y_proba[:, class_index])
            ks_per_class[class_name] = float(np.max(tpr - fpr))
        return ks_per_class

    def _compute_bad_capture_per_class(
        self,
        y_test_bin: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, float]:
        """Compute top-bucket bad capture ratio for each class."""

        capture_per_class: Dict[str, float] = {}
        for class_index, class_name in enumerate(self.class_names_):
            binary_target = y_test_bin[:, class_index]
            total_positive = int(binary_target.sum())
            if total_positive == 0:
                capture_per_class[class_name] = float("nan")
                continue

            ranking_frame = pd.DataFrame(
                {
                    "score": y_proba[:, class_index],
                    "target": binary_target,
                }
            ).sort_values("score", ascending=False)

            top_n = max(1, int(len(ranking_frame) * self.top_capture_ratio))
            captured = int(ranking_frame.head(top_n)["target"].sum())
            capture_per_class[class_name] = float(captured / total_positive)
        return capture_per_class

    def _compute_prediction_distribution(
        self,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """Return normalized class distribution keyed by display names."""

        distribution = pd.Series(labels).value_counts(normalize=True)
        named_distribution: Dict[str, float] = {}
        for class_label, class_name in zip(self.classes_, self.class_names_):
            named_distribution[class_name] = float(distribution.get(class_label, 0.0))
        return named_distribution

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

    def _get_target_class_index(self) -> int:
        """Pick the last class as the default target for SHAP class explanation."""

        if len(self.classes_) == 0:
            raise ValueError("No class labels are available for SHAP explanation.")
        return int(len(self.classes_) - 1)

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
