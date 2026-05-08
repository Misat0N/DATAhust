#!/usr/bin/env python3
"""Low-resource runner for the final four-class model comparison.

This script is designed for machines that cannot sustain the full notebook
workload. It runs the comparison incrementally and saves intermediate results
after each model group, so you can execute only the groups you need.

Default behavior
----------------
- Prefer ``purified_train/val/test.csv`` when available.
- Run light-to-medium workload groups by default.
- Keep the heavy sampling group optional.
- Allow explicit row limits for train/validation/test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.features.feature_selector import CreditFeatureSelector
from src.features.preprocessor import CreditDataPreprocessor
from src.features.sampler import CreditSampler
from src.models.model_evaluator import CreditModelEvaluator
from src.models.risk_classifier import CreditRiskClassifier
from src.models.threshold_adjuster import RiskThresholdAdjuster
from src.models.two_stage_risk_model import TwoStageCreditRiskModel


DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"
MODEL_DIR = PROJECT_ROOT / "src" / "models"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["正常类", "关注类", "次级/可疑类", "损失类"]
MANUAL_CLASS_WEIGHT = {0: 1, 1: 5, 2: 20, 3: 35}


class FixedPredictionModel:
    """Wrap fixed predictions so the shared evaluator can score them."""

    def __init__(self, y_pred: np.ndarray, y_proba: np.ndarray) -> None:
        self._y_pred = np.asarray(y_pred, dtype=int)
        self._y_proba = np.asarray(y_proba, dtype=float)
        self.classes_ = np.arange(self._y_proba.shape[1])

    def predict(self, X):
        return self._y_pred

    def predict_proba(self, X):
        return self._y_proba


def resolve_split_path(split_name: str) -> Path:
    """Prefer the purified split when present."""

    purified_path = DATA_DIR / f"purified_{split_name}.csv"
    if purified_path.exists():
        return purified_path
    return DATA_DIR / f"{split_name}.csv"


def maybe_limit_frame(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    """Optionally cut a dataframe to the first N rows."""

    if limit is None or limit <= 0:
        return frame
    return frame.iloc[: min(limit, len(frame))].copy()


def load_splits(
    train_limit: int | None,
    val_limit: int | None,
    test_limit: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the train/validation/test splits with optional row caps."""

    train_df = pd.read_csv(resolve_split_path("train"), low_memory=False)
    val_df = pd.read_csv(resolve_split_path("val"), low_memory=False)
    test_df = pd.read_csv(resolve_split_path("test"), low_memory=False)

    train_df = maybe_limit_frame(train_df, train_limit)
    val_df = maybe_limit_frame(val_df, val_limit)
    test_df = maybe_limit_frame(test_df, test_limit)
    return train_df, val_df, test_df


def prepare_selected_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    top_k_features: int,
    n_estimators: int,
) -> dict[str, object]:
    """Prepare leak-free processed and selected features."""

    y_train = train_df["preloan_risk_label"].astype(int)
    y_val = val_df["preloan_risk_label"].astype(int)
    y_test = test_df["preloan_risk_label"].astype(int)

    preprocessor = CreditDataPreprocessor(target_column="preloan_risk_label")
    X_train_processed = preprocessor.fit_transform(train_df)
    X_val_processed = preprocessor.transform(val_df)
    X_test_processed = preprocessor.transform(test_df)

    selector = CreditFeatureSelector(
        top_k_features=top_k_features,
        use_pca=False,
        n_estimators=n_estimators,
    )
    X_train_selected = selector.fit_transform(X_train_processed, y_train)
    X_val_selected = selector.transform(X_val_processed)
    X_test_selected = selector.transform(X_test_processed)

    categorical_feature_indices = [
        index
        for index, column in enumerate(X_train_processed.columns)
        if "=" in column and not column.endswith("__target_encoded")
    ]

    return {
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "X_train_processed": X_train_processed,
        "X_val_processed": X_val_processed,
        "X_test_processed": X_test_processed,
        "X_train_selected": X_train_selected,
        "X_val_selected": X_val_selected,
        "X_test_selected": X_test_selected,
        "categorical_feature_indices": categorical_feature_indices,
    }


def evaluate_model(
    model,
    X_test,
    y_test: pd.Series,
) -> dict[str, object]:
    """Evaluate a classifier with the project evaluator."""

    metrics = CreditModelEvaluator(
        model=model,
        X_test=X_test,
        y_test=y_test,
        class_names=CLASS_NAMES,
    ).evaluate_imbalanced_multiclass()
    return metrics


def evaluate_adjusted_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    X_eval,
) -> dict[str, object]:
    """Evaluate externally adjusted predictions."""

    wrapped_model = FixedPredictionModel(y_pred, y_proba)
    return evaluate_model(wrapped_model, X_eval, y_true)


def compute_high_risk_recall(metric_payload: dict[str, object]) -> float:
    """Compute the mean recall across all non-normal classes."""

    report = metric_payload["classification_report_named"]
    risk_recalls = [
        float(report.get(class_name, {}).get("recall", 0.0))
        for class_name in metric_payload["class_names"]
        if class_name != "正常类"
    ]
    return float(np.mean(risk_recalls)) if risk_recalls else 0.0


def summarize_result(
    model_name: str,
    metrics: dict[str, object],
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a compact summary row for final comparison."""

    report = metrics["classification_report_named"]
    summary = {
        "model_name": model_name,
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "ks_value": float(metrics["ks_value"]),
        "normal_precision": float(report.get("正常类", {}).get("precision", 0.0)),
        "high_risk_recall": compute_high_risk_recall(metrics),
    }
    if extra:
        summary.update(extra)
    return summary


def apply_business_rules(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    adjuster: RiskThresholdAdjuster,
) -> np.ndarray:
    """Apply deterministic overrides row by row."""

    final_predictions = []
    for (_, row), prediction in zip(frame.iterrows(), predictions):
        final_predictions.append(adjuster.business_rule_override(row, int(prediction)))
    return np.asarray(final_predictions, dtype=int)


def run_baseline(features: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    """Run the single-stage baseline model."""

    model = CreditRiskClassifier(model_type="lightgbm", class_weight_mode="none")
    model.fit(features["X_train_selected"], features["y_train"])
    metrics = evaluate_model(model, features["X_test_selected"], features["y_test"])
    return summarize_result("原始单阶段4分类", metrics), metrics


def run_weighted(features: dict[str, object], test_df: pd.DataFrame) -> tuple[dict[str, object], dict[str, object]]:
    """Run the weighted single-stage model with threshold and business rules."""

    model = CreditRiskClassifier(
        model_type="lightgbm",
        class_weight_mode="manual",
        custom_class_weight=MANUAL_CLASS_WEIGHT,
    )
    model.fit(features["X_train_selected"], features["y_train"])

    val_proba = model.predict_proba(features["X_val_selected"])
    test_proba = model.predict_proba(features["X_test_selected"])

    adjuster = RiskThresholdAdjuster(normal_class_threshold=0.9)
    best_threshold, _ = adjuster.find_optimal_threshold(
        y_val_proba=val_proba,
        y_val=features["y_val"],
        min_normal_precision=0.8,
    )
    threshold_pred = adjuster.adjust_prediction(test_proba)
    final_pred = apply_business_rules(test_df, threshold_pred, adjuster)

    metrics = evaluate_adjusted_predictions(
        features["y_test"],
        final_pred,
        test_proba,
        features["X_test_selected"],
    )
    return summarize_result(
        "单阶段+权重+阈值规则",
        metrics,
        extra={"normal_threshold": float(best_threshold)},
    ), metrics


def run_sampled(
    features: dict[str, object],
    test_df: pd.DataFrame,
    top_k_features: int,
    n_estimators: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Run the optional sampling-heavy single-stage model."""

    sampler = CreditSampler(
        categorical_features=features["categorical_feature_indices"]
    )
    X_train_sampled, y_train_sampled = sampler.fit_resample(
        features["X_train_processed"],
        features["y_train"],
        sampling_strategy="auto",
        max_oversample_ratio=5,
    )
    sampler.plot_sample_distribution(features["y_train"], y_train_sampled)

    selector = CreditFeatureSelector(
        top_k_features=top_k_features,
        use_pca=False,
        n_estimators=n_estimators,
    )
    X_train_selected = selector.fit_transform(X_train_sampled, y_train_sampled)
    X_val_selected = selector.transform(features["X_val_processed"])
    X_test_selected = selector.transform(features["X_test_processed"])

    model = CreditRiskClassifier(
        model_type="lightgbm",
        class_weight_mode="manual",
        custom_class_weight=MANUAL_CLASS_WEIGHT,
    )
    model.fit(X_train_selected, y_train_sampled)

    val_proba = model.predict_proba(X_val_selected)
    test_proba = model.predict_proba(X_test_selected)

    adjuster = RiskThresholdAdjuster(normal_class_threshold=0.9)
    best_threshold, _ = adjuster.find_optimal_threshold(
        y_val_proba=val_proba,
        y_val=features["y_val"],
        min_normal_precision=0.8,
    )
    threshold_pred = adjuster.adjust_prediction(test_proba)
    final_pred = apply_business_rules(test_df, threshold_pred, adjuster)

    metrics = evaluate_adjusted_predictions(
        features["y_test"],
        final_pred,
        test_proba,
        X_test_selected,
    )
    return summarize_result(
        "单阶段+权重+采样+阈值规则",
        metrics,
        extra={"normal_threshold": float(best_threshold)},
    ), metrics


def run_two_stage(features: dict[str, object], test_df: pd.DataFrame) -> tuple[dict[str, object], dict[str, object]]:
    """Run the two-stage model."""

    model = TwoStageCreditRiskModel()
    model.fit(features["X_train_selected"], features["y_train"])
    test_proba = model.predict_proba(features["X_test_selected"])
    final_pred = model.predict_with_threshold_and_rule(
        features["X_test_selected"],
        user_data=test_df,
        normal_threshold=model.get_threshold(),
    )
    metrics = evaluate_adjusted_predictions(
        features["y_test"],
        final_pred,
        test_proba,
        features["X_test_selected"],
    )
    return summarize_result(
        "两阶段分级+权重+阈值规则",
        metrics,
        extra={"normal_threshold": float(model.get_threshold())},
    ), metrics


def save_outputs(
    summaries: list[dict[str, object]],
    details: dict[str, dict[str, object]],
    figure_name: str,
    report_name: str,
) -> None:
    """Save comparison outputs to JSON, radar figure, and Markdown report."""

    if not summaries:
        return

    comparison_df = pd.DataFrame(summaries)
    radar_metrics = ["macro_f1", "high_risk_recall", "normal_precision", "ks_value"]
    radar_labels = ["Macro-F1", "高风险平均召回率", "正常类精准率", "KS值"]
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for _, row in comparison_df.iterrows():
        values = [float(row[metric]) for metric in radar_metrics]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=row["model_name"])
        ax.fill(angles, values, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("最终模型核心指标对比雷达图")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))

    radar_path = FIGURE_DIR / figure_name
    fig.tight_layout()
    fig.savefig(radar_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    result_json_path = MODEL_DIR / "final_comparison_low_resource.json"
    result_json_path.write_text(
        json.dumps(
            {
                "summaries": summaries,
                "details": details,
                "figure": f"results/figures/{figure_name}",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    best_model_row = comparison_df.sort_values(
        by=["macro_f1", "high_risk_recall", "normal_precision"],
        ascending=False,
    ).iloc[0]
    report_lines = [
        "# 最终模型优化报告",
        "",
        "## 运行方式",
        "- 本报告由低负载脚本分组生成。",
        "- 数据来源优先使用 `purified_train/val/test.csv`。",
        "",
        "## 当前最优",
        f"- 推荐模型：`{best_model_row['model_name']}`",
        f"- 最优 Macro-F1：`{best_model_row['macro_f1']:.6f}`",
        f"- 高风险类平均召回率：`{best_model_row['high_risk_recall']:.6f}`",
        f"- 正常类精准率：`{best_model_row['normal_precision']:.6f}`",
        "",
        "## 指标对比",
        comparison_df.to_markdown(index=False),
        "",
        "## 图表输出",
        f"- `results/figures/{figure_name}`",
        "",
        "## 结果文件",
        "- `src/models/final_comparison_low_resource.json`",
    ]
    report_path = DOCS_DIR / report_name
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run the final four-class comparison in a low-resource way.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=["baseline", "weighted", "sampled", "two_stage"],
        default=["baseline", "weighted", "two_stage"],
        help="Model groups to run. Default skips the heavy sampled group.",
    )
    parser.add_argument("--train-limit", type=int, default=120000)
    parser.add_argument("--val-limit", type=int, default=30000)
    parser.add_argument("--test-limit", type=int, default=30000)
    parser.add_argument("--top-k-features", type=int, default=80)
    parser.add_argument("--selector-estimators", type=int, default=100)
    parser.add_argument(
        "--report-name",
        type=str,
        default="final_model_optimization_report_low_resource.md",
    )
    parser.add_argument(
        "--figure-name",
        type=str,
        default="final_model_comparison_radar_low_resource.png",
    )
    return parser.parse_args()


def main() -> None:
    """Run the requested model groups and save comparison outputs."""

    args = parse_args()
    print("Loading purified-or-default datasets ...")
    train_df, val_df, test_df = load_splits(
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        test_limit=args.test_limit,
    )
    print(
        {
            "train_shape": train_df.shape,
            "val_shape": val_df.shape,
            "test_shape": test_df.shape,
            "groups": args.groups,
        }
    )

    features = prepare_selected_features(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        top_k_features=args.top_k_features,
        n_estimators=args.selector_estimators,
    )

    summaries: list[dict[str, object]] = []
    details: dict[str, dict[str, object]] = {}

    for group in args.groups:
        print(f"Running group: {group}")
        if group == "baseline":
            summary, metrics = run_baseline(features)
        elif group == "weighted":
            summary, metrics = run_weighted(features, test_df)
        elif group == "sampled":
            summary, metrics = run_sampled(
                features,
                test_df,
                top_k_features=args.top_k_features,
                n_estimators=args.selector_estimators,
            )
        elif group == "two_stage":
            summary, metrics = run_two_stage(features, test_df)
        else:
            raise ValueError(f"Unsupported group: {group}")

        summaries.append(summary)
        details[group] = metrics
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    save_outputs(
        summaries=summaries,
        details=details,
        figure_name=args.figure_name,
        report_name=args.report_name,
    )
    print("Low-resource comparison finished.")


if __name__ == "__main__":
    main()
