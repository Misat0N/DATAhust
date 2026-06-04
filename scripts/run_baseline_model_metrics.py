#!/usr/bin/env python3
"""Run baseline comparisons for logistic regression, random forest, and XGBoost."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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
from src.models.model_evaluator import CreditModelEvaluator
from src.models.risk_classifier import CreditRiskClassifier


DATA_DIR = PROJECT_ROOT / "data" / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"
MODEL_DIR = PROJECT_ROOT / "src" / "models"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["正常类", "关注类", "次级/可疑类", "损失类"]
MODEL_LABELS = {
    "logistic_regression": "逻辑回归",
    "random_forest": "随机森林",
    "xgboost": "XGBoost",
}


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

    return {
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "X_train_selected": X_train_selected,
        "X_val_selected": X_val_selected,
        "X_test_selected": X_test_selected,
    }


def build_summary(
    model_key: str,
    metrics: dict[str, object],
) -> dict[str, object]:
    """Extract the three requested baseline metrics."""

    auc_per_class = metrics.get("auc_roc_per_class", {})
    valid_auc_values = [
        float(value) for value in auc_per_class.values() if not np.isnan(float(value))
    ]
    average_auc = (
        float(np.mean(valid_auc_values))
        if valid_auc_values
        else float(metrics.get("auc_roc_ovo_macro", float("nan")))
    )
    return {
        "model_type": model_key,
        "model_name": MODEL_LABELS[model_key],
        "weighted_f1_score": float(metrics["weighted_f1"]),
        "average_auc": average_auc,
        "ks_value": float(metrics["ks_value"]),
    }


def run_single_model(
    model_key: str,
    features: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Train and evaluate one baseline model family."""

    model = CreditRiskClassifier(model_type=model_key, class_weight_mode="none")
    model.fit(features["X_train_selected"], features["y_train"])
    metrics = CreditModelEvaluator(
        model=model,
        X_test=features["X_test_selected"],
        y_test=features["y_test"],
        class_names=CLASS_NAMES,
    ).evaluate_imbalanced_multiclass()
    return build_summary(model_key, metrics), metrics


def save_outputs(
    summaries: list[dict[str, object]],
    details: dict[str, dict[str, object]],
    report_name: str,
) -> None:
    """Save baseline comparison outputs to csv/json/markdown."""

    summary_df = pd.DataFrame(summaries)
    csv_path = MODEL_DIR / "baseline_model_metrics.csv"
    json_path = MODEL_DIR / "baseline_model_metrics.json"
    report_path = DOCS_DIR / report_name

    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(
        json.dumps(
            {
                "summaries": summaries,
                "details": details,
                "csv": "src/models/baseline_model_metrics.csv",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    try:
        result_table = summary_df.to_markdown(index=False)
    except ImportError:
        result_table = summary_df.to_string(index=False)

    report_lines = [
        "# Baseline 模型对比",
        "",
        "## 指标说明",
        "- `weighted_f1_score`: 加权 F1-Score",
        "- `average_auc`: 多分类各类别 AUC 的简单平均",
        "- `ks_value`: 各类别 KS 的平均值",
        "",
        "## 结果表",
        result_table,
        "",
        "## 文件输出",
        "- `src/models/baseline_model_metrics.csv`",
        "- `src/models/baseline_model_metrics.json`",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run baseline metrics for logistic regression, random forest, and XGBoost.",
    )
    parser.add_argument("--train-limit", type=int, default=120000)
    parser.add_argument("--val-limit", type=int, default=30000)
    parser.add_argument("--test-limit", type=int, default=30000)
    parser.add_argument("--top-k-features", type=int, default=80)
    parser.add_argument("--selector-estimators", type=int, default=100)
    parser.add_argument(
        "--report-name",
        type=str,
        default="baseline_model_metrics.md",
    )
    return parser.parse_args()


def main() -> None:
    """Run the three baseline models and print a compact table."""

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
            "models": list(MODEL_LABELS.values()),
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
    for model_key, model_name in MODEL_LABELS.items():
        print(f"Running baseline model: {model_name}")
        summary, metrics = run_single_model(model_key, features)
        summaries.append(summary)
        details[model_key] = metrics
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    save_outputs(summaries, details, args.report_name)
    print("Baseline comparison finished.")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
