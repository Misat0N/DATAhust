#!/usr/bin/env python3
"""Plot the final confusion matrix for the chosen credit-risk configuration.

Final configuration (see best_class_weight_config.json / best_threshold_config.json):
- model: LightGBM
- class weight: mild = {0:1, 1:2, 2:6, 3:10}
- normal-class threshold: 0.50
- business rules: enabled
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.features.feature_selector import CreditFeatureSelector
from src.features.preprocessor import CreditDataPreprocessor
from src.models.risk_classifier import CreditRiskClassifier
from src.models.threshold_adjuster import RiskThresholdAdjuster

DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["正常类", "关注类", "次级/可疑类", "损失类"]
MILD_CLASS_WEIGHT = {0: 1, 1: 2, 2: 6, 3: 10}

plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def resolve_split_path(split_name: str) -> Path:
    purified_path = DATA_DIR / f"purified_{split_name}.csv"
    if purified_path.exists():
        return purified_path
    return DATA_DIR / f"{split_name}.csv"


def maybe_limit(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None or limit <= 0:
        return frame
    return frame.iloc[: min(limit, len(frame))].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the final confusion matrix.")
    parser.add_argument("--train-limit", type=int, default=150000)
    parser.add_argument("--test-limit", type=int, default=40000)
    parser.add_argument("--top-k-features", type=int, default=80)
    parser.add_argument("--selector-estimators", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()

    print("Loading datasets ...")
    train_df = maybe_limit(pd.read_csv(resolve_split_path("train"), low_memory=False), args.train_limit)
    test_df = maybe_limit(pd.read_csv(resolve_split_path("test"), low_memory=False), args.test_limit)

    y_train = train_df["preloan_risk_label"].astype(int)
    y_test = test_df["preloan_risk_label"].astype(int)

    preprocessor = CreditDataPreprocessor(target_column="preloan_risk_label")
    x_train_processed = preprocessor.fit_transform(train_df)
    x_test_processed = preprocessor.transform(test_df)

    selector = CreditFeatureSelector(
        top_k_features=args.top_k_features,
        use_pca=False,
        n_estimators=args.selector_estimators,
    )
    x_train_selected = selector.fit_transform(x_train_processed, y_train)
    x_test_selected = selector.transform(x_test_processed)

    print("Training final model (mild weights, LightGBM) ...")
    model = CreditRiskClassifier(
        model_type="lightgbm",
        class_weight_mode="manual",
        custom_class_weight=MILD_CLASS_WEIGHT,
    )
    model.fit(x_train_selected, y_train)
    test_proba = model.predict_proba(x_test_selected)

    adjuster = RiskThresholdAdjuster(normal_class_threshold=args.threshold)
    threshold_pred = adjuster.adjust_prediction(np.asarray(test_proba, dtype=float))

    final_pred = []
    for (_, row), prediction in zip(test_df.iterrows(), threshold_pred):
        final_pred.append(adjuster.business_rule_override(row, int(prediction)))
    final_pred = np.asarray(final_pred, dtype=int)

    labels = list(range(len(CLASS_NAMES)))
    matrix = confusion_matrix(y_test.to_numpy(), final_pred, labels=labels)

    output_path = FIGURE_DIR / "risk_model_confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(8, 6.5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(CLASS_NAMES, rotation=25, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("风险分类混淆矩阵（mild + 阈值0.50 + 业务规则）")

    max_value = max(matrix.max(), 1)
    for row_index in labels:
        for col_index in labels:
            value = matrix[row_index, col_index]
            text_color = "white" if value > max_value / 2 else "#1e293b"
            ax.text(
                col_index,
                row_index,
                f"{value:,}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
