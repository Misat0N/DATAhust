#!/usr/bin/env python3
"""Recompute SHAP feature importance for the FINAL 3-class risk model.

Pipeline (consistent with the final solution):
- maturity filtering + random sampling
- merge to 3 classes (0 正常类 / 1 风险关注类 / 2 损失类)
- LightGBM with class_weight=auto (no oversampling)

Outputs (written to results/figures/article/):
- shap_summary_plot.png        : mean |SHAP| bar chart across all 3 classes
- shap_feature_importance.csv  : ranked mean |SHAP| per feature
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.features.feature_selector import CreditFeatureSelector
from src.features.preprocessor import CreditDataPreprocessor
from src.label_builder import compute_maturity_mask
from src.models.risk_classifier import CreditRiskClassifier

DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTICLE_DIR = PROJECT_ROOT / "results" / "figures" / "article"
ARTICLE_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES_3 = ["正常类", "风险关注类", "损失类"]

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
    purified = DATA_DIR / f"purified_{split_name}.csv"
    return purified if purified.exists() else DATA_DIR / f"{split_name}.csv"


def to_three_class(label_4: pd.Series) -> pd.Series:
    mapping = {0: 0, 1: 1, 2: 1, 3: 2}
    return label_4.astype(int).map(mapping).astype(int)


def random_sample(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if limit is None or limit <= 0 or limit >= len(shuffled):
        return shuffled
    return shuffled.iloc[:limit].copy()


def extract_mean_abs_shap(shap_values, n_features: int) -> np.ndarray:
    """Aggregate mean |SHAP| across samples and classes into one vector."""

    values = shap_values
    if isinstance(values, list):
        # list of (n_samples, n_features), one per class
        stacked = np.stack([np.abs(np.asarray(v)).mean(axis=0) for v in values], axis=0)
        return stacked.mean(axis=0)
    arr = np.asarray(values)
    if arr.ndim == 3:
        # (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
        if arr.shape[1] == n_features:
            return np.abs(arr).mean(axis=(0, 2))
        return np.abs(arr).mean(axis=(0, 1))
    return np.abs(arr).mean(axis=0)


def extract_class_shap_matrix(shap_values, class_index: int, n_features: int) -> np.ndarray:
    """Return the (n_samples, n_features) SHAP matrix for one class."""

    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_index])
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        if arr.shape[1] == n_features:
            # (n_samples, n_features, n_classes)
            return arr[:, :, class_index]
        # (n_classes, n_samples, n_features)
        return arr[class_index]
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute SHAP for final 3-class model.")
    parser.add_argument("--train-limit", type=int, default=120000)
    parser.add_argument("--top-k-features", type=int, default=80)
    parser.add_argument("--selector-estimators", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shap-sample", type=int, default=2000)
    parser.add_argument("--max-display", type=int, default=20)
    parser.add_argument(
        "--focus-class",
        type=int,
        default=2,
        help="Class index for the beeswarm plot (0 正常 / 1 风险关注 / 2 损失). Default 2.",
    )
    args = parser.parse_args()

    print("Loading datasets ...")
    train_full = pd.read_csv(resolve_split_path("train"), low_memory=False)
    train_full = train_full.loc[compute_maturity_mask(train_full).to_numpy()].copy()
    train_full["risk3"] = to_three_class(train_full["preloan_risk_label"])
    train_df = random_sample(train_full, args.train_limit, args.seed)

    y_train = train_df["risk3"].astype(int)

    preprocessor = CreditDataPreprocessor(target_column="preloan_risk_label")
    x_train_processed = preprocessor.fit_transform(train_df)
    selector = CreditFeatureSelector(
        top_k_features=args.top_k_features,
        use_pca=False,
        n_estimators=args.selector_estimators,
    )
    x_train_selected = selector.fit_transform(x_train_processed, y_train)

    print("Training final 3-class LightGBM ...")
    model = CreditRiskClassifier(model_type="lightgbm", class_weight_mode="auto")
    model.fit(x_train_selected, y_train)
    estimator = model.best_model_

    sample_size = min(args.shap_sample, len(x_train_selected))
    feature_sample = x_train_selected.iloc[:sample_size].copy()
    feature_names = list(feature_sample.columns)

    print(f"Computing SHAP on {sample_size} samples ...")
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(feature_sample)

    mean_abs = extract_mean_abs_shap(shap_values, len(feature_names))
    importance_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    csv_path = ARTICLE_DIR / "shap_feature_importance.csv"
    importance_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {csv_path}")

    # Beeswarm summary plot for the focus class (default: 损失类).
    class_matrix = extract_class_shap_matrix(shap_values, args.focus_class, len(feature_names))
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        class_matrix,
        feature_sample,
        feature_names=feature_names,
        max_display=args.max_display,
        plot_type="dot",
        color_bar=True,
        show=False,
    )
    fig = plt.gcf()
    fig.suptitle(
        f"最终三分类风险模型 SHAP 蜂群图（聚焦：{CLASS_NAMES_3[args.focus_class]}）",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    summary_path = ARTICLE_DIR / "shap_summary_plot.png"
    fig.savefig(summary_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {summary_path}")

    print("\nTop 10 features:")
    print(importance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
