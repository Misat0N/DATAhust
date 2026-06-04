#!/usr/bin/env python3
"""Evaluate the effect of maturity filtering on the final risk model.

This script does NOT overwrite any raw data. It loads the splits, optionally
applies maturity filtering (drop not-yet-matured Current loans whose risk
outcome is unobservable), prints the label distribution before/after, then
trains the final configuration (LightGBM + mild weights + threshold 0.50 +
business rules) and reports the confusion matrix and key metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

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
    purified = DATA_DIR / f"purified_{split_name}.csv"
    return purified if purified.exists() else DATA_DIR / f"{split_name}.csv"


def maybe_limit(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None or limit <= 0:
        return frame
    return frame.iloc[: min(limit, len(frame))].copy()


def random_sample(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    """Shuffle then take up to ``limit`` rows so all classes can appear."""

    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if limit is None or limit <= 0 or limit >= len(shuffled):
        return shuffled
    return shuffled.iloc[:limit].copy()


def print_distribution(tag: str, labels: pd.Series) -> None:
    total = len(labels)
    vc = labels.value_counts().sort_index()
    print(f"-- {tag}: total={total}")
    for k in range(len(CLASS_NAMES)):
        count = int(vc.get(k, 0))
        pct = (count / total * 100) if total else 0.0
        print(f"   label {k} {CLASS_NAMES[k]}: {count:>8d}  {pct:>7.3f}%")


def apply_maturity_filter(frame: pd.DataFrame, tag: str) -> pd.DataFrame:
    before = frame["preloan_risk_label"].astype(int)
    print_distribution(f"{tag} 过滤前", before)
    mask = compute_maturity_mask(frame).to_numpy()
    filtered = frame.loc[mask].copy()
    print_distribution(f"{tag} 过滤后", filtered["preloan_risk_label"].astype(int))
    print(f"   保留比例: {mask.mean() * 100:.2f}%")
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate maturity filtering effect.")
    parser.add_argument("--train-limit", type=int, default=150000)
    parser.add_argument("--test-limit", type=int, default=40000)
    parser.add_argument("--top-k-features", type=int, default=80)
    parser.add_argument("--selector-estimators", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Skip maturity filtering (baseline run for comparison).",
    )
    args = parser.parse_args()

    print("Loading datasets ...")
    train_full = pd.read_csv(resolve_split_path("train"), low_memory=False)
    test_full = pd.read_csv(resolve_split_path("test"), low_memory=False)

    if not args.no_filter:
        print("\n=== 成熟度过滤（先过滤，后随机抽样） ===")
        train_full = apply_maturity_filter(train_full, "train(全量)")
        test_full = apply_maturity_filter(test_full, "test(全量)")
    else:
        print("\n=== 跳过成熟度过滤（对照基线） ===")

    train_df = random_sample(train_full, args.train_limit, args.seed)
    test_df = random_sample(test_full, args.test_limit, args.seed)

    print("\n=== 随机抽样后用于训练/评估的分布 ===")
    print_distribution("train(抽样后)", train_df["preloan_risk_label"].astype(int))
    print_distribution("test(抽样后)", test_df["preloan_risk_label"].astype(int))

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

    print("\nTraining final model (mild + LightGBM) ...")
    model = CreditRiskClassifier(
        model_type="lightgbm",
        class_weight_mode="manual",
        custom_class_weight=MILD_CLASS_WEIGHT,
    )
    model.fit(x_train_selected, y_train)
    test_proba = np.asarray(model.predict_proba(x_test_selected), dtype=float)

    adjuster = RiskThresholdAdjuster(normal_class_threshold=args.threshold)
    threshold_pred = adjuster.adjust_prediction(test_proba)
    final_pred = []
    for (_, row), prediction in zip(test_df.iterrows(), threshold_pred):
        final_pred.append(adjuster.business_rule_override(row, int(prediction)))
    final_pred = np.asarray(final_pred, dtype=int)

    labels = list(range(len(CLASS_NAMES)))
    matrix = confusion_matrix(y_test.to_numpy(), final_pred, labels=labels)

    weighted_f1 = f1_score(y_test, final_pred, average="weighted", zero_division=0)
    macro_f1 = f1_score(y_test, final_pred, average="macro", zero_division=0)

    recalls = matrix.diagonal() / np.clip(matrix.sum(axis=1), 1, None)
    high_risk_recall = float(np.mean([recalls[i] for i in (1, 2, 3) if matrix.sum(axis=1)[i] > 0]))

    print("\n=== 测试集指标 ===")
    print(f"weighted_f1     = {weighted_f1:.4f}")
    print(f"macro_f1        = {macro_f1:.4f}")
    print(f"high_risk_recall= {high_risk_recall:.4f}")
    for i, name in enumerate(CLASS_NAMES):
        support = int(matrix.sum(axis=1)[i])
        print(f"   {name} recall = {recalls[i]:.4f}  (support={support})")

    suffix = "no_filter" if args.no_filter else "maturity_filtered"
    output_path = FIGURE_DIR / f"risk_model_confusion_matrix_{suffix}.png"
    fig, ax = plt.subplots(figsize=(8, 6.5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(CLASS_NAMES, rotation=25, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    title_tag = "未过滤" if args.no_filter else "成熟度过滤后"
    ax.set_title(f"风险分类混淆矩阵（{title_tag}，mild+0.50+规则）")
    max_value = max(matrix.max(), 1)
    for r in labels:
        for c in labels:
            value = matrix[r, c]
            color = "white" if value > max_value / 2 else "#1e293b"
            ax.text(c, r, f"{value:,}", ha="center", va="center", color=color, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
