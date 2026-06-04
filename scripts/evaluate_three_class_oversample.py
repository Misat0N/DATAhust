#!/usr/bin/env python3
"""Validate a 3-class re-labeling + oversampling pipeline.

Changes vs the 4-class setup
----------------------------
- Merge original 关注类(1) and 次级/可疑类(2) into a single 风险关注类(1).
- Original 损失类(3) becomes 损失类(2). Result is a 3-class problem:
    0 正常类, 1 风险关注类, 2 损失类
- Apply maturity filtering, then global random sampling so all classes appear.
- Oversample the TRAINING set only with SMOTE-ENN (CreditSampler) toward a
  balanced distribution. The test set keeps its real distribution.
- Evaluate the plain LightGBM classifier (no 4-class threshold/rule layer,
  which is incompatible with the 3-class scheme).

This script does NOT overwrite any raw data or the formal label_builder output.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.features.feature_selector import CreditFeatureSelector
from src.features.preprocessor import CreditDataPreprocessor
from src.features.sampler import CreditSampler
from src.label_builder import compute_maturity_mask
from src.models.risk_classifier import CreditRiskClassifier
from src.models.threshold_adjuster import RiskThresholdAdjuster

DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

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
    """Map 4-class labels to the 3-class scheme.

    0 -> 0 (正常类)
    1, 2 -> 1 (风险关注类)
    3 -> 2 (损失类)
    """

    mapping = {0: 0, 1: 1, 2: 1, 3: 2}
    return label_4.astype(int).map(mapping).astype(int)


def random_sample(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if limit is None or limit <= 0 or limit >= len(shuffled):
        return shuffled
    return shuffled.iloc[:limit].copy()


def print_distribution(tag: str, labels: pd.Series) -> None:
    total = len(labels)
    vc = labels.value_counts().sort_index()
    print(f"-- {tag}: total={total}")
    for k in range(len(CLASS_NAMES_3)):
        count = int(vc.get(k, 0))
        pct = (count / total * 100) if total else 0.0
        print(f"   label {k} {CLASS_NAMES_3[k]}: {count:>8d}  {pct:>7.3f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="3-class relabel + oversample eval.")
    parser.add_argument("--train-limit", type=int, default=120000)
    parser.add_argument("--test-limit", type=int, default=40000)
    parser.add_argument("--top-k-features", type=int, default=80)
    parser.add_argument("--selector-estimators", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-oversample",
        action="store_true",
        help="Skip SMOTE-ENN oversampling (for comparison).",
    )
    parser.add_argument(
        "--max-oversample-ratio",
        type=float,
        default=3.0,
        help="Cap on SMOTE oversampling relative to majority class.",
    )
    parser.add_argument(
        "--pure-smote",
        action="store_true",
        help="Skip the ENN cleaning step (use plain SMOTE oversampling).",
    )
    parser.add_argument(
        "--loss-threshold",
        type=float,
        default=None,
        help=(
            "If set, predict 损失类 whenever P(损失类) >= this value. "
            "If omitted, a threshold scan is printed and the argmax prediction "
            "is used for the saved confusion matrix."
        ),
    )
    parser.add_argument(
        "--apply-rules",
        action="store_true",
        help=(
            "Apply business-rule overrides on top of the threshold prediction. "
            "Rule outputs are mapped to the 3-class scheme: 1/2 -> 1 风险关注类, "
            "3 -> 2 损失类."
        ),
    )
    args = parser.parse_args()

    print("Loading datasets ...")
    train_full = pd.read_csv(resolve_split_path("train"), low_memory=False)
    test_full = pd.read_csv(resolve_split_path("test"), low_memory=False)

    print("\n=== 成熟度过滤 ===")
    train_full = train_full.loc[compute_maturity_mask(train_full).to_numpy()].copy()
    test_full = test_full.loc[compute_maturity_mask(test_full).to_numpy()].copy()

    # Re-label to 3 classes before sampling.
    train_full["risk3"] = to_three_class(train_full["preloan_risk_label"])
    test_full["risk3"] = to_three_class(test_full["preloan_risk_label"])

    train_df = random_sample(train_full, args.train_limit, args.seed)
    test_df = random_sample(test_full, args.test_limit, args.seed)

    print("\n=== 抽样后分布（3 类） ===")
    print_distribution("train(抽样后)", train_df["risk3"])
    print_distribution("test(抽样后)", test_df["risk3"])

    y_train = train_df["risk3"].astype(int)
    y_test = test_df["risk3"].astype(int)

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

    if not args.no_oversample:
        print("\n=== SMOTE-ENN 过采样（仅训练集） ===")
        # Selector may drop columns, so recompute categorical indices on selected.
        selected_categorical = [
            index
            for index, column in enumerate(x_train_selected.columns)
            if "=" in str(column) and not str(column).endswith("__target_encoded")
        ]
        sampler = CreditSampler(categorical_features=selected_categorical)
        x_train_selected, y_train = sampler.fit_resample(
            x_train_selected,
            y_train,
            sampling_strategy="auto",
            max_oversample_ratio=args.max_oversample_ratio,
            apply_enn=not args.pure_smote,
        )
    else:
        print("\n=== 跳过过采样（对照） ===")

    print("\nTraining LightGBM (3-class) ...")
    model = CreditRiskClassifier(model_type="lightgbm", class_weight_mode="auto")
    model.fit(x_train_selected, y_train)
    test_proba = np.asarray(model.predict_proba(x_test_selected), dtype=float)
    model_classes = np.asarray(model.classes_, dtype=int)
    loss_col = int(np.where(model_classes == 2)[0][0])

    labels = list(range(len(CLASS_NAMES_3)))
    y_test_array = y_test.to_numpy()

    def predict_with_loss_threshold(proba: np.ndarray, threshold: float | None) -> np.ndarray:
        argmax_pred = model_classes[np.argmax(proba, axis=1)]
        if threshold is None:
            return argmax_pred.astype(int)
        force_loss = proba[:, loss_col] >= threshold
        result = argmax_pred.copy()
        result[force_loss] = 2
        return result.astype(int)

    def summarize(pred: np.ndarray) -> dict[str, float]:
        cm = confusion_matrix(y_test_array, pred, labels=labels)
        rec = cm.diagonal() / np.clip(cm.sum(axis=1), 1, None)
        return {
            "weighted_f1": f1_score(y_test_array, pred, average="weighted", zero_division=0),
            "macro_f1": f1_score(y_test_array, pred, average="macro", zero_division=0),
            "normal_recall": rec[0],
            "attention_recall": rec[1],
            "loss_recall": rec[2],
        }

    print("\n=== 损失类阈值扫描（P(损失类) >= t 即判损失类） ===")
    print("  t      weighted_f1  macro_f1  正常recall  关注recall  损失recall")
    scan_thresholds = [None, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15]
    for t in scan_thresholds:
        s = summarize(predict_with_loss_threshold(test_proba, t))
        label = "argmax" if t is None else f"{t:.2f}"
        print(
            f"  {label:<6} {s['weighted_f1']:.4f}      {s['macro_f1']:.4f}   "
            f"{s['normal_recall']:.4f}     {s['attention_recall']:.4f}     {s['loss_recall']:.4f}"
        )

    y_pred = predict_with_loss_threshold(test_proba, args.loss_threshold)

    if args.apply_rules:
        # Business rules emit 0/1/2/3; map to the 3-class scheme.
        # 0 -> 0 正常类, 1/2 -> 1 风险关注类, 3 -> 2 损失类.
        rule_to_3class = {0: 0, 1: 1, 2: 1, 3: 2}
        adjuster = RiskThresholdAdjuster()
        rule_applied = y_pred.copy()
        for position, (_, row) in enumerate(test_df.iterrows()):
            rule_label_4 = adjuster.business_rule_override(row, 0)
            mapped = rule_to_3class[int(rule_label_4)]
            # Rules only escalate risk; never downgrade a model risk prediction.
            if mapped > rule_applied[position]:
                rule_applied[position] = mapped
        y_pred = rule_applied.astype(int)

    matrix = confusion_matrix(y_test_array, y_pred, labels=labels)
    weighted_f1 = f1_score(y_test_array, y_pred, average="weighted", zero_division=0)
    macro_f1 = f1_score(y_test_array, y_pred, average="macro", zero_division=0)
    recalls = matrix.diagonal() / np.clip(matrix.sum(axis=1), 1, None)

    chosen = "argmax" if args.loss_threshold is None else f"{args.loss_threshold:.2f}"
    print(f"\n=== 选定方案（loss_threshold={chosen}）测试集指标 ===")
    print(f"weighted_f1 = {weighted_f1:.4f}")
    print(f"macro_f1    = {macro_f1:.4f}")
    for i, name in enumerate(CLASS_NAMES_3):
        support = int(matrix.sum(axis=1)[i])
        print(f"   {name} recall = {recalls[i]:.4f}  (support={support})")
    print("\n" + classification_report(
        y_test_array, y_pred, labels=labels, target_names=CLASS_NAMES_3, zero_division=0
    ))

    if args.no_oversample:
        suffix = "no_oversample"
    elif args.pure_smote:
        suffix = "pure_smote"
    else:
        suffix = "oversampled"
    if args.loss_threshold is not None:
        suffix += f"_loss{args.loss_threshold:.2f}"
    if args.apply_rules:
        suffix += "_rules"
    output_path = FIGURE_DIR / f"risk_model_confusion_matrix_3class_{suffix}.png"
    fig, ax = plt.subplots(figsize=(7.5, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(CLASS_NAMES_3, rotation=20, ha="right")
    ax.set_yticklabels(CLASS_NAMES_3)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    base_tag = {"no_oversample": "未过采样", "pure_smote": "纯SMOTE", "oversampled": "SMOTE-ENN"}[
        "no_oversample" if args.no_oversample else ("pure_smote" if args.pure_smote else "oversampled")
    ]
    threshold_tag = "argmax" if args.loss_threshold is None else f"损失阈值{args.loss_threshold:.2f}"
    rule_tag = "+业务规则" if args.apply_rules else ""
    ax.set_title(f"三分类混淆矩阵（{base_tag}+{threshold_tag}{rule_tag}）")
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
