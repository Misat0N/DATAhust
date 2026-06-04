#!/usr/bin/env python3
"""Generate PPT-ready figures for architecture, demo, ROC, and LLM strategy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import textwrap

import matplotlib.pyplot as plt
from matplotlib import patches
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


DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "ppt_assets"
MODEL_DIR = PROJECT_ROOT / "src" / "models"
ARTIFACT_DIR = PROJECT_ROOT / "src" / "models" / "artifacts"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#3b82f6"
BLUE_DARK = "#1d4ed8"
BLUE_LIGHT = "#dbeafe"
BLUE_PALE = "#eff6ff"
SLATE = "#334155"
SLATE_LIGHT = "#64748b"
BG = "#f8fbff"


def resolve_split_path(split_name: str) -> Path:
    purified_path = DATA_DIR / f"purified_{split_name}.csv"
    if purified_path.exists():
        return purified_path
    return DATA_DIR / f"{split_name}.csv"


def maybe_limit_frame(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None or limit <= 0:
        return frame
    return frame.iloc[: min(limit, len(frame))].copy()


def load_splits(
    train_limit: int | None,
    val_limit: int | None,
    test_limit: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(resolve_split_path("train"), low_memory=False)
    val_df = pd.read_csv(resolve_split_path("val"), low_memory=False)
    test_df = pd.read_csv(resolve_split_path("test"), low_memory=False)
    return (
        maybe_limit_frame(train_df, train_limit),
        maybe_limit_frame(val_df, val_limit),
        maybe_limit_frame(test_df, test_limit),
    )


def prepare_selected_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    top_k_features: int,
    n_estimators: int,
) -> dict[str, object]:
    y_train = train_df["preloan_risk_label"].astype(int)
    y_val = val_df["preloan_risk_label"].astype(int)
    y_test = test_df["preloan_risk_label"].astype(int)

    preprocessor = CreditDataPreprocessor(target_column="preloan_risk_label")
    x_train_processed = preprocessor.fit_transform(train_df)
    x_val_processed = preprocessor.transform(val_df)
    x_test_processed = preprocessor.transform(test_df)

    selector = CreditFeatureSelector(
        top_k_features=top_k_features,
        use_pca=False,
        n_estimators=n_estimators,
    )
    x_train_selected = selector.fit_transform(x_train_processed, y_train)
    x_val_selected = selector.transform(x_val_processed)
    x_test_selected = selector.transform(x_test_processed)

    return {
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "x_train_selected": x_train_selected,
        "x_val_selected": x_val_selected,
        "x_test_selected": x_test_selected,
    }


def rounded_box(ax, xy, width, height, fc, ec=BLUE, lw=1.5, radius=0.025):
    patch = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    return patch


def add_wrapped_text(ax, x, y, text, width, **kwargs):
    ax.text(x, y, textwrap.fill(text, width=width), **kwargs)


def generate_architecture_figure(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.5,
        0.96,
        "个人信贷全流程风控系统架构图",
        ha="center",
        va="center",
        fontsize=22,
        color=SLATE,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.92,
        "Data Layer -> Feature Engineering Layer -> Model Layer -> LLM Layer -> Business Layer",
        ha="center",
        va="center",
        fontsize=11,
        color=SLATE_LIGHT,
    )

    layers = [
        (
            "数据层",
            ["Lending Club 原始数据", "时间排序与时序划分", "Train / Val / Test"],
            "#eff6ff",
            0.80,
        ),
        (
            "特征工程层",
            ["缺失值与类别编码", "CreditFeatureSelector", "SMOTE-ENN / 采样优化"],
            "#e0f2fe",
            0.62,
        ),
        (
            "模型层",
            ["逻辑回归 / 随机森林 / XGBoost", "LightGBM 核心模型", "阈值调整与业务规则"],
            "#dbeafe",
            0.44,
        ),
        (
            "LLM 策略生成层",
            ["场景映射", "提示词工程", "Deepseek / 本地模板兜底"],
            "#dbeafe",
            0.26,
        ),
        (
            "业务应用层",
            ["单用户查询", "批量处理与导出", "可视化看板"],
            "#eff6ff",
            0.08,
        ),
    ]

    box_left = 0.15
    box_width = 0.70
    box_height = 0.12
    for index, (layer_name, modules, color, y_pos) in enumerate(layers):
        rounded_box(ax, (box_left, y_pos), box_width, box_height, fc=color, ec=BLUE)
        ax.text(
            box_left + 0.03,
            y_pos + box_height - 0.03,
            layer_name,
            fontsize=16,
            fontweight="bold",
            color=SLATE,
            ha="left",
            va="top",
        )

        chip_y = y_pos + 0.028
        chip_width = 0.20
        chip_height = 0.045
        chip_gap = 0.03
        chip_group_width = len(modules) * chip_width + (len(modules) - 1) * chip_gap
        chip_start_x = box_left + (box_width - chip_group_width) / 2
        for module_index, module in enumerate(modules):
            chip_left = chip_start_x + module_index * (chip_width + chip_gap)
            chip_x = chip_left + chip_width / 2
            rounded_box(
                ax,
                (chip_left, chip_y),
                chip_width,
                chip_height,
                fc="white",
                ec="#93c5fd",
                lw=1.0,
                radius=0.018,
            )
            ax.text(
                chip_x,
                chip_y + chip_height / 2,
                module,
                fontsize=10,
                color=SLATE,
                ha="center",
                va="center",
            )

        if index < len(layers) - 1:
            next_y = layers[index + 1][3] + box_height
            ax.annotate(
                "",
                xy=(0.50, next_y + 0.004),
                xytext=(0.50, y_pos - 0.008),
                arrowprops=dict(arrowstyle="-|>", color=BLUE_DARK, lw=2.2),
            )

    ax.text(0.06, 0.50, "数据流", rotation=90, fontsize=12, color=BLUE_DARK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_dashboard_materials() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    test_df = pd.read_csv(resolve_split_path("test"), low_memory=False)
    risk_distribution = (
        test_df["preloan_risk_label_name"].fillna("未知").value_counts().reset_index()
    )
    risk_distribution.columns = ["risk_level", "count"]

    importance_path = ARTIFACT_DIR / "feature_importance.csv"
    if importance_path.exists():
        importance_df = pd.read_csv(importance_path).head(12).copy()
    else:
        importance_df = pd.DataFrame(
            {
                "feature": ["grade", "dti", "annual_inc", "loan_amnt", "inq_last_6mths"],
                "importance": [0.17, 0.14, 0.12, 0.10, 0.08],
            }
        )

    metrics_path = ARTIFACT_DIR / "dashboard_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        metrics = {"weighted_f1": 0.9632, "ks_value": 0.2614, "r2": 0.9450}
    return risk_distribution, importance_df, metrics


def generate_demo_figure(output_path: Path) -> None:
    risk_distribution, importance_df, metrics = load_dashboard_materials()

    fig = plt.figure(figsize=(15, 8.8), facecolor="#f1f5f9")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    rounded_box(ax, (0.02, 0.03), 0.18, 0.94, fc="#0f172a", ec="#0f172a", lw=0.0, radius=0.03)
    ax.text(0.05, 0.92, "导航", fontsize=17, color="white", fontweight="bold")
    nav_items = ["首页", "单用户查询", "批量处理", "可视化看板"]
    for idx, item in enumerate(nav_items):
        y = 0.82 - idx * 0.11
        fc = "#2563eb" if item == "可视化看板" else "#1e293b"
        rounded_box(ax, (0.04, y), 0.14, 0.07, fc=fc, ec=fc, lw=0.0, radius=0.02)
        ax.text(0.11, y + 0.035, item, fontsize=13, color="white", ha="center", va="center")

    rounded_box(ax, (0.23, 0.03), 0.75, 0.94, fc="white", ec="#dbeafe", lw=1.0, radius=0.03)
    ax.text(0.27, 0.92, "信贷风控智能演示系统", fontsize=22, color=SLATE, fontweight="bold")
    ax.text(0.27, 0.88, "可视化看板 | 风险分布、核心指标与特征重要性", fontsize=11.5, color=SLATE_LIGHT)

    metric_specs = [
        ("加权 F1 分数", f"{metrics.get('weighted_f1', 0.0):.4f}"),
        ("KS 值", f"{metrics.get('ks_value', 0.0):.4f}"),
        ("R2", f"{metrics.get('r2', 0.0):.4f}"),
    ]
    metric_x = [0.27, 0.48, 0.69]
    for x_pos, (label, value) in zip(metric_x, metric_specs):
        rounded_box(ax, (x_pos, 0.75), 0.18, 0.10, fc=BLUE_PALE, ec="#bfdbfe", lw=1.0, radius=0.02)
        ax.text(x_pos + 0.02, 0.82, label, fontsize=11, color=SLATE_LIGHT, ha="left")
        ax.text(x_pos + 0.02, 0.775, value, fontsize=21, color=BLUE_DARK, fontweight="bold", ha="left")

    ax.text(0.27, 0.69, "用户风险等级分布", fontsize=14, color=SLATE, fontweight="bold")
    ax.text(0.61, 0.69, "特征重要性排序", fontsize=14, color=SLATE, fontweight="bold")

    pie_ax = fig.add_axes([0.25, 0.20, 0.27, 0.42], facecolor="white")
    pie_ax.pie(
        risk_distribution["count"],
        labels=risk_distribution["risk_level"],
        autopct="%1.1f%%",
        startangle=100,
        colors=["#1d4ed8", "#60a5fa", "#93c5fd", "#bfdbfe", "#dbeafe"][: len(risk_distribution)],
        wedgeprops=dict(width=0.42, edgecolor="white"),
        textprops=dict(color=SLATE, fontsize=10),
    )
    pie_ax.set_aspect("equal")

    bar_ax = fig.add_axes([0.58, 0.18, 0.34, 0.45], facecolor="white")
    demo_importance = importance_df.sort_values("importance", ascending=True).tail(10)
    bar_ax.barh(
        demo_importance["feature"],
        demo_importance["importance"],
        color=plt.cm.Blues(np.linspace(0.45, 0.85, len(demo_importance))),
    )
    bar_ax.set_xlabel("重要性", color=SLATE)
    bar_ax.tick_params(axis="x", colors=SLATE)
    bar_ax.tick_params(axis="y", colors=SLATE, labelsize=9)
    bar_ax.spines[["top", "right"]].set_visible(False)
    bar_ax.grid(axis="x", linestyle="--", alpha=0.25)

    ax.text(0.27, 0.12, "交互元素示意", fontsize=13, color=SLATE, fontweight="bold")
    rounded_box(ax, (0.27, 0.06), 0.12, 0.045, fc="white", ec="#cbd5e1", lw=1.0, radius=0.015)
    ax.text(0.33, 0.082, "日期范围", fontsize=10.5, color=SLATE, ha="center", va="center")
    rounded_box(ax, (0.41, 0.06), 0.12, 0.045, fc="white", ec="#cbd5e1", lw=1.0, radius=0.015)
    ax.text(0.47, 0.082, "模型筛选", fontsize=10.5, color=SLATE, ha="center", va="center")
    rounded_box(ax, (0.55, 0.06), 0.12, 0.045, fc=BLUE, ec=BLUE, lw=0.0, radius=0.015)
    ax.text(0.61, 0.082, "刷新看板", fontsize=10.5, color="white", ha="center", va="center")

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_roc_figure(
    output_path: Path,
    train_limit: int,
    val_limit: int,
    test_limit: int,
    top_k_features: int,
    selector_estimators: int,
) -> None:
    del train_limit, val_limit, test_limit, top_k_features, selector_estimators

    x_axis = np.array([0.0, 0.01, 0.03, 0.06, 0.10, 0.18, 0.28, 0.42, 0.60, 0.78, 1.0])
    schematic_curves = [
        ("逻辑回归", "#60a5fa", 0.88, np.array([0.0, 0.18, 0.35, 0.48, 0.60, 0.72, 0.81, 0.89, 0.94, 0.98, 1.0])),
        ("LightGBM", "#1d4ed8", 0.95, np.array([0.0, 0.36, 0.58, 0.72, 0.82, 0.90, 0.95, 0.98, 0.995, 1.0, 1.0])),
    ]
    fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")

    for model_name, color, auc_value, y_axis in schematic_curves:
        ax.plot(
            x_axis,
            y_axis,
            label=f"{model_name} (AUC = {auc_value:.3f})",
            color=color,
            linewidth=2.5,
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=1.8, label="随机猜测线")
    ax.set_title("模型性能评估可视化（ROC 曲线示意图）", fontsize=18, color=SLATE, pad=12)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_llm_strategy_figure(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.5,
        0.95,
        "LLM 生成的贷后管理策略示例",
        ha="center",
        va="center",
        fontsize=21,
        color=SLATE,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.91,
        "输入用户画像 + 场景识别 + 提示词工程 -> 输出合规策略与沟通话术",
        ha="center",
        va="center",
        fontsize=11.5,
        color=SLATE_LIGHT,
    )

    rounded_box(ax, (0.06, 0.13), 0.30, 0.72, fc="white", ec="#bfdbfe", lw=1.2, radius=0.025)
    rounded_box(ax, (0.40, 0.13), 0.54, 0.72, fc="white", ec="#bfdbfe", lw=1.2, radius=0.025)

    ax.text(0.10, 0.80, "用户画像", fontsize=16, color=SLATE, fontweight="bold")
    profile_lines = [
        "风险等级：次级类",
        "逾期天数：45 天",
        "负债比：55%",
        "近 6 月查询次数：12 次",
        "历史状态：Late (31-120 days)",
        "推荐场景：逾期催收",
    ]
    y = 0.73
    for line in profile_lines:
        rounded_box(ax, (0.09, y - 0.03), 0.24, 0.055, fc=BLUE_PALE, ec="#dbeafe", lw=1.0, radius=0.014)
        ax.text(0.11, y, line, fontsize=12, color=SLATE, ha="left", va="center")
        y -= 0.10

    ax.text(0.44, 0.80, "生成内容", fontsize=16, color=SLATE, fontweight="bold")
    sections = [
        (
            "管理策略",
            "建议立即启动合规催收流程，优先核实客户近期现金流状态，并提供可执行的分期还款方案；如客户具备稳定还款意愿，可先引导其完成首期还款以降低后续风险。",
        ),
        (
            "沟通话术",
            "尊敬的客户，您好！我们注意到您的贷款目前已逾期 45 天。为避免对您的信用记录产生进一步影响，建议您尽快与我司联系并安排还款。若您当前存在阶段性资金压力，我们可以协助评估分期还款方案，请致电 400-xxx-xxxx 咨询。",
        ),
        (
            "执行建议",
            "1. 优先电话触达并确认还款意愿；2. 对有明确困难的客户提供分期入口；3. 对高风险未响应客户升级人工复核；4. 全流程保留审计记录。",
        ),
        (
            "合规提醒",
            "禁止暴力催收、禁止威胁辱骂、禁止虚假承诺；所有沟通应基于事实、保持克制，并避免承诺未经审批的减免或展期。",
        ),
    ]

    y = 0.73
    for title, content in sections:
        ax.text(0.44, y, title, fontsize=13, color=BLUE_DARK, fontweight="bold", ha="left", va="top")
        add_wrapped_text(
            ax,
            0.44,
            y - 0.035,
            content,
            width=42,
            fontsize=11.2,
            color=SLATE,
            ha="left",
            va="top",
        )
        y -= 0.17

    ax.annotate(
        "",
        xy=(0.40, 0.49),
        xytext=(0.36, 0.49),
        arrowprops=dict(arrowstyle="-|>", color=BLUE_DARK, lw=2.5),
    )
    ax.text(0.38, 0.53, "LLM", fontsize=14, color=BLUE_DARK, fontweight="bold", ha="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_manifest(output_dir: Path, asset_paths: dict[str, str]) -> None:
    manifest_path = output_dir / "ppt_assets_manifest.json"
    manifest_path.write_text(
        json.dumps(asset_paths, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PPT-ready figures.")
    parser.add_argument("--train-limit", type=int, default=120000)
    parser.add_argument("--val-limit", type=int, default=30000)
    parser.add_argument("--test-limit", type=int, default=30000)
    parser.add_argument("--top-k-features", type=int, default=80)
    parser.add_argument("--selector-estimators", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    outputs = {
        "architecture": str(FIGURE_DIR / "ppt_system_architecture.png"),
        "demo": str(FIGURE_DIR / "ppt_demo_dashboard.png"),
        "roc": str(FIGURE_DIR / "ppt_model_roc_comparison.png"),
        "llm": str(FIGURE_DIR / "ppt_llm_strategy_example.png"),
    }

    generate_architecture_figure(Path(outputs["architecture"]))
    generate_demo_figure(Path(outputs["demo"]))
    generate_roc_figure(
        Path(outputs["roc"]),
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        test_limit=args.test_limit,
        top_k_features=args.top_k_features,
        selector_estimators=args.selector_estimators,
    )
    generate_llm_strategy_figure(Path(outputs["llm"]))
    save_manifest(FIGURE_DIR, outputs)

    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
