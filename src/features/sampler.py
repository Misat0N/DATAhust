"""Sampling utilities for imbalanced credit risk training data."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import EditedNearestNeighbours


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


class CreditSampler:
    """Handle multiclass imbalance on the training set only.

    The sampler is designed for tabular credit data that may contain binary or
    one-hot categorical columns. It first applies ``SMOTENC`` to oversample
    minority classes while respecting categorical feature positions, then uses
    ``EditedNearestNeighbours`` to remove noisy majority samples near unstable
    decision boundaries.
    """

    def __init__(
        self,
        categorical_features: list[int] | np.ndarray | None = None,
        random_state: int = 42,
    ) -> None:
        """Initialize the sampler with categorical feature indices."""

        self.categorical_features = list(categorical_features or [])
        self.random_state = random_state
        self.oversampler_: SMOTENC | None = None
        self.undersampler_: EditedNearestNeighbours | None = None
        self.before_distribution_: dict[Any, int] | None = None
        self.after_oversample_distribution_: dict[Any, int] | None = None
        self.after_distribution_: dict[Any, int] | None = None

    def fit_resample(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        sampling_strategy: str | dict[Any, int] = "auto",
        max_oversample_ratio: float = 5,
        apply_enn: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Apply SMOTENC (+ optional ENN) to the training set only.

        Processing steps
        ----------------
        1. Oversample minority classes with ``SMOTENC``.
        2. Optionally remove noisy samples with ``EditedNearestNeighbours``.
        3. Limit oversampling so that the post-SMOTE class size does not exceed
           ``max_oversample_ratio * max_original_minority_count``.

        When ``apply_enn`` is ``False`` the ENN cleaning step is skipped, which
        is useful for heavily overlapping classes where ENN would delete most
        of a minority class. Validation and test sets must never be passed in.
        """

        self._validate_sampling_config(
            X_train=X_train,
            y_train=y_train,
            sampling_strategy=sampling_strategy,
            max_oversample_ratio=max_oversample_ratio,
        )

        train_frame, feature_names = self._ensure_dataframe(X_train)
        y_array = np.asarray(y_train)
        self.before_distribution_ = self.get_sample_distribution(y_array, name="采样前")

        target_sampling = self._build_target_sampling_strategy(
            y_array=y_array,
            sampling_strategy=sampling_strategy,
            max_oversample_ratio=max_oversample_ratio,
        )

        if not target_sampling:
            X_after = train_frame.copy()
            y_after = pd.Series(y_array, name="target")
            self.after_distribution_ = self.get_sample_distribution(
                y_after,
                name="采样后",
            )
            return X_after.astype(np.float32), y_after

        minimum_resample_count = min(
            count
            for label, count in Counter(y_array.tolist()).items()
            if label in target_sampling or count > 1
        )
        k_neighbors = max(1, min(5, minimum_resample_count - 1))

        categorical_mask = self._build_categorical_mask(train_frame.shape[1])
        self.oversampler_ = SMOTENC(
            categorical_features=categorical_mask,
            sampling_strategy=target_sampling,
            k_neighbors=k_neighbors,
            random_state=self.random_state,
        )
        X_over, y_over = self.oversampler_.fit_resample(train_frame, y_array)
        self.after_oversample_distribution_ = self.get_sample_distribution(
            y_over,
            name="SMOTENC后",
        )

        if apply_enn:
            self.undersampler_ = EditedNearestNeighbours()
            X_resampled, y_resampled = self.undersampler_.fit_resample(X_over, y_over)
            self.after_distribution_ = self.get_sample_distribution(
                y_resampled,
                name="ENN后",
            )
        else:
            self.undersampler_ = None
            X_resampled, y_resampled = X_over, y_over
            self.after_distribution_ = self.get_sample_distribution(
                y_resampled,
                name="纯SMOTE(跳过ENN)",
            )

        X_resampled_frame = pd.DataFrame(
            X_resampled,
            columns=feature_names,
        ).astype(np.float32)
        y_resampled_series = pd.Series(y_resampled, name="target")
        return X_resampled_frame, y_resampled_series

    def get_sample_distribution(
        self,
        y: pd.Series | np.ndarray,
        name: str = "",
    ) -> dict[Any, int]:
        """Return and print class-count distribution for a label array."""

        distribution = {
            label: int(count)
            for label, count in sorted(Counter(np.asarray(y).tolist()).items())
        }
        if name:
            print(f"{name}样本分布: {distribution}")
        return distribution

    def plot_sample_distribution(
        self,
        y_before: pd.Series | np.ndarray,
        y_after: pd.Series | np.ndarray,
    ) -> str:
        """Plot label distributions before and after sampling."""

        before_distribution = self.get_sample_distribution(y_before)
        after_distribution = self.get_sample_distribution(y_after)

        labels = sorted(set(before_distribution) | set(after_distribution))
        before_values = [before_distribution.get(label, 0) for label in labels]
        after_values = [after_distribution.get(label, 0) for label in labels]
        positions = np.arange(len(labels))
        bar_width = 0.35

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.bar(positions - bar_width / 2, before_values, width=bar_width, label="采样前")
        ax.bar(positions + bar_width / 2, after_values, width=bar_width, label="采样后")
        ax.set_xticks(positions)
        ax.set_xticklabels([str(label) for label in labels])
        ax.set_xlabel("类别标签")
        ax.set_ylabel("样本数")
        ax.set_title("采样前后标签分布对比")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        output_path = FIGURE_DIR / "sample_distribution_comparison.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return "results/figures/sample_distribution_comparison.png"

    def _validate_sampling_config(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        sampling_strategy: str | dict[Any, int],
        max_oversample_ratio: float,
    ) -> None:
        """Validate the sampling configuration before resampling."""

        y_array = np.asarray(y_train)
        if y_array.ndim != 1:
            raise ValueError("y_train must be a one-dimensional label array.")
        if len(y_array) == 0:
            raise ValueError("y_train must contain at least one sample.")
        if len(np.unique(y_array)) < 2:
            raise ValueError("Resampling requires at least two classes.")
        if max_oversample_ratio <= 1:
            raise ValueError("max_oversample_ratio must be greater than 1.")
        if sampling_strategy != "auto" and not isinstance(sampling_strategy, dict):
            raise ValueError(
                "sampling_strategy must be 'auto' or a dictionary of target counts."
            )

        if isinstance(X_train, pd.DataFrame):
            feature_count = X_train.shape[1]
        else:
            feature_count = np.asarray(X_train).shape[1]

        invalid_indices = [
            index
            for index in self.categorical_features
            if index < 0 or index >= feature_count
        ]
        if invalid_indices:
            raise ValueError(
                "categorical_features contains invalid column indices: "
                f"{invalid_indices}"
            )

    def _ensure_dataframe(
        self,
        X_train: pd.DataFrame | np.ndarray,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Convert training features to a DataFrame while preserving column names."""

        if isinstance(X_train, pd.DataFrame):
            return X_train.copy(), X_train.columns.tolist()

        X_array = np.asarray(X_train)
        feature_names = [f"feature_{index:04d}" for index in range(X_array.shape[1])]
        return pd.DataFrame(X_array, columns=feature_names), feature_names

    def _build_target_sampling_strategy(
        self,
        y_array: np.ndarray,
        sampling_strategy: str | dict[Any, int],
        max_oversample_ratio: float,
    ) -> dict[Any, int]:
        """Build the capped oversampling target for ``SMOTENC``."""

        class_counts = Counter(y_array.tolist())
        if len(class_counts) < 2:
            return {}

        majority_count = max(class_counts.values())
        minority_counts = [
            count for count in class_counts.values() if count < majority_count
        ]
        if not minority_counts:
            return {}

        reference_minority_count = max(minority_counts)
        target_cap = max(
            int(reference_minority_count),
            int(np.ceil(majority_count / max_oversample_ratio)),
        )

        if isinstance(sampling_strategy, dict):
            capped_strategy = {}
            for label, requested_count in sampling_strategy.items():
                original_count = class_counts.get(label, 0)
                if requested_count > original_count:
                    capped_strategy[label] = min(int(requested_count), target_cap)
            return capped_strategy

        capped_strategy = {}
        for label, count in class_counts.items():
            if count < majority_count and count > 1:
                target_count = min(majority_count, target_cap)
                if target_count > count:
                    capped_strategy[label] = target_count
        return capped_strategy

    def _build_categorical_mask(self, feature_count: int) -> list[bool]:
        """Convert categorical indices to the boolean mask expected by SMOTENC."""

        categorical_mask = [False] * feature_count
        for column_index in self.categorical_features:
            categorical_mask[int(column_index)] = True
        return categorical_mask
