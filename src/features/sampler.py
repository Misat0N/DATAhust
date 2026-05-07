"""Sampling utilities for imbalanced credit risk training data."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE


class CreditSampler:
    """Handle class imbalance on the training set only using SMOTE-ENN."""

    def __init__(
        self,
        random_state: int = 42,
        sampling_strategy: str = "not majority",
    ) -> None:
        """Initialize the credit sampler."""

        self.random_state = random_state
        self.sampling_strategy = sampling_strategy
        self.sampler_ = SMOTEENN(
            random_state=self.random_state,
            sampling_strategy=self.sampling_strategy,
        )

    def fit_resample(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Apply SMOTE-ENN to the training set only.

        Validation and test sets must remain untouched and therefore should not
        be passed to this class.
        """

        if isinstance(X_train, pd.DataFrame):
            feature_names = X_train.columns.tolist()
        else:
            X_train = np.asarray(X_train)
            feature_names = [
                f"feature_{index:04d}" for index in range(X_train.shape[1])
            ]

        y_array = np.asarray(y_train)
        class_counts = Counter(y_array.tolist())
        majority_count = max(class_counts.values())
        eligible_classes = {
            label: majority_count
            for label, count in class_counts.items()
            if 1 < count < majority_count
        }
        minimum_eligible_count = min(
            (count for count in class_counts.values() if count > 1),
            default=0,
        )

        if not eligible_classes or minimum_eligible_count < 2:
            X_resampled = np.asarray(X_train)
            y_resampled = y_array
        else:
            adaptive_k_neighbors = min(5, minimum_eligible_count - 1)
            adaptive_sampler = SMOTEENN(
                random_state=self.random_state,
                sampling_strategy=eligible_classes,
                smote=SMOTE(
                    random_state=self.random_state,
                    k_neighbors=adaptive_k_neighbors,
                    sampling_strategy=eligible_classes,
                ),
            )
            X_resampled, y_resampled = adaptive_sampler.fit_resample(X_train, y_array)
            self.sampler_ = adaptive_sampler

        X_resampled_frame = pd.DataFrame(
            X_resampled,
            columns=feature_names,
        ).astype(np.float32)
        y_resampled_series = pd.Series(
            y_resampled,
            name="target",
        )
        return X_resampled_frame, y_resampled_series
