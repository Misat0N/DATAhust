"""Feature selection and optional dimensionality reduction utilities."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold


class CreditFeatureSelector:
    """Select informative credit features without peeking at non-train data.

    The selector applies four steps in sequence:

    1. Variance filtering.
    2. Correlation filtering.
    3. Random forest importance ranking.
    4. Optional PCA dimensionality reduction.
    """

    def __init__(
        self,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.8,
        top_k_features: int = 50,
        use_pca: bool = False,
        pca_variance_ratio: float = 0.95,
        random_state: int = 42,
        n_estimators: int = 200,
    ) -> None:
        """Initialize the credit feature selector."""

        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.top_k_features = top_k_features
        self.use_pca = use_pca
        self.pca_variance_ratio = pca_variance_ratio
        self.random_state = random_state
        self.n_estimators = n_estimators

        self.variance_selector_ = VarianceThreshold(threshold=self.variance_threshold)
        self.correlation_drop_columns_: List[str] = []
        self.selected_feature_names_: List[str] = []
        self.pca_feature_names_: List[str] = []
        self.random_forest_: RandomForestClassifier | None = None
        self.pca_: PCA | None = None
        self.is_fitted_ = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
    ) -> "CreditFeatureSelector":
        """Fit feature selection rules on the training set only."""

        train_frame = self._ensure_dataframe(X_train)
        target_array = np.asarray(y_train)

        variance_filtered_frame = self._fit_variance_filter(train_frame)
        correlation_filtered_frame = self._fit_correlation_filter(
            variance_filtered_frame
        )
        self._fit_random_forest_selector(correlation_filtered_frame, target_array)

        selected_frame = correlation_filtered_frame[self.selected_feature_names_]
        if self.use_pca:
            self._fit_pca(selected_frame)
        else:
            self.pca_ = None
            self.pca_feature_names_ = []

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using already fitted feature selection rules."""

        if not self.is_fitted_:
            raise RuntimeError("CreditFeatureSelector must be fitted before transform.")

        input_frame = self._ensure_dataframe(X)
        variance_filtered = self._apply_variance_filter(input_frame)
        correlation_filtered = variance_filtered.drop(
            columns=self.correlation_drop_columns_,
            errors="ignore",
        )

        selected_frame = correlation_filtered.reindex(
            columns=self.selected_feature_names_,
            fill_value=0.0,
        )

        if self.pca_ is None:
            return selected_frame.astype(np.float32)

        pca_array = self.pca_.transform(selected_frame)
        pca_frame = pd.DataFrame(
            pca_array.astype(np.float32),
            index=selected_frame.index,
            columns=self.pca_feature_names_,
        )
        return pca_frame

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
    ) -> pd.DataFrame:
        """Fit the selector and transform the training set."""

        return self.fit(X_train, y_train).transform(X_train)

    def _ensure_dataframe(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Convert the input to a pandas DataFrame if needed."""

        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, np.ndarray):
            columns = [f"feature_{index:04d}" for index in range(X.shape[1])]
            return pd.DataFrame(X, columns=columns)
        raise TypeError("CreditFeatureSelector expects a DataFrame or numpy array.")

    def _fit_variance_filter(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fit the variance filter and return the retained training features."""

        self.variance_selector_.fit(frame)
        kept_mask = self.variance_selector_.get_support()
        kept_columns = frame.columns[kept_mask].tolist()
        return frame[kept_columns].copy()

    def _apply_variance_filter(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the already fitted variance filter."""

        kept_columns = frame.columns[self.variance_selector_.get_support()].tolist()
        return frame.reindex(columns=kept_columns, fill_value=0.0).copy()

    def _fit_correlation_filter(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Remove highly collinear features based on Pearson correlation."""

        correlation_matrix = frame.corr(method="pearson").abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )

        columns_to_drop = [
            column
            for column in upper_triangle.columns
            if any(upper_triangle[column] > self.correlation_threshold)
        ]
        self.correlation_drop_columns_ = columns_to_drop
        return frame.drop(columns=columns_to_drop, errors="ignore").copy()

    def _fit_random_forest_selector(
        self,
        frame: pd.DataFrame,
        y_train: np.ndarray,
    ) -> None:
        """Fit a random forest and keep the most important features."""

        feature_count = min(self.top_k_features, frame.shape[1])
        if feature_count == 0:
            raise ValueError("No features remain after variance and correlation filtering.")

        self.random_forest_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        self.random_forest_.fit(frame, y_train)

        importance_series = pd.Series(
            self.random_forest_.feature_importances_,
            index=frame.columns,
        ).sort_values(ascending=False)
        self.selected_feature_names_ = importance_series.head(feature_count).index.tolist()

    def _fit_pca(self, frame: pd.DataFrame) -> None:
        """Fit PCA on the selected feature subset."""

        self.pca_ = PCA(
            n_components=self.pca_variance_ratio,
            svd_solver="full",
            random_state=self.random_state,
        )
        self.pca_.fit(frame)
        component_count = self.pca_.n_components_
        self.pca_feature_names_ = [
            f"pca_component_{index + 1:03d}" for index in range(component_count)
        ]
