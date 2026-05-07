"""Feature preprocessing utilities for Lending Club credit data.

This module implements a training-only fitted preprocessor to avoid data
leakage. The preprocessor supports:

- Missing value imputation.
- IQR-based outlier clipping for continuous variables.
- Semantic feature grouping into four business categories.
- One-hot encoding for low-cardinality categorical variables.
- Mean target encoding for high-cardinality categorical variables.
- Standard scaling for continuous variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TargetEncodingStats:
    """Container for smoothed target encoding statistics."""

    mapping: Dict[str, float]
    global_mean: float


class CreditDataPreprocessor:
    """Preprocess credit data without leaking validation or test information.

    Parameters
    ----------
    target_column:
        Label column used to fit target encoding for high-cardinality features.
    low_cardinality_threshold:
        Maximum unique value count for a categorical feature to use one-hot
        encoding. Features above this threshold use target encoding.
    excluded_columns:
        Columns that must never be treated as model features because they are
        labels or derived targets.
    target_smoothing:
        Smoothing strength for target encoding. Larger values increase the
        influence of the global target mean for sparse categories.
    """

    def __init__(
        self,
        target_column: str = "preloan_risk_label",
        low_cardinality_threshold: int = 10,
        excluded_columns: Iterable[str] | None = None,
        target_smoothing: float = 20.0,
    ) -> None:
        """Initialize the credit data preprocessor."""

        default_excluded_columns = {
            "preloan_risk_label",
            "preloan_risk_label_name",
            "estimated_overdue_days",
            "calibrated_credit_limit",
            "calibrated_credit_limit_band",
            "postloan_scene_label",
            "postloan_scene_label_name",
        }
        self.target_column = target_column
        self.low_cardinality_threshold = low_cardinality_threshold
        self.excluded_columns = set(excluded_columns or default_excluded_columns)
        self.target_smoothing = target_smoothing

        self.feature_groups_: Dict[str, List[str]] = {
            "user_base_features": [],
            "repayment_capacity_features": [],
            "credit_risk_features": [],
            "loan_product_features": [],
        }
        self.feature_columns_: List[str] = []
        self.categorical_columns_: List[str] = []
        self.continuous_columns_: List[str] = []
        self.low_cardinality_columns_: List[str] = []
        self.high_cardinality_columns_: List[str] = []
        self.categorical_fill_values_: Dict[str, str] = {}
        self.continuous_fill_values_: Dict[str, float] = {}
        self.outlier_bounds_: Dict[str, tuple[float, float]] = {}
        self.low_cardinality_categories_: Dict[str, List[str]] = {}
        self.target_encoding_stats_: Dict[str, TargetEncodingStats] = {}
        self.scaler_ = StandardScaler()
        self.output_columns_: List[str] = []
        self.global_target_mean_: float | None = None
        self.is_fitted_ = False

    def fit(self, X_train: pd.DataFrame) -> "CreditDataPreprocessor":
        """Fit preprocessing rules using the training set only."""

        train_frame = self._validate_input_frame(X_train)
        self.feature_columns_ = self._get_feature_columns(train_frame)

        if self.target_column not in train_frame.columns and self.feature_columns_:
            raise ValueError(
                f"Training data must contain target column '{self.target_column}' "
                "to fit target encoding for high-cardinality features."
            )

        feature_frame = train_frame[self.feature_columns_].copy()
        target_series = pd.to_numeric(
            train_frame[self.target_column],
            errors="coerce",
        )

        self._split_feature_types(feature_frame)
        self._build_feature_groups()
        self._fit_imputers(feature_frame)
        imputed_frame = self._apply_imputation(feature_frame)
        self._fit_outlier_bounds(imputed_frame)
        clipped_frame = self._apply_outlier_clipping(imputed_frame)
        self._fit_low_cardinality_categories(clipped_frame)
        self._fit_target_encoders(clipped_frame, target_series)
        self._fit_scaler(clipped_frame)

        transformed_frame = self._transform_internal(train_frame)
        self.output_columns_ = transformed_frame.columns.tolist()
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform a dataset using already fitted preprocessing rules."""

        if not self.is_fitted_:
            raise RuntimeError("CreditDataPreprocessor must be fitted before transform.")

        input_frame = self._validate_input_frame(X)
        transformed_frame = self._transform_internal(input_frame)
        return transformed_frame.reindex(columns=self.output_columns_, fill_value=0.0)

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Fit the preprocessor and transform the training set."""

        return self.fit(X_train).transform(X_train)

    def _validate_input_frame(self, X: object) -> pd.DataFrame:
        """Ensure the input object is a pandas DataFrame."""

        if not isinstance(X, pd.DataFrame):
            raise TypeError("CreditDataPreprocessor expects a pandas DataFrame input.")
        return X.copy()

    def _get_feature_columns(self, frame: pd.DataFrame) -> List[str]:
        """Return model feature columns after removing excluded label columns."""

        return [
            column
            for column in frame.columns
            if column not in self.excluded_columns and column != self.target_column
        ]

    def _split_feature_types(self, frame: pd.DataFrame) -> None:
        """Split columns into categorical and continuous feature sets."""

        self.categorical_columns_ = [
            column
            for column in frame.columns
            if pd.api.types.is_object_dtype(frame[column])
            or pd.api.types.is_categorical_dtype(frame[column])
            or pd.api.types.is_bool_dtype(frame[column])
        ]
        self.continuous_columns_ = [
            column for column in frame.columns if column not in self.categorical_columns_
        ]

        self.low_cardinality_columns_ = []
        self.high_cardinality_columns_ = []
        for column in self.categorical_columns_:
            unique_count = frame[column].nunique(dropna=True)
            if unique_count <= self.low_cardinality_threshold:
                self.low_cardinality_columns_.append(column)
            else:
                self.high_cardinality_columns_.append(column)

    def _build_feature_groups(self) -> None:
        """Assign each feature to one of the four business feature groups."""

        group_map = {group_name: [] for group_name in self.feature_groups_}
        for column in self.feature_columns_:
            group_name = self._infer_feature_group(column)
            group_map[group_name].append(column)
        self.feature_groups_ = group_map

    def _infer_feature_group(self, column: str) -> str:
        """Infer the business feature group from the feature name."""

        lower_column = column.lower()

        loan_product_keywords = [
            "loan",
            "funded",
            "term",
            "int_rate",
            "grade",
            "sub_grade",
            "purpose",
            "title",
            "installment",
            "policy_code",
            "disbursement",
            "application_type",
            "initial_list_status",
            "issue_d",
        ]
        repayment_keywords = [
            "annual_inc",
            "dti",
            "installment",
            "revol_bal",
            "revol_util",
            "total_bal",
            "tot_cur_bal",
            "avg_cur_bal",
            "bc_open_to_buy",
            "all_util",
            "il_util",
            "total_rev_hi_lim",
            "total_bc_limit",
            "total_il_high_credit_limit",
            "tot_hi_cred_lim",
            "mort_acc",
            "open_acc",
            "num_sats",
        ]
        credit_risk_keywords = [
            "delinq",
            "inq",
            "pub_rec",
            "collection",
            "chargeoff",
            "bankrupt",
            "tax_liens",
            "mths_since",
            "hardship",
            "settlement",
            "acc_now_delinq",
            "num_tl",
            "pct_tl",
            "percent_bc",
            "earliest_cr_line",
        ]
        user_base_keywords = [
            "emp",
            "home_ownership",
            "verification",
            "zip_code",
            "addr_state",
            "desc",
            "pymnt_plan",
            "sec_app",
            "joint",
        ]

        for keyword in loan_product_keywords:
            if keyword in lower_column:
                return "loan_product_features"
        for keyword in repayment_keywords:
            if keyword in lower_column:
                return "repayment_capacity_features"
        for keyword in credit_risk_keywords:
            if keyword in lower_column:
                return "credit_risk_features"
        for keyword in user_base_keywords:
            if keyword in lower_column:
                return "user_base_features"
        return "user_base_features"

    def _fit_imputers(self, frame: pd.DataFrame) -> None:
        """Fit missing value fill rules on the training set."""

        self.categorical_fill_values_ = {}
        for column in self.categorical_columns_:
            mode_series = frame[column].dropna().mode()
            fill_value = (
                str(mode_series.iloc[0])
                if not mode_series.empty
                else "__missing__"
            )
            self.categorical_fill_values_[column] = fill_value

        self.continuous_fill_values_ = {}
        for column in self.continuous_columns_:
            numeric_series = pd.to_numeric(frame[column], errors="coerce")
            if numeric_series.notna().sum() == 0:
                median_value = 0.0
            else:
                median_value = float(numeric_series.median())
            self.continuous_fill_values_[column] = float(median_value)

    def _apply_imputation(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted missing value rules."""

        transformed = frame.copy()
        for column, fill_value in self.categorical_fill_values_.items():
            if column in transformed.columns:
                transformed[column] = transformed[column].fillna(fill_value).astype(str)

        for column, fill_value in self.continuous_fill_values_.items():
            if column in transformed.columns:
                transformed[column] = pd.to_numeric(
                    transformed[column],
                    errors="coerce",
                ).fillna(fill_value)
        return transformed

    def _fit_outlier_bounds(self, frame: pd.DataFrame) -> None:
        """Fit IQR-based clipping bounds for continuous features."""

        self.outlier_bounds_ = {}
        for column in self.continuous_columns_:
            numeric_series = pd.to_numeric(frame[column], errors="coerce")
            if numeric_series.notna().sum() == 0:
                self.outlier_bounds_[column] = (0.0, 0.0)
                continue
            q1 = numeric_series.quantile(0.25)
            q3 = numeric_series.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                lower_bound = float(numeric_series.min())
                upper_bound = float(numeric_series.max())
            else:
                lower_bound = float(q1 - 1.5 * iqr)
                upper_bound = float(q3 + 1.5 * iqr)
            self.outlier_bounds_[column] = (lower_bound, upper_bound)

    def _apply_outlier_clipping(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Clip continuous features using the fitted IQR bounds."""

        transformed = frame.copy()
        for column, (lower_bound, upper_bound) in self.outlier_bounds_.items():
            if column in transformed.columns:
                transformed[column] = pd.to_numeric(
                    transformed[column],
                    errors="coerce",
                ).clip(lower=lower_bound, upper=upper_bound)
        return transformed

    def _fit_low_cardinality_categories(self, frame: pd.DataFrame) -> None:
        """Store categories for stable one-hot encoding across splits."""

        self.low_cardinality_categories_ = {}
        for column in self.low_cardinality_columns_:
            categories = sorted(frame[column].astype(str).unique().tolist())
            if "__unknown__" not in categories:
                categories.append("__unknown__")
            self.low_cardinality_categories_[column] = categories

    def _fit_target_encoders(
        self,
        frame: pd.DataFrame,
        target_series: pd.Series,
    ) -> None:
        """Fit smoothed mean target encoders for high-cardinality features."""

        if self.high_cardinality_columns_ and target_series.isna().any():
            raise ValueError(
                "Target column contains missing values. Target encoding requires "
                "a fully observed training target."
            )

        self.global_target_mean_ = float(target_series.mean()) if not target_series.empty else 0.0
        self.target_encoding_stats_ = {}

        for column in self.high_cardinality_columns_:
            grouped = (
                pd.DataFrame(
                    {
                        "category": frame[column].astype(str),
                        "target": target_series.astype(float),
                    }
                )
                .groupby("category")["target"]
                .agg(["mean", "count"])
            )
            smoothing = grouped["count"] / (grouped["count"] + self.target_smoothing)
            encoded_value = (
                self.global_target_mean_ * (1.0 - smoothing)
                + grouped["mean"] * smoothing
            )
            self.target_encoding_stats_[column] = TargetEncodingStats(
                mapping=encoded_value.to_dict(),
                global_mean=self.global_target_mean_,
            )

    def _fit_scaler(self, frame: pd.DataFrame) -> None:
        """Fit the standard scaler on clipped continuous features."""

        if self.continuous_columns_:
            self.scaler_.fit(frame[self.continuous_columns_])

    def _encode_low_cardinality_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode low-cardinality categorical features."""

        if not self.low_cardinality_columns_:
            return pd.DataFrame(index=frame.index)

        encoded_parts = []
        for column in self.low_cardinality_columns_:
            categories = self.low_cardinality_categories_[column]
            series = frame[column].astype(str)
            series = series.where(series.isin(categories), "__unknown__")
            categorical = pd.Categorical(series, categories=categories)
            encoded_frame = pd.get_dummies(
                categorical,
                prefix=column,
                prefix_sep="=",
                dtype=np.float32,
            )
            encoded_frame.index = frame.index
            encoded_parts.append(encoded_frame)

        return pd.concat(encoded_parts, axis=1) if encoded_parts else pd.DataFrame(index=frame.index)

    def _encode_high_cardinality_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Target encode high-cardinality categorical features."""

        encoded_frame = pd.DataFrame(index=frame.index)
        for column in self.high_cardinality_columns_:
            stats = self.target_encoding_stats_[column]
            encoded_series = (
                frame[column]
                .astype(str)
                .map(stats.mapping)
                .fillna(stats.global_mean)
                .astype(np.float32)
            )
            encoded_frame[f"{column}__target_encoded"] = encoded_series
        return encoded_frame

    def _scale_continuous_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Scale continuous features using the fitted standard scaler."""

        if not self.continuous_columns_:
            return pd.DataFrame(index=frame.index)

        scaled_array = self.scaler_.transform(frame[self.continuous_columns_])
        scaled_frame = pd.DataFrame(
            scaled_array.astype(np.float32),
            index=frame.index,
            columns=[f"{column}__scaled" for column in self.continuous_columns_],
        )
        return scaled_frame

    def _transform_internal(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Shared internal transformation logic used by fit and transform."""

        available_features = [column for column in self.feature_columns_ if column in frame.columns]
        missing_features = sorted(set(self.feature_columns_) - set(available_features))
        transformed = frame.copy()
        if missing_features:
            missing_frame = pd.DataFrame(
                np.nan,
                index=transformed.index,
                columns=missing_features,
            )
            transformed = pd.concat([transformed, missing_frame], axis=1)

        feature_frame = transformed[self.feature_columns_].copy()
        feature_frame = self._apply_imputation(feature_frame)
        feature_frame = self._apply_outlier_clipping(feature_frame)

        scaled_continuous = self._scale_continuous_features(feature_frame)
        one_hot_categorical = self._encode_low_cardinality_features(feature_frame)
        target_encoded = self._encode_high_cardinality_features(feature_frame)

        transformed_frame = pd.concat(
            [scaled_continuous, one_hot_categorical, target_encoded],
            axis=1,
        )
        transformed_frame = transformed_frame.astype(np.float32)
        return transformed_frame
