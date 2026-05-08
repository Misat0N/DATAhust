"""Sample purification utilities for Lending Club credit risk training data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
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


class CreditSamplePurifier:
    """Purify noisy normal-class samples before model training.

    The purifier is designed for Lending Club style loan-book data. Its goal is
    to remove weak or low-information normal samples while keeping the genuine
    distribution of healthy borrowers as intact as possible. The workflow is:

    1. Remove invalid or low-information normal samples.
    2. Apply stratified down-sampling on the remaining normal pool.
    3. Keep all risky samples unchanged.
    """

    NORMAL_STATUSES = {"fully paid", "current"}
    CORE_FEATURES = ["annual_inc", "dti", "delinq_2yrs", "inq_last_6mths"]

    def __init__(self) -> None:
        """Initialize purification thresholds and deterministic sampling state."""

        self.minimum_observation_months = 6
        self.minimum_repayment_periods = 3
        self.minimum_utilization_pct = 1.0
        self.abnormal_dti_threshold = 80.0
        self.target_normal_risk_ratio = 4.0
        self.random_state = 42

    def filter_invalid_normal_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove noisy normal samples and records with invalid core features.

        Parameters
        ----------
        df:
            Input Lending Club style dataframe. The method expects raw or
            processed columns such as ``loan_status``, ``issue_d``,
            ``last_pymnt_d``, ``revol_util``, ``annual_inc``, ``dti``,
            ``delinq_2yrs``, and ``inq_last_6mths``.

        Returns
        -------
        pd.DataFrame
            Purified dataframe with problematic normal samples removed.

        Purification rules
        ------------------
        1. Remove normal samples whose observed loan age is under 6 months.
        2. Remove normal samples with fewer than 3 observed repayment periods.
        3. Remove normal samples whose utilization rate is under 1 percent.
        4. Remove any sample with missing core features:
           ``annual_inc``, ``dti``, ``delinq_2yrs``, ``inq_last_6mths``.
        5. Remove normal samples with ``dti > 80`` and no delinquency records.
        """

        if df.empty:
            return df.copy()

        purified = df.copy()
        normal_mask = self._get_normal_mask(purified)
        observed_months = self._estimate_observed_months(purified)
        repayment_periods = self._estimate_repayment_periods(purified, observed_months)
        utilization_pct = self._get_numeric_series(purified, "revol_util", np.nan)
        dti = self._get_numeric_series(purified, "dti", np.nan)
        delinq_2yrs = self._get_numeric_series(purified, "delinq_2yrs", np.nan)
        acc_now_delinq = self._get_numeric_series(purified, "acc_now_delinq", 0.0)

        core_feature_frame = purified.reindex(columns=self.CORE_FEATURES)
        missing_core_mask = core_feature_frame.isna().any(axis=1)
        too_fresh_mask = normal_mask & observed_months.lt(self.minimum_observation_months)
        too_few_payments_mask = normal_mask & repayment_periods.lt(
            self.minimum_repayment_periods
        )
        zombie_mask = normal_mask & utilization_pct.lt(self.minimum_utilization_pct)
        abnormal_clean_mask = (
            normal_mask
            & dti.gt(self.abnormal_dti_threshold)
            & delinq_2yrs.fillna(0).le(0)
            & acc_now_delinq.fillna(0).le(0)
        )

        removal_mask = (
            missing_core_mask
            | too_fresh_mask
            | too_few_payments_mask
            | zombie_mask
            | abnormal_clean_mask
        )
        return purified.loc[~removal_mask].copy()

    def stratified_sampling_normal_samples(
        self,
        df: pd.DataFrame,
        target_ratio: float = 4,
    ) -> pd.DataFrame:
        """Down-sample normal samples by strata while keeping all risk samples.

        Parameters
        ----------
        df:
            Purified training dataframe.
        target_ratio:
            Desired upper bound for ``normal_count / risk_count``. When the
            current ratio is already below the threshold, the input dataframe is
            returned unchanged.
        """

        if df.empty:
            return df.copy()
        if target_ratio <= 0:
            raise ValueError("target_ratio must be greater than 0.")

        sampled = df.copy()
        normal_mask = self._get_normal_mask(sampled)
        normal_df = sampled.loc[normal_mask].copy()
        risk_df = sampled.loc[~normal_mask].copy()

        if normal_df.empty or risk_df.empty:
            return sampled

        target_normal_count = int(len(risk_df) * target_ratio)
        if len(normal_df) <= target_normal_count:
            return sampled

        stratified_normal = normal_df.copy()
        stratified_normal["grade_bucket"] = (
            stratified_normal.get("grade", pd.Series("Unknown", index=stratified_normal.index))
            .fillna("Unknown")
            .astype(str)
        )
        stratified_normal["annual_inc_bucket"] = self._build_quantile_bucket(
            stratified_normal["annual_inc"],
            prefix="inc",
        )
        stratified_normal["dti_bucket"] = self._build_quantile_bucket(
            stratified_normal["dti"],
            prefix="dti",
        )
        stratified_normal["strata_key"] = (
            stratified_normal["grade_bucket"].astype(str)
            + "|"
            + stratified_normal["annual_inc_bucket"].astype(str)
            + "|"
            + stratified_normal["dti_bucket"].astype(str)
        )

        group_sizes = stratified_normal.groupby("strata_key").size().sort_values(
            ascending=False
        )
        allocations = self._allocate_counts(group_sizes, target_normal_count)

        sampled_groups: list[pd.DataFrame] = []
        for group_index, (strata_key, group_frame) in enumerate(
            stratified_normal.groupby("strata_key", sort=False)
        ):
            sample_size = allocations.get(strata_key, 0)
            if sample_size <= 0:
                continue
            if sample_size >= len(group_frame):
                sampled_groups.append(group_frame)
                continue

            sampled_groups.append(
                group_frame.sample(
                    n=sample_size,
                    random_state=self.random_state + group_index,
                )
            )

        sampled_normal_df = (
            pd.concat(sampled_groups, axis=0).sort_index()
            if sampled_groups
            else stratified_normal.iloc[0:0].copy()
        )

        cleanup_columns = [
            "grade_bucket",
            "annual_inc_bucket",
            "dti_bucket",
            "strata_key",
        ]
        sampled_normal_df = sampled_normal_df.drop(
            columns=[column for column in cleanup_columns if column in sampled_normal_df],
            errors="ignore",
        )

        final_df = (
            pd.concat([sampled_normal_df, risk_df], axis=0)
            .sort_index()
            .reset_index(drop=True)
        )
        return final_df

    def get_sample_distribution(
        self,
        df: pd.DataFrame,
        name: str = "",
    ) -> dict[str, object]:
        """Return and print normal/risk plus detailed label distributions."""

        if df.empty:
            distribution = {
                "total_samples": 0,
                "normal_samples": 0,
                "risk_samples": 0,
                "normal_risk_ratio": float("nan"),
                "label_distribution": {},
                "label_share": {},
            }
            if name:
                print(f"{name}分布: {distribution}")
            return distribution

        normal_mask = self._get_normal_mask(df)
        normal_count = int(normal_mask.sum())
        risk_count = int((~normal_mask).sum())

        if "preloan_risk_label" in df.columns:
            label_distribution = (
                df["preloan_risk_label"].value_counts().sort_index().astype(int).to_dict()
            )
        else:
            label_distribution = {
                0: normal_count,
                1: risk_count,
            }

        total_count = len(df)
        label_share = {
            str(label): round(count / total_count, 6)
            for label, count in label_distribution.items()
        }
        distribution = {
            "total_samples": total_count,
            "normal_samples": normal_count,
            "risk_samples": risk_count,
            "normal_risk_ratio": (
                float(normal_count / risk_count) if risk_count > 0 else float("inf")
            ),
            "label_distribution": label_distribution,
            "label_share": label_share,
        }

        if name:
            print(f"{name}总样本数: {total_count}")
            print(
                f"{name}正常/风险样本数: {normal_count} / {risk_count}, "
                f"比例={distribution['normal_risk_ratio']:.4f}"
                if np.isfinite(distribution["normal_risk_ratio"])
                else f"{name}正常/风险样本数: {normal_count} / {risk_count}, 比例=inf"
            )
            print(f"{name}标签分布: {label_distribution}")
        return distribution

    def plot_distribution_comparison(
        self,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
    ) -> str:
        """Plot class distributions before and after purification."""

        before_distribution = self._get_plot_distribution(df_before)
        after_distribution = self._get_plot_distribution(df_after)

        labels = sorted(set(before_distribution) | set(after_distribution), key=str)
        before_values = [before_distribution.get(label, 0) for label in labels]
        after_values = [after_distribution.get(label, 0) for label in labels]
        positions = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.35
        ax.bar(
            positions - bar_width / 2,
            before_values,
            width=bar_width,
            label="提纯前",
        )
        ax.bar(
            positions + bar_width / 2,
            after_values,
            width=bar_width,
            label="提纯后",
        )
        ax.set_xticks(positions)
        ax.set_xticklabels([str(label) for label in labels])
        ax.set_xlabel("类别")
        ax.set_ylabel("样本数")
        ax.set_title("样本提纯前后标签分布对比")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        output_path = FIGURE_DIR / "sample_purification_comparison.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return "results/figures/sample_purification_comparison.png"

    def _get_normal_mask(self, df: pd.DataFrame) -> pd.Series:
        """Return the normal-sample mask using loan status and risk labels."""

        status = (
            df.get("loan_status", pd.Series("", index=df.index))
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        status_normal_mask = status.isin(self.NORMAL_STATUSES)

        if "preloan_risk_label" in df.columns:
            risk_label = pd.to_numeric(
                df["preloan_risk_label"],
                errors="coerce",
            ).fillna(1)
            return status_normal_mask & risk_label.eq(0)
        return status_normal_mask

    def _estimate_observed_months(self, df: pd.DataFrame) -> pd.Series:
        """Estimate how many months the loan has been observed in the dataset."""

        issue_date = self._get_datetime_series(df, "issue_d")
        last_payment_date = self._get_datetime_series(df, "last_pymnt_d")
        last_pull_date = self._get_datetime_series(df, "last_credit_pull_d")

        observed_end = last_payment_date.fillna(last_pull_date)
        observed_end = observed_end.fillna(issue_date)
        month_delta = (
            (observed_end.dt.year - issue_date.dt.year) * 12
            + (observed_end.dt.month - issue_date.dt.month)
        )
        month_delta = month_delta.fillna(0).clip(lower=0)
        return month_delta.astype(int)

    def _estimate_repayment_periods(
        self,
        df: pd.DataFrame,
        observed_months: pd.Series,
    ) -> pd.Series:
        """Estimate the number of historical repayment periods."""

        total_payment = self._get_numeric_series(df, "total_pymnt", np.nan).clip(lower=0)
        installment = self._get_numeric_series(df, "installment", np.nan)

        inferred_periods = np.floor(
            total_payment / installment.replace(0, np.nan)
        )
        inferred_periods = pd.Series(
            inferred_periods,
            index=df.index,
            dtype="float64",
        ).replace([np.inf, -np.inf], np.nan)

        repayment_periods = inferred_periods.fillna(observed_months.astype(float))
        repayment_periods = np.maximum(repayment_periods, observed_months.astype(float))
        return pd.Series(repayment_periods, index=df.index).fillna(0).clip(lower=0)

    def _build_quantile_bucket(
        self,
        series: pd.Series,
        prefix: str,
    ) -> pd.Series:
        """Create stable quantile buckets with graceful fallback."""

        numeric_series = pd.to_numeric(series, errors="coerce")
        if numeric_series.notna().sum() < 5:
            return pd.Series(f"{prefix}_unknown", index=series.index, dtype="object")

        try:
            bucket = pd.qcut(
                numeric_series,
                q=5,
                duplicates="drop",
            )
            codes = bucket.cat.codes.astype(str)
            codes = codes.where(bucket.notna(), "unknown")
            return prefix + "_" + codes
        except ValueError:
            return pd.Series(f"{prefix}_unknown", index=series.index, dtype="object")

    def _allocate_counts(
        self,
        group_sizes: pd.Series,
        target_total: int,
    ) -> dict[str, int]:
        """Allocate target samples proportionally across strata."""

        if target_total <= 0 or group_sizes.empty:
            return {}

        proportions = group_sizes / group_sizes.sum()
        raw_allocations = proportions * target_total
        allocations = np.floor(raw_allocations).astype(int)
        allocations = allocations.clip(upper=group_sizes)

        remaining = int(target_total - allocations.sum())
        fractional_order = (raw_allocations - allocations).sort_values(ascending=False)

        for strata_key in fractional_order.index:
            if remaining <= 0:
                break
            if allocations.loc[strata_key] >= group_sizes.loc[strata_key]:
                continue
            allocations.loc[strata_key] += 1
            remaining -= 1

        return allocations.astype(int).to_dict()

    def _get_plot_distribution(self, df: pd.DataFrame) -> dict[object, int]:
        """Return label distribution for plotting."""

        if "preloan_risk_label" in df.columns:
            return (
                df["preloan_risk_label"].value_counts().sort_index().astype(int).to_dict()
            )

        normal_mask = self._get_normal_mask(df)
        return {
            "正常": int(normal_mask.sum()),
            "风险": int((~normal_mask).sum()),
        }

    def _get_numeric_series(
        self,
        df: pd.DataFrame,
        column: str,
        default: float,
    ) -> pd.Series:
        """Safely convert a column to numeric values."""

        if column not in df.columns:
            return pd.Series(default, index=df.index, dtype="float64")
        return pd.to_numeric(df[column], errors="coerce")

    def _get_datetime_series(self, df: pd.DataFrame, column: str) -> pd.Series:
        """Safely convert a Lending Club date column to datetime."""

        if column not in df.columns:
            return pd.Series(pd.NaT, index=df.index)
        return pd.to_datetime(df[column], format="%b-%Y", errors="coerce")


def build_purified_splits(target_ratio: float = 4) -> dict[str, dict[str, object]]:
    """Generate purified train/val/test datasets and save them to disk.

    The function always applies rule-based purification to all three splits.
    Only the training split additionally receives stratified normal-class
    sampling so that validation and test remain closer to natural deployment
    distributions.
    """

    purifier = CreditSamplePurifier()
    outputs: dict[str, dict[str, object]] = {}

    for split_name in ("train", "val", "test"):
        source_path = PROCESSED_DIR / f"{split_name}.csv"
        if not source_path.exists():
            raise FileNotFoundError(
                f"Missing processed split: {source_path}. "
                "Please prepare the processed data before purification."
            )

        split_df = pd.read_csv(source_path, low_memory=False)
        before_distribution = purifier.get_sample_distribution(
            split_df,
            name=f"{split_name}提纯前",
        )
        filtered_df = purifier.filter_invalid_normal_samples(split_df)

        if split_name == "train":
            final_df = purifier.stratified_sampling_normal_samples(
                filtered_df,
                target_ratio=target_ratio,
            )
        else:
            final_df = filtered_df.copy()

        output_path = PROCESSED_DIR / f"purified_{split_name}.csv"
        final_df.to_csv(output_path, index=False)
        outputs[split_name] = {
            "path": str(output_path),
            "before_distribution": before_distribution,
            "after_distribution": purifier.get_sample_distribution(
                final_df,
                name=f"{split_name}提纯后",
            ),
        }

    purifier.plot_distribution_comparison(
        pd.read_csv(PROCESSED_DIR / "train.csv", low_memory=False),
        pd.read_csv(PROCESSED_DIR / "purified_train.csv", low_memory=False),
    )
    return outputs


def main() -> None:
    """Entry point for command-line purification of processed datasets."""

    outputs = build_purified_splits(target_ratio=4)
    print("Sample purification completed.")
    for split_name, payload in outputs.items():
        print(f"{split_name}: {payload['path']}")


if __name__ == "__main__":
    main()
