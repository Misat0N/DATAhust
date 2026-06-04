"""Build risk and business labels for Lending Club split datasets.

This module reads the pre-split ``train.csv``, ``val.csv``, and ``test.csv``
files from ``data/processed``, constructs:

1. A 4-class pre-loan risk label with merged class 2/3.
2. A calibrated credit limit label based on requested amount, DTI, income,
   and the risk label.
3. A 4-class post-loan operational scenario label.

The source files can still be large, so labels are added in chunks and the
original split files are overwritten only after successful processing.

Important task-definition note
------------------------------
The labels in this file are derived from post-origination observations such as
``loan_status`` and delinquency-related fields. They are valid as supervised
training targets, but these post-loan fields must never be reused as model
inputs in the pre-loan risk classification or credit limit estimation tasks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHUNK_SIZE = 100_000

RISK_LABEL_NAMES = {
    0: "正常类",
    1: "关注类",
    2: "次级/可疑类",
    3: "损失类",
}

POST_LOAN_LABEL_NAMES = {
    0: "正常维护",
    1: "风险预警",
    2: "逾期催收",
    3: "协商还款",
}


def get_numeric_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    """Safely convert a column to numeric values."""

    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def normalize_status_series(frame: pd.DataFrame) -> pd.Series:
    """Normalize ``loan_status`` for stable rule-based matching."""

    if "loan_status" not in frame.columns:
        return pd.Series("", index=frame.index, dtype="object")
    return frame["loan_status"].fillna("").astype(str).str.strip().str.lower()


def estimate_overdue_days(frame: pd.DataFrame) -> pd.Series:
    """Estimate overdue days from status and delinquency-related fields.

    Lending Club does not expose a single clean "days past due" field for every
    record. This function therefore builds a conservative estimate using:

    - ``loan_status`` as the primary delinquency signal.
    - ``hardship_dpd`` when available because it directly reflects DPD.
    - ``delinq_amnt`` and ``installment`` to infer the approximate number of
      missed installments for records in late buckets.
    """

    status = normalize_status_series(frame)
    hardship_dpd = get_numeric_series(frame, "hardship_dpd", default=0.0).clip(lower=0)
    delinq_amnt = get_numeric_series(frame, "delinq_amnt", default=0.0).clip(lower=0)
    installment = get_numeric_series(frame, "installment", default=np.nan)

    inferred_from_amount = (
        np.ceil(delinq_amnt / installment.replace(0, np.nan)) * 30
    )
    inferred_from_amount = pd.Series(
        inferred_from_amount,
        index=frame.index,
        dtype="float64",
    ).replace([np.inf, -np.inf], np.nan)

    overdue_days = hardship_dpd.astype("float64")

    in_grace_mask = status.eq("in grace period")
    late_16_30_mask = status.eq("late (16-30 days)")
    late_31_120_mask = status.eq("late (31-120 days)")
    charged_off_mask = status.eq("charged off")
    default_mask = status.eq("default")
    policy_charged_off_mask = status.eq(
        "does not meet the credit policy. status:charged off"
    )

    overdue_days = overdue_days.mask(in_grace_mask & overdue_days.eq(0), 15)
    overdue_days = overdue_days.mask(late_16_30_mask & overdue_days.eq(0), 23)

    late_31_120_estimate = inferred_from_amount.clip(lower=31, upper=120).fillna(60)
    overdue_days = overdue_days.mask(
        late_31_120_mask & overdue_days.eq(0),
        late_31_120_estimate,
    )

    severe_loss_mask = charged_off_mask | default_mask | policy_charged_off_mask
    overdue_days = overdue_days.mask(severe_loss_mask & overdue_days.lt(181), 181)

    residual_delinq_mask = overdue_days.eq(0) & delinq_amnt.gt(0)
    overdue_days = overdue_days.mask(residual_delinq_mask, 15)

    return overdue_days.fillna(0).astype(int)


TERMINAL_STATUSES = {
    "fully paid",
    "charged off",
    "default",
    "does not meet the credit policy. status:fully paid",
    "does not meet the credit policy. status:charged off",
}


def parse_month_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Parse a Lending Club ``Mon-YYYY`` date column into datetimes."""

    if column not in frame.columns:
        return pd.Series(pd.NaT, index=frame.index)
    return pd.to_datetime(frame[column], format="%b-%Y", errors="coerce")


def parse_term_months(frame: pd.DataFrame) -> pd.Series:
    """Extract the loan term length in months from the ``term`` column."""

    if "term" not in frame.columns:
        return pd.Series(36.0, index=frame.index, dtype="float64")
    return (
        frame["term"]
        .astype(str)
        .str.extract(r"(\d+)")[0]
        .astype("float64")
        .fillna(36.0)
    )


def compute_maturity_mask(
    frame: pd.DataFrame,
    observation_date: pd.Timestamp | None = None,
) -> pd.Series:
    """Return a boolean mask marking records whose risk outcome is observable.

    Rationale
    ---------
    Risk labels are derived from post-loan performance. Loans that are still
    ``Current`` and have not yet reached their scheduled maturity are labeled
    ``0`` (正常类) only because they have not had time to default. Keeping these
    "not-yet-matured" records makes the recent (test) period look artificially
    safe and trains the model to predict the normal class for everyone.

    A record is considered observable when ANY of the following holds:
    - Its ``loan_status`` is a terminal state (outcome already known).
    - It already shows a delinquency signal (estimated overdue days > 0).
    - Its scheduled maturity (``issue_d`` + ``term``) is on or before the
      observation/snapshot date, i.e. it has had a full window to perform.
    """

    status = normalize_status_series(frame)
    issue_date = parse_month_series(frame, "issue_d")
    term_months = parse_term_months(frame)
    scheduled_maturity = issue_date + pd.to_timedelta(term_months * 30, unit="D")

    if observation_date is None:
        last_pull = parse_month_series(frame, "last_credit_pull_d")
        observation_date = last_pull.max()
        if pd.isna(observation_date):
            observation_date = issue_date.max()

    terminal_mask = status.isin(TERMINAL_STATUSES)

    overdue_days = (
        frame["estimated_overdue_days"]
        if "estimated_overdue_days" in frame.columns
        else estimate_overdue_days(frame)
    )
    delinquency_mask = pd.Series(overdue_days, index=frame.index).fillna(0).gt(0)

    matured_mask = scheduled_maturity.le(observation_date)

    return (terminal_mask | delinquency_mask | matured_mask).fillna(False)


def build_risk_labels(frame: pd.DataFrame, overdue_days: pd.Series) -> pd.DataFrame:
    """Create the formal 4-class pre-loan risk label.

    The label is constructed from observed post-loan performance. This is
    acceptable for target generation, but the source fields used here must be
    excluded from model features to avoid target leakage.

    Final mapping
    -------------
    - 0: 正常类, no delinquency and no realized loss signal.
    - 1: 关注类, delinquency bucket from 1 to 30 days.
    - 2: 次级/可疑类, merged delinquency bucket from 31 to 180 days.
    - 3: 损失类, charged-off / default / settlement / recoveries / >180 DPD.
    """

    status = normalize_status_series(frame)
    recoveries = get_numeric_series(frame, "recoveries", default=0.0)
    debt_settlement_flag = (
        frame.get("debt_settlement_flag", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
        .str.upper()
    )
    settlement_status = frame.get(
        "settlement_status",
        pd.Series(np.nan, index=frame.index),
    )

    risk_label = np.zeros(len(frame), dtype=int)
    risk_label[(overdue_days >= 1) & (overdue_days <= 30)] = 1
    risk_label[(overdue_days >= 31) & (overdue_days <= 180)] = 2

    loss_status_mask = status.isin(
        {
            "charged off",
            "default",
            "does not meet the credit policy. status:charged off",
        }
    )
    loss_mask = (
        loss_status_mask
        | overdue_days.gt(180)
        | debt_settlement_flag.eq("Y")
        | settlement_status.notna()
        | recoveries.gt(0)
    )
    risk_label[loss_mask] = 3

    frame["preloan_risk_label"] = risk_label
    frame["preloan_risk_label_name"] = frame["preloan_risk_label"].map(
        RISK_LABEL_NAMES
    )
    frame["estimated_overdue_days"] = overdue_days
    return frame


def build_credit_limit_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Create calibrated credit limit labels using amount, DTI, income, and risk."""

    loan_amnt = get_numeric_series(frame, "loan_amnt", default=0.0).clip(lower=0)
    annual_inc = get_numeric_series(frame, "annual_inc", default=0.0).clip(lower=0)
    dti = get_numeric_series(frame, "dti", default=0.0).clip(lower=0)

    # Lending Club stores DTI as percentages such as 18.24, so convert to ratio.
    dti_ratio = (dti / 100.0).clip(lower=0, upper=1.5)

    # Higher DTI reduces affordable credit room. The floor avoids collapsing
    # every high-DTI account to zero, which would be too aggressive for a label.
    affordability_ratio = (0.45 - 0.30 * dti_ratio).clip(lower=0.08, upper=0.45)
    affordability_limit = annual_inc * affordability_ratio

    risk_multiplier_map = {
        0: 1.00,
        1: 0.85,
        2: 0.55,
        3: 0.20,
    }
    risk_multiplier = frame["preloan_risk_label"].map(risk_multiplier_map).fillna(0.20)

    base_limit = np.minimum(loan_amnt, affordability_limit)
    calibrated_limit = base_limit * risk_multiplier
    calibrated_limit = np.where(calibrated_limit > 0, np.maximum(calibrated_limit, 500), 0)
    calibrated_limit = (np.round(calibrated_limit / 100.0) * 100).astype(int)

    frame["calibrated_credit_limit"] = calibrated_limit
    frame["calibrated_credit_limit_band"] = pd.cut(
        frame["calibrated_credit_limit"],
        bins=[-1, 5_000, 10_000, 20_000, 50_000, np.inf],
        labels=["<=5k", "5k-10k", "10k-20k", "20k-50k", "50k+"],
    ).astype(str)
    return frame


def build_post_loan_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the 4-class post-loan operational scenario label."""

    hardship_flag = (
        frame.get("hardship_flag", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
        .str.upper()
    )
    debt_settlement_flag = (
        frame.get("debt_settlement_flag", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
        .str.upper()
    )
    settlement_status = frame.get(
        "settlement_status",
        pd.Series(np.nan, index=frame.index),
    )

    risk_label = frame["preloan_risk_label"]
    overdue_days = frame["estimated_overdue_days"]

    post_loan_label = np.zeros(len(frame), dtype=int)

    negotiation_mask = (
        hardship_flag.eq("Y")
        | debt_settlement_flag.eq("Y")
        | settlement_status.notna()
        | risk_label.eq(3)
    )
    collection_mask = (~negotiation_mask) & (
        overdue_days.gt(30) | risk_label.eq(2)
    )
    warning_mask = (~negotiation_mask) & (~collection_mask) & (
        overdue_days.gt(0) | risk_label.eq(1)
    )

    post_loan_label[warning_mask] = 1
    post_loan_label[collection_mask] = 2
    post_loan_label[negotiation_mask] = 3

    frame["postloan_scene_label"] = post_loan_label
    frame["postloan_scene_label_name"] = frame["postloan_scene_label"].map(
        POST_LOAN_LABEL_NAMES
    )
    return frame


def enrich_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Add all requested labels to a chunk."""

    overdue_days = estimate_overdue_days(frame)
    frame = build_risk_labels(frame, overdue_days)
    frame = build_credit_limit_labels(frame)
    frame = build_post_loan_labels(frame)
    return frame


def overwrite_with_labeled_file(input_path: Path) -> None:
    """Read a split file in chunks, add labels, and overwrite the original file."""

    temp_output_path = input_path.with_suffix(".tmp")
    if temp_output_path.exists():
        temp_output_path.unlink()

    chunk_index = 0
    for chunk in pd.read_csv(input_path, chunksize=CHUNK_SIZE, low_memory=False):
        labeled_chunk = enrich_labels(chunk)
        labeled_chunk.to_csv(
            temp_output_path,
            mode="a",
            index=False,
            header=chunk_index == 0,
        )
        chunk_index += 1

    temp_output_path.replace(input_path)


def main() -> None:
    """Build labels for all processed data splits."""

    for file_name in ("train.csv", "val.csv", "test.csv"):
        input_path = PROCESSED_DIR / file_name
        if not input_path.exists():
            raise FileNotFoundError(
                f"Expected split file does not exist: {input_path}. "
                "Please run src/data_loader.py first."
            )

        print(f"Building labels for {file_name} ...")
        overwrite_with_labeled_file(input_path)

    print("Label construction completed for train.csv, val.csv, and test.csv.")


if __name__ == "__main__":
    main()
