"""Download, inspect, anonymize, and temporally split Lending Club data.

This module uses ``kagglehub`` to download the Lending Club dataset, removes
direct identifier columns, and creates strict 70% / 15% / 15% chronological
train, validation, and test splits based on ``issue_d``.

The source CSV is large, so the implementation uses chunked processing to keep
memory usage under control while preserving ascending time order in outputs.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import kagglehub
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TEMP_DIR = PROCESSED_DIR / "_temp_split_parts"

CHUNK_SIZE = 50_000
IDENTIFIER_COLUMNS = {
    "id",
    "member_id",
    "url",
}


@dataclass(frozen=True)
class SplitTargets:
    """Target row counts for the train/validation/test partitions."""

    train: int
    val: int
    test: int


def download_dataset() -> Path:
    """Download the dataset via kagglehub and return the source CSV path."""

    dataset_dir = Path(
        kagglehub.dataset_download("adarshsng/lending-club-loan-data-csv")
    )
    csv_candidates = sorted(dataset_dir.glob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(
            f"No CSV file was found in downloaded dataset directory: {dataset_dir}"
        )

    csv_path = csv_candidates[0]
    metadata = {
        "dataset_dir": str(dataset_dir),
        "source_csv": str(csv_path),
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "source_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return csv_path


def inspect_dataset(csv_path: Path) -> None:
    """Print basic dataset information using a lightweight sample."""

    print("=== Dataset info sample ===")
    sample_df = pd.read_csv(csv_path, nrows=1000, low_memory=False)
    sample_df.info()
    print("\n=== Dataset head ===")
    print(sample_df.head())


def get_columns_to_keep(csv_path: Path) -> list[str]:
    """Return the output columns after dropping direct identifier fields."""

    all_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    return [column for column in all_columns if column not in IDENTIFIER_COLUMNS]


def parse_issue_d(series: pd.Series) -> pd.Series:
    """Parse Lending Club ``issue_d`` values such as ``Dec-2018`` to timestamps."""

    return pd.to_datetime(series, format="%b-%Y", errors="coerce")


def build_issue_month_counts(csv_path: Path) -> tuple[pd.Series, int]:
    """Count valid records by issue month and return invalid row count."""

    month_counts: Dict[pd.Timestamp, int] = {}
    invalid_rows = 0

    for chunk in pd.read_csv(
        csv_path,
        usecols=["issue_d"],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        issue_dt = parse_issue_d(chunk["issue_d"])
        invalid_rows += int(issue_dt.isna().sum())
        valid_counts = issue_dt.dropna().value_counts()

        for issue_month, count in valid_counts.items():
            month_counts[issue_month] = month_counts.get(issue_month, 0) + int(count)

    counts_series = pd.Series(month_counts, dtype="int64").sort_index()
    return counts_series, invalid_rows


def compute_split_targets(total_rows: int) -> SplitTargets:
    """Convert the requested 70/15/15 ratio into exact row counts."""

    train_rows = int(total_rows * 0.70)
    val_rows = int(total_rows * 0.15)
    test_rows = total_rows - train_rows - val_rows
    return SplitTargets(train=train_rows, val=val_rows, test=test_rows)


def allocate_rows_by_month(
    month_counts: pd.Series,
    targets: SplitTargets,
) -> dict[pd.Timestamp, dict[str, int]]:
    """Allocate each issue month to train/val/test while keeping exact counts.

    Allocation follows ascending issue month order. When a month crosses a split
    boundary, the month is split internally in file order so that final counts
    still match the exact requested ratio.
    """

    remaining_train = targets.train
    remaining_val = targets.val
    remaining_test = targets.test
    allocation: dict[pd.Timestamp, dict[str, int]] = {}

    for issue_month, count in month_counts.items():
        rows_left = int(count)
        train_take = min(remaining_train, rows_left)
        rows_left -= train_take
        remaining_train -= train_take

        val_take = min(remaining_val, rows_left)
        rows_left -= val_take
        remaining_val -= val_take

        test_take = min(remaining_test, rows_left)
        rows_left -= test_take
        remaining_test -= test_take

        allocation[issue_month] = {
            "train": train_take,
            "val": val_take,
            "test": test_take,
        }

        if rows_left != 0:
            raise ValueError(
                f"Unexpected unallocated rows for month {issue_month}: {rows_left}"
            )

    if any(value != 0 for value in (remaining_train, remaining_val, remaining_test)):
        raise ValueError(
            "Split allocation failed to match requested train/val/test counts."
        )

    return allocation


def reset_output_paths(split_names: Iterable[str]) -> None:
    """Remove old output files and temporary split fragments."""

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for split_name in split_names:
        output_path = PROCESSED_DIR / f"{split_name}.csv"
        if output_path.exists():
            output_path.unlink()


def get_month_part_path(split_name: str, issue_month: pd.Timestamp) -> Path:
    """Return the temporary CSV path for one split and one issue month."""

    month_key = issue_month.strftime("%Y-%m")
    month_dir = TEMP_DIR / split_name
    month_dir.mkdir(parents=True, exist_ok=True)
    return month_dir / f"{month_key}.csv"


def append_partition_rows(part_path: Path, frame: pd.DataFrame) -> None:
    """Append rows to a temporary partition CSV fragment."""

    frame.to_csv(part_path, mode="a", index=False, header=not part_path.exists())


def write_partition_parts(
    csv_path: Path,
    keep_columns: list[str],
    allocation: dict[pd.Timestamp, dict[str, int]],
) -> None:
    """Write split-specific monthly CSV fragments using chunked processing."""

    remaining_allocation = {
        month: split_counts.copy() for month, split_counts in allocation.items()
    }

    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE, low_memory=False):
        chunk["issue_d_parsed"] = parse_issue_d(chunk["issue_d"])
        chunk = chunk.loc[chunk["issue_d_parsed"].notna()].copy()
        if chunk.empty:
            continue

        chunk = chunk.drop(columns=list(IDENTIFIER_COLUMNS), errors="ignore")
        chunk = chunk[keep_columns + ["issue_d_parsed"]]

        for issue_month, month_frame in chunk.groupby("issue_d_parsed", sort=False):
            available_counts = remaining_allocation.get(issue_month)
            if available_counts is None:
                raise KeyError(f"Allocation plan missing issue month: {issue_month}")

            month_frame = month_frame.drop(columns=["issue_d_parsed"]).copy()
            cursor = 0

            for split_name in ("train", "val", "test"):
                rows_for_split = available_counts[split_name]
                if rows_for_split <= 0:
                    continue

                split_frame = month_frame.iloc[cursor : cursor + rows_for_split]
                if not split_frame.empty:
                    append_partition_rows(
                        get_month_part_path(split_name, issue_month),
                        split_frame,
                    )
                    cursor += len(split_frame)
                    available_counts[split_name] -= len(split_frame)

            if cursor != len(month_frame):
                raise ValueError(
                    "Monthly partitioning failed because not all rows were consumed "
                    f"for month {issue_month}."
                )


def concatenate_parts(
    split_name: str,
    ordered_months: Iterable[pd.Timestamp],
    keep_columns: list[str],
) -> int:
    """Concatenate monthly fragments into a final chronologically sorted CSV."""

    output_path = PROCESSED_DIR / f"{split_name}.csv"
    rows_written = 0
    header_written = False

    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        for issue_month in ordered_months:
            part_path = get_month_part_path(split_name, issue_month)
            if not part_path.exists():
                continue

            with part_path.open("r", encoding="utf-8") as part_file:
                for line_number, line in enumerate(part_file):
                    if line_number == 0:
                        if header_written:
                            continue
                        header_written = True
                    else:
                        rows_written += 1
                    output_file.write(line)

    if not header_written:
        pd.DataFrame(columns=keep_columns).to_csv(output_path, index=False)

    return rows_written


def build_temporal_splits(csv_path: Path) -> None:
    """Create strict chronological train/val/test CSV files."""

    keep_columns = get_columns_to_keep(csv_path)
    month_counts, invalid_rows = build_issue_month_counts(csv_path)
    total_valid_rows = int(month_counts.sum())
    targets = compute_split_targets(total_valid_rows)
    allocation = allocate_rows_by_month(month_counts, targets)

    print("\n=== Split planning summary ===")
    print(f"Valid rows with parsable issue_d: {total_valid_rows:,}")
    print(f"Rows dropped due to invalid issue_d: {invalid_rows:,}")
    print(
        "Target split sizes: "
        f"train={targets.train:,}, val={targets.val:,}, test={targets.test:,}"
    )

    reset_output_paths(("train", "val", "test"))
    write_partition_parts(csv_path, keep_columns, allocation)

    ordered_months = list(month_counts.index)
    train_rows = concatenate_parts("train", ordered_months, keep_columns)
    val_rows = concatenate_parts("val", ordered_months, keep_columns)
    test_rows = concatenate_parts("test", ordered_months, keep_columns)

    print("\n=== Output summary ===")
    print(f"train.csv rows: {train_rows:,}")
    print(f"val.csv rows: {val_rows:,}")
    print(f"test.csv rows: {test_rows:,}")

    if (train_rows, val_rows, test_rows) != (
        targets.train,
        targets.val,
        targets.test,
    ):
        raise ValueError("Final split counts do not match the requested ratios.")

    shutil.rmtree(TEMP_DIR, ignore_errors=True)


def main() -> None:
    """Run the end-to-end data loading workflow."""

    csv_path = download_dataset()
    print(f"Downloaded source CSV: {csv_path}")
    inspect_dataset(csv_path)
    build_temporal_splits(csv_path)


if __name__ == "__main__":
    main()
