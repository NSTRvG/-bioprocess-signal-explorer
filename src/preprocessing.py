"""CSV loading, timestamp validation, and bounded interpolation."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_and_preprocess(
    file_path,
    timestamp_col: str = "timestamp",
    max_gap_minutes: int = 30,
) -> pd.DataFrame:
    """Load a CSV and return sorted numeric signals indexed by timestamp."""
    df = pd.read_csv(file_path)
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column {timestamp_col!r} was not found")

    timestamps = pd.to_datetime(df.pop(timestamp_col), errors="coerce", utc=True)
    invalid_timestamps = timestamps.isna()
    if invalid_timestamps.any():
        raise ValueError(
            f"Found {int(invalid_timestamps.sum())} invalid or missing timestamps"
        )
    df.index = pd.DatetimeIndex(timestamps, name=timestamp_col)
    df = df.sort_index(kind="stable")
    if not df.index.is_unique:
        raise ValueError("Duplicate timestamps are not supported")

    numeric_columns = {}
    for column in df.columns:
        converted = pd.to_numeric(df[column], errors="coerce")
        invalid_values = df[column].notna() & converted.isna()
        if invalid_values.any() and converted.notna().any():
            invalid_count = int(invalid_values.sum())
            raise ValueError(
                f"Found {invalid_count} non-numeric value(s) in measurement "
                f"column {column!r}"
            )
        if (
            pd.api.types.is_numeric_dtype(df[column])
            or converted.notna().any()
        ):
            numeric_columns[column] = converted

    numeric = pd.DataFrame(numeric_columns, index=df.index)
    if numeric.empty:
        raise ValueError("No numeric measurement columns were found")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return handle_missing_values(numeric, max_gap_minutes=max_gap_minutes)


def handle_missing_values(
    df: pd.DataFrame,
    max_gap_minutes: int = 30,
) -> pd.DataFrame:
    """Interpolate only internal missing runs whose bounding gap is short."""
    if max_gap_minutes < 0:
        raise ValueError("max_gap_minutes cannot be negative")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    if not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise ValueError("DataFrame timestamps must be sorted and unique")

    result = df.copy()
    for column in result.columns:
        values = result[column].copy()
        missing_positions = np.flatnonzero(values.isna().to_numpy())
        if missing_positions.size == 0:
            continue

        breaks = np.flatnonzero(np.diff(missing_positions) > 1) + 1
        for run in np.split(missing_positions, breaks):
            left = int(run[0]) - 1
            right = int(run[-1]) + 1
            if left < 0 or right >= len(values):
                continue
            bounding_gap = (
                result.index[right] - result.index[left]
            ).total_seconds() / 60
            if bounding_gap <= max_gap_minutes:
                segment = values.iloc[left : right + 1].interpolate(method="time")
                values.iloc[left : right + 1] = segment
        result[column] = values
    return result


def validate_data(df: pd.DataFrame) -> dict:
    """Return a summary suitable for the dashboard's validation panel."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    if len(df) == 0:
        start = end = None
        duration_hours = 0.0
    else:
        start = df.index.min()
        end = df.index.max()
        duration_hours = (end - start).total_seconds() / 3600

    missing_counts = df.isna().sum()
    denominator = max(len(df), 1)
    return {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": df.columns.tolist(),
        "time_range": {
            "start": start,
            "end": end,
            "duration_hours": duration_hours,
        },
        "missing_values": missing_counts.to_dict(),
        "missing_percentage": (missing_counts / denominator * 100)
        .round(2)
        .to_dict(),
    }
