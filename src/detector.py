"""Causal rolling-band excursion detection."""

import numpy as np
import pandas as pd

from .events import group_events
from .models import (
    DetectionConfig,
    DetectionMethod,
    DetectionResult,
)


def _validate_series(series: pd.Series, min_periods: int, window: str) -> None:
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("series must have a DatetimeIndex")
    if not series.index.is_monotonic_increasing:
        raise ValueError("series timestamps must be sorted chronologically")
    if not series.index.is_unique:
        raise ValueError("series timestamps must be unique")
    if min_periods < 1:
        raise ValueError("min_periods must be at least 1")
    try:
        if pd.Timedelta(window) <= pd.Timedelta(0):
            raise ValueError("window must be a positive duration")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid rolling window: {window!r}") from exc


def _median_absolute_deviation(values: np.ndarray) -> float:
    center = np.median(values)
    return float(np.median(np.abs(values - center)))


def calculate_robust_band(
    series: pd.Series,
    window: str,
    k: float,
    min_periods: int,
) -> tuple[pd.Series, pd.Series]:
    """Calculate causal rolling median +/- scaled rolling MAD bands."""
    _validate_series(series, min_periods, window)
    if not np.isfinite(k) or k <= 0:
        raise ValueError("k must be a finite number greater than zero")

    numeric = pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    rolling = numeric.rolling(
        window=window,
        min_periods=min_periods,
        closed="left",
    )
    rolling_median = rolling.median()
    rolling_mad = rolling.apply(_median_absolute_deviation, raw=True)
    scaled_mad = rolling_mad * 1.4826
    return rolling_median + k * scaled_mad, rolling_median - k * scaled_mad


def calculate_classic_band(
    series: pd.Series,
    window: str,
    k: float,
    min_periods: int,
) -> tuple[pd.Series, pd.Series]:
    """Calculate causal rolling mean +/- standard-deviation bands."""
    _validate_series(series, min_periods, window)
    if not np.isfinite(k) or k <= 0:
        raise ValueError("k must be a finite number greater than zero")

    numeric = pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    rolling = numeric.rolling(
        window=window,
        min_periods=min_periods,
        closed="left",
    )
    rolling_mean = rolling.mean()
    rolling_std = rolling.std()
    return rolling_mean + k * rolling_std, rolling_mean - k * rolling_std


def detect_excursions(
    df: pd.DataFrame,
    variable: str,
    config: DetectionConfig,
) -> DetectionResult:
    """Detect out-of-band points and group them into events."""
    if variable not in df.columns:
        raise ValueError(f"Variable {variable!r} not found in DataFrame")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame timestamps must be sorted chronologically")
    if not df.index.is_unique:
        raise ValueError("DataFrame timestamps must be unique")

    series = pd.to_numeric(df[variable], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    band_calculator = (
        calculate_robust_band
        if config.method == DetectionMethod.ROBUST
        else calculate_classic_band
    )
    upper_band, lower_band = band_calculator(
        series, config.window, config.k, config.min_periods
    )
    is_outlier = (
        ((series > upper_band) | (series < lower_band))
        & upper_band.notna()
        & lower_band.notna()
        & series.notna()
    ).astype(bool)

    event_df = df.copy()
    event_df[variable] = series
    events = group_events(
        df=event_df,
        variable=variable,
        is_outlier=is_outlier,
        upper_band=upper_band,
        lower_band=lower_band,
        config=config,
    )
    return DetectionResult(
        variable=variable,
        timestamps=df.index.to_list(),
        values=series.to_list(),
        upper_band=upper_band.to_list(),
        lower_band=lower_band.to_list(),
        is_outlier=is_outlier.to_list(),
        events=events,
    )
