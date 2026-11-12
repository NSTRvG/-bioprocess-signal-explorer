"""Unit tests for causal rolling bands and event grouping."""

import numpy as np
import pandas as pd
import pytest

from src.detector import (
    calculate_classic_band,
    calculate_robust_band,
    detect_excursions,
)
from src.models import DetectionConfig, DetectionMethod


def create_test_series(n_points: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=n_points, freq="5min")
    return pd.DataFrame({"pH": rng.normal(7.0, 0.1, n_points)}, index=timestamps)


def test_stable_series_does_not_create_events() -> None:
    df = create_test_series()
    result = detect_excursions(df, "pH", DetectionConfig())
    assert result.events == []


def test_spike_is_detected_and_grouped() -> None:
    df = create_test_series()
    df.iloc[50, df.columns.get_loc("pH")] = 10.0
    config = DetectionConfig(min_event_points=1)
    result = detect_excursions(df, "pH", config)
    assert any(event.start_time == df.index[50] for event in result.events)


def test_robust_band_uses_previous_samples_and_true_window_mad() -> None:
    series = pd.Series(
        [1.0, 2.0, 3.0, 4.0, 1000.0],
        index=pd.date_range("2024-01-01", periods=5, freq="5min"),
    )
    upper, lower = calculate_robust_band(series, "20min", 1.0, 4)
    expected_median = 2.5
    expected_mad = 1.0 * 1.4826
    assert upper.iloc[-1] == pytest.approx(expected_median + expected_mad)
    assert lower.iloc[-1] == pytest.approx(expected_median - expected_mad)

    changed_current = series.copy()
    changed_current.iloc[-1] = -1000.0
    changed_upper, changed_lower = calculate_robust_band(
        changed_current, "20min", 1.0, 4
    )
    assert changed_upper.iloc[-1] == upper.iloc[-1]
    assert changed_lower.iloc[-1] == lower.iloc[-1]


@pytest.mark.parametrize(
    ("calculator", "min_periods", "position"),
    [
        (calculate_robust_band, 1, 2),
        (calculate_classic_band, 2, 3),
    ],
)
def test_time_window_uses_original_observation_timestamps(
    calculator,
    min_periods: int,
    position: int,
) -> None:
    timestamps = pd.to_datetime(
        [
            "2024-01-01T00:00:00Z",
            "2024-01-01T02:00:00Z",
            "2024-01-01T02:01:00Z",
            "2024-01-01T02:02:00Z",
        ]
    )
    series = pd.Series([100.0, 1.0, 1.0, 1.0], index=timestamps)

    upper, lower = calculator(
        series,
        window="60min",
        k=3.0,
        min_periods=min_periods,
    )

    assert upper.iloc[position] == pytest.approx(1.0)
    assert lower.iloc[position] == pytest.approx(1.0)


def test_opposite_directions_are_not_merged_into_one_event() -> None:
    timestamps = pd.date_range("2024-01-01", periods=12, freq="5min")
    df = pd.DataFrame({"signal": np.zeros(12)}, index=timestamps)
    df.loc[timestamps[6], "signal"] = 10
    df.loc[timestamps[7], "signal"] = -10
    result = detect_excursions(
        df,
        "signal",
        DetectionConfig(
            window="20min",
            min_periods=3,
            min_event_points=1,
            max_gap_minutes=10,
        ),
    )
    directions = {event.direction for event in result.events}
    assert directions == {"high", "low"}


def test_invalid_or_unsorted_time_index_is_rejected() -> None:
    df = create_test_series(5).iloc[::-1]
    with pytest.raises(ValueError, match="sorted chronologically"):
        detect_excursions(df, "pH", DetectionConfig(min_periods=2))
