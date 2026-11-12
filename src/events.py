"""Grouping of out-of-band samples into excursion events."""

import pandas as pd

from .models import (
    DetectionConfig,
    ExcursionEvent,
    Severity,
    calculate_severity,
)


def group_events(
    df: pd.DataFrame,
    variable: str,
    is_outlier: pd.Series,
    upper_band: pd.Series,
    lower_band: pd.Series,
    config: DetectionConfig,
) -> list[ExcursionEvent]:
    """Group outliers separated by no more than the configured time gap."""
    outlier_positions = [
        position
        for position, is_outlier_value in enumerate(is_outlier.fillna(False))
        if is_outlier_value
    ]
    if not outlier_positions:
        return []

    groups: list[list[int]] = []
    current_group = [outlier_positions[0]]
    previous_position = outlier_positions[0]
    previous_direction = _direction(
        df, variable, upper_band, lower_band, previous_position
    )

    for position in outlier_positions[1:]:
        time_gap = (
            df.index[position] - df.index[previous_position]
        ).total_seconds() / 60
        direction = _direction(df, variable, upper_band, lower_band, position)
        if (
            time_gap <= config.max_gap_minutes
            and direction == previous_direction
        ):
            current_group.append(position)
        else:
            groups.append(current_group)
            current_group = [position]
        previous_position = position
        previous_direction = direction
    groups.append(current_group)

    events: list[ExcursionEvent] = []
    for positions in groups:
        if len(positions) < config.min_event_points:
            continue
        events.append(
            create_event(
                event_id=len(events) + 1,
                variable=variable,
                positions=positions,
                df=df,
                upper_band=upper_band,
                lower_band=lower_band,
            )
        )
    return events


def _direction(
    df: pd.DataFrame,
    variable: str,
    upper_band: pd.Series,
    lower_band: pd.Series,
    position: int,
) -> str:
    if df[variable].iloc[position] > upper_band.iloc[position]:
        return "high"
    if df[variable].iloc[position] < lower_band.iloc[position]:
        return "low"
    raise ValueError("An outlier must be outside one of its control bands")


def create_event(
    event_id: int,
    variable: str,
    positions: list[int],
    df: pd.DataFrame,
    upper_band: pd.Series,
    lower_band: pd.Series,
) -> ExcursionEvent:
    """Build an event from outlier positions that share one direction."""
    values = df[variable].iloc[positions]
    uppers = upper_band.iloc[positions]
    lowers = lower_band.iloc[positions]
    direction = _direction(df, variable, upper_band, lower_band, positions[0])

    if direction == "high":
        deviations = values - uppers
        max_position = int(deviations.to_numpy().argmax())
        max_value = float(values.iloc[max_position])
    else:
        deviations = lowers - values
        max_position = int(deviations.to_numpy().argmax())
        max_value = float(values.iloc[max_position])

    start_time = df.index[positions[0]]
    end_time = df.index[positions[-1]]
    duration = (end_time - start_time).total_seconds() / 60
    band_width = float((uppers - lowers).mean())
    max_deviation = float(deviations.iloc[max_position])

    return ExcursionEvent(
        event_id=event_id,
        variable=variable,
        start_time=start_time,
        end_time=end_time,
        duration_minutes=duration,
        direction=direction,
        max_value=max_value,
        max_deviation=max_deviation,
        severity=calculate_severity(max_deviation, band_width),
        n_points=len(positions),
    )
