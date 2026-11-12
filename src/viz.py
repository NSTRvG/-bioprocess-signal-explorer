"""Plotly visualizations for detector results."""

import pandas as pd
import plotly.graph_objects as go

from .models import DetectionResult, ExcursionEvent


def create_band_chart(
    result: DetectionResult,
    df: pd.DataFrame,
    show_events: bool = True,
) -> go.Figure:
    """Create a time-series chart with control bands and flagged samples."""
    del df  # Retained in the signature for compatibility with the documented API.
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.timestamps,
            y=result.values,
            mode="lines",
            name=result.variable,
            line={"color": "#2E86AB", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.timestamps,
            y=result.upper_band,
            mode="lines",
            name="Upper band",
            line={"color": "#A23B72", "width": 1, "dash": "dash"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.timestamps,
            y=result.lower_band,
            mode="lines",
            name="Lower band",
            line={"color": "#A23B72", "width": 1, "dash": "dash"},
            fill="tonexty",
            fillcolor="rgba(162, 59, 114, 0.1)",
        )
    )

    outlier_points = [
        (timestamp, value)
        for timestamp, value, is_outlier in zip(
            result.timestamps, result.values, result.is_outlier
        )
        if is_outlier
    ]
    if outlier_points:
        times, values = zip(*outlier_points)
        fig.add_trace(
            go.Scatter(
                x=times,
                y=values,
                mode="markers",
                name="Excursions",
                marker={"color": "#F18F01", "size": 8, "symbol": "circle-open"},
            )
        )

    if show_events:
        for event in result.events:
            fig.add_vrect(
                x0=event.start_time,
                x1=event.end_time,
                fillcolor="rgba(241, 143, 1, 0.2)",
                layer="below",
                line_width=0,
            )

    fig.update_layout(
        title=f"Excursion detection: {result.variable}",
        xaxis_title="Time",
        yaxis_title=result.variable,
        hovermode="x unified",
        height=400,
    )
    return fig


def create_events_table(events: list[ExcursionEvent]) -> pd.DataFrame:
    """Convert events into a displayable and downloadable table."""
    columns = [
        "ID",
        "Variable",
        "Start",
        "End",
        "Duration (min)",
        "Direction",
        "Extreme value",
        "Deviation",
        "Severity",
    ]
    rows = [
        {
            "ID": event.event_id,
            "Variable": event.variable,
            "Start": event.start_time.strftime("%Y-%m-%d %H:%M"),
            "End": event.end_time.strftime("%Y-%m-%d %H:%M"),
            "Duration (min)": round(event.duration_minutes, 1),
            "Direction": "High" if event.direction == "high" else "Low",
            "Extreme value": round(event.max_value, 2),
            "Deviation": round(event.max_deviation, 2),
            "Severity": event.severity.value,
        }
        for event in events
    ]
    return pd.DataFrame(rows, columns=columns)
