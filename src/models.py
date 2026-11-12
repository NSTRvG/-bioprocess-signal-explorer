"""Typed data models for excursion detection."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import math


class DetectionMethod(Enum):
    """Method used to calculate control bands."""

    ROBUST = "robust"
    CLASSIC = "classic"


class Severity(Enum):
    """Severity level assigned to an excursion."""

    LOW = "low"
    MODERATE = "moderate"
    CRITICAL = "critical"


@dataclass
class DetectionConfig:
    """Parameters controlling the detector."""

    window: str = "60min"
    k: float = 3.0
    method: DetectionMethod = DetectionMethod.ROBUST
    min_periods: int = 10
    min_event_points: int = 2
    max_gap_minutes: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.window, str) or not self.window.strip():
            raise ValueError("window must be a non-empty pandas-compatible duration")
        if not math.isfinite(self.k) or self.k <= 0:
            raise ValueError("k must be a finite number greater than zero")
        if self.min_periods < 1:
            raise ValueError("min_periods must be at least 1")
        if self.min_event_points < 1:
            raise ValueError("min_event_points must be at least 1")
        if self.max_gap_minutes < 0:
            raise ValueError("max_gap_minutes cannot be negative")
        if not isinstance(self.method, DetectionMethod):
            raise ValueError("method must be a DetectionMethod")


@dataclass
class ExcursionEvent:
    """A contiguous set of out-of-band measurements."""

    event_id: int
    variable: str
    start_time: datetime
    end_time: datetime
    duration_minutes: float
    direction: str
    max_value: float
    max_deviation: float
    severity: Severity
    n_points: int


@dataclass
class DetectionResult:
    """Bands, outlier flags, and events for one variable."""

    variable: str
    timestamps: list[datetime]
    values: list[float]
    upper_band: list[float]
    lower_band: list[float]
    is_outlier: list[bool]
    events: list[ExcursionEvent]


def calculate_severity(max_deviation: float, band_width: float) -> Severity:
    """Classify a deviation relative to half the width of its control band."""
    if not math.isfinite(band_width) or band_width <= 0:
        return Severity.MODERATE

    normalized = abs(max_deviation) / (band_width / 2)
    if normalized > 2.0:
        return Severity.CRITICAL
    if normalized > 1.0:
        return Severity.MODERATE
    return Severity.LOW
