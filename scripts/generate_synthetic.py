"""Generate plausible bioreactor data with labeled synthetic anomalies."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_batch(
    duration_hours: float = 180.0,
    sampling_minutes: float = 5.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a synthetic batch and anomaly labels (not experimental data)."""
    if duration_hours <= 0 or sampling_minutes <= 0:
        raise ValueError("duration_hours and sampling_minutes must be positive")
    if not np.isfinite(duration_hours) or not np.isfinite(sampling_minutes):
        raise ValueError("duration_hours and sampling_minutes must be finite")

    n_points = int(duration_hours * 60 / sampling_minutes)
    if n_points < 2:
        raise ValueError("The requested batch must contain at least two samples")

    rng = np.random.default_rng(seed)
    elapsed_minutes = np.arange(n_points) * sampling_minutes
    timestamps = pd.Timestamp("2024-01-01") + pd.to_timedelta(
        elapsed_minutes, unit="min"
    )
    time_hours = elapsed_minutes / 60

    biomass = 50.0 / (1 + np.exp(-0.08 * (time_hours - 36)))
    biomass = np.clip(biomass + rng.normal(0, 0.5, n_points), 0.1, None)

    ph = 7.0 - 0.002 * time_hours
    for pulse_time in range(20, int(duration_hours), 20):
        pulse = (time_hours >= pulse_time) & (time_hours < pulse_time + 2)
        ph[pulse] += 0.1
    ph += rng.normal(0, 0.02, n_points)

    dissolved_oxygen = 100.0 - 18.0 * biomass / 50.0
    dissolved_oxygen = np.where(
        dissolved_oxygen < 30,
        30 + rng.normal(0, 2, n_points),
        dissolved_oxygen,
    )
    dissolved_oxygen += rng.normal(0, 1.5, n_points)
    temperature = 37.0 + rng.normal(0, 0.15, n_points)

    anomalies = []
    spike_start = 120.0
    spike = (time_hours >= spike_start) & (time_hours < spike_start + 0.5)
    if spike.any():
        ph[spike] += 0.8
        anomalies.append(
            {
                "event_id": 1,
                "variable": "pH",
                "type": "spike",
                "start_hour": spike_start,
                "end_hour": min(spike_start + 0.5, duration_hours),
                "magnitude": 0.8,
                "description": "Synthetic transient base-pump fault",
            }
        )

    do_start, do_end = 50.0, 52.5
    do_fault = (time_hours >= do_start) & (time_hours < do_end)
    if do_fault.any():
        dissolved_oxygen[do_fault] -= 15.0
        anomalies.append(
            {
                "event_id": 2,
                "variable": "DO",
                "type": "step_down",
                "start_hour": do_start,
                "end_hour": min(do_end, duration_hours),
                "magnitude": -15.0,
                "description": "Synthetic increased oxygen demand",
            }
        )

    temp_start, temp_end = 80.0, 90.0
    temp_fault = (time_hours >= temp_start) & (time_hours < temp_end)
    if temp_fault.any():
        temperature[temp_fault] += np.linspace(0, 1.2, int(temp_fault.sum()))
        anomalies.append(
            {
                "event_id": 3,
                "variable": "temperature",
                "type": "drift",
                "start_hour": temp_start,
                "end_hour": min(temp_end, duration_hours),
                "magnitude": 1.2,
                "description": "Synthetic partial temperature-control fault",
            }
        )

    batch = pd.DataFrame(
        {
            "timestamp": timestamps,
            "pH": ph,
            "DO": dissolved_oxygen,
            "temperature": temperature,
            "biomass": biomass,
        }
    )
    return batch, pd.DataFrame(anomalies)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "synthetic",
    )
    parser.add_argument("--duration-hours", type=float, default=180.0)
    parser.add_argument("--sampling-minutes", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    batch, ground_truth = generate_synthetic_batch(
        args.duration_hours, args.sampling_minutes, args.seed
    )
    batch.to_csv(args.output_dir / "synthetic_batch.csv", index=False)
    ground_truth.to_csv(args.output_dir / "ground_truth.csv", index=False)
    print(f"Synthetic data written to {args.output_dir.resolve()}")
    print(f"  synthetic_batch.csv: {len(batch)} samples")
    print(f"  ground_truth.csv: {len(ground_truth)} synthetic anomalies")


if __name__ == "__main__":
    main()
