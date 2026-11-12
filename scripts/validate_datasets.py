"""Reproduce synthetic and IndPenSim sample validation summaries."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import detect_excursions
from src.models import DetectionConfig
from src.preprocessing import load_and_preprocess, validate_data


def _classification_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, int | float]:
    true_positive = int(np.count_nonzero(truth & prediction))
    false_positive = int(np.count_nonzero(~truth & prediction))
    false_negative = int(np.count_nonzero(truth & ~prediction))
    true_negative = int(np.count_nonzero(~truth & ~prediction))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def _validate_synthetic(data_dir: Path) -> dict:
    data = load_and_preprocess(data_dir / "synthetic_batch.csv")
    labels = pd.read_csv(data_dir / "ground_truth.csv")
    config = DetectionConfig(
        window="60min",
        k=3.0,
        min_periods=10,
        min_event_points=2,
        max_gap_minutes=15,
    )
    origin = data.index[0]
    variable_results = {}
    for variable in data.columns:
        result = detect_excursions(data, variable, config)
        outlier_flags = pd.Series(result.is_outlier, index=data.index).fillna(False)
        variable_labels = labels[labels["variable"] == variable]
        labeled_mask = pd.Series(False, index=data.index)
        intervals = []
        for label in variable_labels.itertuples(index=False):
            start = origin + pd.Timedelta(hours=float(label.start_hour))
            end = origin + pd.Timedelta(hours=float(label.end_hour))
            interval = (data.index >= start) & (data.index < end)
            labeled_mask |= pd.Series(interval, index=data.index)
            intervals.append((start, end))
        labeled_points = int(labeled_mask.sum())
        hits = int((outlier_flags & labeled_mask).sum())
        variable_results[variable] = {
            "sample_count": len(data),
            "labeled_anomaly_intervals": len(intervals),
            "labeled_anomaly_samples": labeled_points,
            "outlier_samples": int(outlier_flags.sum()),
            "outliers_inside_labels": hits,
            "outliers_outside_labels": int((outlier_flags & ~labeled_mask).sum()),
            "labeled_sample_coverage": (
                round(hits / labeled_points, 6) if labeled_points else None
            ),
            "events": len(result.events),
            "events_overlapping_labels": sum(
                any(event.start_time < end and event.end_time >= start for start, end in intervals)
                for event in result.events
            ),
        }
    return {
        "data": "repository-generated synthetic batch",
        "ground_truth": "injected synthetic anomaly intervals",
        "detector_config": {
            "window": config.window,
            "k": config.k,
            "min_periods": config.min_periods,
            "method": config.method.value,
        },
        "data_validation": validate_data(data),
        "by_variable": variable_results,
        "interpretation": (
            "Coverage measures whether outlier points fall inside injected intervals; "
            "it is not a false-alarm-adjusted accuracy score."
        ),
    }


def _validate_indpensim(data_dir: Path) -> dict:
    data_path = data_dir / "indpensim_batch_095.csv"
    labels_path = data_dir / "indpensim_batch_095_ground_truth.csv"
    manifest_path = data_dir / "indpensim_batch_095_manifest.json"
    data = load_and_preprocess(data_path)
    labels = pd.read_csv(labels_path)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], utc=True)
    labels = labels.set_index("timestamp")
    if not data.index.equals(labels.index):
        raise ValueError("IndPenSim measurements and fault labels are misaligned")

    fault_reference = pd.to_numeric(labels["fault_ref"], errors="coerce").fillna(0)
    truth = (fault_reference > 0).to_numpy()
    cadence_minutes = (
        data.index.to_series().diff().dropna().dt.total_seconds().median() / 60
    )
    config = DetectionConfig(
        window="180min",
        k=3.0,
        min_periods=10,
        min_event_points=1,
        max_gap_minutes=max(15, int(np.ceil(cadence_minutes * 1.25))),
    )

    predictions = []
    outliers_by_variable = {}
    for variable in data.columns:
        result = detect_excursions(data, variable, config)
        flags = pd.Series(result.is_outlier, index=data.index).fillna(False).to_numpy()
        predictions.append(flags)
        outliers_by_variable[variable] = {
            "outlier_samples": int(flags.sum()),
            "events": len(result.events),
        }
    union_prediction = np.logical_or.reduce(predictions)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "data": "published IndPenSim benchmark sample",
        "source": {
            "doi": manifest["doi"],
            "version": manifest["dataset_version"],
            "batch_reference": manifest["sample_batch_reference"],
            "license": manifest["license"],
            "simulation_generated": True,
            "source_rows": manifest["sample_rows"],
            "fault_reference_positive_rows": manifest[
                "fault_reference_positive_rows"
            ],
        },
        "detector_config": {
            "window": config.window,
            "k": config.k,
            "min_periods": config.min_periods,
            "method": config.method.value,
            "event_gap_minutes": config.max_gap_minutes,
        },
        "data_validation": validate_data(data),
        "cadence_minutes_median": cadence_minutes,
        "outliers_by_variable": outliers_by_variable,
        "union_outliers_vs_process_fault_reference": _classification_metrics(
            truth, union_prediction
        ),
        "interpretation": (
            "Fault reference is a process-level simulator label, not a "
            "variable-specific ground truth. This exploratory pointwise comparison "
            "does not establish detector accuracy or production readiness."
        ),
    }


def validate_datasets(project_root: str | Path) -> dict:
    """Evaluate the bundled synthetic data and optional IndPenSim sample."""
    root = Path(project_root)
    data_dir = root / "data"
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic": _validate_synthetic(data_dir / "synthetic"),
    }
    real_data_dir = data_dir / "sample"
    if all(
        (real_data_dir / filename).is_file()
        for filename in (
            "indpensim_batch_095.csv",
            "indpensim_batch_095_ground_truth.csv",
            "indpensim_batch_095_manifest.json",
        )
    ):
        report["indpensim"] = _validate_indpensim(real_data_dir)
    else:
        report["indpensim"] = {
            "available": False,
            "message": (
                "Run scripts/download_indpensim_sample.py to retrieve the "
                "licensed benchmark sample."
            ),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON output path (default: data/sample/validation_report.json)",
    )
    args = parser.parse_args()
    report = validate_datasets(args.project_root)
    output = args.output or args.project_root / "data" / "sample" / "validation_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Validation report written to {output.resolve()}")
    if "indpensim" in report and report["indpensim"].get(
        "union_outliers_vs_process_fault_reference"
    ):
        metrics = report["indpensim"]["union_outliers_vs_process_fault_reference"]
        print(
            "IndPenSim process-label comparison "
            f"(exploratory): precision={metrics['precision']:.3f}, "
            f"recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}"
        )


if __name__ == "__main__":
    main()
