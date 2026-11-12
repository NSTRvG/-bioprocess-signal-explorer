"""Tests for CSV preprocessing and the synthetic data generator."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.generate_synthetic import generate_synthetic_batch
from src.detector import detect_excursions
from scripts.preprocess_indpensim import preprocess_indpensim
from src.indpensim import resolve_data_indices
from src.models import DetectionConfig
from src.preprocessing import handle_missing_values, load_and_preprocess, validate_data


def test_load_sorts_timestamps_and_interpolates_short_gaps(tmp_path) -> None:
    csv_path = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01T00:10:00Z",
                "2024-01-01T00:00:00Z",
                "2024-01-01T00:05:00Z",
            ],
            "pH": [3.0, 1.0, np.nan],
            "label": ["a", "b", "c"],
        }
    ).to_csv(csv_path, index=False)

    result = load_and_preprocess(csv_path)
    assert result.index.is_monotonic_increasing
    assert result["pH"].tolist() == [1.0, 2.0, 3.0]
    assert "label" not in result.columns


def test_long_missing_gap_is_not_interpolated() -> None:
    index = pd.to_datetime(
        ["2024-01-01T00:00Z", "2024-01-01T00:05Z", "2024-01-01T01:00Z"]
    )
    df = pd.DataFrame({"signal": [1.0, np.nan, 3.0]}, index=index)
    result = handle_missing_values(df, max_gap_minutes=30)
    assert pd.isna(result.iloc[1, 0])


def test_duplicate_and_invalid_timestamps_are_rejected(tmp_path) -> None:
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-01"],
            "signal": [1, 2],
        }
    ).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="Duplicate timestamps"):
        load_and_preprocess(csv_path)


def test_mixed_numeric_and_text_measurements_are_rejected(tmp_path) -> None:
    csv_path = tmp_path / "mixed.csv"
    pd.DataFrame(
        {
            "timestamp": ["2024-01-01T00:00Z", "2024-01-01T00:05Z"],
            "signal": ["7.0", "not-a-number"],
            "label": ["first", "second"],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="non-numeric value.*'signal'"):
        load_and_preprocess(csv_path)


def test_validate_data_handles_empty_frames() -> None:
    df = pd.DataFrame(
        {"signal": pd.Series(dtype=float)},
        index=pd.DatetimeIndex([], name="timestamp"),
    )
    report = validate_data(df)
    assert report["n_rows"] == 0
    assert report["time_range"]["duration_hours"] == 0.0


def test_synthetic_batch_is_deterministic_and_labeled() -> None:
    first, labels = generate_synthetic_batch(duration_hours=24, seed=7)
    second, _ = generate_synthetic_batch(duration_hours=24, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == ["timestamp", "pH", "DO", "temperature", "biomass"]
    assert labels.empty


def test_synthetic_injected_anomalies_are_detected() -> None:
    batch, labels = generate_synthetic_batch()
    df = batch.set_index("timestamp")
    df.index = pd.DatetimeIndex(df.index, name="timestamp")
    config = DetectionConfig()

    for label in labels.itertuples(index=False):
        events = detect_excursions(df, label.variable, config).events
        anomaly_start = df.index[0] + pd.Timedelta(hours=label.start_hour)
        anomaly_end = df.index[0] + pd.Timedelta(hours=label.end_hour)
        assert any(
            event.start_time <= anomaly_end and event.end_time >= anomaly_start
            for event in events
        ), f"No detected event overlaps synthetic {label.variable} anomaly"


def test_preprocess_rejects_missing_timestamp_column(tmp_path) -> None:
    csv_path = tmp_path / "missing.csv"
    pd.DataFrame({"signal": [1.0]}).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="Timestamp column"):
        load_and_preprocess(csv_path)


def test_indpensim_preprocessing_uses_explicit_columns_and_time_units(tmp_path) -> None:
    source = tmp_path / "source.csv"
    pd.DataFrame(
        {
            "batch_no": [1, 1, 2],
            "elapsed": [0.0, 0.5, 0.0],
            "pH_value": [7.0, 7.1, 6.9],
            "Fault reference(Fault_ref:Fault ref)": [0, 1, 0],
        }
    ).to_csv(source, index=False)

    written = preprocess_indpensim(
        source,
        output_dir=tmp_path / "sample",
        target_batches=[1],
        batch_column="batch_no",
        time_column="elapsed",
        variable_columns=["pH_value"],
        time_unit="h",
    )
    result = pd.read_csv(written[0])
    assert result.columns.tolist() == ["timestamp", "pH_value"]
    assert len(result) == 2
    assert pd.to_datetime(result["timestamp"]).iloc[1] - pd.to_datetime(
        result["timestamp"]
    ).iloc[0] == pd.Timedelta(minutes=30)


def test_indpensim_published_header_gap_is_mapped_explicitly() -> None:
    header = [f"column_{index}" for index in range(40)]
    header[31] = "Fault reference(Fault_ref:Fault ref)"
    header[34] = " 1-Raman spec recorded"
    header[35] = "2-PAT control(PAT_ref:PAT ref)"
    header[36] = "Batch reference(Batch_ref:Batch ref)"
    header[37] = "Batch ID"
    header[38] = "Fault flag"

    indices = resolve_data_indices(header, row_width=38)
    assert 34 not in indices
    assert 35 not in indices
    assert indices[header[36]] == 34
    assert indices[header[37]] == 35
    assert indices[header[38]] == 36
    assert indices[header[39]] == 37


def test_bundled_indpensim_sample_has_aligned_labels_and_provenance() -> None:
    sample_dir = Path(__file__).resolve().parents[1] / "data" / "sample"
    data = load_and_preprocess(sample_dir / "indpensim_batch_095.csv")
    labels = pd.read_csv(sample_dir / "indpensim_batch_095_ground_truth.csv")
    manifest = json.loads(
        (sample_dir / "indpensim_batch_095_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(data) == len(labels) == 1055
    assert data.index.equals(pd.DatetimeIndex(pd.to_datetime(labels["timestamp"])))
    assert labels["batch_reference"].unique().tolist() == [95]
    assert int((labels["fault_ref"] > 0).sum()) == 188
    assert manifest["doi"] == "10.17632/npt257bjxn.1"
    assert manifest["license"] == "CC BY 4.0"
    assert manifest["simulation_generated"]
    for filename, expected_hash in manifest["sample_sha256"].items():
        actual_hash = hashlib.sha256(
            (sample_dir / filename).read_bytes()
        ).hexdigest()
        assert actual_hash == expected_hash
