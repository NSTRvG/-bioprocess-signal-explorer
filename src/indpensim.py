"""Schema-aware, streaming helpers for the published IndPenSim CSV."""

import csv
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import tempfile
from typing import Iterable


INDPENSIM_TIME = "Time (h)"
INDPENSIM_DO = "Dissolved oxygen concentration(DO2:mg/L)"
INDPENSIM_PH = "pH(pH:pH)"
INDPENSIM_TEMPERATURE = "Temperature(T:K)"
INDPENSIM_FAULT_REFERENCE = "Fault reference(Fault_ref:Fault ref)"
INDPENSIM_BATCH_REFERENCE = "Batch reference(Batch_ref:Batch ref)"
INDPENSIM_BATCH_ID = "Batch ID"
INDPENSIM_FAULT_FLAG = "Fault flag"
INDPENSIM_DEFAULT_VARIABLES = [
    INDPENSIM_PH,
    INDPENSIM_DO,
    INDPENSIM_TEMPERATURE,
]
INDPENSIM_OUTPUT_NAMES = {
    INDPENSIM_PH: "pH",
    INDPENSIM_DO: "DO_mg_L",
    INDPENSIM_TEMPERATURE: "temperature_K",
}
TIME_ORIGIN = datetime(2000, 1, 1, tzinfo=timezone.utc)


def resolve_data_indices(header: list[str], row_width: int) -> dict[str, int]:
    """Map CSV header names to row fields, accounting for the published header gap."""
    if row_width == len(header):
        return {name: index for index, name in enumerate(header)}

    is_known_indpensim_layout = (
        row_width == len(header) - 2
        and len(header) > 39
        and header[31].startswith("Fault reference(")
        and header[34].strip() == "1-Raman spec recorded"
        and header[35].startswith("2-PAT control(")
        and header[36].startswith("Batch reference(")
        and header[37] == INDPENSIM_BATCH_ID
        and header[38] == INDPENSIM_FAULT_FLAG
        and len(header[39:]) == row_width - 37
    )
    if not is_known_indpensim_layout:
        raise ValueError(
            f"CSV header has {len(header)} fields but data rows have {row_width}; "
            "the mismatch does not match the documented IndPenSim layout"
        )

    return {
        name: index if index < 34 else index - 2
        for index, name in enumerate(header)
        if index not in {34, 35}
    }


def _batch_key(value: str | int) -> str:
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return text


def _process_timestamp(value: str, time_unit: str, time_column: str) -> datetime:
    unit = time_unit
    if unit == "auto":
        unit = "h" if "(h)" in time_column.lower() else "m"
    if unit == "datetime":
        try:
            timestamp = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Invalid datetime value in {time_column!r}: {value!r}") from exc
        return (
            timestamp.replace(tzinfo=timezone.utc)
            if timestamp.tzinfo is None
            else timestamp.astimezone(timezone.utc)
        )

    multipliers = {"s": 1.0, "m": 60.0, "h": 3600.0}
    try:
        numeric_time = float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric time in {time_column!r}: {value!r}") from exc
    if not math.isfinite(numeric_time):
        raise ValueError(f"Non-finite process time in {time_column!r}: {value!r}")
    try:
        return TIME_ORIGIN + timedelta(seconds=numeric_time * multipliers[unit])
    except (OverflowError, KeyError) as exc:
        raise ValueError(f"Unsupported time unit: {time_unit!r}") from exc


def process_indpensim_rows(
    header: list[str],
    rows: Iterable[list[str]],
    output_dir: str | Path,
    target_batches: list[str | int] | None = None,
    batch_column: str = INDPENSIM_BATCH_REFERENCE,
    time_column: str = INDPENSIM_TIME,
    variable_columns: list[str] | None = None,
    time_unit: str = "h",
) -> list[Path]:
    """Stream requested batches into narrow app-ready CSVs and separate labels."""
    if target_batches is None:
        target_batches = [1, 95]
    if time_unit not in {"auto", "datetime", "s", "m", "h"}:
        raise ValueError("time_unit must be auto, datetime, s, m, or h")

    row_iterator = iter(rows)
    first_row = next(row_iterator, None)
    if first_row is None:
        raise ValueError("Input dataset contains no data rows")
    column_indices = resolve_data_indices(header, len(first_row))
    required = [batch_column, time_column, INDPENSIM_FAULT_REFERENCE]
    missing = [column for column in required if column not in column_indices]
    if missing:
        raise ValueError(
            f"Required IndPenSim columns not found: {missing}. "
            f"Available fields include: {header[:40]}"
        )

    if variable_columns is None:
        variable_columns = [
            column
            for column in INDPENSIM_DEFAULT_VARIABLES
            if column in column_indices
        ]
    if not variable_columns:
        raise ValueError("Select at least one numeric process variable")
    missing_variables = [
        column for column in variable_columns if column not in column_indices
    ]
    if missing_variables:
        raise ValueError(f"Variable column(s) not found: {missing_variables}")

    targets = list(dict.fromkeys(_batch_key(batch) for batch in target_batches))
    if not targets:
        raise ValueError("Select at least one batch")
    if len(targets) > 10:
        raise ValueError("At most 10 batches may be extracted in one run")
    batch_index = column_indices[batch_column]
    time_index = column_indices[time_column]
    fault_index = column_indices[INDPENSIM_FAULT_REFERENCE]
    variable_indices = [column_indices[name] for name in variable_columns]
    variable_output_names = [
        INDPENSIM_OUTPUT_NAMES.get(name, name) for name in variable_columns
    ]
    if len(set(variable_output_names)) != len(variable_output_names):
        raise ValueError("Selected variables produce duplicate output column names")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    handles = {}
    temporary_paths: list[Path] = []
    counts = {target: 0 for target in targets}
    previous_timestamps: dict[str, datetime] = {}
    previous_batch_number: int | None = None
    numeric_targets = []
    for target in targets:
        try:
            numeric_targets.append(int(target))
        except ValueError:
            pass
    max_numeric_target = (
        max(numeric_targets) if len(numeric_targets) == len(targets) else None
    )
    final_paths = []

    try:
        for target in targets:
            safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", target)
            stem = f"indpensim_batch_{safe_id.zfill(3)}"
            data_destination = output_path / f"{stem}.csv"
            labels_destination = output_path / f"{stem}_ground_truth.csv"
            data_temp = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                prefix=f".{stem}-",
                suffix=".tmp",
                dir=output_path,
                delete=False,
            )
            labels_temp = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                prefix=f".{stem}-labels-",
                suffix=".tmp",
                dir=output_path,
                delete=False,
            )
            temporary_paths.extend([Path(data_temp.name), Path(labels_temp.name)])
            data_writer = csv.writer(data_temp)
            labels_writer = csv.writer(labels_temp)
            data_writer.writerow(["timestamp", *variable_output_names])
            labels_writer.writerow(
                [
                    "timestamp",
                    "process_time",
                    "fault_ref",
                    "batch_reference",
                    "batch_id",
                    "fault_flag",
                ]
            )
            handles[target] = (
                data_temp,
                labels_temp,
                data_writer,
                labels_writer,
                data_destination,
                labels_destination,
            )

        for row in _prepend(first_row, row_iterator):
            if len(row) != len(first_row):
                raise ValueError("IndPenSim data rows have inconsistent field counts")
            raw_batch = row[batch_index]
            batch = _batch_key(raw_batch)
            try:
                batch_number = int(batch)
            except ValueError:
                batch_number = None
            if (
                previous_batch_number is not None
                and batch_number is not None
                and batch_number < previous_batch_number
            ):
                raise ValueError("Batch references are not in ascending order")
            if batch_number is not None:
                previous_batch_number = batch_number
            if (
                max_numeric_target is not None
                and batch_number is not None
                and batch_number > max_numeric_target
            ):
                break
            if batch not in handles:
                continue

            try:
                timestamp = _process_timestamp(
                    row[time_index], time_unit, time_column
                )
            except IndexError as exc:
                raise ValueError("CSV row is missing the process-time field") from exc
            previous = previous_timestamps.get(batch)
            if previous is not None and timestamp <= previous:
                raise ValueError(
                    f"Batch {batch} timestamps are not strictly increasing"
                )
            previous_timestamps[batch] = timestamp
            timestamp_text = timestamp.isoformat()
            variable_values = [row[index] for index in variable_indices]
            for name, value in zip(variable_columns, variable_values):
                if value.strip().lower() not in {"", "nan"}:
                    try:
                        numeric_value = float(value)
                    except ValueError as exc:
                        raise ValueError(
                            f"Non-numeric measurement in {name!r}: {value!r}"
                        ) from exc
                    if math.isinf(numeric_value):
                        raise ValueError(f"Infinite measurement in {name!r}")

            data_temp, labels_temp, data_writer, labels_writer, _, _ = handles[batch]
            data_writer.writerow([timestamp_text, *variable_values])
            fault_ref = row[fault_index]
            batch_id = (
                row[column_indices[INDPENSIM_BATCH_ID]]
                if INDPENSIM_BATCH_ID in column_indices
                else raw_batch
            )
            fault_flag = (
                row[column_indices[INDPENSIM_FAULT_FLAG]]
                if INDPENSIM_FAULT_FLAG in column_indices
                else ""
            )
            labels_writer.writerow(
                [
                    timestamp_text,
                    row[time_index],
                    fault_ref,
                    raw_batch,
                    batch_id,
                    fault_flag,
                ]
            )
            counts[batch] += 1

        for target, (data_temp, labels_temp, *_rest) in handles.items():
            data_temp.flush()
            labels_temp.flush()
            os.fsync(data_temp.fileno())
            os.fsync(labels_temp.fileno())
            data_temp.close()
            labels_temp.close()
            if counts[target] == 0:
                raise ValueError(f"Batch {target!r} was not found in the input")

        for target in targets:
            _, _, _, _, data_destination, labels_destination = handles[target]
            temporary_index = targets.index(target) * 2
            temporary_paths[temporary_index].replace(data_destination)
            temporary_paths[temporary_index + 1].replace(labels_destination)
            final_paths.extend([data_destination, labels_destination])
    except Exception:
        for item in handles.values():
            for handle in item[:2]:
                if not handle.closed:
                    handle.close()
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)
        raise

    for target in targets:
        print(f"Batch {target}: {counts[target]} rows")
    return final_paths


def _prepend(first: list[str], rest: Iterable[list[str]]) -> Iterable[list[str]]:
    yield first
    yield from rest
