"""Fetch one faulted IndPenSim batch using bounded HTTP byte-range requests."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.indpensim import (
    INDPENSIM_BATCH_REFERENCE,
    process_indpensim_rows,
    resolve_data_indices,
)


DATASET_ID = "npt257bjxn"
DATASET_VERSION = 1
DATASET_DOI = "10.17632/npt257bjxn.1"
SOURCE_FILENAME = "100_Batches_IndPenSim_V3.csv"
MAX_ROW_BYTES = 256 * 1024
MAX_RANGE_PROBES = 64
DATASET_API = f"https://data.mendeley.com/api/datasets-v2/datasets/{DATASET_ID}"
FILES_API = f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}/files"


def _validate_content_range(
    response: requests.Response,
    expected_start: int,
    source_size: int,
) -> None:
    if response.status_code != 206:
        raise ValueError(
            "The provider did not honor the bounded byte-range request "
            f"(HTTP {response.status_code})"
        )
    match = re.fullmatch(
        r"bytes (\d+)-(\d+)/(\d+)",
        response.headers.get("Content-Range", ""),
    )
    if (
        not match
        or int(match.group(1)) != expected_start
        or int(match.group(3)) != source_size
    ):
        raise ValueError("The provider returned an unexpected Content-Range")
    if not response.url.startswith("https://"):
        raise ValueError("The file request redirected away from HTTPS")


def _fetch_range(
    session: requests.Session,
    url: str,
    start: int,
    end: int,
    source_size: int,
) -> bytes:
    with session.get(
        url,
        headers={"Range": f"bytes={start}-{end}"},
        stream=True,
        timeout=(10, 60),
    ) as response:
        _validate_content_range(response, start, source_size)
        content = bytearray()
        for chunk in response.iter_content(chunk_size=64 * 1024):
            content.extend(chunk)
            if len(content) > end - start + 1:
                raise ValueError("Range response exceeded the requested byte limit")
        if len(content) != end - start + 1:
            raise ValueError("Range response was shorter than requested")
        return bytes(content)


def _parse_row(raw_line: bytes) -> list[str]:
    try:
        return next(csv.reader([raw_line.decode("utf-8-sig").rstrip("\r")]))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError("Could not parse an IndPenSim CSV row") from exc


def _row_at(
    session: requests.Session,
    url: str,
    offset: int,
    data_start: int,
    source_size: int,
    header: list[str],
) -> tuple[int, int, list[str]]:
    start = max(data_start, offset - MAX_ROW_BYTES)
    end = min(source_size - 1, offset + MAX_ROW_BYTES)
    content = _fetch_range(session, url, start, end, source_size)
    relative_offset = offset - start

    if relative_offset == 0 or content[relative_offset - 1 : relative_offset] == b"\n":
        line_start = relative_offset
    else:
        line_start = content.rfind(b"\n", 0, relative_offset) + 1
        if line_start == 0:
            raise ValueError("CSV row exceeds the backward probe size limit")

    line_end = content.find(b"\n", line_start)
    if line_end < 0:
        if end != source_size - 1:
            raise ValueError("CSV row exceeds the forward probe size limit")
        line_end = len(content)
    row = _parse_row(content[line_start:line_end])
    indices = resolve_data_indices(header, len(row))
    if INDPENSIM_BATCH_REFERENCE not in indices:
        raise ValueError("IndPenSim batch-reference field was not found")
    return (
        start + line_start,
        start + line_end,
        row,
    )


def _locate_batch_start(
    session: requests.Session,
    url: str,
    source_size: int,
    header: list[str],
    data_start: int,
    target_batch: int,
) -> tuple[int, int]:
    indices = resolve_data_indices(
        header,
        len(
            _row_at(
                session, url, data_start, data_start, source_size, header
            )[2]
        ),
    )
    batch_index = indices[INDPENSIM_BATCH_REFERENCE]
    checkpoints = []
    for fraction in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99):
        offset = max(data_start, min(source_size - 1, int(source_size * fraction)))
        _, _, row = _row_at(
            session, url, offset, data_start, source_size, header
        )
        try:
            checkpoints.append(int(float(row[batch_index])))
        except (ValueError, IndexError) as exc:
            raise ValueError("Invalid batch reference in source data") from exc
    if checkpoints != sorted(checkpoints):
        raise ValueError(
            "Batch references are not monotonically ordered; "
            "refusing to binary-search this source file"
        )
    if target_batch < checkpoints[0] or target_batch > checkpoints[-1]:
        raise ValueError(f"Batch reference {target_batch} is outside this dataset")

    low, high = data_start, source_size - 1
    probes = 0
    while low < high:
        offset = (low + high) // 2
        row_start, row_end, row = _row_at(
            session, url, offset, data_start, source_size, header
        )
        try:
            batch_reference = int(float(row[batch_index]))
        except (ValueError, IndexError) as exc:
            raise ValueError("Invalid batch reference in source data") from exc
        if batch_reference < target_batch:
            low = row_end + 1
        else:
            high = row_start
        probes += 1
        if probes > MAX_RANGE_PROBES:
            raise ValueError("Batch lookup exceeded its bounded request budget")
        time.sleep(0.05)

    row_start, _, row = _row_at(
        session, url, low, data_start, source_size, header
    )
    if row_start != low or int(float(row[batch_index])) != target_batch:
        raise ValueError(f"Batch reference {target_batch} was not found")
    return low, probes + len(checkpoints) + 2


def _stream_rows(
    session: requests.Session,
    url: str,
    start: int,
    source_size: int,
):
    with session.get(
        url,
        headers={"Range": f"bytes={start}-{source_size - 1}"},
        stream=True,
        timeout=(10, 60),
    ) as response:
        _validate_content_range(response, start, source_size)
        for raw_line in response.iter_lines():
            if raw_line:
                yield _parse_row(raw_line)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_indpensim_sample(
    output_dir: str | Path,
    batch_reference: int = 95,
) -> dict:
    """Download only one batch and its selected signals from Mendeley Data."""
    if batch_reference < 1 or batch_reference > 100:
        raise ValueError("batch_reference must be between 1 and 100")

    with requests.Session() as session:
        dataset_response = session.get(
            DATASET_API,
            params={"version": DATASET_VERSION, "fields": "articles.*"},
            timeout=30,
        )
        dataset_response.raise_for_status()
        dataset = dataset_response.json()
        if dataset.get("doi", {}).get("id") != DATASET_DOI:
            raise ValueError("Mendeley returned an unexpected dataset version")
        license_name = dataset.get("data_licence", {}).get("short_name")
        license_url = dataset.get("data_licence", {}).get("url")
        if license_name != "CC BY 4.0":
            raise ValueError(
                f"Expected CC BY 4.0 license; provider reports {license_name!r}"
            )

        files_response = session.get(
            FILES_API,
            params={
                "folder_id": "root",
                "version": DATASET_VERSION,
                "$start": 0,
                "$limit": 1000,
            },
            timeout=30,
        )
        files_response.raise_for_status()
        files = files_response.json()
        if len(files) >= 1000:
            raise ValueError("Dataset file listing exceeded the bounded first page")
        matches = [item for item in files if item.get("filename") == SOURCE_FILENAME]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one {SOURCE_FILENAME!r} in the dataset file listing"
            )
        source_file = matches[0]
        details = source_file.get("content_details", {})
        source_size = details.get("size")
        source_hash = details.get("sha256_hash")
        download_url = details.get("download_url")
        if (
            source_file.get("status") != "COMPLETED"
            or not isinstance(source_size, int)
            or not isinstance(source_hash, str)
            or not isinstance(download_url, str)
            or not download_url.startswith("https://")
        ):
            raise ValueError("Mendeley source file metadata is incomplete or invalid")

        first_range_end = min(source_size - 1, MAX_ROW_BYTES - 1)
        initial = _fetch_range(
            session, download_url, 0, first_range_end, source_size
        )
        header_end = initial.find(b"\n")
        if header_end < 0:
            raise ValueError("Could not find the CSV header terminator")
        header = _parse_row(initial[:header_end])
        first_data_line_end = initial.find(b"\n", header_end + 1)
        if first_data_line_end < 0:
            raise ValueError("Could not find the first CSV data row")
        first_row = _parse_row(initial[header_end + 1 : first_data_line_end])
        resolve_data_indices(header, len(first_row))
        data_start = header_end + 1

        batch_start, range_probes = _locate_batch_start(
            session,
            download_url,
            source_size,
            header,
            data_start,
            batch_reference,
        )
        output_path = Path(output_dir)
        output_files = process_indpensim_rows(
            header=header,
            rows=_stream_rows(session, download_url, batch_start, source_size),
            output_dir=output_path,
            target_batches=[batch_reference],
            time_column="Time (h)",
            batch_column=INDPENSIM_BATCH_REFERENCE,
            time_unit="h",
        )

    data_path, labels_path = output_files
    with labels_path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        labels = list(reader)
    fault_values = [
        float(row["fault_ref"])
        for row in labels
        if row["fault_ref"].strip().lower() not in {"", "nan"}
    ]
    manifest = {
        "source": "Mendeley Data",
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "doi": DATASET_DOI,
        "simulation_generated": True,
        "source_file": SOURCE_FILENAME,
        "source_file_id": source_file["id"],
        "source_file_size_bytes": source_size,
        "source_file_sha256_from_provider": source_hash,
        "source_file_sha256_verified_locally": False,
        "license": license_name,
        "license_url": license_url,
        "sample_batch_reference": batch_reference,
        "sample_rows": len(labels),
        "sample_time_start_h": float(labels[0]["process_time"]),
        "sample_time_end_h": float(labels[-1]["process_time"]),
        "fault_reference_positive_rows": sum(value > 0 for value in fault_values),
        "range_requests_for_lookup": range_probes,
        "retrieval_endpoints": {
            "metadata": (
                f"{DATASET_API}?version={DATASET_VERSION}&fields=articles.*"
            ),
            "file_listing": (
                f"{FILES_API}?folder_id=root&version={DATASET_VERSION}"
                "&$start=0&$limit=1000"
            ),
            "file_listing_records_returned": len(files),
            "file_matches_for_source_csv": len(matches),
            "byte_range_download": (
                f"https://data.mendeley.com/public-files/datasets/{DATASET_ID}"
                f"/files/{source_file['id']}/file_downloaded"
            ),
        },
        "retrieved_full_source": False,
        "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
        "transformations": [
            "Selected one batch by the Batch reference field.",
            "The original source describes IndPenSim as a mathematical simulation, not observed plant measurements.",
            "Kept pH, dissolved oxygen concentration, and temperature plus a separate fault-label file.",
            "Mapped two obsolete header fields absent from the source data rows.",
            "Converted elapsed process hours to UTC timestamps anchored at 2000-01-01; these are not calendar measurement dates.",
            "Preserved temperature in kelvin and dissolved oxygen in mg/L.",
        ],
        "attribution": (
            "Goldrick, S.; Jankauskas, K.; Lennox, B.; Farid, S.; Lovett, D.; "
            "Duran-Villalobos, C. A. Data for: Modern day monitoring and control "
            "challenges outlined on an industrial-scale benchmark fermentation "
            "process. Mendeley Data, version 1. "
            "https://doi.org/10.17632/npt257bjxn.1"
        ),
        "sample_sha256": {
            data_path.name: _hash_file(data_path),
            labels_path.name: _hash_file(labels_path),
        },
    }
    manifest_path = output_path / f"indpensim_batch_{batch_reference:03d}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Retrieved {len(labels)} rows from IndPenSim batch {batch_reference}; "
        f"did not download the full {source_size:,}-byte source file."
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-reference", type=int, default=95)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "sample",
    )
    args = parser.parse_args()
    download_indpensim_sample(args.output_dir, args.batch_reference)


if __name__ == "__main__":
    main()
