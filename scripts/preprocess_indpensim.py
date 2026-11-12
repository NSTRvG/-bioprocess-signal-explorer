"""Stream selected batches from a locally downloaded IndPenSim CSV."""

import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.indpensim import (
    INDPENSIM_BATCH_REFERENCE,
    INDPENSIM_TIME,
    process_indpensim_rows,
)


def preprocess_indpensim(
    input_file: str | Path,
    output_dir: str | Path = "data/sample",
    target_batches: list[str | int] | None = None,
    batch_column: str = INDPENSIM_BATCH_REFERENCE,
    time_column: str = INDPENSIM_TIME,
    variable_columns: list[str] | None = None,
    time_unit: str = "h",
) -> list[Path]:
    """Process selected batches without loading the full source file into memory."""
    input_path = Path(input_file)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")

    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        header = next(reader, None)
        if header is None:
            raise ValueError("Input CSV is empty")
        return process_indpensim_rows(
            header=header,
            rows=reader,
            output_dir=output_dir,
            target_batches=target_batches,
            batch_column=batch_column,
            time_column=time_column,
            variable_columns=variable_columns,
            time_unit=time_unit,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "full"
        / "100_Batches_IndPenSim_V3.csv",
    )
    parser.add_argument("--output-dir", type=Path, default="data/sample")
    parser.add_argument("--batch-column", default=INDPENSIM_BATCH_REFERENCE)
    parser.add_argument("--time-column", default=INDPENSIM_TIME)
    parser.add_argument(
        "--time-unit",
        choices=["auto", "datetime", "s", "m", "h"],
        default="h",
        help="Numeric process-time unit; auto infers hours from a '(h)' column",
    )
    parser.add_argument("--batch-ids", default="1,95")
    parser.add_argument(
        "--variables",
        help="Comma-separated source columns (default: pH, DO, temperature)",
    )
    args = parser.parse_args()
    if not args.input_file.is_file():
        parser.error(f"Input CSV does not exist: {args.input_file}")

    variables = (
        [item.strip() for item in args.variables.split(",") if item.strip()]
        if args.variables
        else None
    )
    batches = [item.strip() for item in args.batch_ids.split(",") if item.strip()]
    preprocess_indpensim(
        input_file=args.input_file,
        output_dir=args.output_dir,
        target_batches=batches,
        batch_column=args.batch_column,
        time_column=args.time_column,
        variable_columns=variables,
        time_unit=args.time_unit,
    )


if __name__ == "__main__":
    main()
