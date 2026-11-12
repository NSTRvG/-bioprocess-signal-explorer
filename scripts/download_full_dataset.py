"""Download and safely extract an IndPenSim ZIP from a direct HTTPS URL."""

import argparse
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
import tempfile
import zipfile

import requests
from tqdm import tqdm


DEFAULT_MAX_DOWNLOAD_BYTES = 4 * 1024**3
MAX_EXTRACTED_BYTES = 10 * 1024**3
MAX_ARCHIVE_MEMBERS = 100_000


def _validate_zip_members(archive: zipfile.ZipFile, output_dir: Path) -> None:
    """Reject paths, links, or archive sizes that could escape or exhaust storage."""
    root = output_dir.resolve()
    members = archive.infolist()
    if len(members) > MAX_ARCHIVE_MEMBERS:
        raise ValueError("Archive contains too many entries")

    total_size = 0
    for member in members:
        name = member.filename
        posix_name = PurePosixPath(name)
        windows_name = PureWindowsPath(name)
        if (
            not name
            or "\\" in name
            or posix_name.is_absolute()
            or windows_name.is_absolute()
            or windows_name.drive
            or any(part in {"", ".", ".."} for part in posix_name.parts)
        ):
            raise ValueError(f"Unsafe archive path: {name!r}")

        destination = (root / Path(*posix_name.parts)).resolve()
        if destination != root and root not in destination.parents:
            raise ValueError(f"Archive path escapes the output directory: {name!r}")

        mode = member.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ValueError(f"Symbolic links are not allowed in archives: {name!r}")
        total_size += member.file_size
        if total_size > MAX_EXTRACTED_BYTES:
            raise ValueError("Archive expands beyond the configured size limit")


def download_indpensim(
    url: str,
    output_dir: Path,
    max_download_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
) -> None:
    """Stream, validate, and extract a ZIP from a user-supplied HTTPS URL."""
    if not url.startswith("https://"):
        raise ValueError("The download URL must use HTTPS")
    if max_download_bytes <= 0:
        raise ValueError("max_download_bytes must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="indpensim-", suffix=".zip", dir=output_dir, delete=False
    ) as temporary:
        temp_path = Path(temporary.name)

    try:
        with requests.get(
            url, stream=True, timeout=(10, 60), allow_redirects=True
        ) as response:
            response.raise_for_status()
            if not response.url.startswith("https://"):
                raise ValueError("The download redirected to a non-HTTPS URL")
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > max_download_bytes:
                raise ValueError("Download exceeds the configured size limit")

            downloaded = 0
            with temp_path.open("wb") as file_handle:
                progress = tqdm(
                    total=int(content_length) if content_length else None,
                    unit="B",
                    unit_scale=True,
                    desc="Downloading",
                )
                try:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        downloaded += len(chunk)
                        if downloaded > max_download_bytes:
                            raise ValueError("Download exceeds the configured size limit")
                        file_handle.write(chunk)
                        progress.update(len(chunk))
                finally:
                    progress.close()

        with zipfile.ZipFile(temp_path) as archive:
            _validate_zip_members(archive, output_dir)
            archive.extractall(output_dir)
    finally:
        temp_path.unlink(missing_ok=True)

    print(f"Dataset extracted to {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        required=True,
        help="Direct HTTPS ZIP URL copied from the dataset provider",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "full",
    )
    parser.add_argument(
        "--max-download-gb",
        type=float,
        default=4.0,
        help="Maximum compressed download size in GiB (default: 4)",
    )
    args = parser.parse_args()
    if args.max_download_gb <= 0:
        parser.error("--max-download-gb must be positive")
    download_indpensim(
        args.url,
        args.output_dir,
        max_download_bytes=int(args.max_download_gb * 1024**3),
    )


if __name__ == "__main__":
    main()
