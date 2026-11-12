"""Security tests for validating downloaded dataset archives."""

from pathlib import Path
import stat
import zipfile

import pytest

import scripts.download_full_dataset as downloader
from scripts.download_full_dataset import _validate_zip_members


def test_zip_path_traversal_is_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.txt", "not allowed")

    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ValueError, match="Unsafe archive path"):
            _validate_zip_members(archive, tmp_path / "output")


def test_absolute_zip_path_is_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("C:/outside.txt", "not allowed")

    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ValueError, match="Unsafe archive path"):
            _validate_zip_members(archive, tmp_path / "output")


def test_zip_symbolic_link_is_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    link = zipfile.ZipInfo("link")
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(link, "../outside.txt")

    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ValueError, match="Symbolic links"):
            _validate_zip_members(archive, tmp_path / "output")


def test_zip_expansion_limit_is_enforced(tmp_path: Path, monkeypatch) -> None:
    archive_path = tmp_path / "large.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("data.txt", "more than one byte")

    monkeypatch.setattr(downloader, "MAX_EXTRACTED_BYTES", 1)
    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ValueError, match="expands beyond"):
            _validate_zip_members(archive, tmp_path / "output")


def test_non_https_download_url_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        downloader.download_indpensim("http://example.test/data.zip", tmp_path)
