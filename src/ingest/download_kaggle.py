"""Download raw text from confirmed Kaggle sources, land unmodified in MinIO."""

# src/ingest/download_kaggle.py

# RUN: uv run --group ingest python -m src.ingest.download_kaggle

import hashlib
import os
import sys
import tempfile
from typing import TYPE_CHECKING

from src.ingest.minio_client import MinioClient
from src.utils.logging_config import configure_logging, get_logger

if TYPE_CHECKING:
    from kaggle.api.kaggle_api_extended import KaggleApi

logger = get_logger(__name__)

KAGGLE_SOURCES: list[str] = [
    "cosmos98/twitter-and-reddit-sentimental-analysis-dataset",
    "tariqsays/sentiment-dataset-with-1-million-tweets",
]

BUCKET = "raw-data"


def _dataset_version(api: KaggleApi, dataset: str) -> str:
    """Best-effort lookup of the current Kaggle dataset version.

    Kaggle's download endpoint doesn't return a version itself, so this
    searches for the exact `ref` via `dataset_list`. Falls back to "unknown"
    if the search doesn't surface an exact match — checksum still recorded
    either way, so provenance isn't lost.

    `kaggle`'s bundled type stubs are incomplete (`dataset_list` typed as
    possibly-`None`, list items possibly-`None`, `ApiDataset` missing
    `currentVersionNumber`) even though all exist at runtime as documented —
    hence the `or []`, the `is None` skip, and `getattr`.
    """
    for result in api.dataset_list(search=dataset.split("/")[-1]) or []:
        if result is None:
            continue
        if result.ref == dataset:
            return str(getattr(result, "currentVersionNumber", "unknown"))
    return "unknown"


def download_kaggle_dataset(dataset: str, path: str) -> str:
    """Authenticate + download+unzip one Kaggle dataset into `path`."""
    from kaggle.api.kaggle_api_extended import KaggleApi  # env-var auth on import

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=path, unzip=True)
    return _dataset_version(api, dataset)


class KaggleDownloader:
    """Downloads confirmed Kaggle sources, lands each raw in MinIO."""

    def __init__(self, minio_client: MinioClient, bucket: str = BUCKET) -> None:
        """Store the MinIO client to upload through and target bucket."""
        self._minio_client = minio_client
        self._bucket = bucket

    def download_all(self) -> list[str]:
        """Download each configured Kaggle source, upload raw files to MinIO.

        Returns names of sources that failed — doesn't stop on one bad source.
        """
        failures: list[str] = []
        for dataset in KAGGLE_SOURCES:
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    version = download_kaggle_dataset(dataset, tmpdir)
                    for filename in os.listdir(tmpdir):
                        filepath = os.path.join(tmpdir, filename)
                        with open(filepath, "rb") as f:
                            checksum = hashlib.sha256(f.read()).hexdigest()
                        key = f"kaggle/{dataset}/{filename}"
                        self._minio_client.upload_file(
                            self._bucket,
                            key,
                            filepath,
                            metadata={"kaggle_version": version, "sha256": checksum},
                        )
            except Exception:
                logger.exception("Failed to download/upload Kaggle source: %s", dataset)
                failures.append(dataset)
        return failures


if __name__ == "__main__":
    configure_logging()
    failed = KaggleDownloader(minio_client=MinioClient()).download_all()
    if failed:
        sys.exit(1)
