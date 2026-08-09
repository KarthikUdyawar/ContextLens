"""Download raw text from confirmed Kaggle sources, land unmodified in MinIO."""

# src/ingest/download_kaggle.py

# RUN: uv run --group ingest python -m src.ingest.download_kaggle

import os
import sys
import tempfile

from src.ingest.minio_client import MinioClient
from src.utils.logging_config import configure_logging, get_logger

logger = get_logger(__name__)

KAGGLE_SOURCES: list[str] = [
    "cosmos98/twitter-and-reddit-sentimental-analysis-dataset",
    "tariqsays/sentiment-dataset-with-1-million-tweets",
]

BUCKET = "raw-data"


def download_kaggle_dataset(dataset: str, path: str) -> None:
    """Authenticate + download+unzip one Kaggle dataset into `path`."""
    from kaggle.api.kaggle_api_extended import KaggleApi  # env-var auth on import

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=path, unzip=True)


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
                    download_kaggle_dataset(dataset, tmpdir)
                    for filename in os.listdir(tmpdir):
                        filepath = os.path.join(tmpdir, filename)
                        key = f"kaggle/{dataset}/{filename}"
                        self._minio_client.upload_file(self._bucket, key, filepath)
            except Exception:
                logger.exception("Failed to download/upload Kaggle source: %s", dataset)
                failures.append(dataset)
        return failures


if __name__ == "__main__":
    configure_logging()
    failed = KaggleDownloader(minio_client=MinioClient()).download_all()
    if failed:
        sys.exit(1)
