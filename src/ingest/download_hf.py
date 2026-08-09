"""Download raw text from confirmed HF sources, land unmodified in MinIO."""

# src/ingest/download_hf.py

# RUN: uv run --group ingest python -m src.ingest.download_hf

import hashlib
import sys
import tempfile

from datasets import load_dataset
from huggingface_hub import HfApi

from src.ingest.minio_client import MinioClient
from src.utils.logging_config import configure_logging, get_logger

logger = get_logger(__name__)

# (dataset name, config) — config=None means no sub-config needed.
HF_SOURCES: list[tuple[str, str | None]] = [
    ("stanfordnlp/sentiment140", None),
    ("cardiffnlp/tweet_eval", "sentiment"),
    ("bdstar/twitter-sentiment-analysis", None),
    ("bdstar/Tweets-Sentiment-Analysis", None),
]

BUCKET = "raw-data"


class HfDownloader:
    """Downloads confirmed HF sources, lands each raw in MinIO."""

    def __init__(self, minio_client: MinioClient, bucket: str = BUCKET) -> None:
        """Store the MinIO client to upload through and target bucket."""
        self._minio_client = minio_client
        self._bucket = bucket

    def download_all(self) -> list[str]:
        """Download each configured HF source, upload raw to MinIO.

        Returns names of sources that failed — doesn't stop on one bad source.
        """
        failures: list[str] = []
        for name, config in HF_SOURCES:
            try:
                args = (name, config) if config else (name,)
                # Resolve the moving `refs/convert/parquet` ref to its current
                # commit SHA and pin the actual load_dataset() call to that SHA —
                # gives an immutable, reproducible snapshot instead of a moving ref.
                sha = HfApi().dataset_info(
                    name, revision="refs/convert/parquet"
                ).sha
                if sha is None:
                    raise ValueError(f"HF returned no commit sha for {name}")
                revision = sha
                dataset = load_dataset(
                    *args,
                    split="train",
                    revision=revision,
                )
                with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
                    dataset.to_parquet(tmp.name)
                    with open(tmp.name, "rb") as f:
                        checksum = hashlib.sha256(f.read()).hexdigest()
                    key = f"hf/{name}/data.parquet"
                    self._minio_client.upload_file(
                        self._bucket,
                        key,
                        tmp.name,
                        metadata={"revision": revision, "sha256": checksum},
                    )
            except Exception:
                logger.exception("Failed to download/upload HF source: %s", name)
                failures.append(name)
        return failures


if __name__ == "__main__":
    configure_logging()
    failed = HfDownloader(minio_client=MinioClient()).download_all()
    if failed:
        sys.exit(1)
