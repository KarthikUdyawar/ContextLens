"""Behavior: upload_file puts bytes at exact key in exact bucket."""
# tests/unit/test_minio_client.py

from unittest.mock import MagicMock, patch

from src.ingest.minio_client import MinioClient


def test_upload_file_puts_bytes_at_exact_key_in_bucket():
    fake_client = MagicMock()

    with patch("src.ingest.minio_client.Minio", return_value=fake_client):
        client = MinioClient()
        client.upload_file(
            "raw-data", "hf/sentiment140/data.parquet", "/tmp/data.parquet"
        )

    fake_client.fput_object.assert_called_once_with(
        "raw-data", "hf/sentiment140/data.parquet", "/tmp/data.parquet"
    )
