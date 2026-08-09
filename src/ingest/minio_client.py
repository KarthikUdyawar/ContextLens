"""MinIO client — thin wrapper for landing raw ingest files."""

# src/ingest/minio_client.py

import os

from minio import Minio


class MinioClient:
    """Wraps a MinIO connection for landing raw ingest files."""

    def __init__(
        self,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        secure: bool = False,
    ) -> None:
        """Build the underlying MinIO client.

        Falls back to env vars, matching docker-compose.yml defaults.
        """
        self._client = Minio(
            endpoint or os.environ.get("MINIO_ENDPOINT", "localhost:9000"),
            access_key=access_key or os.environ.get("MINIO_ROOT_USER", "minioadmin"),
            secret_key=secret_key
            or os.environ.get("MINIO_ROOT_PASSWORD", "minioadmin"),
            secure=secure,
        )

    def upload_file(self, bucket: str, key: str, filepath: str) -> None:
        """Upload a local file to bucket/key, unmodified."""
        self._client.fput_object(bucket, key, filepath)
