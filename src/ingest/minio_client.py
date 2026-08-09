"""MinIO client — thin wrapper for landing raw ingest files."""

# src/ingest/minio_client.py

import os
from typing import Any, cast

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

    def upload_file(
        self,
        bucket: str,
        key: str,
        filepath: str,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Upload a local file to bucket/key, unmodified.

        `metadata` (e.g. source revision, content checksum) is stored as
        MinIO object metadata for provenance — queryable via `mc stat`.

        `cast` below: minio's `fput_object` types metadata as an invariant
        `Dict[str, str | list[str] | tuple[str]]`; our `dict[str, str]` is
        structurally fine (every value is a plain str) but dict invariance
        makes Pylance reject it without the cast.
        """
        self._client.fput_object(
            bucket, key, filepath, metadata=cast("dict[str, Any] | None", metadata)
        )
