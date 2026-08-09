"""Behavior: download_all uploads one object per file per source, correctly prefixed."""

import os
from unittest.mock import MagicMock, patch

from src.ingest.download_kaggle import KaggleDownloader


def fake_download(dataset: str, path: str) -> None:
    """Stand-in for the real Kaggle download — drops one file in `path`."""
    with open(os.path.join(path, "data.csv"), "w") as f:
        f.write("id,text\n1,hello\n")


def test_download_all_uploads_one_object_per_source_with_correct_prefix():
    fake_minio = MagicMock()

    with patch(
        "src.ingest.download_kaggle.download_kaggle_dataset",
        side_effect=fake_download,
    ):
        downloader = KaggleDownloader(minio_client=fake_minio)
        downloader.download_all()

    uploaded_keys = [call.args[1] for call in fake_minio.upload_file.call_args_list]
    assert uploaded_keys == [
        "kaggle/cosmos98/twitter-and-reddit-sentimental-analysis-dataset/data.csv",
        "kaggle/tariqsays/sentiment-dataset-with-1-million-tweets/data.csv",
    ]
    assert fake_minio.upload_file.call_count == 2
