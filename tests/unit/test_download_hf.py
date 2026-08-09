"""HfDownloader.download_all behavior tests.

Uploads one object per configured source, correctly prefixed; survives
a failing source without stopping the rest.
"""
# tests/unit/test_download_hf.py

from unittest.mock import MagicMock, patch

from src.ingest.download_hf import HfDownloader


def test_download_all_uploads_one_object_per_source_with_correct_prefix():
    fake_minio = MagicMock()
    fake_dataset = MagicMock()

    with patch("src.ingest.download_hf.load_dataset", return_value=fake_dataset):
        downloader = HfDownloader(minio_client=fake_minio)
        downloader.download_all()

    uploaded_keys = [call.args[1] for call in fake_minio.upload_file.call_args_list]
    assert uploaded_keys == [
        "hf/sentiment140/data.parquet",
        "hf/cardiffnlp/tweet_eval/data.parquet",
        "hf/bdstar/twitter-sentiment-analysis/data.parquet",
        "hf/bdstar/Tweets-Sentiment-Analysis/data.parquet",
    ]
    assert fake_minio.upload_file.call_count == 4


def test_download_all_continues_past_a_failing_source_and_reports_it():
    fake_minio = MagicMock()

    def load_dataset_side_effect(name, *args, **kwargs):
        if name == "cardiffnlp/tweet_eval":
            raise RuntimeError("HF fetch failed")
        return MagicMock()

    with patch(
        "src.ingest.download_hf.load_dataset", side_effect=load_dataset_side_effect
    ):
        downloader = HfDownloader(minio_client=fake_minio)
        failures = downloader.download_all()

    uploaded_keys = [call.args[1] for call in fake_minio.upload_file.call_args_list]
    assert uploaded_keys == [
        "hf/sentiment140/data.parquet",
        "hf/bdstar/twitter-sentiment-analysis/data.parquet",
        "hf/bdstar/Tweets-Sentiment-Analysis/data.parquet",
    ]
    assert failures == ["cardiffnlp/tweet_eval"]
