import pytest
from urllib.parse import urlsplit, urlunsplit

from airflow.providers.huggingface.asset import create_asset, sanitize_uri


def test_sanitize_dataset_asset_uri_string_and_airflow_split_result():
    uri = "hf://datasets/org/example/"
    assert sanitize_uri(uri) == "hf://datasets/org/example"
    normalized = sanitize_uri(urlsplit(uri))
    assert urlunsplit(normalized) == "hf://datasets/org/example"
    assert create_asset(uri).uri == "hf://datasets/org/example"


def test_rejects_non_dataset_asset_uri():
    with pytest.raises(ValueError):
        sanitize_uri("hf://models/org/example")
