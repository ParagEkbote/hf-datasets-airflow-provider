"""Airflow asset URI and OpenLineage integration for Hub datasets."""
from __future__ import annotations

from urllib.parse import SplitResult, urlsplit


def sanitize_uri(uri: str | SplitResult) -> str | SplitResult:
    """Normalize the canonical dataset URI; Airflow passes a SplitResult."""
    parsed = uri if isinstance(uri, SplitResult) else urlsplit(uri)
    if parsed.scheme != "hf" or parsed.netloc != "datasets" or not parsed.path.strip("/"):
        raise ValueError(f"Invalid Hugging Face dataset asset URI: {uri}")
    normalized = parsed._replace(
        scheme="hf", netloc="datasets", path=f"/{parsed.path.strip('/')}", query="", fragment=""
    )
    if isinstance(uri, SplitResult):
        return normalized
    from urllib.parse import urlunsplit
    return urlunsplit(normalized)


def create_asset(uri: str):
    try:
        from airflow.sdk import Asset
    except ImportError:
        from airflow.datasets import Dataset as Asset
    return Asset(uri=sanitize_uri(uri))


def to_openlineage_dataset(uri: str, **kwargs):
    from openlineage.client import Dataset
    canonical = sanitize_uri(uri)
    return Dataset(namespace="https://huggingface.co", name=canonical.removeprefix("hf://"), facets={})
