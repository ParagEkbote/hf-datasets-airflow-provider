from pathlib import Path

import pytest

from airflow.providers.huggingface.operators.datasets import _artifact_path


def test_artifact_path_decodes_file_uri():
    assert _artifact_path("file:///tmp/a%20b") == Path("/tmp/a b")


def test_artifact_path_rejects_unimplemented_schemes():
    with pytest.raises(ValueError, match="file://"):
        _artifact_path("s3://bucket/path")
