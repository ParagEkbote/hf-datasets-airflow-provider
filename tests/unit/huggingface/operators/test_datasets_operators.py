# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Regression coverage for durable load/map/publish behavior."""
from unittest.mock import patch

import pytest
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, Value
from huggingface_hub import CommitInfo

from airflow.providers.huggingface.operators.datasets import (
    LoadDatasetOperator,
    MapDatasetOperator,
    PushToHubOperator,
    _load_artifact,
    _save,
)


def add_one(example):
    return {"value": example["value"] + 1}


def commit_info(repo_id="org/data", oid="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef", message="publish"):
    url = f"https://huggingface.co/datasets/{repo_id}/commit/{oid}"
    return CommitInfo(commit_url=url, commit_message=message, commit_description="", oid=oid)



@pytest.fixture
def hub_dependencies(monkeypatch):
    """Use a real Airflow Connection while mocking only Hub and Datasets APIs."""
    monkeypatch.setenv(
        "AIRFLOW_CONN_HUGGINGFACE_TEST",
        '{"conn_type":"huggingface","password":"operator-test-token"}',
    )
    with (
        patch("airflow.providers.huggingface.hooks.huggingface.HfApi", autospec=True) as hf_api,
        patch("datasets.Dataset.push_to_hub", autospec=True) as push_to_hub,
    ):
        hf_api.return_value.list_repo_commits.return_value = []
        yield hf_api, push_to_hub


def missing_readme_error():
    from huggingface_hub.utils import EntryNotFoundError
    return EntryNotFoundError()


def test_load_pins_and_records_resolved_revision_and_honors_download_mode(tmp_path, monkeypatch):
    output = (tmp_path / "loaded.json").as_uri()
    dataset = Dataset.from_dict({"value": [1, 2]})
    monkeypatch.setenv("AIRFLOW_CONN_HUGGINGFACE_TEST", '{"conn_type":"huggingface","password":"operator-test-token"}')
    with (
        patch("airflow.providers.huggingface.hooks.huggingface.HfApi", autospec=True) as hf_api,
        patch("airflow.providers.huggingface.hooks.huggingface.load_dataset", autospec=True) as load_dataset,
    ):
        hf_api.return_value.dataset_info.return_value.sha = "resolved-sha"
        load_dataset.return_value = dataset
        operator = LoadDatasetOperator(
            task_id="load", path="org/data", revision="release-tag", output_uri=output,
            download_mode="force_redownload", cache_dir=str(tmp_path / "cache"),
            huggingface_conn_id="huggingface_test",
        )
        reference = operator.execute({})
    hf_api.return_value.dataset_info.assert_called_once_with("org/data", revision="release-tag")
    load_dataset.assert_called_once_with(
        "org/data", revision="resolved-sha", download_mode="force_redownload", cache_dir=str(tmp_path / "cache"),
        token="operator-test-token",
    )
    assert reference["revision"] == "resolved-sha"
    assert _load_artifact(reference).num_rows == 2


def test_non_hub_load_does_not_resolve_hub_revision(tmp_path):
    with patch("airflow.providers.huggingface.hooks.huggingface.load_dataset", autospec=True) as load_dataset:
        load_dataset.return_value = Dataset.from_dict({"x": [1]})
        operator = LoadDatasetOperator(task_id="csv", path="csv", output_uri=(tmp_path / "csv.json").as_uri(),
                                       huggingface_conn_id=None)
        result = operator.execute({})
    load_dataset.assert_called_once_with("csv", download_mode="reuse_dataset_if_exists")
    assert result["revision"] is None


def test_parquet_manifest_preserves_order_features_and_dataset_splits(tmp_path):
    features = Features({"z_first": Value("int64"), "a_second": Value("string"),
                         "kind": ClassLabel(names=["first", "second"])})
    source = DatasetDict({
        "train": Dataset.from_dict({"z_first": [1], "a_second": ["one"], "kind": [0]}, features=features),
        "validation": Dataset.from_dict({"z_first": [2], "a_second": ["two"], "kind": [1]}, features=features),
    })
    reference = _save(source, (tmp_path / "splits.json").as_uri(), {"revision": "abc"})
    restored = _load_artifact(reference)
    assert list(restored) == ["train", "validation"]
    assert restored["train"].column_names == ["z_first", "a_second", "kind"]
    assert restored["validation"].features == features
    assert restored["train"].features["kind"].int2str(0) == "first"
    assert reference["format"] == "parquet"
    assert reference["fingerprint"]
    assert reference["splits"]["train"]["rows"] == 1


def test_parquet_rejects_media_feature_types(tmp_path):
    dataset = Dataset.from_dict({"image": [None]}, features=Features({"image": Image()}))
    with pytest.raises(ValueError, match="unsupported Parquet feature type"):
        _save(dataset, (tmp_path / "image.json").as_uri())


def test_map_uses_parse_time_callable_and_persists_reference(tmp_path):
    source = _save(Dataset.from_dict({"value": [1, 2]}), (tmp_path / "input.json").as_uri(),
                   {"repository_id": "org/data", "revision": "source-sha"})
    source.update(repository_id="org/data", revision="source-sha")
    operator = MapDatasetOperator(task_id="map", input_uri=source,
                                  output_uri=(tmp_path / "mapped.json").as_uri(), transform=add_one)
    reference = operator.execute({})
    assert _load_artifact(reference)["value"] == [2, 3]
    assert reference["revision"] == "source-sha"
    assert "transform" not in operator.template_fields
    with pytest.raises(TypeError, match="callable"):
        MapDatasetOperator(task_id="bad", input_uri=source, output_uri="file:///tmp/out.json", transform="x:y")


def test_map_transform_exception_propagates(tmp_path):
    source = _save(Dataset.from_dict({"value": [1]}), (tmp_path / "input.json").as_uri())
    error = LookupError("transform failed")
    def fail(example):
        raise error
    operator = MapDatasetOperator(task_id="map_fail", input_uri=source,
                                  output_uri=(tmp_path / "mapped.json").as_uri(), transform=fail)
    with pytest.raises(LookupError) as raised:
        operator.execute({})
    assert raised.value is error


def test_fingerprint_and_split_checksum_mismatches_are_rejected(tmp_path):
    reference = _save(Dataset.from_dict({"value": [1]}), (tmp_path / "input.json").as_uri())
    changed_reference = dict(reference, fingerprint="wrong")
    with pytest.raises(ValueError, match="artifact fingerprint"):
        _load_artifact(changed_reference)
    split_path = tmp_path / "input.json.split-0.parquet"
    split_path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="split fingerprint"):
        _load_artifact(reference)


def test_push_returns_commit_info_and_emits_matching_asset_event(tmp_path, hub_dependencies):
    hf_api, push_to_hub = hub_dependencies
    reference = _save(Dataset.from_dict({"value": [1]}), (tmp_path / "input.json").as_uri())
    commit = commit_info()
    push_to_hub.return_value = commit
    operator = PushToHubOperator(task_id="publish", input_uri=reference, repo_id="org/data",
                                 commit_message="publish source", huggingface_conn_id="huggingface_test")
    event = type("Event", (), {"extra": None})()
    class Events:
        def __getitem__(self, asset):
            assert asset.uri == "hf://datasets/org/data"
            return event
    result = operator.execute({"outlet_events": Events()})
    push_to_hub.assert_called_once()
    kwargs = push_to_hub.call_args.kwargs
    assert reference["fingerprint"] in kwargs["commit_message"]
    hf_api.return_value.list_repo_commits.assert_called_once_with("org/data", repo_type="dataset")
    assert result["commit_sha"] == commit.oid
    assert result["commit_url"] == commit.commit_url
    assert event.extra["commit_sha"] == result["commit_sha"]
    assert event.extra["commit_url"] == result["commit_url"]


def test_push_failure_propagates_and_emits_no_event(tmp_path, hub_dependencies):
    _, push_to_hub = hub_dependencies
    reference = _save(Dataset.from_dict({"value": [1]}), (tmp_path / "input.json").as_uri())
    error = ValueError("publication denied")
    push_to_hub.side_effect = error
    operator = PushToHubOperator(task_id="failed", input_uri=reference, repo_id="org/data",
                                 huggingface_conn_id="huggingface_test")
    event = type("Event", (), {"extra": None})()
    class Events:
        def __getitem__(self, asset): return event
    with pytest.raises(ValueError) as raised:
        operator.execute({"outlet_events": Events()})
    assert raised.value is error
    assert event.extra is None


def test_push_requires_reference_and_validates_fingerprint(tmp_path):
    reference = _save(Dataset.from_dict({"value": [1]}), (tmp_path / "input.json").as_uri())
    operator = PushToHubOperator(task_id="publish", input_uri=reference["uri"], repo_id="org/data")
    with pytest.raises(ValueError, match="requires an artifact reference"):
        operator.execute({})
    reference["fingerprint"] = "bad"
    operator = PushToHubOperator(task_id="publish_bad", input_uri=reference, repo_id="org/data")
    with pytest.raises(ValueError, match="artifact fingerprint"):
        operator.execute({})


def test_airflow_30_path_without_state_store_and_retry_with_store(tmp_path, hub_dependencies):
    hf_api, _ = hub_dependencies
    reference = _save(Dataset.from_dict({"value": [1]}), (tmp_path / "input.json").as_uri())
    marker = f"airflow-artifact-sha256:{reference['fingerprint']}"
    commit = commit_info(oid="a" * 40, message=f"Publish [{marker}]")
    hf_api.return_value.list_repo_commits.return_value = [commit]
    operator = PushToHubOperator(task_id="legacy", input_uri=reference, repo_id="org/data",
                                 huggingface_conn_id="huggingface_test")
    result = operator.execute({})
    assert result["commit_sha"] == "a" * 40
    store = type("Store", (), {"get": lambda self, key: result,
                                "set": lambda self, key, value: None})()
    event = type("Event", (), {"extra": None})()
    class Events:
        def __getitem__(self, asset): return event
    assert operator.execute({"task_state_store": store, "outlet_events": Events()}) == result
    assert event.extra["commit_sha"] == result["commit_sha"]


def test_datasets_floor_returns_hub_commit_info():
    import inspect
    import tomllib
    from pathlib import Path
    from datasets import Dataset as HFDataset
    from huggingface_hub import CommitInfo as HubCommitInfo

    project = tomllib.loads(Path("pyproject.toml").read_text())
    requirement = next(item for item in project["project"]["dependencies"] if item.startswith("datasets"))
    assert ">=3.4.0" in requirement
    assert inspect.signature(HFDataset.push_to_hub).return_annotation is HubCommitInfo


def test_state_store_write_failure_retry_does_not_push_twice(tmp_path, hub_dependencies):
    hf_api, push_to_hub = hub_dependencies
    reference = _save(Dataset.from_dict({"value": [1]}), (tmp_path / "retry_after_push.json").as_uri())
    marker = f"airflow-artifact-sha256:{reference['fingerprint']}"
    commit = commit_info(oid="a" * 40, message=f"Publish [{marker}]")
    hf_api.return_value.list_repo_commits.side_effect = [[], [commit]]
    push_to_hub.return_value = commit
    operator = PushToHubOperator(task_id="retry_after_push", input_uri=reference, repo_id="org/data",
                                 huggingface_conn_id="huggingface_test")
    class FailingStore:
        def get(self, key): return None
        def set(self, key, value): raise RuntimeError("worker failed after Hub commit")
    with pytest.raises(RuntimeError, match="worker failed"):
        operator.execute({"task_state_store": FailingStore()})
    class Store:
        def get(self, key): return None
        def set(self, key, value): pass
    retry_result = operator.execute({"task_state_store": Store()})
    push_to_hub.assert_called_once()
    assert retry_result["commit_sha"] == "a" * 40


def test_dataset_card_is_created_and_published_after_data_commit(tmp_path, hub_dependencies):
    hf_api, push_to_hub = hub_dependencies
    reference = _save(Dataset.from_dict({"value": [1]}), (tmp_path / "card.json").as_uri())
    commit = commit_info(oid="b" * 40)
    push_to_hub.return_value = commit
    hf_api.return_value.upload_file.return_value = commit_info(oid="e" * 40)
    with patch("airflow.providers.huggingface.hooks.huggingface.hf_hub_download", autospec=True,
               side_effect=missing_readme_error()):
        operator = PushToHubOperator(task_id="card", input_uri=reference, repo_id="org/data", dataset_card=True,
                                     card_metadata={"license": "mit"}, card_description="A test dataset",
                                     huggingface_conn_id="huggingface_test")
        result = operator.execute({})
    uploaded = hf_api.return_value.upload_file.call_args.kwargs["path_or_fileobj"].decode()
    assert "license: \"mit\"" in uploaded
    assert "A test dataset" in uploaded
    assert result["dataset_card"] == {
        "enabled": True, "action": "created", "path": "README.md", "commit_sha": "e" * 40
    }


def test_existing_card_is_preserved_and_card_policy_is_validated(tmp_path, hub_dependencies):
    _, push_to_hub = hub_dependencies
    reference = _save(Dataset.from_dict({"value": [1]}), (tmp_path / "preserve.json").as_uri())
    commit = commit_info(oid="b" * 40)
    push_to_hub.return_value = commit
    readme = tmp_path / "README.md"
    readme.write_text("user documentation")
    with patch("airflow.providers.huggingface.hooks.huggingface.hf_hub_download", autospec=True,
               return_value=str(readme)) as download:
        operator = PushToHubOperator(task_id="preserve", input_uri=reference, repo_id="org/data", dataset_card=True,
                                     huggingface_conn_id="huggingface_test")
        result = operator.execute({})
        download.assert_called_once()
    assert result["dataset_card"]["action"] == "preserved"
    with pytest.raises(ValueError, match="card_update_policy"):
        PushToHubOperator(task_id="invalid_card", input_uri=reference, repo_id="org/data",
                          card_update_policy="replace")


def test_object_storage_memory_scheme_handoff(monkeypatch):
    """Exercise non-local ObjectStoragePath open semantics without cloud credentials."""
    from fsspec.implementations.memory import MemoryFileSystem
    from airflow.sdk.io import fs as airflow_fs

    monkeypatch.setattr(airflow_fs, "_register_filesystems", lambda: {
        "file": lambda conn_id, options: __import__("fsspec").implementations.local.LocalFileSystem(**options),
        "memory": lambda conn_id, options: MemoryFileSystem(**options),
    })
    uri = "memory://hf-provider-tests/round-trip.json"
    reference = _save(Dataset.from_dict({"value": [3]}), uri)
    restored = _load_artifact(reference)
    assert restored["value"] == [3]
