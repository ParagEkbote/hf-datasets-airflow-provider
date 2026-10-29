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
    StreamDatasetOperator,
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


def test_stream_operator_resolves_revision_and_enforces_both_bounds(tmp_path):
    rows = [{"value": value} for value in range(1, 10)]
    with patch("airflow.providers.huggingface.operators.datasets.HuggingFaceHook") as hook_type:
        hook = hook_type.return_value
        hook.get_dataset_info.return_value.sha = "pinned-stream-sha"
        hook.load_dataset.return_value = iter(rows)
        operator = StreamDatasetOperator(
            task_id="stream",
            path="org/data",
            split="train",
            revision="main",
            output_uri=(tmp_path / "stream.json").as_uri(),
            max_source_rows=5,
            max_output_rows=2,
            transform=lambda row, increment: {"value": row["value"] + increment},
            transform_kwargs={"increment": 1},
            predicate=lambda row: row["value"] % 2 == 0,
            feature_types={"value": "int64"},
            huggingface_conn_id=None,
        )
        reference = operator.execute({})

    hook.get_dataset_info.assert_called_once_with("org/data", "main")
    hook.load_dataset.assert_called_once_with(
        "org/data", streaming=True, split="train", revision="pinned-stream-sha"
    )
    dataset = _load_artifact(reference)
    assert dataset["value"] == [2, 4]
    assert dataset.features["value"].dtype == "int64"
    assert reference["revision"] == "pinned-stream-sha"
    assert reference["source_rows_examined"] == 3
    assert reference["streaming"] is True


def test_stream_operator_preserves_split_names_for_unspecified_split(tmp_path):
    with patch("airflow.providers.huggingface.operators.datasets.HuggingFaceHook") as hook_type:
        hook = hook_type.return_value
        hook.load_dataset.return_value = {
            "train": iter([{"value": 1}, {"value": 2}]),
            "validation": iter([{"value": 3}]),
        }
        operator = StreamDatasetOperator(
            task_id="stream_splits",
            path="csv",
            output_uri=(tmp_path / "splits.json").as_uri(),
            max_source_rows=3,
            max_output_rows=3,
            transform=lambda row: row,
            huggingface_conn_id=None,
        )
        reference = operator.execute({})

    dataset = _load_artifact(reference)
    assert list(dataset) == ["train", "validation"]
    assert dataset["train"]["value"] == [1, 2]
    assert dataset["validation"]["value"] == [3]


@pytest.mark.parametrize("limit", [0, -1])
def test_stream_operator_requires_positive_row_bounds(tmp_path, limit):
    with pytest.raises(ValueError, match="row bounds must be positive"):
        StreamDatasetOperator(
            task_id="invalid_stream",
            path="csv",
            output_uri=(tmp_path / "invalid.json").as_uri(),
            max_source_rows=limit,
            max_output_rows=1,
            transform=lambda row: row,
        )


def test_stream_operator_fails_when_no_rows_pass_the_filter(tmp_path):
    with patch("airflow.providers.huggingface.operators.datasets.HuggingFaceHook") as hook_type:
        hook_type.return_value.load_dataset.return_value = iter([{"value": 1}, {"value": 2}])
        operator = StreamDatasetOperator(
            task_id="empty_stream",
            path="csv",
            output_uri=(tmp_path / "empty.json").as_uri(),
            max_source_rows=2,
            max_output_rows=2,
            transform=lambda row: row,
            predicate=lambda row: False,
            huggingface_conn_id=None,
        )
        with pytest.raises(ValueError, match="No streaming rows"):
            operator.execute({})


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
    split_path = tmp_path / "input.json.split-00000.parquet"
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


def test_load_retry_reuses_resolved_revision_from_task_state_store(tmp_path, monkeypatch):
    monkeypatch.setenv("AIRFLOW_CONN_HUGGINGFACE_TEST", '{"conn_type":"huggingface","password":"operator-test-token"}')
    dataset = Dataset.from_dict({"value": [1]})
    store_data = {}
    class Store:
        def get(self, key): return store_data.get(key)
        def set(self, key, value): store_data[key] = value
    ti = type("TaskInstance", (), {"dag_id": "d", "task_id": "load"})()
    context = {"task_state_store": Store(), "ti": ti}
    with (
        patch("airflow.providers.huggingface.hooks.huggingface.HfApi", autospec=True) as api,
        patch("airflow.providers.huggingface.hooks.huggingface.load_dataset", autospec=True) as load_dataset,
    ):
        api.return_value.dataset_info.side_effect = [
            type("Info", (), {"sha": "pinned-sha"})(),
            type("Info", (), {"sha": "repinned-sha"})(),
        ]
        load_dataset.return_value = dataset
        first = LoadDatasetOperator(task_id="load", path="org/data", revision="main",
                                    output_uri=(tmp_path / "first.json").as_uri(),
                                    huggingface_conn_id="huggingface_test")
        first_result = first.execute(context)
        second = LoadDatasetOperator(task_id="load", path="org/data", revision="other-tag",
                                     output_uri=(tmp_path / "second.json").as_uri(),
                                     huggingface_conn_id="huggingface_test")
        second_result = second.execute(context)
    assert first_result["revision"] == "pinned-sha"
    assert second_result["revision"] == "repinned-sha"
    assert api.return_value.dataset_info.call_args_list == [
        ( ("org/data",), {"revision": "main"} ),
        ( ("org/data",), {"revision": "other-tag"} ),
    ]
    assert [call.kwargs["revision"] for call in load_dataset.call_args_list] == [
        "pinned-sha", "repinned-sha"
    ]


def test_push_preserves_user_outlets_and_changes_commit_for_new_fingerprint(tmp_path, hub_dependencies):
    hf_api, push_to_hub = hub_dependencies
    reference = _save(Dataset.from_dict({"value": [1]}), (tmp_path / "input.json").as_uri())
    caller_asset = type("Asset", (), {"uri": "custom://caller"})()
    first_commit = commit_info(oid="1" * 40)
    second_commit = commit_info(oid="2" * 40)
    push_to_hub.side_effect = [first_commit, second_commit]
    operator = PushToHubOperator(task_id="publish", input_uri=reference, repo_id="org/data",
                                 huggingface_conn_id="huggingface_test", outlets=[caller_asset])
    assert caller_asset in operator.outlets
    assert operator.published_asset in operator.outlets
    operator.execute({})
    changed = _save(Dataset.from_dict({"value": [2]}), (tmp_path / "changed.json").as_uri())
    operator.input_uri = changed
    result = operator.execute({})
    assert push_to_hub.call_count == 2
    assert result["commit_sha"] == second_commit.oid


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
