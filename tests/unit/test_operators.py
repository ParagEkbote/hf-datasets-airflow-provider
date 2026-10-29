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
import pytest
from datasets import Dataset, load_from_disk

from airflow.sdk import Asset
from airflow.providers.huggingface.operators.datasets import (
    LoadDatasetOperator,
    MapDatasetOperator,
    PushToHubOperator,
)


def add_one(example):
    return {"value": example["value"] + 1}


def keep_even(example):
    return example["value"] % 2 == 0


def test_load_operator_persists_compact_reference(tmp_path, monkeypatch):
    from airflow.providers.huggingface.operators import datasets as operators

    seen, calls = {}, []
    dataset = Dataset.from_dict({"text": ["a", "b"]})

    class FakeHook:
        def __init__(self, conn_id):
            pass

        def load_dataset(self, path, **kwargs):
            calls.append("load_dataset")
            seen.update(path=path, **kwargs)
            return dataset

        def get_dataset_info(self, path, revision):
            calls.append(("get_dataset_info", revision))
            return type("Info", (), {"sha": "abc123"})()

    monkeypatch.setattr(operators, "HuggingFaceDatasetsHook", FakeHook)
    output = (tmp_path / "loaded").as_uri()
    operator = LoadDatasetOperator(
        task_id="load", path="org/data", name="default", split="train",
        revision="main", cache_dir=str(tmp_path / "cache"), output_uri=output,
    )

    result = operator.execute({})

    assert seen["download_mode"] == "reuse_dataset_if_exists"
    # The requested ref is pinned to a commit first, and that commit is what gets loaded and recorded.
    assert calls == [("get_dataset_info", "main"), "load_dataset"]
    assert seen["revision"] == "abc123"
    assert result["revision"] == "abc123"
    assert result["uri"] == output
    assert result["splits"]["dataset"]["rows"] == 2
    assert load_from_disk(str(tmp_path / "loaded")).num_rows == 2


def test_load_operator_can_force_redownload(tmp_path, monkeypatch):
    from airflow.providers.huggingface.operators import datasets as operators

    seen = {}

    class FakeHook:
        def __init__(self, conn_id):
            pass

        def load_dataset(self, path, **kwargs):
            seen.update(kwargs)
            return Dataset.from_dict({"value": [1]})

        def get_dataset_info(self, path, revision):
            return type("Info", (), {"sha": "abc123"})()

    monkeypatch.setattr(operators, "HuggingFaceDatasetsHook", FakeHook)
    operator = LoadDatasetOperator(
        task_id="load", path="org/data", output_uri=(tmp_path / "data").as_uri(),
        download_mode="force_redownload",
    )

    operator.execute({})

    assert seen["download_mode"] == "force_redownload"


def test_load_operator_does_not_resolve_revision_for_local_path(tmp_path, monkeypatch):
    from airflow.providers.huggingface.operators import datasets as operators

    class FakeHook:
        def __init__(self, conn_id):
            pass

        def load_dataset(self, path, **kwargs):
            return Dataset.from_dict({"value": [1]})

        def get_dataset_info(self, path, revision):
            raise AssertionError("a local path must not be looked up on the Hub")

    monkeypatch.setattr(operators, "HuggingFaceDatasetsHook", FakeHook)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "local" / "data").mkdir(parents=True)
    operator = LoadDatasetOperator(task_id="load", path="local/data", output_uri=(tmp_path / "out").as_uri())

    result = operator.execute({})

    assert result["revision"] is None


def test_map_operator_maps_and_persists_arrow(tmp_path):
    source = tmp_path / "input"
    output = tmp_path / "output"
    Dataset.from_dict({"value": [1, 2]}).save_to_disk(str(source))
    operator = MapDatasetOperator(
        task_id="map", input_uri=source.as_uri(), output_uri=output.as_uri(),
        transform="tests.unit.test_operators:add_one",
    )

    result = operator.execute({})

    mapped = load_from_disk(str(output))
    assert mapped["value"] == [2, 3]
    assert result["format"] == "arrow"
    assert result["splits"]["dataset"]["rows"] == 2


def test_filter_operator_uses_dataset_filter(tmp_path):
    from airflow.providers.huggingface.operators.datasets import FilterDatasetOperator

    source, output = tmp_path / "source", tmp_path / "filtered"
    Dataset.from_dict({"value": [1, 2, 3, 4]}).save_to_disk(str(source))
    operator = FilterDatasetOperator(
        task_id="filter", input_uri=source.as_uri(), output_uri=output.as_uri(),
        predicate="tests.unit.test_operators:keep_even",
    )
    reference = operator.execute({})
    assert load_from_disk(str(output))["value"] == [2, 4]
    assert reference["uri"] == output.as_uri()


def test_assemble_operator_combines_named_artifacts(tmp_path):
    from airflow.providers.huggingface.operators.datasets import AssembleDatasetOperator

    train, valid, output = tmp_path / "train", tmp_path / "valid", tmp_path / "all"
    Dataset.from_dict({"value": [1]}).save_to_disk(str(train))
    Dataset.from_dict({"value": [2]}).save_to_disk(str(valid))
    operator = AssembleDatasetOperator(
        task_id="assemble", inputs={"train": train.as_uri(), "validation": valid.as_uri()},
        output_uri=output.as_uri(),
    )
    reference = operator.execute({})
    assembled = load_from_disk(str(output))
    assert list(assembled) == ["train", "validation"]
    assert assembled["validation"]["value"] == [2]
    assert reference["splits"]["validation"]["rows"] == 1


def test_stream_operator_bounds_transforms_and_persists(tmp_path, monkeypatch):
    from airflow.providers.huggingface.operators import datasets as operators

    class FakeHook:
        def __init__(self, conn_id): pass
        def load_dataset(self, path, **kwargs):
            assert kwargs["streaming"] is True
            return iter({"text": f"  {'word ' * 100} {n}  "} for n in range(8))

    monkeypatch.setattr(operators, "HuggingFaceDatasetsHook", FakeHook)
    from airflow.providers.huggingface.operators.datasets import StreamDatasetOperator
    output = tmp_path / "streamed"
    operator = StreamDatasetOperator(
        task_id="stream", path="org/data", output_uri=output.as_uri(),
        max_source_rows=5, max_output_rows=2,
        transform="examples.transforms:clean_fineweb_document",
        predicate="examples.transforms:is_long_fineweb_document",
        feature_types={"text": "string", "word_count": "int64"},
    )
    reference = operator.execute({})
    result = load_from_disk(str(output))
    assert result.num_rows == 2
    assert reference["source_rows_examined"] == 2
    assert result.features["word_count"].dtype == "int64"


def test_push_returns_commit_metadata_and_adds_asset_event(tmp_path, monkeypatch):
    from airflow.providers.huggingface.operators import datasets as operators

    source = tmp_path / "input"
    Dataset.from_dict({"value": [1]}).save_to_disk(str(source))
    published = {}

    class FakeHook:
        def __init__(self, conn_id):
            pass

        def push_to_hub(self, dataset, repo_id, **kwargs):
            published.update(repo_id=repo_id, **kwargs)
            return type("Commit", (), {"oid": "deadbeef"})()

    monkeypatch.setattr(operators, "HuggingFaceDatasetsHook", FakeHook)
    operator = PushToHubOperator(
        task_id="publish", input_uri=source.as_uri(), repo_id="org/data",
        private=True, commit_message="update",
    )
    event = type("Event", (), {"extra": None})()
    class OutletEvents:
        def __getitem__(self, asset):
            assert asset.uri == "huggingface://datasets/org/data"
            return event

    context = {"outlet_events": OutletEvents()}

    result = operator.execute(context)

    assert published["private"] is True
    assert result["commit_sha"] == "deadbeef"
    assert event.extra["commit_sha"] == "deadbeef"
    assert result["url"] == "https://huggingface.co/datasets/org/data"


def test_failed_push_raises_without_event(tmp_path, monkeypatch):
    from airflow.providers.huggingface.operators import datasets as operators

    source = tmp_path / "input"
    Dataset.from_dict({"value": [1]}).save_to_disk(str(source))

    class FailedHook:
        def __init__(self, conn_id):
            pass

        def push_to_hub(self, dataset, repo_id, **kwargs):
            raise RuntimeError("upload failed")

    monkeypatch.setattr(operators, "HuggingFaceDatasetsHook", FailedHook)
    operator = PushToHubOperator(task_id="publish", input_uri=source.as_uri(), repo_id="org/data")

    try:
        operator.execute({})
    except RuntimeError as exc:
        assert "upload failed" in str(exc)
    else:
        raise AssertionError("publication failure must fail the task")


def test_push_can_create_dataset_card_with_compact_result(tmp_path, monkeypatch):
    from airflow.providers.huggingface.operators import datasets as operators

    source = tmp_path / "input"
    Dataset.from_dict({"value": [1]}).save_to_disk(str(source))
    published = {}

    class FakeHook:
        def __init__(self, conn_id): pass
        def get_dataset_readme(self, repo_id): return None
        def push_to_hub(self, dataset, repo_id, **kwargs): return type("Commit", (), {"oid": "data-sha"})()
        def upload_dataset_readme(self, repo_id, content, commit_message):
            published.update(repo_id=repo_id, content=content)
            return "card-sha"

    monkeypatch.setattr(operators, "HuggingFaceDatasetsHook", FakeHook)
    operator = PushToHubOperator(task_id="publish", input_uri=source.as_uri(), repo_id="org/data",
                                 dataset_card=True, card_metadata={"license": "mit"}, card_description="A dataset.")
    result = operator.execute({})
    assert "license: \"mit\"" in published["content"]
    assert "A dataset." in published["content"]
    assert result["dataset_card"] == {"enabled": True, "action": "created", "path": "README.md", "commit_sha": "card-sha"}


def test_preserve_existing_readme_and_validate_policy(tmp_path, monkeypatch):
    from airflow.providers.huggingface.operators import datasets as operators

    source = tmp_path / "input"
    Dataset.from_dict({"value": [1]}).save_to_disk(str(source))
    class FakeHook:
        def __init__(self, conn_id): pass
        def get_dataset_readme(self, repo_id): return "user docs"
        def push_to_hub(self, dataset, repo_id, **kwargs): return None
    monkeypatch.setattr(operators, "HuggingFaceDatasetsHook", FakeHook)
    operator = PushToHubOperator(task_id="publish", input_uri=source.as_uri(), repo_id="org/data", dataset_card=True)
    assert operator.execute({})["dataset_card"]["action"] == "preserved"
    try:
        PushToHubOperator(task_id="bad", input_uri=source.as_uri(), repo_id="org/data", card_update_policy="overwrite")
    except ValueError as exc:
        assert "card_update_policy" in str(exc)
    else:
        raise AssertionError("invalid card policy must fail during initialization")


def test_push_keeps_user_outlets_alongside_hub_asset():
    extra = Asset("s3://bucket/training-manifest")

    operator = PushToHubOperator(task_id="publish", input_uri="file:///tmp/in", repo_id="org/data", outlets=[extra])

    assert [asset.uri for asset in operator.outlets] == ["s3://bucket/training-manifest", "huggingface://datasets/org/data"]


def test_push_rejects_templated_repo_id():
    with pytest.raises(ValueError, match="repo_id cannot be templated"):
        PushToHubOperator(task_id="publish", input_uri="file:///tmp/in", repo_id="{{ params.repo }}")


class FakeTaskStateStore:
    """Dict-backed stand-in for ``context["task_state_store"]`` (Airflow 3.3+)."""

    def __init__(self):
        self.values = {}

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value


def _resolving_hook(resolved, sha="abc123"):
    class FakeHook:
        def __init__(self, conn_id):
            pass

        def load_dataset(self, path, **kwargs):
            return Dataset.from_dict({"value": [1]})

        def get_dataset_info(self, path, revision):
            resolved.append(revision)
            return type("Info", (), {"sha": sha})()

    return FakeHook


def test_durable_load_retry_reuses_commit_pinned_by_first_try(tmp_path, monkeypatch):
    from airflow.providers.huggingface.operators import datasets as operators

    resolved, store = [], FakeTaskStateStore()
    operator = LoadDatasetOperator(task_id="load", path="org/data", revision="main", output_uri=(tmp_path / "out").as_uri())
    monkeypatch.setattr(operators, "HuggingFaceDatasetsHook", _resolving_hook(resolved, sha="first"))
    operator.execute({"task_state_store": store})

    # The branch moves on before the retry; the retry must still load the first try's commit.
    monkeypatch.setattr(operators, "HuggingFaceDatasetsHook", _resolving_hook(resolved, sha="second"))
    result = operator.execute({"task_state_store": store})

    assert resolved == ["main"]
    assert result["revision"] == "first"


@pytest.mark.parametrize(
    ("durable", "retry_revision", "expected"),
    [
        pytest.param(False, "main", "second", id="durable-disabled"),
        pytest.param(True, "v2", "second", id="requested-revision-changed"),
    ],
)
def test_load_resolves_again_when_pin_does_not_apply(tmp_path, monkeypatch, durable, retry_revision, expected):
    from airflow.providers.huggingface.operators import datasets as operators

    resolved, store = [], FakeTaskStateStore()
    output = (tmp_path / "out").as_uri()
    monkeypatch.setattr(operators, "HuggingFaceDatasetsHook", _resolving_hook(resolved, sha="first"))
    LoadDatasetOperator(task_id="load", path="org/data", revision="main", output_uri=output, durable=durable).execute(
        {"task_state_store": store}
    )

    monkeypatch.setattr(operators, "HuggingFaceDatasetsHook", _resolving_hook(resolved, sha="second"))
    result = LoadDatasetOperator(
        task_id="load", path="org/data", revision=retry_revision, output_uri=output, durable=durable
    ).execute({"task_state_store": store})

    assert result["revision"] == expected


def _push_hook(pushes):
    class FakeHook:
        def __init__(self, conn_id):
            pass

        def push_to_hub(self, dataset, repo_id, **kwargs):
            pushes.append(repo_id)
            return type("Commit", (), {"oid": f"sha-{len(pushes)}"})()

    return FakeHook


def _outlet_events():
    event = type("Event", (), {"extra": None})()

    class OutletEvents:
        def __getitem__(self, asset):
            return event

    return OutletEvents(), event


def test_durable_push_retry_does_not_upload_again(tmp_path, monkeypatch):
    from airflow.providers.huggingface.operators import datasets as operators

    source = tmp_path / "input"
    dataset = Dataset.from_dict({"value": [1]})
    dataset.save_to_disk(str(source))
    reference = {"uri": source.as_uri(), "fingerprint": dataset._fingerprint}
    pushes, store = [], FakeTaskStateStore()
    monkeypatch.setattr(operators, "HuggingFaceDatasetsHook", _push_hook(pushes))
    operator = PushToHubOperator(task_id="publish", input_uri=reference, repo_id="org/data")
    operator.execute({"task_state_store": store})

    outlet_events, event = _outlet_events()
    result = operator.execute({"task_state_store": store, "outlet_events": outlet_events})

    assert pushes == ["org/data"]
    assert result["commit_sha"] == "sha-1"
    assert event.extra["commit_sha"] == "sha-1"


def test_durable_push_uploads_again_when_input_data_changed(tmp_path, monkeypatch):
    from airflow.providers.huggingface.operators import datasets as operators

    source = tmp_path / "input"
    Dataset.from_dict({"value": [1]}).save_to_disk(str(source))
    pushes, store = [], FakeTaskStateStore()
    monkeypatch.setattr(operators, "HuggingFaceDatasetsHook", _push_hook(pushes))
    PushToHubOperator(
        task_id="publish", input_uri={"uri": source.as_uri(), "fingerprint": "before-fix"}, repo_id="org/data"
    ).execute({"task_state_store": store})

    # e.g. the whole Dag run was cleared after fixing the transform, which keeps task state.
    result = PushToHubOperator(
        task_id="publish", input_uri={"uri": source.as_uri(), "fingerprint": "after-fix"}, repo_id="org/data"
    ).execute({"task_state_store": store})

    assert pushes == ["org/data", "org/data"]
    assert result["commit_sha"] == "sha-2"


def test_saved_reference_carries_dataset_fingerprint(tmp_path):
    source, output = tmp_path / "input", tmp_path / "output"
    Dataset.from_dict({"value": [1, 2]}).save_to_disk(str(source))
    operator = MapDatasetOperator(
        task_id="map", input_uri=source.as_uri(), output_uri=output.as_uri(),
        transform="tests.unit.test_operators:add_one",
    )

    reference = operator.execute({})

    assert reference["fingerprint"] == load_from_disk(str(source)).map(add_one)._fingerprint
