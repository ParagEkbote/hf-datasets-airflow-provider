from datasets import Dataset, load_from_disk

from airflow.providers.huggingface.operators.datasets import (
    LoadDatasetOperator,
    MapDatasetOperator,
    PushToHubOperator,
)


def add_one(example):
    return {"value": example["value"] + 1}


def test_load_operator_persists_compact_reference(tmp_path, monkeypatch):
    from airflow.providers.huggingface.operators import datasets as operators

    seen = {}
    dataset = Dataset.from_dict({"text": ["a", "b"]})

    class FakeHook:
        def __init__(self, conn_id):
            pass

        def load_dataset(self, path, **kwargs):
            seen.update(path=path, **kwargs)
            return dataset

        def get_dataset_info(self, path, revision):
            return type("Info", (), {"sha": "abc123"})()

    monkeypatch.setattr(operators, "HuggingFaceDatasetsHook", FakeHook)
    output = (tmp_path / "loaded").as_uri()
    operator = LoadDatasetOperator(
        task_id="load", path="org/data", name="default", split="train",
        revision="main", cache_dir=str(tmp_path / "cache"), output_uri=output,
    )

    result = operator.execute({})

    assert seen["download_mode"] == "reuse_dataset_if_exists"
    assert seen["revision"] == "main"
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

    monkeypatch.setattr(operators, "HuggingFaceDatasetsHook", FakeHook)
    operator = LoadDatasetOperator(
        task_id="load", path="org/data", output_uri=(tmp_path / "data").as_uri(),
        download_mode="force_redownload",
    )

    operator.execute({})

    assert seen["download_mode"] == "force_redownload"


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
