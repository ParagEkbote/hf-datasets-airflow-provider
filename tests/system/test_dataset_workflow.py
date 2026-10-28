import csv

from datasets import load_from_disk

from airflow.providers.huggingface.operators.datasets import LoadDatasetOperator, MapDatasetOperator


def add_one(row):
    return {"value": row["value"] + 1}


def test_local_load_transform_persist_workflow(tmp_path):
    source_file = tmp_path / "source.csv"
    with source_file.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["value"])
        writer.writeheader()
        writer.writerows([{"value": 4}, {"value": 9}])

    loaded_uri = (tmp_path / "loaded").as_uri()
    mapped_uri = (tmp_path / "mapped").as_uri()
    load = LoadDatasetOperator(
        task_id="load", path="csv", output_uri=loaded_uri,
        cache_dir=str(tmp_path / "cache"),
        load_kwargs={"data_files": str(source_file)},
    )
    load_result = load.execute({})
    mapped = MapDatasetOperator(
        task_id="map", input_uri=load_result, output_uri=mapped_uri,
        transform="tests.system.test_dataset_workflow:add_one",
    )
    map_result = mapped.execute({})
    dataset = load_from_disk(str(tmp_path / "mapped"))

    assert sum(split["rows"] for split in load_result["splits"].values()) == 2
    assert map_result["source_uri"] == loaded_uri
    assert map_result["revision"] is None
    assert dataset["train"]["value"] == [5, 10]
    assert dataset["train"].features["value"].dtype == "int64"
