"""Load and map a bounded, preselected FineWeb sample with the public API."""
from airflow.sdk import DAG
from airflow.providers.huggingface.operators.datasets import LoadDatasetOperator, MapDatasetOperator, PushToHubOperator
import os

def clean_fineweb_document(example):
    text = " ".join(example["text"].split())
    return {"text": text, "word_count": len(text.split())}


with DAG("huggingface_fineweb_sample", tags=["huggingface", "fineweb", "example"]) as dag:
    load = LoadDatasetOperator(task_id="load", path="HuggingFaceFW/fineweb-edu",
        name="sample-10BT", split="train[:1000]",
        revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        output_uri="file:///tmp/hf-datasets/fineweb/source.json")
    transform = MapDatasetOperator(task_id="clean", input_uri=load.output,
        output_uri="file:///tmp/hf-datasets/fineweb/clean.json", transform=clean_fineweb_document)
    publish = PushToHubOperator(task_id="publish", input_uri=transform.output,
        repo_id=os.environ.get("HF_FINEWEB_REPO_ID", "my-org/fineweb-edu-sample"))
    load >> transform >> publish
