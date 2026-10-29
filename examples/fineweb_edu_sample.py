"""Stream, clean, and publish a bounded FineWeb sample."""
from airflow.sdk import DAG
from airflow.providers.huggingface.operators.datasets import PushToHubOperator, StreamDatasetOperator
import os

def clean_fineweb_document(example):
    text = " ".join(example["text"].split())
    return {"text": text, "word_count": len(text.split())}


def is_long_fineweb_document(example):
    return example["word_count"] >= 100


with DAG("huggingface_fineweb_sample", tags=["huggingface", "fineweb", "example"]) as dag:
    sample = StreamDatasetOperator(task_id="stream_bounded_sample", path="HuggingFaceFW/fineweb-edu",
        name="sample-10BT", split="train",
        revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        output_uri="file:///tmp/hf-datasets/fineweb/clean.json",
        max_source_rows=10_000, max_output_rows=1_000,
        transform=clean_fineweb_document, predicate=is_long_fineweb_document,
        feature_types={"text": "string", "word_count": "int64"})
    publish = PushToHubOperator(task_id="publish", input_uri=sample.output,
        repo_id=os.environ.get("HF_FINEWEB_REPO_ID", "my-org/fineweb-edu-sample"))
    sample >> publish
