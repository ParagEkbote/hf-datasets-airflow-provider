"""Transform and publish a streamed Hugging Face dataset directly to the Hub."""
import os

from airflow.sdk import DAG
from airflow.providers.huggingface.operators.datasets import StreamDatasetOperator


def normalize_review(example):
    return {"text": " ".join(example["text"].split()), "label": example["label"]}


def keep_nonempty_review(example):
    return bool(example["text"])


with DAG("huggingface_rotten_tomatoes_stream", tags=["huggingface", "streaming", "example"]) as dag:
    publish = StreamDatasetOperator(
        task_id="stream_transform_publish",
        path="cornell-movie-review-data/rotten_tomatoes",
        split="train",
        repo_id=os.environ.get("HF_STREAM_REPO_ID", "my-org/rotten-tomatoes-streamed"),
        transform=normalize_review,
        predicate=keep_nonempty_review,
    )
