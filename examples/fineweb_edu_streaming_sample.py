"""Persist a bounded, provider-managed FineWeb-Edu streaming sample."""

from datetime import datetime

from airflow.sdk import DAG
from airflow.providers.huggingface.operators.datasets import PushToHubOperator, StreamDatasetOperator

MAX_SOURCE_ROWS = 10_000
MAX_OUTPUT_ROWS = 1_000
OUTPUT_URI = "file:///tmp/hf-datasets/fineweb_edu_clean_sample"

with DAG(
    dag_id="huggingface_fineweb_edu_streaming_sample",
    start_date=datetime(2024, 1, 1), schedule=None, catchup=False,
    tags=["huggingface", "streaming", "example"], doc_md=__doc__,
) as dag:
    sample = StreamDatasetOperator(
        task_id="stream_bounded_sample", path="HuggingFaceFW/fineweb-edu",
        name="sample-10BT", split="train", revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        output_uri=OUTPUT_URI, max_source_rows=MAX_SOURCE_ROWS, max_output_rows=MAX_OUTPUT_ROWS,
        transform="examples.transforms:clean_fineweb_document",
        predicate="examples.transforms:is_long_fineweb_document",
        feature_types={"text": "string", "word_count": "int64"},
    )
    publish = PushToHubOperator(
        task_id="publish_sample", input_uri=sample.output,
        repo_id="AINovice2005/fineweb-edu-clean-sample", private=True,
        commit_message="Airflow: bounded FineWeb-Edu sample",
    )
    sample >> publish
