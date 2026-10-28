"""Stream a bounded sample from the large FineWeb-Edu corpus and transform it."""

from datetime import datetime
from pathlib import Path

from airflow.sdk import DAG, task

from airflow.providers.huggingface.hooks.datasets import HuggingFaceDatasetsHook


MAX_OUTPUT_ROWS = 1_000
OUTPUT_DIRECTORY = "/mnt/shared/huggingface/fineweb_edu_clean_sample"


def clean_document(example: dict) -> dict:
    """Normalize document whitespace and add a word count for quality filtering."""
    text = " ".join(example["text"].split())
    return {"text": text, "word_count": len(text.split())}


with DAG(
    dag_id="huggingface_fineweb_edu_streaming_sample",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    doc_md=__doc__,
) as dag:

    @task
    def stream_transform_and_persist() -> dict:
        """Lazily clean/filter documents and persist at most 1,000 output rows."""
        from datasets import Dataset

        dataset = HuggingFaceDatasetsHook().load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
            revision="main",
        )
        transformed = dataset.map(clean_document).filter(lambda row: row["word_count"] >= 100)
        bounded_rows = list(transformed.take(MAX_OUTPUT_ROWS))
        output = Path(OUTPUT_DIRECTORY)
        output.parent.mkdir(parents=True, exist_ok=True)
        Dataset.from_list(bounded_rows).save_to_disk(str(output))
        return {
            "uri": output.as_uri(),
            "format": "arrow",
            "repository_id": "HuggingFaceFW/fineweb-edu",
            "configuration": "sample-10BT",
            "split": "train",
            "streaming": True,
            "rows": len(bounded_rows),
            "transformation": "normalized whitespace, added word_count, filtered word_count >= 100",
        }

    stream_transform_and_persist()
