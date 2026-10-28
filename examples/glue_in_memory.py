"""Load, normalize, and validate the GLUE MRPC training split."""


from datetime import datetime

from airflow.sdk import DAG, task
from datasets import load_dataset


OUTPUT_REPO = "AINovice2005/glue-mrpc-normalized"


with DAG(
    dag_id="huggingface_glue_mrpc_direct_push",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
) as dag:

    @task
    def transform_and_push():
        dataset = load_dataset(
            "nyu-mll/glue",
            "mrpc",
            split="train",
        )

        def normalize_pair(example):
            return {
                "sentence1": " ".join(example["sentence1"].split()),
                "sentence2": " ".join(example["sentence2"].split()),
            }

        normalized = dataset.map(
            normalize_pair,
            remove_columns=["sentence1", "sentence2", "idx"],
        )

        normalized.push_to_hub(
            OUTPUT_REPO,
            split="train",
            private=True,
            commit_message="Publish normalized GLUE MRPC training split",
        )

        return {
            "repo_id": OUTPUT_REPO,
            "split": "train",
            "rows": len(normalized),
        }

    transform_and_push()
