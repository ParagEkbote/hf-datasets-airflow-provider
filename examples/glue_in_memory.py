"""Load and transform the small GLUE MRPC training split in memory."""

from datetime import datetime

from airflow.sdk import DAG

from airflow.providers.huggingface.operators.datasets import LoadDatasetOperator, MapDatasetOperator


with DAG(
    dag_id="huggingface_glue_mrpc_in_memory",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    doc_md=__doc__,
) as dag:
    load_mrpc = LoadDatasetOperator(
        task_id="load_mrpc",
        path="nyu-mll/glue",
        name="mrpc",
        split="train",
        output_uri="file:///mnt/shared/huggingface/glue_mrpc_train",
    )

    normalize_pairs = MapDatasetOperator(
        task_id="normalize_sentence_pairs",
        input_uri=load_mrpc.output,
        output_uri="file:///mnt/shared/huggingface/glue_mrpc_normalized",
        transform="examples.transforms:normalize_glue_pair",
        map_kwargs={"remove_columns": ["sentence1", "sentence2", "idx"]},
    )

    load_mrpc >> normalize_pairs
