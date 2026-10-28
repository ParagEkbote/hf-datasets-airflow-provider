"""
examples/glue_mrpc_normalize.py
--------------------------------
Load the GLUE MRPC training split, combine sentence pairs into a single
``text`` field, and publish the result to the Hub.

Use-case
~~~~~~~~
MRPC (Microsoft Research Paraphrase Corpus) ships two sentence columns and
a binary similarity label.  Fine-tuning encoders on it typically needs a
single fused text field.  This DAG shows the canonical load → map →
publish path using in-memory ``Dataset`` objects (no streaming) for a
dataset small enough to fit in worker memory.

Prerequisites
~~~~~~~~~~~~~
* Install the provider::

    pip install hf-datasets-airflow-provider

* Create an Airflow connection ``huggingface_default`` with type
  ``huggingface`` and a write-access token in the password field.

* Ensure the example artifact directory is visible to all task workers. The
  local tutorial paths use ``/tmp/hf-datasets``; change them for a shared mount
  in a multi-worker deployment.

* Create a module ``examples.transforms`` importable on every task worker
  (or adjust the ``transform`` argument below).  The normalize function is
  reproduced in a comment at the bottom of this file.

Airflow 3 Asset scheduling
~~~~~~~~~~~~~~~~~~~~~~~~~~
``published`` is registered as an ``Asset``.  Any downstream DAG that
declares ``schedule=[published]`` will be triggered automatically after a
successful run of the ``publish`` task.
"""

from datetime import datetime
import os

from airflow.sdk import DAG, Asset
from airflow.providers.huggingface.operators.datasets import (
    LoadDatasetOperator,
    MapDatasetOperator,
    PushToHubOperator,
)

# Stable Asset URI that downstream DAGs can schedule on.
MRPC_REPO_ID = os.environ.get("HF_MRPC_REPO_ID", "AINovice2005/glue-mrpc-normalized")
published = Asset(f"huggingface://datasets/{MRPC_REPO_ID}")

with DAG(
    dag_id="glue_mrpc_normalize",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["huggingface", "glue", "example"],
) as dag:

    load = LoadDatasetOperator(
        task_id="load_mrpc",
        path="nyu-mll/glue",
        name="mrpc",          # dataset configuration
        split="train",
        revision="bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
        output_uri="file:///tmp/hf-datasets/mrpc_raw",
        # cache_dir keeps downloaded Arrow files across retries.
        cache_dir="/tmp/hf-datasets/cache",
    )

    transform = MapDatasetOperator(
        task_id="normalize_pairs",
        # Pass the whole XCom artifact reference so provenance is forwarded.
        input_uri=load.output,
        output_uri="file:///tmp/hf-datasets/mrpc_normalized",
        # Dotted import path: module:callable
        transform="examples.transforms:normalize_glue_pair",
        map_kwargs={
            # Combine both sentences in a single pass and drop the raw columns.
            "batched": True,
            "batch_size": 512,
            "remove_columns": ["sentence1", "sentence2", "idx"],
        },
    )

    publish = PushToHubOperator(
        task_id="publish",
        input_uri=transform.output,
        repo_id=MRPC_REPO_ID,
        private=True,
        commit_message="Airflow: normalized MRPC training split",
        # Generate a dataset card, preserving any README already present.
        dataset_card=True,
        card_description=(
            "MRPC sentence pairs fused into a single ``text`` field while "
            "retaining the binary similarity label.  Produced by the "
            "``glue_mrpc_normalize`` Airflow DAG."
        ),
        card_metadata={
            "pretty_name": "Normalized MRPC training pairs",
            "license": "mit",
            "task_categories": ["text-classification"],
            "tags": ["glue", "mrpc", "paraphrase"],
        },
        card_update_policy="preserve",
    )

    load >> transform >> publish


