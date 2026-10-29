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
publish path using durable Parquet artifacts for
a dataset small enough to fit in worker memory.

Prerequisites
~~~~~~~~~~~~~
* Install the provider::

    pip install hf-datasets-airflow-provider

* Create an Airflow connection ``huggingface_default`` with type
  ``huggingface`` and a write-access token in the password field.

* Ensure the example artifact directory is visible to all task workers. The
  local tutorial paths use ``/tmp/hf-datasets``; change them for a shared mount
  in a multi-worker deployment.


Airflow 3 Asset scheduling
~~~~~~~~~~~~~~~~~~~~~~~~~~
``published`` is registered as an ``Asset``.  Any downstream DAG that
declares ``schedule=[published]`` will be triggered automatically after a
successful run of the ``publish`` task.
"""

import os

from airflow.sdk import DAG, Asset
from airflow.providers.huggingface.operators.datasets import (
    LoadDatasetOperator,
    MapDatasetOperator,
    PushToHubOperator,
)

def normalize_glue_pair(batch):
    """Fuse MRPC sentence pairs while retaining their labels."""
    return {
        "text": [f"{' '.join(a.split()).lower()} [SEP] {' '.join(b.split()).lower()}"
                 for a, b in zip(batch["sentence1"], batch["sentence2"])],
        "label": batch["label"],
    }


# Stable Asset URI that downstream DAGs can schedule on.
MRPC_REPO_ID = os.environ.get("HF_MRPC_REPO_ID", "my-org/glue-mrpc-normalized")
published = Asset(f"hf://datasets/{MRPC_REPO_ID}")

with DAG(
    dag_id="glue_mrpc_normalize",
    tags=["huggingface", "glue", "example"],
) as dag:

    load = LoadDatasetOperator(
        task_id="load_mrpc",
        path="nyu-mll/glue",
        name="mrpc",          # dataset configuration
        split="train",
        revision="bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
        output_uri="file:///tmp/hf-datasets/mrpc_raw.json",
        # cache_dir keeps downloaded Arrow files across retries.
        cache_dir="/tmp/hf-datasets/cache",
    )

    transform = MapDatasetOperator(
        task_id="normalize_pairs",
        # Pass the whole XCom artifact reference so provenance is forwarded.
        input_uri=load.output,
        output_uri="file:///tmp/hf-datasets/mrpc_normalized.json",
        transform=normalize_glue_pair,
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
    )

    load >> transform >> publish


