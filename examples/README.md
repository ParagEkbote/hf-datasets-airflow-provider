# Provider-native example DAGs

These examples use the public provider operators for dataset loading, mapping,
filtering, bounded streaming, split assembly, and Hub publication. Transform
functions live in `examples/transforms.py`; install or distribute this module
to every worker that executes the DAGs.

## Prerequisites and configuration

Install Airflow 3 and this provider into the Airflow environment:

```bash
pip install 'apache-airflow>=3,<4' .
```

Create an Airflow connection named `huggingface_default` (connection type
`huggingface`) and put the Hub token in its password field. The token is used
only by provider hooks during task execution. For publication examples, change
the `AINovice2005/...` repository IDs to repositories you can write to. Set
`HF_MRPC_REPO_ID` to override the MRPC destination. Source datasets are pinned
to commit SHAs in the DAG files; review those revisions when updating examples.

The examples write local artifacts under `/tmp/hf-datasets`. This works for a
single-worker tutorial. Multi-worker deployments need a shared writable
filesystem at the same path on each worker; change the `file://` URIs to that
deployment's shared location. This provider release does not configure object
storage. Operators exchange only compact JSON artifact
references through XCom, never dataset contents.

## Run

Put this repository (including `examples/transforms.py`) on the Airflow
`PYTHONPATH`, set the examples folder as the DAG folder, start Airflow, then
list and trigger a DAG:

```bash
export PYTHONPATH="$PWD"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/examples"
airflow dags list
airflow dags trigger glue_mrpc_normalize
```

Other DAG IDs are `huggingface_fineweb_edu_streaming_sample`,
`huggingface_squad_multi_split_pipeline`,
`huggingface_daily_refresh_producer`, and
`huggingface_daily_refresh_consumer`. The Asset consumer starts automatically
after a successful producer publication; trigger the producer to demonstrate
the event-driven flow.

## Expected outputs

* **GLUE MRPC:** `file:///tmp/hf-datasets/mrpc_normalized`, with `text` and `label`
  columns, followed by publication to the configured Hub repository.
* **FineWeb-Edu:** no more than 10,000 streamed source rows are examined and
  at most 1,000 documents with at least 100 words are saved at
  `file:///tmp/hf-datasets/fineweb_edu_clean_sample`, then published.
* **SQuAD:** train and validation are loaded and mapped in parallel, assembled
  into one Arrow `DatasetDict`, then published.
* **Daily refresh:** the producer publishes long Alpaca output records; its
  successful publication emits `huggingface://datasets/AINovice2005/alpaca-long-output`.
  The consumer loads that Hub repository, maps it, and publishes its own result.

Artifact references contain `uri`, `format`, and compact split/row/feature
metadata. Downstream tasks pass the upstream operator output directly, so the
URI and available provenance remain in XCom. Dataset objects stay within the
task worker. All dataset network access and transformations happen at task
runtime, not while Airflow parses the DAG.
