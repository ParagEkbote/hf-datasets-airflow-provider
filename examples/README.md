# Provider-native example DAGs

These examples use the initial public API for dataset loading, mapping, and Hub publication. Each example defines its map callable in the DAG module so it is available when the DAG is parsed and on workers.

## Prerequisites and configuration

Install Airflow 3 and this provider into the Airflow environment:

```bash
pip install 'apache-airflow>=3,<4' .
```

Create an Airflow connection named `huggingface_default` (connection type
`huggingface`) and put the Hub token in its password field. The token is used
only by provider hooks during task execution. Publication targets use `my-org/...` defaults. Set `HF_MRPC_REPO_ID`,
`HF_FINEWEB_REPO_ID`, `HF_SQUAD_REPO_ID`, `HF_PRODUCER_REPO_ID`, and
`HF_CONSUMER_REPO_ID` to repositories where the Airflow token has write access. Source datasets are pinned
to commit SHAs in the DAG files; review those revisions when updating examples.

The examples write local artifacts under `/tmp/hf-datasets`. This works for a
single-worker tutorial. Multi-worker deployments need a shared writable
filesystem at the same path on each worker; change the `file://` URIs to that
deployment's shared location. For object storage, install the matching Airflow provider and filesystem dependency, configure the connection, and use an ``s3://`` or ``gs://`` artifact URI. Operators exchange compact references through XCom, never dataset contents.

## Run

Set the examples folder as the DAG folder, start Airflow, then
list and trigger a DAG:

```bash
export PYTHONPATH="$PWD"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/examples"
airflow dags list
airflow dags trigger glue_mrpc_normalize
```

Other DAG IDs are `huggingface_fineweb_sample`,
`huggingface_squad_multi_split_pipeline`,
`huggingface_daily_refresh_producer`, and
`huggingface_daily_refresh_consumer`. The Asset consumer starts automatically
after a successful producer publication; trigger the producer to demonstrate
the event-driven flow.

## Expected outputs

* **GLUE MRPC:** `file:///tmp/hf-datasets/mrpc_normalized.json`, with `text` and `label`
  columns, followed by publication to the configured Hub repository.
* **FineWeb-Edu:** a selected 1,000-row sample is cleaned and published.
* **SQuAD:** train and validation are mapped together and published as a DatasetDict.
* **Daily refresh:** successful publication emits `hf://datasets/<HF_PRODUCER_REPO_ID>`.
  The consumer loads that Hub repository, maps it, and publishes its own result.

Each `output_uri` identifies a JSON manifest. Its neighboring `.split-N.parquet` objects contain data, and the returned artifact reference includes the manifest URI, split names, feature schemas, and SHA-256 fingerprints. Downstream tasks pass the upstream operator output directly, so the
URI and available provenance remain in XCom. Dataset objects stay within the
task worker. All dataset network access and transformations happen at task
runtime, not while Airflow parses the DAG.
