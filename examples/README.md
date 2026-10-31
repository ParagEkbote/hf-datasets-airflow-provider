# Provider-native example DAGs

These examples use the public API for dataset loading, mapping, streaming directly to the Hub, and artifact publication. Each example defines its callable in the DAG module so it is available when the DAG is parsed and on workers.

## Prerequisites and configuration

Install Airflow 3 and this provider into the Airflow environment:

```bash
pip install 'apache-airflow>=3,<4' .
```

Create an Airflow connection named `huggingface_default` (connection type
`huggingface`) and put the Hub token in its password field. The token is used
only by provider hooks during task execution. Publication targets use `my-org/...` defaults. Set `HF_MRPC_REPO_ID`,
`HF_STREAM_REPO_ID`, `HF_SQUAD_REPO_ID`, `HF_PRODUCER_REPO_ID`, and
`HF_CONSUMER_REPO_ID` to repositories where the Airflow token has write access.
Where an example sets ``revision``, it pins the source to a commit SHA; review
those revisions when updating examples.

Examples that create durable artifacts write under `/tmp/hf-datasets`. This
works for a single-worker tutorial. Multi-worker deployments need a shared
writable filesystem at the same path on each worker; change the `file://` URIs
to that deployment's shared location. For object storage, install the matching
Airflow provider and filesystem dependency, configure the connection, and use
an ``s3://`` or ``gs://`` artifact URI. Operators exchange compact references
through XCom, never dataset contents. The streaming publisher sends the
``IterableDataset`` directly to the Hub and returns only commit metadata.

## Run

Set the examples folder as the DAG folder, start Airflow, then
list and trigger a DAG:

```bash
export PYTHONPATH="$PWD"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/examples"
airflow dags list
airflow dags trigger glue_mrpc_normalize
```

Other DAG IDs are `huggingface_rotten_tomatoes_stream`,
`huggingface_squad_multi_split_pipeline`,
`huggingface_daily_refresh_producer`, and
`huggingface_daily_refresh_consumer`. The Asset consumer starts automatically
after a successful producer publication; trigger the producer to demonstrate
the event-driven flow.

## Expected outputs

* **GLUE MRPC:** `file:///tmp/hf-datasets/mrpc_normalized.json`, with `text` and `label`
  columns, followed by publication to the configured Hub repository.
* **Rotten Tomatoes streaming:** the train split is transformed and filtered lazily, then published progressively to the configured Hub repository without a local artifact.
* **SQuAD:** train and validation are mapped together and published as a DatasetDict.
* **Daily refresh:** successful publication emits `hf://datasets/<HF_PRODUCER_REPO_ID>`.
  The consumer loads that Hub repository, maps it, and publishes its own result.

Each durable `output_uri` identifies a JSON manifest. Its neighboring
`.split-N.parquet` objects contain data, and the returned artifact reference
includes the manifest URI, split names, feature schemas, and SHA-256
fingerprints. Downstream tasks pass the upstream operator output directly, so
the URI and available provenance remain in XCom. Dataset objects stay within
the task worker. All dataset network access and transformations happen at task
runtime, not while Airflow parses the DAG.
