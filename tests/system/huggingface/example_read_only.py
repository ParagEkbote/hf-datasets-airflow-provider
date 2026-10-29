"""Read a pinned public dataset, map it, and validate the durable XCom handoff."""
from airflow.sdk import DAG, task
from airflow.sdk.execution_time.context import get_current_context

from airflow.providers.huggingface.operators.datasets import LoadDatasetOperator, MapDatasetOperator
from airflow.providers.huggingface.operators.datasets import _load_artifact

SOURCE_REPO = "nyu-mll/glue"
SOURCE_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"


def add_pair_text(example):
    return {"pair_text": f"{example['sentence1']} [SEP] {example['sentence2']}"}


with DAG(dag_id="huggingface_system_read_only", tags=["huggingface", "system-test"]) as dag:
    load = LoadDatasetOperator(
        task_id="load_public_dataset",
        path=SOURCE_REPO,
        name="mrpc",
        split="train[:8]",
        revision=SOURCE_REVISION,
        output_uri="file:///tmp/hf-provider-system/read-only-source.json",
        huggingface_conn_id=None,
    )
    mapped = MapDatasetOperator(
        task_id="map_public_dataset",
        input_uri=load.output,
        output_uri="file:///tmp/hf-provider-system/read-only-mapped.json",
        transform=add_pair_text,
    )

    @task
    def verify_xcom_and_artifact(mapped_ref):
        context = get_current_context()
        load_ref = context["ti"].xcom_pull(task_ids="load_public_dataset")
        assert load_ref["revision"] == SOURCE_REVISION
        result = _load_artifact(mapped_ref)
        assert result.num_rows == 8
        assert "pair_text" in result.column_names
        return {"revision": load_ref["revision"], "rows": result.num_rows}

    verify_xcom_and_artifact(mapped.output)

    from tests_common.test_utils.watcher import watcher

    list(dag.tasks) >> watcher()

from tests_common.test_utils.system_tests import get_test_run

test_run = get_test_run(dag)
