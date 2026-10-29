"""Verify an Airflow 3.3+ retry reuses a successful Hub data commit."""
import os

import airflow
import pytest
from packaging.version import Version
from airflow.sdk import DAG, task
from airflow.task.trigger_rule import TriggerRule

from airflow.providers.huggingface.hooks.huggingface import HuggingFaceHook
from airflow.providers.huggingface.operators.datasets import (
    LoadDatasetOperator,
    MapDatasetOperator,
    PushToHubOperator,
)

pytestmark = pytest.mark.skipif(
    Version(airflow.__version__) < Version("3.3"),
    reason="Task state store retry acceptance requires Airflow 3.3+",
)
HUB_ORG = os.environ.get("HF_TEST_HUB_ORG", "configure-org")
REPO_ID = os.environ.get("HF_TEST_DURABLE_REPO_ID", f"{HUB_ORG}/airflow-provider-system-durable")
CONN_ID = os.environ.get("HF_TEST_HUB_CONN_ID", "huggingface_test")
REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"


def add_row_marker(example):
    return {"row_marker": "durable-retry-test"}


class FailAfterSuccessfulFirstPushOperator(PushToHubOperator):
    """Raise after the first successful commit; retries must reuse that commit."""

    def execute(self, context):
        result = super().execute(context)
        if context["ti"].try_number == 1:
            raise RuntimeError("intentional post-publication failure for retry test")
        return result


with DAG(dag_id="huggingface_system_durable", tags=["huggingface", "system-test"]) as dag:

    @task
    def create_private_repo():
        if HUB_ORG == "configure-org":
            raise RuntimeError("Set HF_TEST_HUB_ORG to a dedicated Hub test organization")
        HuggingFaceHook(CONN_ID).get_conn().create_repo(
            REPO_ID, repo_type="dataset", private=True, exist_ok=False
        )
        return REPO_ID

    load = LoadDatasetOperator(
        task_id="load_public_dataset",
        path="nyu-mll/glue",
        name="mrpc",
        split="train[:8]",
        revision=REVISION,
        output_uri="file:///tmp/hf-provider-system/durable-source.json",
        huggingface_conn_id=None,
    )
    mapped = MapDatasetOperator(
        task_id="map_public_dataset",
        input_uri=load.output,
        output_uri="file:///tmp/hf-provider-system/durable-mapped.json",
        transform=add_row_marker,
    )
    publish = FailAfterSuccessfulFirstPushOperator(
        task_id="publish_then_fail_once",
        input_uri=mapped.output,
        repo_id=REPO_ID,
        private=True,
        retries=1,
        retry_delay=0,
        huggingface_conn_id=CONN_ID,
    )

    @task
    def verify_single_data_commit(mapped_ref, publication):
        fingerprint = mapped_ref["fingerprint"]
        marker = f"airflow-artifact-sha256:{fingerprint}"
        commits = HuggingFaceHook(CONN_ID).get_conn().list_repo_commits(REPO_ID, repo_type="dataset")
        matches = [commit for commit in commits if marker in (commit.message or "")]
        assert len(matches) == 1
        assert publication["commit_sha"] == matches[0].oid

    @task(trigger_rule=TriggerRule.ALL_DONE)
    def delete_private_repo(created_repo):
        if created_repo:
            HuggingFaceHook(CONN_ID).get_conn().delete_repo(created_repo, repo_type="dataset")

    created = create_private_repo()
    verified = verify_single_data_commit(mapped.output, publish.output)
    cleanup = delete_private_repo(created)
    created >> load >> mapped >> publish >> verified
    [created, publish, verified] >> cleanup

    from tests_common.test_utils.watcher import watcher

    list(dag.tasks) >> watcher()

from tests_common.test_utils.system_tests import get_test_run

test_run = get_test_run(dag)
