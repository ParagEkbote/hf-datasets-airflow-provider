"""Publish to a temporary private Hub repo and validate XCom, asset, and Hub head."""
import os

from airflow.sdk import DAG, task
from airflow.task.trigger_rule import TriggerRule

from airflow.providers.huggingface.hooks.huggingface import HuggingFaceHook
from airflow.providers.huggingface.operators.datasets import (
    LoadDatasetOperator,
    MapDatasetOperator,
    PushToHubOperator,
)

HUB_ORG = os.environ.get("HF_TEST_HUB_ORG", "configure-org")
REPO_ID = os.environ.get("HF_TEST_WRITE_REPO_ID", f"{HUB_ORG}/airflow-provider-system-write")
CONN_ID = os.environ.get("HF_TEST_HUB_CONN_ID", "huggingface_test")
REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"


def keep_columns(example):
    return {"sentence": example["sentence1"], "label": example["label"]}


class AssertAssetPushOperator(PushToHubOperator):
    """Assert that outlet metadata and the publication result agree."""

    def execute(self, context):
        result = super().execute(context)
        event = context["outlet_events"][self.published_asset]
        assert event.extra["commit_sha"] == result["commit_sha"]
        assert event.extra["commit_url"] == result["commit_url"]
        return result


with DAG(dag_id="huggingface_system_write", tags=["huggingface", "system-test"]) as dag:

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
        output_uri="file:///tmp/hf-provider-system/write-source.json",
        huggingface_conn_id=None,
    )
    mapped = MapDatasetOperator(
        task_id="map_public_dataset",
        input_uri=load.output,
        output_uri="file:///tmp/hf-provider-system/write-mapped.json",
        transform=keep_columns,
    )
    publish = AssertAssetPushOperator(
        task_id="publish_private_dataset",
        input_uri=mapped.output,
        repo_id=REPO_ID,
        private=True,
        huggingface_conn_id=CONN_ID,
    )

    @task
    def verify_xcom_matches_hub_head(publication):
        assert publication["commit_sha"]
        head = HuggingFaceHook(CONN_ID).get_dataset_info(REPO_ID).sha
        assert publication["commit_sha"] == head
        assert publication["commit_url"].endswith(head)

    @task(trigger_rule=TriggerRule.ALL_DONE)
    def delete_private_repo(created_repo):
        if created_repo:
            HuggingFaceHook(CONN_ID).get_conn().delete_repo(created_repo, repo_type="dataset")

    created = create_private_repo()
    verified = verify_xcom_matches_hub_head(publish.output)
    cleanup = delete_private_repo(created)
    created >> load >> mapped >> publish >> verified
    [created, publish, verified] >> cleanup

    from tests_common.test_utils.watcher import watcher

    list(dag.tasks) >> watcher()

from tests_common.test_utils.system_tests import get_test_run

test_run = get_test_run(dag)
