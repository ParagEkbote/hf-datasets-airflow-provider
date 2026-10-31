# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Publish to a temporary private Hub repo and validate XCom, asset, and Hub head."""
import os
import uuid

from airflow.sdk import Asset, DAG, get_current_context, task
from airflow.providers.common.compat.sdk import TriggerRule

from airflow.providers.huggingface.hooks.huggingface import HuggingFaceHook
from airflow.providers.huggingface.operators.datasets import (
    LoadDatasetOperator,
    MapDatasetOperator,
    PushToHubOperator,
)

HUB_ORG = os.environ.get("HF_TEST_HUB_ORG", "configure-org")
SYSTEM_TESTS_ENV_ID = os.environ.get("SYSTEM_TESTS_ENV_ID", uuid.uuid4().hex)
REPO_BASE = os.environ.get("HF_TEST_WRITE_REPO_ID", f"{HUB_ORG}/airflow-provider-system-write")
REPO_ID = f"{REPO_BASE}-{SYSTEM_TESTS_ENV_ID}"
CONN_ID = os.environ.get("HF_TEST_HUB_CONN_ID", "huggingface_default")
REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
PUBLISHED_ASSET = Asset(uri=f"hf://datasets/{REPO_ID}")


def keep_columns(example):
    return {"sentence": example["sentence1"], "label": example["label"]}


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
    publish = PushToHubOperator(
        task_id="publish_private_dataset",
        input_uri=mapped.output,
        repo_id=REPO_ID,
        private=True,
        huggingface_conn_id=CONN_ID,
    )

    @task(inlets=[PUBLISHED_ASSET])
    def verify_xcom_matches_hub_head(publication):
        context = get_current_context()
        event = context["inlet_events"][PUBLISHED_ASSET][-1]
        assert publication["commit_sha"]
        assert event.extra["commit_sha"] == publication["commit_sha"]
        head = HuggingFaceHook(CONN_ID).get_dataset_info(REPO_ID).sha
        assert publication["commit_sha"] == head
        assert event.extra["commit_url"].endswith(head)

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
