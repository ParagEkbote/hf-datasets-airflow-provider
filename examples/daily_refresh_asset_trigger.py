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
"""Publish a daily refreshed dataset and trigger a separate consumer by Asset."""

from datetime import datetime

from airflow.sdk import Asset, DAG
from airflow.providers.huggingface.operators.datasets import (
    FilterDatasetOperator, LoadDatasetOperator, MapDatasetOperator, PushToHubOperator,
)

SOURCE_REPO = "tatsu-lab/alpaca"
PRODUCER_REPO = "AINovice2005/alpaca-long-output"
CONSUMER_REPO = "AINovice2005/alpaca-long-output-consumed"
PUBLISHED = Asset(f"huggingface://datasets/{PRODUCER_REPO}")

with DAG(
    dag_id="huggingface_daily_refresh_producer", start_date=datetime(2024, 1, 1),
    schedule="@daily", catchup=False, tags=["huggingface", "assets", "example"],
    doc_md="Load and enrich the pinned Alpaca source, filter long outputs, then publish. Publication emits the Asset only on success.",
) as producer:
    load = LoadDatasetOperator(
        task_id="load_source", path=SOURCE_REPO, split="train",
        revision="dce01c9b08f87459cf36a430d809084718273017", output_uri="file:///tmp/hf-datasets/alpaca/source",
    )
    enrich = MapDatasetOperator(
        task_id="add_character_count", input_uri=load.output,
        output_uri="file:///tmp/hf-datasets/alpaca/enriched",
        transform="examples.transforms:add_char_count", map_kwargs={"batched": True},
    )
    long_outputs = FilterDatasetOperator(
        task_id="filter_long_outputs", input_uri=enrich.output,
        output_uri="file:///tmp/hf-datasets/alpaca/long_outputs",
        predicate="examples.transforms:keep_long_outputs",
    )
    publish = PushToHubOperator(
        task_id="publish", input_uri=long_outputs.output, repo_id=PRODUCER_REPO,
        private=True, commit_message="Airflow daily refresh: long Alpaca outputs",
    )
    load >> enrich >> long_outputs >> publish

with DAG(
    dag_id="huggingface_daily_refresh_consumer", start_date=datetime(2024, 1, 1),
    schedule=[PUBLISHED], catchup=False, tags=["huggingface", "assets", "example"],
    doc_md="Load the producer's published Hub dataset after its Asset event and add consumer metadata.",
) as consumer:
    consume = LoadDatasetOperator(
        task_id="load_published_dataset", path=PRODUCER_REPO, split="train",
        output_uri="file:///tmp/hf-datasets/alpaca/consumer_input",
    )
    remap = MapDatasetOperator(
        task_id="refresh_character_count", input_uri=consume.output,
        output_uri="file:///tmp/hf-datasets/alpaca/consumer_output",
        transform="examples.transforms:add_char_count", map_kwargs={"batched": True},
    )
    republish = PushToHubOperator(
        task_id="publish_consumer_output", input_uri=remap.output, repo_id=CONSUMER_REPO,
        private=True, commit_message="Airflow Asset consumer refresh",
    )
    consume >> remap >> republish
