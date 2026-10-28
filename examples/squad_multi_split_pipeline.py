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
"""Map SQuAD splits in parallel, assemble them, and publish the DatasetDict."""

from datetime import datetime

from airflow.sdk import DAG, TaskGroup
from airflow.providers.huggingface.operators.datasets import (
    AssembleDatasetOperator, LoadDatasetOperator, MapDatasetOperator, PushToHubOperator,
)

with DAG(
    dag_id="huggingface_squad_multi_split_pipeline", start_date=datetime(2024, 1, 1),
    schedule=None, catchup=False, tags=["huggingface", "squad", "example"], doc_md=__doc__,
) as dag:
    with TaskGroup(group_id="process_splits"):
        load_train = LoadDatasetOperator(
            task_id="load_train", path="rajpurkar/squad", split="train",
            revision="7b6d24c440a36b6815f21b70d25016731768db1f", output_uri="file:///tmp/hf-datasets/squad/train",
        )
        map_train = MapDatasetOperator(
            task_id="map_train", input_uri=load_train.output,
            output_uri="file:///tmp/hf-datasets/squad/train_flattened",
            transform="examples.transforms:flatten_squad_context",
            map_kwargs={"batched": True, "batch_size": 512,
                        "remove_columns": ["id", "title", "context", "question", "answers"]},
        )
        load_validation = LoadDatasetOperator(
            task_id="load_validation", path="rajpurkar/squad", split="validation",
            revision="7b6d24c440a36b6815f21b70d25016731768db1f", output_uri="file:///tmp/hf-datasets/squad/validation",
        )
        map_validation = MapDatasetOperator(
            task_id="map_validation", input_uri=load_validation.output,
            output_uri="file:///tmp/hf-datasets/squad/validation_flattened",
            transform="examples.transforms:flatten_squad_context",
            map_kwargs={"batched": True, "batch_size": 512,
                        "remove_columns": ["id", "title", "context", "question", "answers"]},
        )
        load_train >> map_train
        load_validation >> map_validation

    assemble = AssembleDatasetOperator(
        task_id="assemble_splits", inputs={"train": map_train.output, "validation": map_validation.output},
        output_uri="file:///tmp/hf-datasets/squad/assembled",
    )
    publish = PushToHubOperator(
        task_id="publish", input_uri=assemble.output, repo_id="AINovice2005/squad-flattened",
        private=True, commit_message="Airflow: flatten SQuAD splits",
    )
    [map_train, map_validation] >> assemble >> publish
