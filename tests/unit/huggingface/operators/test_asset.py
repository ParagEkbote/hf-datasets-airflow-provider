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
import pytest
from urllib.parse import urlsplit, urlunsplit

from airflow.providers.huggingface.asset import create_asset, sanitize_uri


def test_sanitize_dataset_asset_uri_string_and_airflow_split_result():
    uri = "hf://datasets/org/example/"
    assert sanitize_uri(uri) == "hf://datasets/org/example"
    normalized = sanitize_uri(urlsplit(uri))
    assert urlunsplit(normalized) == "hf://datasets/org/example"
    assert create_asset(uri).uri == "hf://datasets/org/example"


@pytest.mark.parametrize("uri", ["hf://org/model", "hf://spaces/org/app"])
def test_non_dataset_hub_asset_uris_pass_through(uri):
    assert sanitize_uri(uri) == uri
    normalized = sanitize_uri(urlsplit(uri))
    assert urlunsplit(normalized) == uri
    assert create_asset(uri).uri == uri


def test_rejects_malformed_dataset_asset_uri():
    with pytest.raises(ValueError):
        sanitize_uri("hf://datasets/")
