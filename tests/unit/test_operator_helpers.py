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
from pathlib import Path

import pytest

from airflow.providers.huggingface.operators.datasets import _artifact_path


def test_artifact_path_decodes_file_uri():
    assert _artifact_path("file:///tmp/a%20b") == Path("/tmp/a b")


def test_artifact_path_rejects_unimplemented_schemes():
    with pytest.raises(ValueError, match="file://"):
        _artifact_path("s3://bucket/path")
