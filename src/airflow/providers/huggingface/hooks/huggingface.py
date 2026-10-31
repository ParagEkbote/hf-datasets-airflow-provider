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
"""Connection and client helpers for Hugging Face services."""
from __future__ import annotations

from typing import Any

from datasets import load_dataset
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

try:
    from airflow.sdk.bases.hook import BaseHook
except ImportError:
    from airflow.hooks.base import BaseHook


class HuggingFaceHook(BaseHook):
    """Access Hugging Face Datasets and Hub APIs using an Airflow connection.

    If the connection is absent, huggingface_hub may use ``HF_TOKEN`` or a
    cached worker login.
    """

    conn_name_attr = "huggingface_conn_id"
    default_conn_name = "huggingface_default"
    conn_type = "huggingface"
    hook_name = "Hugging Face"

    def __init__(self, huggingface_conn_id: str | None = default_conn_name) -> None:
        super().__init__()
        self.huggingface_conn_id = huggingface_conn_id
        self._allow_ambient_fallback = huggingface_conn_id in (None, self.default_conn_name)
        self._client: HfApi | None = None

    def get_token(self) -> str | None:
        if not self.huggingface_conn_id:
            return None
        try:
            connection = self.get_connection(self.huggingface_conn_id)
        except Exception as exc:
            from airflow.exceptions import AirflowNotFoundException
            if isinstance(exc, AirflowNotFoundException) and self._allow_ambient_fallback:
                return None
            raise
        return connection.password or None

    def get_conn(self) -> HfApi:
        if self._client is None:
            self._client = HfApi(token=self.get_token())
        return self._client

    def test_connection(self) -> tuple[bool, str]:
        try:
            self.get_conn().whoami()
            return True, "Connection successfully tested"
        except Exception as exc:
            return False, str(exc)

    def load_dataset(self, path: str, **kwargs: Any) -> Any:
        token = self.get_token()
        if token and "token" not in kwargs:
            kwargs["token"] = token
        return load_dataset(path, **kwargs)

    def get_dataset_info(self, repo_id: str, revision: str | None = None) -> Any:
        return self.get_conn().dataset_info(repo_id, revision=revision)

    def find_publication_commit(self, repo_id: str, fingerprint: str) -> Any | None:
        """Find a prior commit for this artifact after a worker crash."""
        return self.find_commit_by_marker(repo_id, f"airflow-artifact-sha256:{fingerprint}")

    def find_commit_by_marker(self, repo_id: str, marker: str) -> Any | None:
        """Find a prior dataset commit containing a caller-provided marker."""
        try:
            commits = self.get_conn().list_repo_commits(repo_id, repo_type="dataset")
        except RepositoryNotFoundError:
            return None
        for commit in commits:
            # huggingface_hub's CommitInfo exposes ``commit_message`` while
            # some API response versions and test doubles expose ``message``.
            message = getattr(commit, "message", None) or getattr(commit, "commit_message", "")
            if marker in (message or ""):
                return commit
        return None

    def push_to_hub(self, dataset: Any, repo_id: str, **kwargs: Any) -> Any:
        token = self.get_token()
        if token and "token" not in kwargs:
            kwargs["token"] = token
        return dataset.push_to_hub(repo_id, **kwargs)
