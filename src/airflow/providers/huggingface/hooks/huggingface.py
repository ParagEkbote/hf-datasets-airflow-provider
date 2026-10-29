"""Connection and client helpers for Hugging Face services."""
from __future__ import annotations

from typing import Any

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

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
        marker = f"airflow-artifact-sha256:{fingerprint}"
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

    def get_dataset_readme(self, repo_id: str) -> str | None:
        try:
            path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="README.md", token=self.get_token())
        except EntryNotFoundError:
            return None
        from pathlib import Path
        return Path(path).read_text(encoding="utf-8")

    def upload_dataset_readme(self, repo_id: str, content: str, commit_message: str) -> str | None:
        result = self.get_conn().upload_file(
            path_or_fileobj=content.encode("utf-8"), path_in_repo="README.md", repo_id=repo_id,
            repo_type="dataset", commit_message=commit_message,
        )
        return getattr(result, "oid", None)
