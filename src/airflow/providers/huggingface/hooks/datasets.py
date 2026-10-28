"""Connection and client helpers for Hugging Face Datasets."""

from __future__ import annotations

from typing import Any

try:
    from airflow.sdk.bases.hook import BaseHook
except ImportError:  # Airflow 3.0 exposes the public base class at the legacy path.
    from airflow.hooks.base import BaseHook


class HuggingFaceDatasetsHook(BaseHook):
    """Resolve an optional Airflow connection and access Datasets and Hub APIs."""

    conn_name_attr = "huggingface_conn_id"
    default_conn_name = "huggingface_default"
    conn_type = "huggingface"
    hook_name = "Hugging Face"

    def __init__(self, huggingface_conn_id: str | None = default_conn_name) -> None:
        super().__init__()
        self.huggingface_conn_id = huggingface_conn_id

    def get_token(self) -> str | None:
        """Get a Hub token from the connection password or ``extra.token``."""
        if not self.huggingface_conn_id:
            return None
        try:
            connection = self.get_connection(self.huggingface_conn_id)
        except Exception as exc:
            # Public access works without a connection. Avoid swallowing backend/config errors.
            from airflow.exceptions import AirflowNotFoundException

            if isinstance(exc, AirflowNotFoundException):
                return None
            raise
        extra = connection.extra_dejson
        token = connection.password or extra.get("token")
        if token:
            self.log.debug("Using Hugging Face credentials from connection %s", self.huggingface_conn_id)
        return token

    @staticmethod
    def _safe_error(exc: Exception, token: str | None) -> str:
        message = str(exc)
        return message.replace(token, "[redacted]") if token else message

    def load_dataset(self, path: str, **kwargs: Any) -> Any:
        """Call ``datasets.load_dataset`` with the configured token when present."""
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("Install the provider dependencies with `pip install apache-airflow-providers-huggingface`.") from exc
        token = self.get_token()
        if token and "token" not in kwargs:
            kwargs["token"] = token
        try:
            return load_dataset(path, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"Failed to load Hugging Face dataset {path!r}: {self._safe_error(exc, token)}") from exc

    def get_dataset_info(self, repo_id: str, revision: str | None = None) -> Any:
        """Resolve Hub repository metadata, including the concrete commit SHA."""
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ImportError("Install `huggingface-hub` to resolve dataset repository metadata.") from exc
        token = self.get_token()
        try:
            return HfApi(token=token).dataset_info(repo_id, revision=revision)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to resolve Hugging Face dataset metadata for {repo_id!r}: {self._safe_error(exc, token)}"
            ) from exc

    def push_to_hub(self, dataset: Any, repo_id: str, **kwargs: Any) -> Any:
        """Publish a Dataset or DatasetDict to the Hub."""
        token = self.get_token()
        if token and "token" not in kwargs:
            kwargs["token"] = token
        try:
            return dataset.push_to_hub(repo_id, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"Failed to publish dataset to {repo_id!r}: {self._safe_error(exc, token)}") from exc
