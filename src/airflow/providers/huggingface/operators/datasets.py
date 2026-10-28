"""Task operators for loading, transforming, and publishing datasets."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from airflow.sdk import Asset
from airflow.sdk.bases.operator import BaseOperator

from airflow.providers.huggingface.hooks.datasets import HuggingFaceDatasetsHook


def _artifact_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError("MVP artifacts support file:// URIs only; workers must share the referenced filesystem.")
    return Path(unquote(parsed.path))


def _reference_uri(reference: str | dict[str, Any]) -> str:
    """Accept a URI or the compact artifact reference returned by an operator."""
    if isinstance(reference, dict):
        if "uri" not in reference:
            raise ValueError("Artifact reference must contain a 'uri' field.")
        return reference["uri"]
    return reference


def _save(dataset: Any, uri: str) -> dict[str, Any]:
    path = _artifact_path(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(path))
    return {
        "uri": uri,
        "format": "arrow",
        "splits": {str(name): {"rows": len(part), "features": str(part.features)} for name, part in dataset.items()}
        if hasattr(dataset, "items") else {"dataset": {"rows": len(dataset), "features": str(dataset.features)}},
    }


def _load_artifact(reference: str | dict[str, Any]) -> Any:
    uri = _reference_uri(reference)
    path = _artifact_path(uri)
    try:
        from datasets import load_from_disk, load_dataset
    except ImportError as exc:
        raise ImportError("Install the provider dependencies to load dataset artifacts.") from exc
    if path.is_dir():
        return load_from_disk(str(path))
    if path.suffix.lower() == ".parquet":
        return load_dataset("parquet", data_files=str(path))
    raise ValueError(f"Artifact must be a Datasets save_to_disk directory or Parquet file: {uri}")


class LoadDatasetOperator(BaseOperator):
    """Load a Hub or local dataset and persist it as an Arrow artifact."""

    template_fields = ("path", "name", "split", "revision", "cache_dir", "output_uri")

    def __init__(self, *, path: str, output_uri: str, name: str | None = None, split: str | None = None,
                 revision: str | None = None, cache_dir: str | None = None,
                 download_mode: str = "reuse_dataset_if_exists", huggingface_conn_id: str | None = "huggingface_default",
                 load_kwargs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.path, self.output_uri, self.name, self.split = path, output_uri, name, split
        self.revision, self.cache_dir, self.download_mode = revision, cache_dir, download_mode
        self.huggingface_conn_id, self.load_kwargs = huggingface_conn_id, load_kwargs or {}

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        kwargs = dict(self.load_kwargs)
        kwargs.update(download_mode=self.download_mode)
        if self.name is not None: kwargs["name"] = self.name
        if self.split is not None: kwargs["split"] = self.split
        if self.revision is not None: kwargs["revision"] = self.revision
        if self.cache_dir is not None: kwargs["cache_dir"] = self.cache_dir
        dataset = HuggingFaceDatasetsHook(self.huggingface_conn_id).load_dataset(self.path, **kwargs)
        saved = _save(dataset, self.output_uri)
        result = {"uri": self.output_uri, "format": "arrow", "repository_id": self.path,
                  "revision": self.revision, "configuration": self.name, "split": self.split, **saved}
        if "/" in self.path and not self.path.startswith(("/", "./", "../")):
            try:
                result["revision"] = HuggingFaceDatasetsHook(self.huggingface_conn_id).get_dataset_info(self.path, self.revision).sha
            except Exception:
                self.log.warning("Could not resolve source revision metadata; using the requested revision value.")
        return result


class MapDatasetOperator(BaseOperator):
    """Apply an importable callable inside the worker and persist the result."""

    template_fields = ("input_uri", "output_uri", "transform", "transform_kwargs")

    def __init__(self, *, input_uri: str | dict[str, Any], output_uri: str, transform: str,
                 transform_kwargs: dict[str, Any] | None = None, map_kwargs: dict[str, Any] | None = None,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.input_uri, self.output_uri, self.transform = input_uri, output_uri, transform
        self.transform_kwargs, self.map_kwargs = transform_kwargs or {}, map_kwargs or {}

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        if ":" not in self.transform:
            raise ValueError("transform must use the import path 'package.module:function'")
        module_name, function_name = self.transform.split(":", 1)
        function = getattr(importlib.import_module(module_name), function_name)
        dataset = _load_artifact(self.input_uri)
        mapped = dataset.map(function, fn_kwargs=self.transform_kwargs, **self.map_kwargs)
        saved = _save(mapped, self.output_uri)
        source_ref = self.input_uri if isinstance(self.input_uri, dict) else {}
        provenance = {key: source_ref[key] for key in ("repository_id", "revision", "configuration", "split") if key in source_ref}
        return {"uri": self.output_uri, "format": "arrow", "source_uri": _reference_uri(self.input_uri), **provenance, **saved}


class PushToHubOperator(BaseOperator):
    """Synchronously publish an artifact and emit a successful Airflow Asset event."""

    template_fields = ("input_uri", "repo_id", "config", "commit_message")

    def __init__(self, *, input_uri: str | dict[str, Any], repo_id: str, config: str | None = None,
                 private: bool | None = None, commit_message: str | None = None,
                 huggingface_conn_id: str | None = "huggingface_default", push_kwargs: dict[str, Any] | None = None,
                 **kwargs: Any) -> None:
        self.input_uri, self.repo_id, self.config = input_uri, repo_id, config
        self.private, self.commit_message = private, commit_message
        self.huggingface_conn_id, self.push_kwargs = huggingface_conn_id, push_kwargs or {}
        self.published_asset = Asset(uri=f"huggingface://datasets/{repo_id}")
        super().__init__(outlets=[self.published_asset], **kwargs)

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        dataset = _load_artifact(self.input_uri)
        kwargs = dict(self.push_kwargs)
        if self.config is not None: kwargs["config_name"] = self.config
        if self.private is not None: kwargs["private"] = self.private
        if self.commit_message is not None: kwargs["commit_message"] = self.commit_message
        result = HuggingFaceDatasetsHook(self.huggingface_conn_id).push_to_hub(dataset, self.repo_id, **kwargs)
        commit_url = getattr(result, "url", None)
        commit_sha = getattr(result, "oid", None)
        if isinstance(result, str):
            commit_url = result
        if commit_url and not commit_sha:
            commit_sha = commit_url.rstrip("/").rsplit("/", 1)[-1]
        output = {"repository_id": self.repo_id, "commit_sha": commit_sha,
                  "url": f"https://huggingface.co/datasets/{self.repo_id}"}
        outlet_events = context.get("outlet_events")
        if outlet_events is not None:
            outlet_events[self.published_asset].extra = {"repository_id": self.repo_id, "commit_sha": commit_sha}
        source_ref = self.input_uri if isinstance(self.input_uri, dict) else {}
        output["source_repository_id"] = source_ref.get("repository_id")
        output["source_revision"] = source_ref.get("revision")
        return output
