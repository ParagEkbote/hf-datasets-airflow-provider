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
"""Operators to load, transform, and publish Hugging Face datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pyarrow.parquet as parquet
from datasets import ClassLabel, Dataset, DatasetDict, DatasetInfo, Features, Sequence, Value

from airflow.sdk.bases.operator import BaseOperator
try:
    from airflow.sdk import Asset, ObjectStoragePath
except ImportError:  # pragma: no cover - compatibility for early Airflow 3 releases
    from airflow.sdk import Asset
    from airflow.providers.common.io.path import ObjectStoragePath

from airflow.providers.huggingface.hooks.huggingface import HuggingFaceHook


def _path(uri: str) -> ObjectStoragePath:
    """Resolve a URI through Airflow's configured object-storage filesystems."""
    return ObjectStoragePath(uri)


def _reference(reference: str | dict[str, Any]) -> str:
    if isinstance(reference, dict):
        uri = reference.get("uri")
        if not uri:
            raise ValueError("Artifact reference must contain a non-empty 'uri'.")
        return uri
    if not isinstance(reference, str) or not reference:
        raise ValueError("Artifact reference must be a URI or contain a 'uri'.")
    return reference


def _sha256_path(uri: str) -> str:
    digest = hashlib.sha256()
    with _path(uri).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_feature(feature: Any, field_name: str) -> None:
    """Allow Parquet-safe scalar, class-label, and nested sequence features."""
    if isinstance(feature, (Value, ClassLabel)):
        return
    if isinstance(feature, Sequence):
        _validate_feature(feature.feature, field_name)
        return
    if isinstance(feature, dict):
        for key, value in feature.items():
            _validate_feature(value, f"{field_name}.{key}")
        return
    if isinstance(feature, (list, tuple)):
        for value in feature:
            _validate_feature(value, field_name)
        return
    raise ValueError(
        f"unsupported Parquet feature type for {field_name!r}: {type(feature).__name__}; "
        "supported types are Value, ClassLabel, and nested Sequence features"
    )


def _dataset_parts(dataset: Dataset | DatasetDict) -> list[tuple[str, Dataset]]:
    if isinstance(dataset, DatasetDict):
        return list(dataset.items())
    if isinstance(dataset, Dataset):
        return [("dataset", dataset)]
    raise TypeError("Expected a datasets.Dataset or datasets.DatasetDict.")


def _save(
    dataset: Dataset | DatasetDict,
    uri: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write split Parquet objects and a JSON manifest through ObjectStoragePath."""
    parts = _dataset_parts(dataset)
    written: list[ObjectStoragePath] = []
    manifest_path = _path(uri)
    split_manifest: dict[str, Any] = {}
    try:
        for index, (name, part) in enumerate(parts):
            for field_name, feature in part.features.items():
                _validate_feature(feature, field_name)
            split_uri = f"{uri}.split-{index:05d}.parquet"
            split_path = _path(split_uri)
            split_path.parent.mkdir(parents=True, exist_ok=True)
            with split_path.open("wb") as stream:
                parquet.write_table(part.data.table, stream)
            written.append(split_path)
            split_manifest[name] = {
                "uri": split_uri,
                "rows": len(part),
                "features": part.features.to_dict(),
                "sha256": _sha256_path(split_uri),
            }

        manifest = {
            "format": "parquet",
            "version": 1,
            "splits": split_manifest,
            "metadata": metadata or {},
        }
        manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("wb") as stream:
            stream.write(manifest_bytes)
        return {
            "uri": uri,
            "format": "parquet",
            "splits": {
                name: {"rows": part_info["rows"], "features": part_info["features"]}
                for name, part_info in split_manifest.items()
            },
            **(metadata or {}),
            "fingerprint": hashlib.sha256(manifest_bytes).hexdigest(),
        }
    except Exception:
        # Remove partial split outputs. A failed write must not leave an artifact
        # reference that a downstream task could mistake for a complete dataset.
        for path in written:
            try:
                path.unlink(missing_ok=True)
            except (OSError, NotImplementedError):
                pass
        try:
            manifest_path.unlink(missing_ok=True)
        except (OSError, NotImplementedError):
            pass
        raise


def _load_artifact(reference: str | dict[str, Any]) -> Dataset | DatasetDict:
    uri = _reference(reference)
    manifest_path = _path(uri)
    with manifest_path.open("rb") as stream:
        manifest_bytes = stream.read()
    if isinstance(reference, dict) and reference.get("fingerprint"):
        actual = hashlib.sha256(manifest_bytes).hexdigest()
        if actual != reference["fingerprint"]:
            raise ValueError("Dataset artifact fingerprint does not match its reference")

    manifest = json.loads(manifest_bytes)
    if manifest.get("format") != "parquet" or manifest.get("version") != 1:
        raise ValueError(f"Unsupported dataset artifact manifest: {uri}")
    parts: dict[str, Dataset] = {}
    for name, item in manifest["splits"].items():
        split_uri = item["uri"]
        if _sha256_path(split_uri) != item.get("sha256"):
            raise ValueError(f"Dataset artifact split fingerprint mismatch: {split_uri}")
        with _path(split_uri).open("rb") as stream:
            table = parquet.read_table(stream)
        features = Features.from_dict(item["features"])
        parts[name] = Dataset(table, info=DatasetInfo(features=features))
    return parts["dataset"] if list(parts) == ["dataset"] else DatasetDict(parts)


def _task_key(context: dict[str, Any], prefix: str, identity: str) -> str:
    ti = context.get("ti") or context.get("task_instance")
    dag_id = getattr(ti, "dag_id", "unknown-dag")
    task_id = getattr(ti, "task_id", "unknown-task")
    return f"huggingface-{prefix}:{dag_id}:{task_id}:{identity}"


def _is_hub_repo(path: str) -> bool:
    """Match a Hub dataset repo ID without treating an existing local path as remote."""
    return path.count("/") == 1 and not path.startswith((".", "/")) and not Path(path).exists()


class LoadDatasetOperator(BaseOperator):
    """Load a dataset revision and persist it as a Parquet artifact."""

    template_fields = ("path", "name", "split", "revision", "cache_dir", "output_uri")
    _PINNED_REVISION_KEY = "pinned_revision"

    def __init__(
        self,
        *,
        path: str,
        output_uri: str,
        name: str | None = None,
        split: str | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
        download_mode: str = "reuse_dataset_if_exists",
        huggingface_conn_id: str | None = "huggingface_default",
        load_kwargs: dict[str, Any] | None = None,
        durable: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.path, self.output_uri, self.name, self.split = path, output_uri, name, split
        self.revision, self.cache_dir, self.download_mode = revision, cache_dir, download_mode
        self.huggingface_conn_id, self.load_kwargs = huggingface_conn_id, load_kwargs or {}
        self.durable = durable

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        hook = HuggingFaceHook(self.huggingface_conn_id)
        requested_revision = self.revision
        is_hub_repo = _is_hub_repo(self.path)
        state = context.get("task_state_store") if self.durable else None
        state_identity = json.dumps([self.path, requested_revision], separators=(",", ":"))
        state_key = _task_key(context, "load", state_identity)
        revision = state.get(state_key) if state is not None else None
        if revision is None:
            revision = requested_revision
            if is_hub_repo:
                revision = hook.get_dataset_info(self.path, requested_revision).sha
            if state is not None and revision is not None:
                state.set(state_key, revision)

        kwargs = dict(self.load_kwargs)
        kwargs["download_mode"] = self.download_mode
        if self.name is not None:
            kwargs["name"] = self.name
        if self.split is not None:
            kwargs["split"] = self.split
        if revision is not None:
            kwargs["revision"] = revision
        if self.cache_dir is not None:
            kwargs["cache_dir"] = self.cache_dir
        dataset = hook.load_dataset(self.path, **kwargs)
        metadata = {
            "repository_id": self.path if is_hub_repo else None,
            "revision": revision,
            "configuration": self.name,
            "split": self.split,
        }
        return _save(dataset, self.output_uri, metadata)


class StreamDatasetOperator(BaseOperator):
    """Read a bounded streaming sample, transform/filter rows, and persist it."""

    template_fields = ("path", "name", "split", "revision", "output_uri", "transform_kwargs")

    def __init__(
        self,
        *,
        path: str,
        output_uri: str,
        max_source_rows: int,
        max_output_rows: int,
        transform: Callable[..., dict[str, Any]],
        name: str | None = None,
        split: str | None = None,
        revision: str | None = None,
        predicate: Callable[..., bool] | None = None,
        transform_kwargs: dict[str, Any] | None = None,
        feature_types: dict[str, str] | None = None,
        huggingface_conn_id: str | None = "huggingface_default",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if max_source_rows <= 0 or max_output_rows <= 0:
            raise ValueError("Streaming row bounds must be positive")
        if not callable(transform):
            raise TypeError("transform must be a callable")
        if predicate is not None and not callable(predicate):
            raise TypeError("predicate must be a callable or None")
        self.path, self.output_uri = path, output_uri
        self.max_source_rows, self.max_output_rows = max_source_rows, max_output_rows
        self.transform, self.predicate = transform, predicate
        self.name, self.split, self.revision = name, split, revision
        self.transform_kwargs = transform_kwargs or {}
        self.feature_types = feature_types or {}
        self.huggingface_conn_id = huggingface_conn_id

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        hook = HuggingFaceHook(self.huggingface_conn_id)
        revision = self.revision
        is_hub_repo = _is_hub_repo(self.path)
        if is_hub_repo:
            revision = hook.get_dataset_info(self.path, revision).sha

        kwargs: dict[str, Any] = {"streaming": True}
        if self.name is not None:
            kwargs["name"] = self.name
        if self.split is not None:
            kwargs["split"] = self.split
        if revision is not None:
            kwargs["revision"] = revision
        stream = hook.load_dataset(self.path, **kwargs)

        rows_by_split: dict[str, list[dict[str, Any]]] = {}
        examined = 0
        streams = stream.items() if isinstance(stream, dict) else [("dataset", stream)]
        output_count = 0
        for split_name, split_stream in streams:
            split_rows: list[dict[str, Any]] = []
            iterator = iter(split_stream)
            while examined < self.max_source_rows and output_count < self.max_output_rows:
                try:
                    row = next(iterator)
                except StopIteration:
                    break
                examined += 1
                transformed = self.transform(row, **self.transform_kwargs)
                if not isinstance(transformed, dict):
                    raise TypeError("stream transform must return a dict")
                if self.predicate is None or self.predicate(transformed):
                    split_rows.append(transformed)
                    output_count += 1
            if split_rows:
                rows_by_split[str(split_name)] = split_rows
            if examined >= self.max_source_rows or output_count >= self.max_output_rows:
                break
        if not rows_by_split:
            raise ValueError("No streaming rows passed the configured bounds and filter")

        features = Features({key: Value(kind) for key, kind in self.feature_types.items()}) if self.feature_types else None
        parts = {
            name: Dataset.from_list(rows, features=features)
            for name, rows in rows_by_split.items()
        }
        dataset = parts["dataset"] if list(parts) == ["dataset"] else DatasetDict(parts)
        metadata = {
            "repository_id": self.path if is_hub_repo else None,
            "revision": revision,
            "configuration": self.name,
            "split": self.split,
            "streaming": True,
            "source_rows_examined": examined,
        }
        return _save(dataset, self.output_uri, metadata)


class MapDatasetOperator(BaseOperator):
    """Apply a Python callable and persist the transformed artifact."""

    template_fields = ("input_uri", "output_uri", "transform_kwargs")

    def __init__(
        self,
        *,
        input_uri: str | dict[str, Any],
        output_uri: str,
        transform: Callable[..., Any],
        transform_kwargs: dict[str, Any] | None = None,
        map_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not callable(transform):
            raise TypeError("transform must be a callable")
        self.input_uri, self.output_uri, self.transform = input_uri, output_uri, transform
        self.transform_kwargs, self.map_kwargs = transform_kwargs or {}, map_kwargs or {}

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        dataset = _load_artifact(self.input_uri)
        mapped = dataset.map(self.transform, fn_kwargs=self.transform_kwargs, **self.map_kwargs)
        source = self.input_uri if isinstance(self.input_uri, dict) else {}
        provenance = {
            key: source[key]
            for key in ("repository_id", "revision", "configuration", "split")
            if key in source
        }
        return _save(mapped, self.output_uri, provenance)


class PushToHubOperator(BaseOperator):
    """Synchronously publish an artifact and emit its successful asset event."""

    template_fields = ("input_uri", "config", "commit_message")

    def __init__(
        self,
        *,
        input_uri: str | dict[str, Any],
        repo_id: str,
        config: str | None = None,
        private: bool | None = None,
        commit_message: str | None = None,
        huggingface_conn_id: str | None = "huggingface_default",
        push_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if "{{" in repo_id:
            raise ValueError("repo_id cannot be templated: it defines the Asset that downstream DAGs schedule on.")
        self.input_uri, self.repo_id, self.config = input_uri, repo_id, config
        self.private, self.commit_message = private, commit_message
        self.huggingface_conn_id, self.push_kwargs = huggingface_conn_id, push_kwargs or {}
        self.published_asset = Asset(uri=f"hf://datasets/{repo_id}")
        user_outlets = kwargs.pop("outlets", None)
        if user_outlets is None:
            user_outlets = []
        elif not isinstance(user_outlets, (list, tuple)):
            user_outlets = [user_outlets]
        super().__init__(outlets=[*user_outlets, self.published_asset], **kwargs)

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        dataset = _load_artifact(self.input_uri)
        source = self.input_uri if isinstance(self.input_uri, dict) else {}
        fingerprint = source.get("fingerprint")
        if not fingerprint:
            raise ValueError("Publication requires an artifact reference with a fingerprint")
        marker = f"airflow-artifact-sha256:{fingerprint}"
        state = context.get("task_state_store")
        state_key = _task_key(context, "publication", f"{self.repo_id}:{fingerprint}")
        if state is not None:
            prior_result = state.get(state_key)
            if prior_result:
                self._emit_asset_event(context, prior_result)
                return prior_result

        hook = HuggingFaceHook(self.huggingface_conn_id)
        commit = hook.find_publication_commit(self.repo_id, fingerprint)
        if commit is None:
            kwargs = dict(self.push_kwargs)
            if self.config is not None:
                kwargs["config_name"] = self.config
            if self.private is not None:
                kwargs["private"] = self.private
            message = self.commit_message or "Publish dataset from Airflow"
            kwargs["commit_message"] = f"{message} [{marker}]"
            commit = hook.push_to_hub(dataset, self.repo_id, **kwargs)

        result = {
            "repository_id": self.repo_id,
            "commit_sha": commit.oid,
            "commit_url": commit.commit_url,
            "url": f"https://huggingface.co/datasets/{self.repo_id}",
            "fingerprint": fingerprint,
            "source_repository_id": source.get("repository_id"),
            "source_revision": source.get("revision"),
        }
        if state is not None:
            state.set(state_key, result)
        self._emit_asset_event(context, result)
        return result

    def _emit_asset_event(self, context: dict[str, Any], publication: dict[str, Any]) -> None:
        events = context.get("outlet_events")
        if events is not None:
            events[self.published_asset].extra = {
                "repository_id": self.repo_id,
                "commit_sha": publication["commit_sha"],
                "commit_url": publication["commit_url"],
                "fingerprint": publication["fingerprint"],
            }
