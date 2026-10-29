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
"""Task operators for loading, transforming, and publishing datasets."""

from __future__ import annotations

import importlib
from pathlib import Path
import json
import re
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


def _is_hub_repo_id(path: str) -> bool:
    """Tell a Hub ``org/name`` repository from a local path, as ``load_dataset`` does."""
    return path.count("/") == 1 and not path.startswith((".", "/")) and not Path(path).exists()


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
        # Datasets' content fingerprint; a re-run that produces different data gets a different one.
        "fingerprint": {str(name): part._fingerprint for name, part in dataset.items()}
        if hasattr(dataset, "items") else dataset._fingerprint,
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

    _PINNED_REVISION_KEY = "pinned_revision"

    def __init__(self, *, path: str, output_uri: str, name: str | None = None, split: str | None = None,
                 revision: str | None = None, cache_dir: str | None = None,
                 download_mode: str = "reuse_dataset_if_exists", huggingface_conn_id: str | None = "huggingface_default",
                 load_kwargs: dict[str, Any] | None = None, durable: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.path, self.output_uri, self.name, self.split = path, output_uri, name, split
        self.revision, self.cache_dir, self.download_mode = revision, cache_dir, download_mode
        self.huggingface_conn_id, self.load_kwargs = huggingface_conn_id, load_kwargs or {}
        self.durable = durable

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        hook = HuggingFaceDatasetsHook(self.huggingface_conn_id)
        revision = self.revision
        if _is_hub_repo_id(self.path):
            revision = self._pin_revision(hook, context)
        kwargs = dict(self.load_kwargs)
        kwargs.update(download_mode=self.download_mode)
        if self.name is not None: kwargs["name"] = self.name
        if self.split is not None: kwargs["split"] = self.split
        if revision is not None: kwargs["revision"] = revision
        if self.cache_dir is not None: kwargs["cache_dir"] = self.cache_dir
        dataset = hook.load_dataset(self.path, **kwargs)
        saved = _save(dataset, self.output_uri)
        return {"uri": self.output_uri, "format": "arrow", "repository_id": self.path,
                "revision": revision, "configuration": self.name, "split": self.split, **saved}

    def _pin_revision(self, hook: HuggingFaceDatasetsHook, context: dict[str, Any]) -> str:
        """
        Pin a moving ref such as ``main`` to its commit before loading.

        The recorded revision is then the one that was loaded. With ``durable`` on Airflow 3.3+,
        the commit is kept in the task state store, so every retry loads the commit the first
        try pinned. Clearing the task from the UI or REST API discards task state, so the task
        pins afresh; clearing the whole Dag run keeps it (apache/airflow#72929).
        """
        store = context.get("task_state_store") if self.durable else None
        pinned = store.get(self._PINNED_REVISION_KEY) if store is not None else None
        if pinned and pinned["path"] == self.path and pinned["requested"] == self.revision:
            self.log.info("Loading commit %s of %s, pinned by an earlier try.", pinned["sha"], self.path)
            return pinned["sha"]
        sha = hook.get_dataset_info(self.path, self.revision).sha
        if store is not None:
            store.set(self._PINNED_REVISION_KEY, {"path": self.path, "requested": self.revision, "sha": sha})
        return sha


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


class FilterDatasetOperator(BaseOperator):
    """Filter a persisted dataset using Datasets' supported filter semantics."""

    template_fields = ("input_uri", "output_uri", "predicate")

    def __init__(self, *, input_uri: str | dict[str, Any], output_uri: str,
                 predicate: str, filter_kwargs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.input_uri, self.output_uri, self.predicate = input_uri, output_uri, predicate
        self.filter_kwargs = filter_kwargs or {}

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        if ":" not in self.predicate:
            raise ValueError("predicate must use the import path 'package.module:function'")
        module_name, function_name = self.predicate.split(":", 1)
        function = getattr(importlib.import_module(module_name), function_name)
        dataset = _load_artifact(self.input_uri)
        filtered = dataset.filter(function, **self.filter_kwargs)
        saved = _save(filtered, self.output_uri)
        return {"uri": self.output_uri, "format": "arrow", "source_uri": _reference_uri(self.input_uri), **saved}


class StreamDatasetOperator(BaseOperator):
    """Read a bounded streaming sample, transform/filter it, and persist Arrow."""

    template_fields = ("path", "name", "split", "revision", "output_uri", "transform", "predicate")

    def __init__(self, *, path: str, output_uri: str, max_source_rows: int, max_output_rows: int,
                 transform: str, name: str | None = None, split: str | None = None,
                 revision: str | None = None, predicate: str | None = None,
                 transform_kwargs: dict[str, Any] | None = None,
                 feature_types: dict[str, str] | None = None,
                 huggingface_conn_id: str | None = "huggingface_default", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if max_source_rows <= 0 or max_output_rows <= 0:
            raise ValueError("Streaming row bounds must be positive")
        self.path, self.output_uri = path, output_uri
        self.max_source_rows, self.max_output_rows = max_source_rows, max_output_rows
        self.transform, self.predicate = transform, predicate
        self.name, self.split, self.revision = name, split, revision
        self.transform_kwargs, self.feature_types = transform_kwargs or {}, feature_types or {}
        self.huggingface_conn_id = huggingface_conn_id

    @staticmethod
    def _callable(spec: str):
        if ":" not in spec:
            raise ValueError("callable must use the import path 'package.module:function'")
        module, name = spec.split(":", 1)
        return getattr(importlib.import_module(module), name)

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        from datasets import Dataset, Features, Value
        kwargs = {"streaming": True}
        if self.name is not None: kwargs["name"] = self.name
        if self.split is not None: kwargs["split"] = self.split
        if self.revision is not None: kwargs["revision"] = self.revision
        stream = HuggingFaceDatasetsHook(self.huggingface_conn_id).load_dataset(self.path, **kwargs)
        transform = self._callable(self.transform)
        predicate = self._callable(self.predicate) if self.predicate else None
        rows, examined = [], 0
        for row in stream:
            if examined >= self.max_source_rows or len(rows) >= self.max_output_rows:
                break
            examined += 1
            result = transform(row, **self.transform_kwargs)
            if predicate is None or predicate(result):
                rows.append(result)
        if not rows:
            raise ValueError("No streaming rows passed the configured bounds and filter")
        features = Features({key: Value(kind) for key, kind in self.feature_types.items()}) if self.feature_types else None
        dataset = Dataset.from_list(rows, features=features)
        saved = _save(dataset, self.output_uri)
        return {"uri": self.output_uri, "format": "arrow", "repository_id": self.path,
                "revision": self.revision, "configuration": self.name, "split": self.split,
                "streaming": True, "source_rows_examined": examined, **saved}


class AssembleDatasetOperator(BaseOperator):
    """Assemble named persisted split artifacts into a DatasetDict artifact."""

    template_fields = ("inputs", "output_uri")

    def __init__(self, *, inputs: dict[str, str | dict[str, Any]], output_uri: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not inputs:
            raise ValueError("inputs must contain at least one named split artifact")
        self.inputs, self.output_uri = inputs, output_uri

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        from datasets import DatasetDict
        parts = {name: _load_artifact(reference) for name, reference in self.inputs.items()}
        assembled = DatasetDict(parts)
        saved = _save(assembled, self.output_uri)
        return {"uri": self.output_uri, "format": "arrow", "splits": list(parts), **saved}


class PushToHubOperator(BaseOperator):
    """Synchronously publish an artifact and emit a successful Airflow Asset event."""

    # repo_id is not templated: it defines the published Asset, which downstream Dags schedule on.
    template_fields = ("input_uri", "config", "commit_message", "card_template", "card_metadata", "card_description")

    _PUSHED_COMMIT_KEY = "pushed_commit"
    _CARD_START = "<!-- BEGIN AIRFLOW HUGGINGFACE DATASET CARD -->"
    _CARD_END = "<!-- END AIRFLOW HUGGINGFACE DATASET CARD -->"
    _CARD_FIELDS = {"pretty_name": str, "license": str, "language": list, "task_categories": list,
                    "tags": list, "size_categories": list}

    def __init__(self, *, input_uri: str | dict[str, Any], repo_id: str, config: str | None = None,
                 private: bool | None = None, commit_message: str | None = None,
                 huggingface_conn_id: str | None = "huggingface_default", push_kwargs: dict[str, Any] | None = None,
                 dataset_card: bool = False, card_template: str | None = None,
                 card_metadata: dict[str, Any] | None = None, card_description: str | None = None,
                 card_update_policy: str = "preserve", durable: bool = True,
                 **kwargs: Any) -> None:
        if card_update_policy not in {"preserve", "update", "replace"}:
            raise ValueError("card_update_policy must be one of: 'preserve', 'update', or 'replace'.")
        if not isinstance(dataset_card, bool):
            raise TypeError("dataset_card must be a bool.")
        if card_metadata is not None and not isinstance(card_metadata, dict):
            raise TypeError("card_metadata must be a dict or None.")
        for key, value in (card_metadata or {}).items():
            expected = self._CARD_FIELDS.get(key)
            if expected is None or not isinstance(value, expected) or (expected is list and not all(isinstance(x, str) for x in value)):
                raise ValueError(f"Invalid dataset card metadata field or value: {key!r}.")
        if not dataset_card and card_update_policy != "preserve":
            raise ValueError("card_update_policy can be changed only when dataset_card=True.")
        if "{{" in repo_id:
            raise ValueError("repo_id cannot be templated: it defines the Asset that downstream Dags schedule on.")
        self.input_uri, self.repo_id, self.config = input_uri, repo_id, config
        self.private, self.commit_message = private, commit_message
        self.huggingface_conn_id, self.push_kwargs = huggingface_conn_id, push_kwargs or {}
        self.dataset_card, self.card_template = dataset_card, card_template
        self.card_metadata, self.card_description = card_metadata or {}, card_description
        self.card_update_policy, self.durable = card_update_policy, durable
        self.published_asset = Asset(uri=f"huggingface://datasets/{repo_id}")
        outlets = kwargs.pop("outlets", None) or []
        if not isinstance(outlets, (list, tuple)):
            outlets = [outlets]
        super().__init__(outlets=[*outlets, self.published_asset], **kwargs)

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        dataset = _load_artifact(self.input_uri)
        hook = HuggingFaceDatasetsHook(self.huggingface_conn_id)
        card_result = {"enabled": False, "action": "disabled"}
        existing_readme = None
        if self.dataset_card:
            existing_readme = hook.get_dataset_readme(self.repo_id)
            if existing_readme is not None and self.card_update_policy == "preserve":
                card_result = {"enabled": True, "action": "preserved", "path": "README.md"}
        card_text = None
        if self.dataset_card and card_result["action"] != "preserved":
            card_text = self._render_card(dataset)
            if existing_readme is not None and self.card_update_policy == "update":
                pattern = re.compile(re.escape(self._CARD_START) + r".*?" + re.escape(self._CARD_END), re.S)
                if not pattern.search(existing_readme):
                    raise ValueError("Cannot safely update existing README.md: provider-managed card markers are missing. Add the markers or use card_update_policy='replace'.")
                card_text = pattern.sub(lambda _: card_text.strip(), existing_readme)
            elif existing_readme is not None and self.card_update_policy == "replace":
                pass
            elif existing_readme is not None:
                raise ValueError("Existing README.md would be overwritten; select preserve, update, or explicit replace policy.")
        kwargs = dict(self.push_kwargs)
        if self.config is not None: kwargs["config_name"] = self.config
        if self.private is not None: kwargs["private"] = self.private
        if self.commit_message is not None: kwargs["commit_message"] = self.commit_message
        # With durable on Airflow 3.3+, a retry after a successful push (for example, when the card
        # upload or the worker failed afterwards) reuses that commit instead of uploading again. The
        # stored commit is tied to the input's fingerprint, so different data is always pushed.
        store = context.get("task_state_store") if self.durable else None
        fingerprint = self.input_uri.get("fingerprint") if isinstance(self.input_uri, dict) else None
        pushed = store.get(self._PUSHED_COMMIT_KEY) if store is not None and fingerprint is not None else None
        if pushed and pushed["repo_id"] == self.repo_id and pushed["fingerprint"] == fingerprint:
            commit_sha = pushed["commit_sha"]
            self.log.info("Commit %s was pushed to %s by an earlier try; not pushing again.", commit_sha, self.repo_id)
        else:
            result = hook.push_to_hub(dataset, self.repo_id, **kwargs)
            commit_url = getattr(result, "url", None)
            commit_sha = getattr(result, "oid", None)
            if isinstance(result, str):
                commit_url = result
            if commit_url and not commit_sha:
                commit_sha = commit_url.rstrip("/").rsplit("/", 1)[-1]
            if store is not None and fingerprint is not None and commit_sha:
                store.set(self._PUSHED_COMMIT_KEY, {"repo_id": self.repo_id, "fingerprint": fingerprint, "commit_sha": commit_sha})
        if card_text is not None:
            action = ("replaced" if self.card_update_policy == "replace" else "updated") if existing_readme is not None else "created"
            try:
                card_commit = hook.upload_dataset_readme(self.repo_id, card_text, commit_message="Update dataset card")
            except Exception as exc:
                raise RuntimeError(f"Dataset was published to {self.repo_id!r} (commit {commit_sha or 'unknown'}), but README.md publication failed: {exc}") from exc
            card_result = {"enabled": True, "action": action, "path": "README.md"}
            if card_commit:
                card_result["commit_sha"] = card_commit
        output = {"repository_id": self.repo_id, "commit_sha": commit_sha,
                  "url": f"https://huggingface.co/datasets/{self.repo_id}"}
        output["dataset_card"] = card_result
        outlet_events = context.get("outlet_events")
        if outlet_events is not None:
            outlet_events[self.published_asset].extra = {"repository_id": self.repo_id, "commit_sha": commit_sha}
        source_ref = self.input_uri if isinstance(self.input_uri, dict) else {}
        output["source_repository_id"] = source_ref.get("repository_id")
        output["source_revision"] = source_ref.get("revision")
        return output

    def _render_card(self, dataset: Any) -> str:
        metadata = dict(self.card_metadata)
        description = self.card_description
        structure = {}
        parts = dataset.items() if hasattr(dataset, "items") else [("dataset", dataset)]
        for name, part in parts:
            structure[str(name)] = {"rows": len(part), "features": {key: str(value) for key, value in part.features.items()}}
        source = self.input_uri if isinstance(self.input_uri, dict) else {}
        provenance = {key: source[key] for key in ("repository_id", "revision", "configuration", "split") if source.get(key) is not None}
        if self.card_template:
            path = Path(self.card_template)
            try:
                template_text = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError(f"Unable to read dataset card template {str(path)!r}: {exc}") from exc
            try:
                from jinja2 import Environment, StrictUndefined
            except ImportError as exc:
                raise ImportError("Custom dataset card templates require Jinja2.") from exc
            body = Environment(undefined=StrictUndefined, autoescape=False).from_string(template_text).render(
                repo_id=self.repo_id, metadata=metadata, description=description, structure=structure, provenance=provenance)
            return body
        yaml = "\n".join(f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in metadata.items())
        sections = ["---", yaml, "---", "", f"{self._CARD_START}", "# Dataset Description", "",
                    description or f"Dataset published to `{self.repo_id}`."]
        if provenance:
            sections += ["", "## Source and Provenance", ""]
            sections += [f"- **{key.replace('_', ' ').title()}:** {value}" for key, value in provenance.items()]
        sections += ["", "## Dataset Structure", "", "| Split | Rows | Features |", "|---|---:|---|"]
        sections += [f"| {name} | {info['rows']} | {', '.join(f'{k}: {v}' for k, v in info['features'].items())} |" for name, info in structure.items()]
        if self.config:
            sections += ["", "## Usage", "", "```python", "from datasets import load_dataset", f'dataset = load_dataset("{self.repo_id}", "{self.config}")', "```"]
        sections += [self._CARD_END, ""]
        return "\n".join(sections)
