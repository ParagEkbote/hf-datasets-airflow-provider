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

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable

from airflow.sdk.bases.operator import BaseOperator
try:
    from airflow.sdk import Asset
except ImportError:
    from airflow.datasets import Dataset as Asset
try:
    from airflow.sdk import ObjectStoragePath
except ImportError:
    try:
        from airflow.io.path import ObjectStoragePath
    except ImportError:
        from airflow.providers.common.io.path import ObjectStoragePath

from airflow.providers.huggingface.hooks.huggingface import HuggingFaceHook


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
    from datasets import Dataset, DatasetDict, DatasetInfo, Features
    import pyarrow.parquet as parquet
    uri = _reference(reference)
    with _path(uri).open("rb") as stream:
        manifest_bytes = stream.read()
    manifest = json.loads(manifest_bytes)
    if isinstance(reference, dict) and reference.get("fingerprint"):
        actual = hashlib.sha256(manifest_bytes).hexdigest()
        if actual != reference["fingerprint"]:
            raise ValueError("Dataset artifact fingerprint does not match its reference")
    if manifest.get("format") != "parquet" or manifest.get("version") != 1:
        raise ValueError(f"Unsupported dataset artifact manifest: {uri}")
    parts = {}
    for name, item in manifest["splits"].items():
        if _sha256_path(item["uri"]) != item.get("sha256"):
            raise ValueError(f"Dataset artifact split fingerprint mismatch: {item['uri']}")
        with _path(item["uri"]).open("rb") as stream:
            table = parquet.read_table(stream)
        features = Features.from_dict(item["features"])
        parts[name] = Dataset(table, info=DatasetInfo(features=features))
    return parts["dataset"] if list(parts) == ["dataset"] else DatasetDict(parts)


class LoadDatasetOperator(BaseOperator):
    """Load a dataset and persist it as a durable Parquet artifact."""
    template_fields = ("path", "name", "split", "revision", "cache_dir", "output_uri")

    _PINNED_REVISION_KEY = "pinned_revision"

    def __init__(self, *, path: str, output_uri: str, name: str | None = None,
                 split: str | None = None, revision: str | None = None, cache_dir: str | None = None,
                 download_mode: str = "reuse_dataset_if_exists", huggingface_conn_id: str | None = "huggingface_default",
                 load_kwargs: dict[str, Any] | None = None, durable: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.path, self.output_uri, self.name, self.split = path, output_uri, name, split
        self.revision, self.cache_dir, self.download_mode = revision, cache_dir, download_mode
        self.huggingface_conn_id, self.load_kwargs = huggingface_conn_id, load_kwargs or {}
        self.durable = durable

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        kwargs = dict(self.load_kwargs)
        kwargs.update(download_mode=self.download_mode)
        if self.name is not None: kwargs["name"] = self.name
        if self.split is not None: kwargs["split"] = self.split
        if self.revision is not None: kwargs["revision"] = self.revision
        if self.cache_dir is not None: kwargs["cache_dir"] = self.cache_dir
        dataset = HuggingFaceHook(self.huggingface_conn_id).load_dataset(self.path, **kwargs)
        saved = _save(dataset, self.output_uri)
        result = {"uri": self.output_uri, "format": "arrow", "repository_id": self.path,
                  "revision": self.revision, "configuration": self.name, "split": self.split, **saved}
        if "/" in self.path and not self.path.startswith(("/", "./", "../")):
            try:
                result["revision"] = HuggingFaceHook(self.huggingface_conn_id).get_dataset_info(self.path, self.revision).sha
            except Exception:
                self.log.warning("Could not resolve source revision metadata; using the requested revision value.")
        return result


class MapDatasetOperator(BaseOperator):
    """Apply a Python callable and persist the result as a Parquet artifact."""
    template_fields = ("input_uri", "output_uri", "transform_kwargs")

    def __init__(self, *, input_uri: str | dict[str, Any], output_uri: str,
                 transform: Callable[..., Any], transform_kwargs: dict[str, Any] | None = None,
                 map_kwargs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not callable(transform):
            raise TypeError("transform must be a callable")
        self.input_uri, self.output_uri, self.transform = input_uri, output_uri, transform
        self.transform_kwargs, self.map_kwargs = transform_kwargs or {}, map_kwargs or {}

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        dataset = _load_artifact(self.input_uri)
        mapped = dataset.map(self.transform, fn_kwargs=self.transform_kwargs, **self.map_kwargs)
        source = self.input_uri if isinstance(self.input_uri, dict) else {}
        provenance = {k: source[k] for k in ("repository_id", "revision", "configuration", "split") if k in source}
        return {**_save(mapped, self.output_uri, provenance), "source_uri": _reference(self.input_uri), **provenance}


class PushToHubOperator(BaseOperator):
    """Synchronously publish an artifact and emit a successful Airflow Asset event."""

    template_fields = ("input_uri", "repo_id", "config", "commit_message", "card_template", "card_metadata", "card_description")

    _PUSHED_COMMIT_KEY = "pushed_commit"
    _CARD_START = "<!-- BEGIN AIRFLOW HUGGINGFACE DATASET CARD -->"
    _CARD_END = "<!-- END AIRFLOW HUGGINGFACE DATASET CARD -->"
    _CARD_FIELDS = {"pretty_name": str, "license": str, "language": list, "task_categories": list,
                    "tags": list, "size_categories": list}

    def __init__(self, *, input_uri: str | dict[str, Any], repo_id: str, config: str | None = None,
                 private: bool | None = None, commit_message: str | None = None,
                 huggingface_conn_id: str | None = "huggingface_default",
                 push_kwargs: dict[str, Any] | None = None,
                 dataset_card: bool = False, card_template: str | None = None,
                 card_metadata: dict[str, Any] | None = None, card_description: str | None = None,
                 card_update_policy: str = "preserve",
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
        self.card_update_policy = card_update_policy
        self.published_asset = Asset(uri=f"huggingface://datasets/{repo_id}")
        super().__init__(outlets=[self.published_asset], **kwargs)

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        dataset = _load_artifact(self.input_uri)
        source_ref = self.input_uri if isinstance(self.input_uri, dict) else {}
        if not source_ref.get("fingerprint"):
            raise ValueError("Publication requires an artifact reference with a fingerprint")
        fingerprint = source_ref["fingerprint"]
        state = context.get("task_state_store")
        state_key = f"huggingface-publication:{self.repo_id}:{fingerprint}" if fingerprint else None
        if state is not None and state_key:
            prior = state.get(state_key)
            if prior:
                self._emit_asset_event(context, prior)
                return prior
        hook = HuggingFaceHook(self.huggingface_conn_id)
        card_result = {"enabled": False, "action": "disabled"}
        existing_readme = hook.get_dataset_readme(self.repo_id) if self.dataset_card else None
        card_text = None
        if self.dataset_card and existing_readme is not None and self.card_update_policy == "preserve":
            card_result = {"enabled": True, "action": "preserved", "path": "README.md"}
        elif self.dataset_card:
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
        result = hook.push_to_hub(dataset, self.repo_id, **kwargs)
        commit_url = getattr(result, "url", None)
        commit_sha = getattr(result, "oid", None)
        if isinstance(result, str):
            commit_url = result
        if commit_url and not commit_sha:
            commit_sha = commit_url.rstrip("/").rsplit("/", 1)[-1]
        if card_text is not None:
            action = ("replaced" if self.card_update_policy == "replace" else "updated") if existing_readme is not None else "created"
            card_commit = hook.upload_dataset_readme(self.repo_id, card_text, commit_message="Update dataset card")
            card_result = {"enabled": True, "action": action, "path": "README.md"}
            if card_commit:
                card_result["commit_sha"] = card_commit
        output = {"repository_id": self.repo_id, "commit_sha": result.oid,
                  "commit_url": result.commit_url, "url": f"https://huggingface.co/datasets/{self.repo_id}",
                  "fingerprint": fingerprint, "source_repository_id": source_ref.get("repository_id"),
                  "source_revision": source_ref.get("revision"), "dataset_card": card_result}
        if state is not None and state_key:
            state.set(state_key, output)
        self._emit_asset_event(context, output)
        return output

    def _emit_asset_event(self, context: dict[str, Any], publication: dict[str, Any]) -> None:
        events = context.get("outlet_events")
        if events is not None:
            events[self.published_asset].extra = {
                "repository_id": self.repo_id,
                "commit_sha": publication["commit_sha"],
                "commit_url": publication["commit_url"],
                "fingerprint": publication["fingerprint"],
            }

    def _render_card(self, dataset: Any) -> str:
        metadata = dict(self.card_metadata)
        structure = {}
        parts = dataset.items() if hasattr(dataset, "items") else [("dataset", dataset)]
        for name, part in parts:
            structure[str(name)] = {"rows": len(part), "features": {key: str(value) for key, value in part.features.items()}}
        source = self.input_uri if isinstance(self.input_uri, dict) else {}
        provenance = {key: source[key] for key in ("repository_id", "revision", "configuration", "split") if source.get(key) is not None}
        if self.card_template:
            try:
                template_text = Path(self.card_template).read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError(f"Unable to read dataset card template {self.card_template!r}: {exc}") from exc
            from jinja2 import Environment, StrictUndefined
            return Environment(undefined=StrictUndefined, autoescape=False).from_string(template_text).render(
                repo_id=self.repo_id, metadata=metadata, description=self.card_description,
                structure=structure, provenance=provenance)
        yaml = "\n".join(f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in metadata.items())
        sections = ["---", yaml, "---", "", self._CARD_START, "# Dataset Description", "",
                    self.card_description or f"Dataset published to `{self.repo_id}`."]
        if provenance:
            sections += ["", "## Source and Provenance", ""]
            sections += [f"- **{key.replace('_', ' ').title()}:** {value}" for key, value in provenance.items()]
        sections += ["", "## Dataset Structure", "", "| Split | Rows | Features |", "|---|---:|---|"]
        sections += [f"| {name} | {info['rows']} | {', '.join(f'{k}: {v}' for k, v in info['features'].items())} |"
                     for name, info in structure.items()]
        if self.config:
            sections += ["", "## Usage", "", "```python", "from datasets import load_dataset",
                         f'dataset = load_dataset("{self.repo_id}", "{self.config}")', "```"]
        sections += [self._CARD_END, ""]
        return "\n".join(sections)
