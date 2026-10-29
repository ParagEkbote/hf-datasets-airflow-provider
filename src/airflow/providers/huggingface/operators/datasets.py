"""Operators for loading, transforming, and publishing Hugging Face datasets."""
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


def _path(uri: str):
    return ObjectStoragePath(uri)


def _sha256_path(uri: str) -> str:
    digest = hashlib.sha256()
    with _path(uri).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _splits(dataset: Any) -> dict[str, Any]:
    return dict(dataset.items()) if hasattr(dataset, "items") else {"dataset": dataset}


def _validate_feature(feature: Any, column: str) -> None:
    """Reject feature encodings whose semantics are not guaranteed by Parquet."""
    if isinstance(feature, dict):
        for key, value in feature.items():
            _validate_feature(value, f"{column}.{key}")
        return
    if isinstance(feature, (list, tuple)):
        for value in feature:
            _validate_feature(value, column)
        return
    kind = type(feature).__name__
    if kind not in {"Value", "ClassLabel", "Sequence", "LargeList", "Features"}:
        raise ValueError(f"Feature {column!r} uses unsupported Parquet feature type {kind!r}")
    if kind in {"Sequence", "LargeList"}:
        child = getattr(feature, "feature", None)
        if child is not None:
            _validate_feature(child, column)


def _save(dataset: Any, uri: str, provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    """Persist each split as Parquet and a JSON manifest at ``uri``."""
    manifest = {"format": "parquet", "version": 1, "splits": {}, "provenance": provenance or {}}
    base = _path(uri)
    for index, (name, part) in enumerate(_splits(dataset).items()):
        for column, feature in part.features.items():
            _validate_feature(feature, column)
        data_uri = f"{uri}.split-{index}.parquet"
        _path(data_uri).parent.mkdir(parents=True, exist_ok=True)
        with _path(data_uri).open("wb") as stream:
            part.to_parquet(stream)
        manifest["splits"][str(name)] = {
            "uri": data_uri,
            "sha256": _sha256_path(data_uri),
            "rows": len(part),
            "features": part.features.to_dict(),
        }
    encoded = json.dumps(manifest, separators=(",", ":")).encode()
    base.parent.mkdir(parents=True, exist_ok=True)
    with base.open("wb") as stream:
        stream.write(encoded)
    return {"uri": uri, "format": "parquet", "fingerprint": hashlib.sha256(encoded).hexdigest(),
            "splits": {name: {"rows": part["rows"], "features": str(part["features"])}
                       for name, part in manifest["splits"].items()}}


def _reference(reference: str | dict[str, Any]) -> str:
    if isinstance(reference, dict):
        if "uri" not in reference:
            raise ValueError("Artifact reference must contain a 'uri' field.")
        return reference["uri"]
    return reference


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

    def __init__(self, *, path: str, output_uri: str, name: str | None = None,
                 split: str | None = None, revision: str | None = None, cache_dir: str | None = None,
                 download_mode: str = "reuse_dataset_if_exists", huggingface_conn_id: str | None = "huggingface_default",
                 load_kwargs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.path, self.output_uri, self.name, self.split = path, output_uri, name, split
        self.revision, self.cache_dir, self.download_mode = revision, cache_dir, download_mode
        self.huggingface_conn_id, self.load_kwargs = huggingface_conn_id, load_kwargs or {}

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        kwargs = dict(self.load_kwargs, download_mode=self.download_mode)
        for key, value in (("name", self.name), ("split", self.split), ("revision", self.revision), ("cache_dir", self.cache_dir)):
            if value is not None:
                kwargs[key] = value
        hook = HuggingFaceHook(self.huggingface_conn_id)
        revision = self.revision
        if "/" in self.path and not self.path.startswith(("/", "./", "../")):
            revision = hook.get_dataset_info(self.path, self.revision).sha
            kwargs["revision"] = revision
        dataset = hook.load_dataset(self.path, **kwargs)
        provenance = {"repository_id": self.path, "revision": revision, "configuration": self.name, "split": self.split}
        result = _save(dataset, self.output_uri, provenance)
        return {**result, **provenance}


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
    """Publish a Parquet artifact and emit its canonical dataset asset."""
    template_fields = ("input_uri", "repo_id", "config", "commit_message", "card_template", "card_metadata", "card_description")

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
                 card_update_policy: str = "preserve", **kwargs: Any) -> None:
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
        self.input_uri, self.repo_id, self.config = input_uri, repo_id, config
        self.private, self.commit_message = private, commit_message
        self.huggingface_conn_id, self.push_kwargs = huggingface_conn_id, push_kwargs or {}
        self.dataset_card, self.card_template = dataset_card, card_template
        self.card_metadata, self.card_description = card_metadata or {}, card_description
        self.card_update_policy = card_update_policy
        self.published_asset = Asset(uri=f"hf://datasets/{repo_id}")
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
        prior = hook.find_publication_commit(self.repo_id, fingerprint)
        if prior is not None:
            result = prior
        else:
            kwargs = dict(self.push_kwargs)
            if self.config is not None: kwargs["config_name"] = self.config
            if self.private is not None: kwargs["private"] = self.private
            marker = f"airflow-artifact-sha256:{fingerprint}"
            message = self.commit_message or "Publish dataset from Airflow"
            kwargs["commit_message"] = f"{message} [{marker}]"
            result = hook.push_to_hub(dataset, self.repo_id, **kwargs)
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
