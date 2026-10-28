from types import SimpleNamespace

import pytest

from airflow.providers.huggingface.hooks.datasets import HuggingFaceDatasetsHook


def test_token_is_resolved_from_connection_without_logging(caplog):
    hook = HuggingFaceDatasetsHook("hf_test")
    hook.get_connection = lambda _: SimpleNamespace(password="secret-token", extra_dejson={})

    assert hook.get_token() == "secret-token"
    assert "secret-token" not in caplog.text


def test_extra_token_is_supported():
    hook = HuggingFaceDatasetsHook("hf_test")
    hook.get_connection = lambda _: SimpleNamespace(password=None, extra_dejson={"token": "secret-token"})

    assert hook.get_token() == "secret-token"


def test_public_load_does_not_require_connection(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "datasets.load_dataset",
        lambda path, **kwargs: calls.append((path, kwargs)) or "loaded",
    )
    hook = HuggingFaceDatasetsHook(huggingface_conn_id=None)

    assert hook.load_dataset("public/example", split="train") == "loaded"
    assert calls == [("public/example", {"split": "train"})]


def test_hub_client_uses_connection_token(monkeypatch):
    from huggingface_hub import HfApi

    constructed = {}

    class FakeApi:
        def __init__(self, token):
            constructed["token"] = token

        def dataset_info(self, repo_id, revision):
            constructed.update(repo_id=repo_id, revision=revision)
            return "info"

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    hook = HuggingFaceDatasetsHook("hf_test")
    hook.get_token = lambda: "secret-token"

    assert hook.get_dataset_info("org/data", "main") == "info"
    assert constructed == {"token": "secret-token", "repo_id": "org/data", "revision": "main"}


def test_load_error_redacts_access_token(monkeypatch):
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad secret-token")))
    hook = HuggingFaceDatasetsHook("hf_test")
    hook.get_token = lambda: "secret-token"

    with pytest.raises(RuntimeError, match=r"bad \[redacted\]"):
        hook.load_dataset("private/data")


def test_missing_datasets_dependency_has_actionable_error(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "datasets", None)
    with pytest.raises(ImportError, match="Install the provider dependencies"):
        HuggingFaceDatasetsHook(None).load_dataset("public/data")
