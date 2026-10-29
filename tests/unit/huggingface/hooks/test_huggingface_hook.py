"""Connection-backed authentication tests for HuggingFaceHook."""
from unittest.mock import patch

import pytest
from airflow.exceptions import AirflowNotFoundException
from airflow.hooks.base import BaseHook

from airflow.providers.huggingface.hooks.huggingface import HuggingFaceHook


@pytest.fixture
def connection_env(monkeypatch):
    # Airflow resolves this as a real Connection, without a metadata DB fixture.
    monkeypatch.setenv(
        "AIRFLOW_CONN_HUGGINGFACE_TEST",
        '{"conn_type":"huggingface","password":"test-secret"}',
    )


def test_connection_resolution_uses_password_only(connection_env):
    hook = HuggingFaceHook("huggingface_test")
    connection = BaseHook.get_connection("huggingface_test")
    assert connection.conn_type == "huggingface"
    assert hook.get_token() == "test-secret"


def test_get_conn_caches_autospecced_hf_api(connection_env):
    with patch("airflow.providers.huggingface.hooks.huggingface.HfApi", autospec=True) as api:
        hook = HuggingFaceHook("huggingface_test")
        first = hook.get_conn()
        assert hook.get_conn() is first
        api.assert_called_once_with(token="test-secret")


def test_connection_test_success_and_failure(connection_env):
    with patch("airflow.providers.huggingface.hooks.huggingface.HfApi", autospec=True) as api:
        hook = HuggingFaceHook("huggingface_test")
        assert hook.test_connection() == (True, "Connection successfully tested")
        api.return_value.whoami.assert_called_once_with()
        api.return_value.whoami.side_effect = ValueError("invalid token")
        assert hook.test_connection() == (False, "invalid token")


def test_no_connection_defers_to_huggingface_hub_fallback(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "ambient-token")
    with patch("airflow.providers.huggingface.hooks.huggingface.HfApi", autospec=True) as api:
        hook = HuggingFaceHook(None)
        hook.get_conn()
        api.assert_called_once_with(token=None)


def test_default_missing_connection_allows_sdk_fallback(monkeypatch):
    monkeypatch.delenv("AIRFLOW_CONN_HUGGINGFACE_DEFAULT", raising=False)
    with patch.object(BaseHook, "get_connection", side_effect=AirflowNotFoundException("missing")):
        assert HuggingFaceHook().get_token() is None


def test_explicit_invalid_connection_fails(monkeypatch):
    with patch.object(BaseHook, "get_connection", side_effect=AirflowNotFoundException("missing")):
        with pytest.raises(AirflowNotFoundException):
            HuggingFaceHook("explicit_bad_id").get_token()


def test_load_dataset_passes_token_and_propagates_library_exception(connection_env):
    from airflow.providers.huggingface.hooks import huggingface

    error = ValueError("invalid or gated dataset")
    with patch.object(huggingface, "load_dataset", autospec=True, side_effect=error) as load:
        hook = HuggingFaceHook("huggingface_test")
        with pytest.raises(ValueError) as raised:
            hook.load_dataset("org/private-data")
    assert raised.value is error
    load.assert_called_once_with("org/private-data", token="test-secret")


def test_hub_repository_exception_propagates(connection_env):
    from huggingface_hub.utils import RepositoryNotFoundError

    hook = HuggingFaceHook("huggingface_test")
    import httpx
    error = RepositoryNotFoundError("missing", response=httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co")))
    with patch("airflow.providers.huggingface.hooks.huggingface.HfApi", autospec=True) as api:
        api.return_value.dataset_info.side_effect = error
        with pytest.raises(RepositoryNotFoundError) as raised:
            hook.get_dataset_info("org/missing")
    assert raised.value is error


def test_gated_repository_error_propagates(connection_env):
    from huggingface_hub.utils import GatedRepoError
    from airflow.providers.huggingface.hooks import huggingface

    import httpx
    error = GatedRepoError("gated", response=httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co")))
    with patch.object(huggingface, "load_dataset", autospec=True, side_effect=error):
        with pytest.raises(GatedRepoError) as raised:
            HuggingFaceHook("huggingface_test").load_dataset("org/gated")
    assert raised.value is error


def test_connection_extra_token_is_ignored(monkeypatch):
    monkeypatch.setenv("AIRFLOW_CONN_HUGGINGFACE_EXTRA_TEST",
                       '{"conn_type":"huggingface","extra":{"token":"ignored"}}')
    assert HuggingFaceHook("huggingface_extra_test").get_token() is None
