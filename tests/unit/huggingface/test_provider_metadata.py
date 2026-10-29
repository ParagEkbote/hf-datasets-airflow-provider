"""Keep provider discovery metadata aligned with this distribution."""
import tomllib
from pathlib import Path

from airflow.providers.huggingface.get_provider_info import get_provider_info


def test_packaging_dependencies_and_provider_discovery():
    root = Path(__file__).resolve().parents[3]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    deps = project["project"]["dependencies"]
    assert "apache-airflow>=3.0,<4" in deps
    assert "datasets>=3.4.0,<5" in deps
    assert "huggingface-hub>=0.24,<2" in deps
    assert "apache-airflow-providers-common-io>=1.3.0" in deps
    entry = project["project"]["entry-points"]["apache_airflow_provider"]["provider_info"]
    assert entry == "airflow.providers.huggingface.get_provider_info:get_provider_info"

    info = get_provider_info()
    connection = info["connection-types"][0]
    assert connection["connection-type"] == "huggingface"
    assert connection["hook-class-name"].endswith("HuggingFaceHook")
    asset_scheme = info["asset-uris"][0]["schemes"]
    assert asset_scheme == ["hf"]
    assert info["asset-uris"][0]["factory"].endswith("asset.create_asset")


def test_generated_runtime_metadata_matches_provider_yaml():
    import yaml

    root = Path(__file__).resolve().parents[3]
    declaration = yaml.safe_load((root / "provider.yaml").read_text())
    info = get_provider_info()
    assert declaration["package-name"] == info["package-name"]
    declared_connection = declaration["connection-types"][0]
    runtime_connection = info["connection-types"][0]
    assert declared_connection["connection-type"] == runtime_connection["connection-type"]
    assert declared_connection["hook-class-name"] == runtime_connection["hook-class-name"]
    assert declared_connection["ui-field-behaviour"] == runtime_connection["ui-field-behaviour"]
    assert declaration["asset-uris"][0]["schemes"] == info["asset-uris"][0]["schemes"]
