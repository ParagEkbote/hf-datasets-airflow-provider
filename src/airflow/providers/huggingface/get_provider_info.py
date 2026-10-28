"""Provider discovery metadata."""


def get_provider_info() -> dict:
    return {
        "package-name": "apache-airflow-providers-huggingface",
        "name": "Hugging Face",
        "description": "Load, transform, and publish Hugging Face Datasets.",
        "versions": ["0.1.0"],
        "integrations": [
            {"integration-name": "Hugging Face Datasets", "external-doc-url": "https://huggingface.co/docs/datasets/"}
        ],
        "hooks": [
            {
                "integration-name": "Hugging Face Datasets",
                "python-modules": ["airflow.providers.huggingface.hooks.datasets"],
            }
        ],
        "connection-types": [
            {
                "connection-type": "huggingface",
                "hook-class-name": "airflow.providers.huggingface.hooks.datasets.HuggingFaceDatasetsHook",
                "hook-name": "Hugging Face",
            }
        ],
        "operators": [
            {"integration-name": "Hugging Face Datasets", "python-modules": ["airflow.providers.huggingface.operators.datasets"]}
        ],
    }
