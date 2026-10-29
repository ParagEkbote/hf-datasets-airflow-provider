"""Backward-compatible import for the consolidated Hugging Face hook."""

from airflow.providers.huggingface.hooks.huggingface import HuggingFaceHook

HuggingFaceDatasetsHook = HuggingFaceHook

__all__ = ["HuggingFaceDatasetsHook", "HuggingFaceHook"]
