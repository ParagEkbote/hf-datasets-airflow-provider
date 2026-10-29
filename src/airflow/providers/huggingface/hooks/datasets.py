"""Backward-compatible import for the former datasets hook."""
from airflow.providers.huggingface.hooks.huggingface import HuggingFaceHook

HuggingFaceDatasetsHook = HuggingFaceHook

__all__ = ["HuggingFaceHook", "HuggingFaceDatasetsHook"]
