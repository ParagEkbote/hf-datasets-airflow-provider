"""Structural checks for provider-native example DAGs."""

from importlib import import_module


def test_examples_import_and_use_provider_tasks():
    modules = [
        "examples.glue_mrpc_normalize",
        "examples.fineweb_edu_streaming_sample",
        "examples.squad_multi_split_pipeline",
        "examples.daily_refresh_asset_trigger",
    ]
    for name in modules:
        module = import_module(name)
        dags = [value for value in vars(module).values() if hasattr(value, "task_dict")]
        assert dags, name
        for dag in dags:
            assert dag.tasks, dag.dag_id
            assert all(task.__class__.__module__.startswith("airflow.providers.huggingface")
                       or task.__class__.__name__ == "EmptyOperator" for task in dag.tasks)


def test_daily_asset_consumer_uses_published_asset_uri():
    module = import_module("examples.daily_refresh_asset_trigger")
    assert module.producer.get_task("publish").outlets[0].uri == module.consumer.schedule[0].uri
