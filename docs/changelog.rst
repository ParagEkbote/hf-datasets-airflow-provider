Changelog
=========

Version 0.1.0
------------

Initial public release of the Hugging Face provider for Apache Airflow.

* Add ``HuggingFaceHook`` and the ``huggingface`` connection.
* Add ``LoadDatasetOperator``, ``MapDatasetOperator``, and
  ``PushToHubOperator``.
* Add revision pinned loads, Parquet artifacts, fingerprints, Hub publication
  commit metadata, and Hugging Face dataset assets.
* Set the initial artifact contract to JSON manifests and per split Parquet
  objects. Streaming, filtering, and split assembly operators are not included
  in the v1 public API.
