Operators
=========

The initial public operator API is ``LoadDatasetOperator``,
``MapDatasetOperator``, ``StreamDatasetOperator``, and
``PushToHubOperator``. The operators exchange the manifest and split Parquet
objects described in :doc:`artifacts`.

LoadDatasetOperator
-------------------

``LoadDatasetOperator`` accepts ``path``, ``output_uri``, optional ``name``
(dataset configuration), ``split``, ``revision``, ``cache_dir``,
``download_mode`` and ``load_kwargs``. It persists the result and returns the
artifact URI plus source ID, resolved revision, configuration, split, row
counts, and feature descriptions. It resolves a Hub source to a commit before
loading; pin a commit SHA in ``revision`` for reproducible inputs.

The default ``reuse_dataset_if_exists`` mode reuses Datasets' valid cache.
``force_redownload`` forces a new download. ``cache_dir`` isolates cache data;
it is separate from the output artifact destination. Keeping a stable cache
directory allows retries and cleared task instances to reuse downloaded data.

MapDatasetOperator
------------------

Required parameters are ``input_uri``, ``output_uri``, and ``transform``.
``transform`` must be a Python callable available when the DAG is parsed and
when the task worker executes. The operator does not template or import the
callable from a string. ``transform_kwargs`` are passed as Datasets ``fn_kwargs``;
``map_kwargs`` are passed to ``Dataset.map`` or ``DatasetDict.map``. The output
preserves compatible provenance and records a new artifact fingerprint.

StreamDatasetOperator
---------------------

``StreamDatasetOperator`` loads with Datasets ``streaming=True`` and requires
positive ``max_source_rows`` and ``max_output_rows`` bounds. It applies a
Python ``transform`` to each row and can optionally apply a Python ``predicate``
to each transformed row. Both callables must be available at DAG parse time and
on workers. ``transform_kwargs`` are passed to the transform. ``feature_types``
can declare a ``Features`` schema for the materialized sample. The selected
bounded rows are persisted through the same manifest and per-split Parquet
artifact contract as load and map tasks; dataset rows are never put in XCom.
An empty result fails the task. The resulting feature types must be supported
by the Parquet artifact writer.

PushToHubOperator
-----------------

Required parameters are ``input_uri`` (the full artifact reference, including
fingerprint) and ``repo_id``. ``config``, ``private``, ``commit_message``,
``huggingface_conn_id``, and ``push_kwargs`` configure publication. The
operator verifies the manifest fingerprint and per-split checksums before
uploading. It returns ``repository_id``, the Hub ``CommitInfo.oid`` as
``commit_sha``, ``CommitInfo.commit_url`` as ``commit_url``, the artifact
fingerprint, and source provenance. Hub exceptions propagate without wrapping.

The commit message includes an artifact fingerprint marker. Airflow 3.3 and
later task state store support caches the result when that facility is
available. The marker also lets a retry find a successfully created data commit
if the worker exited before saving task state. The task emits an asset event
with the same commit SHA and URL only after successful publication. Dataset
card generation and README updates are not part of this operator's scope.

Example
-------

.. code-block:: python

   from airflow.providers.huggingface.operators.datasets import (
       LoadDatasetOperator, MapDatasetOperator, PushToHubOperator,
   )
   from my_project.transforms import normalize

   load = LoadDatasetOperator(
       task_id="load", path="nyu-mll/glue", name="mrpc", split="train",
       revision="<commit-sha>", output_uri="s3://airflow/data/mrpc.json",
   )
   transform = MapDatasetOperator(
       task_id="normalize", input_uri=load.output,
       output_uri="s3://airflow/data/mrpc-normalized.json", transform=normalize,
   )
   publish = PushToHubOperator(
       task_id="publish", input_uri=transform.output,
       repo_id="my-org/mrpc-normalized",
   )
   load >> transform >> publish
