Operators
=========

The initial public operator API is ``LoadDatasetOperator``,
``MapDatasetOperator``, and ``PushToHubOperator``. The operators exchange the
manifest and split Parquet objects described in :doc:`artifacts`.

LoadDatasetOperator
-------------------

Required parameters:

* ``path``: Datasets source, commonly a Hub dataset repository ID.
* ``output_uri``: ObjectStoragePath URI where the manifest will be written.

Optional parameters:

* ``name`` and ``split``: Datasets configuration and split.
* ``revision``: source branch, tag, or commit SHA.
* ``cache_dir`` and ``download_mode``: Datasets cache controls.
* ``huggingface_conn_id``: connection ID, defaulting to
  ``huggingface_default``.
* ``load_kwargs``: additional keyword arguments passed to
  ``datasets.load_dataset``.

For a Hub repository, the operator resolves ``revision`` through Hub metadata,
then passes the resolved commit SHA to ``load_dataset``. The result reference
records that SHA with the artifact fingerprint, source ID, configuration, and
split. A load or revision lookup error is propagated and no artifact reference
is returned.

MapDatasetOperator
------------------

Required parameters are ``input_uri``, ``output_uri``, and ``transform``.
``transform`` must be a Python callable available when the DAG is parsed and
when the task worker executes. The operator does not template or import the
callable from a string. ``transform_kwargs`` are passed as Datasets ``fn_kwargs``;
``map_kwargs`` are passed to ``Dataset.map`` or ``DatasetDict.map``. The output
preserves compatible provenance and records a new artifact fingerprint.

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
with the same commit SHA and URL only after successful publication (including
optional card handling).

Optional card parameters retain the existing README behavior:

* ``dataset_card`` enables a generated ``README.md``; default ``False``.
* ``card_metadata`` accepts ``pretty_name`` and ``license`` strings, and
  ``language``, ``task_categories``, ``tags``, and ``size_categories`` lists
  of strings.
* ``card_description`` supplies human-authored description text.
* ``card_template`` selects a trusted local Jinja2 template. Install Jinja2
  when using it. Its context is ``repo_id``, ``metadata``, ``description``,
  ``structure``, and ``provenance``.
* ``card_update_policy`` is ``preserve`` (default), ``update`` (replace only
  provider marked content), or ``replace`` (replace the whole README).

A card upload is a separate Hub commit. If it fails, the task fails and emits
no event, while retry recovery prevents the dataset data commit from being
created again. Connection and Hub errors retain their original exception types.

Streaming decision
------------------

Streaming, standalone filtering, and split assembly operators are not part of
the initial v1 API. A streaming ``IterableDataset`` cannot be written through
the current materialized Parquet handoff without changing memory and failure
semantics. Use a separately designed streaming-to-durable-artifact workflow
when one is needed; this release does not claim streaming support.

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
