Operators
=========

Artifact references
-------------------

Operators exchange small JSON references through XCom. An Arrow artifact
reference has the form ``{"uri": "file:///mnt/data/name", "format": "arrow",
"repository_id": "org/data", "revision": "<commit-sha>"}``; the provenance
fields are included when available. Pass the upstream task's whole output (for
example, ``input_uri=load.output``) to preserve this metadata, or pass its URI
when metadata is not needed.
The URI points to a directory written by Datasets ``save_to_disk``. Workers
executing consecutive tasks must see the same filesystem at that path. A local
path on one ephemeral worker is not shared storage; configure a mounted shared
volume in multi-worker deployments. This MVP does not configure cloud storage.
Mapping also accepts a ``file://`` URI to a Parquet file and reads it through
Datasets' Parquet builder.

LoadDatasetOperator
-------------------

``LoadDatasetOperator`` accepts ``path``, ``output_uri``, optional ``name``
(dataset configuration), ``split``, ``revision``, ``cache_dir``,
``download_mode`` and ``load_kwargs``. It persists the result and returns the
artifact URI plus source ID, requested/resolved revision, configuration, split,
row counts, and feature descriptions. Resolution of a source commit SHA is
best-effort; pin a commit SHA in ``revision`` for reproducible inputs.

The default ``reuse_dataset_if_exists`` mode reuses Datasets' valid cache.
``force_redownload`` forces a new download. ``cache_dir`` isolates cache data;
it is separate from the output artifact destination. Keeping a stable cache
directory allows retries and cleared task instances to reuse downloaded data.
Run-specific cache directories are possible by templating ``cache_dir`` but are
not the default.

MapDatasetOperator
------------------

``MapDatasetOperator`` accepts ``input_uri``, ``output_uri``, an import path in
``module:function`` form, optional ``transform_kwargs``, and ``map_kwargs``.
The callable is imported only in task execution, never during DAG parsing.
``map_kwargs`` are passed to ``Dataset.map``/``DatasetDict.map`` and include
options such as ``batched``, ``batch_size``, ``num_proc``, and
``load_from_cache_file``. Datasets retains split structure and Arrow features
where the transformation output permits it.

PushToHubOperator and assets
----------------------------

``PushToHubOperator`` accepts the artifact URI, target ``repo_id``, optional
``config``, ``private``, ``commit_message`` and ``push_kwargs``. It uploads
synchronously and returns repository ID, URL, and commit SHA when the Datasets
API provides one. A successful task emits an Airflow 3 Asset event with the
repository ID and commit SHA in event metadata. The stable asset URI is
``huggingface://datasets/<repo_id>``; use the same ``Asset`` in a downstream
DAG's schedule. A failed upload raises and emits no successful event.

Uploads and transformations run in the task worker. Large datasets require
adequate worker memory, temporary disk, and network bandwidth. Arrow artifacts
are the native persisted format; Parquet is accepted as map input for
interoperability, but is not the internal handoff format.

Examples
--------

The repository's ``examples/glue_in_memory.py`` DAG loads the GLUE MRPC
training split without streaming, then combines and normalizes its sentence
pairs while retaining the label. This demonstrates the ordinary in-memory
``Dataset`` path through the load and map operators.

``examples/streaming_fineweb_edu.py`` uses the hook from a task to load
``HuggingFaceFW/fineweb-edu``'s ``sample-10BT`` configuration with
``streaming=True``. It lazily normalizes whitespace, adds a word count, filters
short documents, then persists at most 1,000 selected records as an Arrow
artifact. FineWeb-Edu's sample config is a large multi-shard dataset; the
bounded output keeps this runnable as an example. The example worker needs
network access and enough disk for its bounded output.

The standard load and map operators persist map-style datasets with
``save_to_disk``. They do not currently persist a complete ``IterableDataset``;
use a task-local streaming sink or a bounded sample as shown in the streaming
example. Replace ``revision="main"`` with a dataset commit SHA when reproducible
input is required.
