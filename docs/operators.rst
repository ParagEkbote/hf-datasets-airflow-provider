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
row counts, and feature descriptions. For a Hub dataset, the operator resolves
``revision`` (or the default branch) to a commit SHA before loading, loads that
commit, and records it. The task fails if the revision cannot be resolved.
Pin a commit SHA in ``revision`` for reproducible inputs across runs.

The default ``reuse_dataset_if_exists`` mode reuses Datasets' valid cache.
``force_redownload`` forces a new download. ``cache_dir`` isolates cache data;
it is separate from the output artifact destination. Keeping a stable cache
directory allows retries and cleared task instances to reuse downloaded data.
Run-specific cache directories are possible by templating ``cache_dir`` but are
not the default. Datasets keys its cache on the resolved commit, so reusing it
for the same commit returns the same data.

Durable execution
-----------------

On Airflow 3.3 and later, ``LoadDatasetOperator`` and ``PushToHubOperator``
keep what they did in the task state store, so a retry continues instead of
starting over. ``durable`` is ``True`` by default. Set ``durable=False`` to
turn it off. On earlier Airflow versions there is no task state store and
``durable`` has no effect.

* ``LoadDatasetOperator`` stores the commit it pinned. Every retry loads that
  commit, even if ``main`` has moved on, so one Dag run never mixes two
  versions of the source. A retry pins again if ``path`` or ``revision``
  changed.
* ``PushToHubOperator`` stores the commit SHA once the upload succeeds. If the
  task fails after that, for example while uploading the dataset card, the
  retry does not upload the data again. It emits the asset event with the
  stored SHA. The stored commit is tied to the input's Datasets fingerprint,
  which every operator records in its output, so different data is always
  pushed. An input given as a plain URI has no fingerprint and is pushed again.

Clearing a task from the UI or REST API discards its task state, so the task
starts over: the load pins the current commit and the push uploads again.
Clearing a whole Dag run, or running ``airflow tasks clear``, keeps task state
(`apache/airflow#72929 <https://github.com/apache/airflow/issues/72929>`__).
A load in a cleared run then reuses its pinned commit. A push uploads again
whenever the upstream data changed.

MapDatasetOperator
------------------

``MapDatasetOperator`` accepts ``input_uri``, ``output_uri``, an import path in
``module:function`` form, optional ``transform_kwargs``, and ``map_kwargs``.
The callable is imported only in task execution, never during DAG parsing.
``map_kwargs`` are passed to ``Dataset.map``/``DatasetDict.map`` and include
options such as ``batched``, ``batch_size``, ``num_proc``, and
``load_from_cache_file``. Datasets retains split structure and Arrow features
where the transformation output permits it.

FilterDatasetOperator
---------------------

``FilterDatasetOperator`` accepts an importable ``predicate`` in
``module:function`` form and passes it to ``Dataset.filter`` (or
``DatasetDict.filter``). Predicates return a boolean for each record, so row
selection uses Datasets filtering semantics rather than dropping rows inside a
mapping function.

StreamDatasetOperator
---------------------

``StreamDatasetOperator`` owns bounded iterable loading and persistence. It
accepts source limits, an importable row transform, an optional row predicate,
and an explicit output ``feature_types`` schema (for example ``string`` and
``int64``). It persists an Arrow artifact and returns its compact reference;
the stream and row objects never enter XCom.

AssembleDatasetOperator
-----------------------

``AssembleDatasetOperator`` accepts a mapping of split names to artifact
references and persists a ``DatasetDict``. Split artifacts must all be visible
to the executing worker under the same documented shared-storage contract.

PushToHubOperator and assets
----------------------------

``PushToHubOperator`` accepts the artifact URI, target ``repo_id``, optional
``config``, ``private``, ``commit_message`` and ``push_kwargs``. It uploads
synchronously and returns repository ID, URL, and commit SHA when the Datasets
API provides one. A successful task emits an Airflow 3 Asset event with the
repository ID and commit SHA in event metadata. The stable asset URI is
``huggingface://datasets/<repo_id>``; use the same ``Asset`` in a downstream
DAG's schedule. A failed upload raises and emits no successful event.

Dataset cards
~~~~~~~~~~~~~

Card generation is disabled by default. Set ``dataset_card=True`` to generate
a root ``README.md``. ``card_metadata`` accepts the Hub fields ``pretty_name``
and ``license`` (strings), plus ``language``, ``task_categories``, ``tags`` and
``size_categories`` (lists of strings). The operator validates these types and
does not infer licensing or other descriptive metadata. ``card_description``
is human-authored text. The built-in card records the destination, observed
split names, row counts and feature types, along with source ID, revision,
configuration and split when present in the input artifact reference.

``card_template`` may name a trusted Jinja template file. Its context contains
only ``repo_id``, ``metadata``, ``description``, ``structure`` and ``provenance``;
Jinja2 must be installed. Strict undefined-variable handling makes template
mistakes fail before dataset publication.

``card_update_policy`` defaults to ``preserve``: an existing README is left
alone, while a missing README is created. ``update`` replaces only a region
bracketed by ``<!-- BEGIN AIRFLOW HUGGINGFACE DATASET CARD -->`` and
``<!-- END AIRFLOW HUGGINGFACE DATASET CARD -->``; it fails if those markers
are absent. ``replace`` explicitly replaces the whole README. Use it only when
you intend to discard existing documentation. With cards disabled, no README
is inspected or changed.

Dataset and card commits are separate Hub operations and are not atomic. If
the dataset upload succeeds and README upload fails, the task fails with the
dataset commit (when available) in its error. A retry using ``preserve`` will
not overwrite a README that appeared in the meantime. XCom reports the card
action and separate card commit when the Hub returns one; it never includes
README contents. The configured Hugging Face connection must have permission
to create or update repository files. Missing source provenance is simply
omitted.

Example with card generation disabled (the default):

.. code-block:: python

   from airflow.providers.huggingface.operators.datasets import (
       LoadDatasetOperator, MapDatasetOperator, PushToHubOperator,
   )

   load = LoadDatasetOperator(
       task_id="load", path="nyu-mll/glue", name="mrpc", split="train",
       output_uri="file:///mnt/shared/mrpc",
   )
   transform = MapDatasetOperator(
       task_id="normalize_pairs", input_uri=load.output,
       output_uri="file:///mnt/shared/mrpc_normalized",
       transform="examples.transforms:normalize_glue_pair",
       map_kwargs={"remove_columns": ["sentence1", "sentence2", "idx"]},
   )
   publish = PushToHubOperator(
       task_id="publish", input_uri=transform.output,
       repo_id="my-org/mrpc-normalized", dataset_card=False,
   )
   load >> transform >> publish

Example generating a card while preserving a README already in the repository:

.. code-block:: python

   from airflow.providers.huggingface.operators.datasets import (
       LoadDatasetOperator, MapDatasetOperator, PushToHubOperator,
   )

   load = LoadDatasetOperator(
       task_id="load", path="nyu-mll/glue", name="mrpc", split="train",
       output_uri="file:///mnt/shared/mrpc",
   )
   transform = MapDatasetOperator(
       task_id="normalize_pairs", input_uri=load.output,
       output_uri="file:///mnt/shared/mrpc_normalized",
       transform="examples.transforms:normalize_glue_pair",
       map_kwargs={"remove_columns": ["sentence1", "sentence2", "idx"]},
   )
   publish = PushToHubOperator(
       task_id="publish", input_uri=transform.output,
       repo_id="my-org/mrpc-normalized", dataset_card=True,
       card_description="MRPC sentence pairs normalized into a single text field while retaining the similarity label.",
       card_metadata={"pretty_name": "Normalized MRPC training pairs", "tags": ["glue", "mrpc", "text-classification"]},
       card_update_policy="preserve",
   )
   load >> transform >> publish

Uploads and transformations run in the task worker. Large datasets require
adequate worker memory, temporary disk, and network bandwidth. Arrow artifacts
are the native persisted format; Parquet is accepted as map input for
interoperability, but is not the internal handoff format.

Examples
--------

The provider-native DAGs are documented in ``examples/README.md``. They cover
GLUE normalization, bounded FineWeb streaming, parallel SQuAD split processing
and assembly, and a producer-consumer Asset workflow. All source revisions
should be configured as commit SHAs before execution.
