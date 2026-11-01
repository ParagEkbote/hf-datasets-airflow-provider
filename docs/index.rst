Apache Airflow Hugging Face Provider
====================================

The Hugging Face provider connects Airflow tasks to Hugging Face Datasets and
Hub repositories. It supplies a shared ``HuggingFaceHook`` and four operators
to load a dataset, map a Python callable, stream directly to the Hub, and
publish a durable artifact. Durable data moves between tasks as Parquet objects
and a small JSON manifest, so tasks can run on separate workers when their
artifact store is shared.

Supported capabilities
----------------------

* Airflow connection based Hub authentication, with Hub SDK fallback when the
default connection is absent.
* Revision pinning and source provenance for Hub dataset loads.
* Split aware Parquet handoff with features metadata and content fingerprints.
* Synchronous publication of materialized or streamed datasets with the Hub
  commit SHA and URL in results.
* ``hf://datasets/<repo_id>`` assets and OpenLineage conversion.

``StreamDatasetOperator`` lazily transforms and filters an ``IterableDataset``
and publishes it directly to the Hub. Datasets uploads Parquet shards
progressively; the operator does not retain all rows or create an intermediate
artifact. Standalone filtering and split assembly operators are not included
in the v1 public API.

Installation
------------

Install into an Airflow environment with Python 3.10 through 3.13:

.. code-block:: bash

   pip install apache-airflow-providers-huggingface

The provider requires Apache Airflow 3.0 or later, ``datasets>=3.4.0,<5``,
and ``huggingface-hub>=0.24,<2``. For ``s3://`` and ``gs://`` artifacts,
install the corresponding storage provider and filesystem dependency. Select
the matching Airflow connection in the URI's user-info section, for example
``s3://aws_default@my-bucket/path/data.json`` or
``gs://google_cloud_default@my-bucket/path/data.json``. Without a connection
ID in the URI, the filesystem uses its default credential chain; configuring
an Airflow connection alone does not select it.

Guides
------

.. toctree::
   :maxdepth: 2

   connections
   operators
   artifacts
   security
   changelog

The system-test DAGs under ``tests/system/huggingface/`` also serve as runnable
examples. The read-only workflow uses a pinned public dataset. The publication
workflows require ``HF_TEST_HUB_ORG`` for a dedicated organization and
``HF_TEST_HUB_CONN_ID`` (defaults to ``huggingface_default``) for a write-scoped
connection. ``HF_TEST_WRITE_REPO_ID`` and ``HF_TEST_DURABLE_REPO_ID`` can
override the base repository names. Each run appends ``SYSTEM_TESTS_ENV_ID``
(or a generated UUID) to its repository name to avoid collisions. Cleanup tasks
use ``ALL_DONE`` so they are attempted after failures.
