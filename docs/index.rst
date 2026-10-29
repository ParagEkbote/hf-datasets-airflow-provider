Apache Airflow Hugging Face Provider
====================================

The Hugging Face provider connects Airflow tasks to Hugging Face Datasets and
Hub repositories. It supplies a shared ``HuggingFaceHook`` and three operators
to load a dataset, map a Python callable, and publish a durable artifact. Data
moves between tasks as Parquet objects and a small JSON manifest, so tasks can
run on separate workers when their artifact store is shared.

Supported capabilities
----------------------

* Airflow connection based Hub authentication, with Hub SDK fallback when the
default connection is absent.
* Revision pinning and source provenance for Hub dataset loads.
* Split aware Parquet handoff with features metadata and content fingerprints.
* Synchronous dataset publication with the Hub commit SHA and URL in results.
* ``hf://datasets/<repo_id>`` assets and OpenLineage conversion.
* Optional dataset card generation and conservative README update policies.

The v1 public operator API does not include streaming, filtering, or split
assembly operators. Use Datasets streaming outside this handoff contract only
when a future provider API explicitly supports its durable output semantics.

Installation
------------

Install into an Airflow environment with Python 3.10 through 3.13:

.. code-block:: bash

   pip install apache-airflow-providers-huggingface

The provider requires Apache Airflow 3.0 or later, ``datasets>=3.4.0,<5``,
``huggingface-hub>=0.24,<2``, and Airflow common I/O. For ``s3://`` and
``gs://`` artifacts, also install the corresponding storage provider and
filesystem dependency and configure its Airflow connection.

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
``HF_TEST_HUB_CONN_ID`` (or ``huggingface_default``) for a write-scoped
connection. ``HF_TEST_WRITE_REPO_ID`` and ``HF_TEST_DURABLE_REPO_ID`` can
override the temporary repository names. Cleanup tasks use ``ALL_DONE`` so
they are attempted after failures.
