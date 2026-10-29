Dataset artifact contract
========================

Load and map tasks write a JSON manifest at the requested ``output_uri`` and
one Parquet object per split beside it (``<uri>.split-N.parquet``). The
returned reference contains the manifest URI, its SHA-256 fingerprint, source
revision, split names, row counts, and serialized Hugging Face ``Features``.
Downstream tasks should pass the complete reference, rather than only its URI,
to retain fingerprint validation and provenance.

The manifest and per-split SHA-256 checksums detect changed or incomplete artifacts. The manifest preserves column names and order through the serialized Features
schema and records ``DatasetDict`` split names. Reconstruction uses
``Features.from_dict`` and a PyArrow Parquet reader before rebuilding each
Datasets ``Dataset`` object. Features must be
representable by the installed Datasets and PyArrow Parquet implementations.
Arrow primitive, list, struct, and class-label data are the intended supported
set. Media and extension features (including Image, Audio, Video, and nested
array features) can have serialization semantics that Parquet cannot retain;
validate these explicitly for a workload before relying on them. If a feature
cannot be encoded or reconstructed, the task fails and does not return a
reference.

The format is designed for ``file://``, ``s3://``, and ``gs://`` paths through
Airflow ``ObjectStoragePath``. Remote schemes need their matching Airflow
storage provider, filesystem dependency, connection, and permissions installed
on every worker. This package does not claim a distributed storage backend is
available merely because the URI parses. Local files require a shared path if
tasks run on different workers. A failed write can leave an incomplete
manifest or orphaned split objects; rerunning the writer overwrites those
objects. Configure object-store lifecycle cleanup for abandoned temporary
objects. Publication checks the manifest fingerprint before upload.

The canonical Airflow asset URI for a dataset repository is
``hf://datasets/<repo_id>``. Hugging Face Hub supplies ``HfFileSystem`` for
``hf://`` file access; Airflow asset URI registration is a separate feature
and does not enable or configure that filesystem.
