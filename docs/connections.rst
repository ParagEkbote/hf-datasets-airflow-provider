Connections
===========

The optional connection ID defaults to ``huggingface_default``. Create a connection
with type ``huggingface`` and put the access token in the password field (or in
the ``token`` field of Extra JSON). Credentials can also come from Airflow's
configured secrets backend because the hook resolves connections through
``BaseHook``. Do not place tokens in DAG source or operator arguments.

Public datasets can be loaded without a connection. For private or gated datasets,
configure a read token. Publishing requires a token with write access. The hook
does not log token values.
