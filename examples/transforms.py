"""Importable transformations used by the example DAGs."""


def normalize_glue_pair(example: dict) -> dict:
    """Normalize an MRPC sentence pair into a single text field."""
    first = " ".join(example["sentence1"].split()).lower()
    second = " ".join(example["sentence2"].split()).lower()
    return {"text": f"{first} [SEP] {second}", "label": example["label"]}
