"""Importable, side-effect-free transformations used by example DAGs."""


def normalize_glue_pair(batch: dict) -> dict:
    """Fuse MRPC sentence pairs while retaining the binary label."""
    return {
        "text": [f"{' '.join(a.split()).lower()} [SEP] {' '.join(b.split()).lower()}"
                 for a, b in zip(batch["sentence1"], batch["sentence2"])],
        "label": batch["label"],
    }


def flatten_squad_context(batch: dict) -> dict:
    """Combine SQuAD context and question and encode answerability."""
    inputs, labels = [], []
    for context, question, answers in zip(batch["context"], batch["question"], batch["answers"]):
        inputs.append(f"context: {context}  question: {question}")
        labels.append(0 if not answers["text"] else 1)
    return {"input_text": inputs, "answerable": labels}


def add_char_count(batch: dict) -> dict:
    """Add output character count while preserving the source columns."""
    instructions = batch["instruction"]
    inputs = batch.get("input", [""] * len(instructions))
    outputs = batch["output"]
    batch["char_count"] = [len(i) + len(inp) + len(out) for i, inp, out in zip(instructions, inputs, outputs)]
    return batch


def keep_long_outputs(example: dict) -> bool:
    """Keep rows whose output is longer than 200 characters (Dataset.filter)."""
    return len(example["output"]) > 200


def clean_fineweb_document(example: dict) -> dict:
    """Normalize a FineWeb document and add a word count."""
    text = " ".join(example["text"].split())
    return {"text": text, "word_count": len(text.split())}


def is_long_fineweb_document(example: dict) -> bool:
    return example["word_count"] >= 100
