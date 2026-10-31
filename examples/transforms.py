# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
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
