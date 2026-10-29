"""Load, map, and publish the SQuAD train and validation splits together."""
from airflow.sdk import DAG
from airflow.providers.huggingface.operators.datasets import LoadDatasetOperator, MapDatasetOperator, PushToHubOperator
import os

def flatten_squad_context(batch):
    """Combine contexts and questions while recording answerability."""
    inputs, labels = [], []
    for context, question, answers in zip(batch["context"], batch["question"], batch["answers"]):
        inputs.append(f"context: {context}  question: {question}")
        labels.append(0 if not answers["text"] else 1)
    return {"input_text": inputs, "answerable": labels}


with DAG("huggingface_squad_multi_split_pipeline", tags=["huggingface", "squad", "example"], doc_md=__doc__) as dag:
    load = LoadDatasetOperator(
        task_id="load_splits", path="rajpurkar/squad",
        revision="7b6d24c440a36b6815f21b70d25016731768db1f",
        output_uri="file:///tmp/hf-datasets/squad/source.json",
    )
    mapped = MapDatasetOperator(
        task_id="flatten_splits", input_uri=load.output,
        output_uri="file:///tmp/hf-datasets/squad/flattened.json",
        transform=flatten_squad_context,
        map_kwargs={"batched": True, "batch_size": 512,
                    "remove_columns": ["id", "title", "context", "question", "answers"]},
    )
    publish = PushToHubOperator(task_id="publish", input_uri=mapped.output,
                                 repo_id=os.environ.get("HF_SQUAD_REPO_ID", "my-org/squad-flattened"), private=True)
    load >> mapped >> publish
