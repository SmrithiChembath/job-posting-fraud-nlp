"""
Fine-tunes DistilBERT for fraud/not-fraud classification, with a class-weighted loss
to handle the ~5% positive rate (plain fine-tuning on imbalanced data tends to just
predict "real" for everything).

Run: python src/bert_train.py

If you're iterating on the script and want a fast smoke test before a full run, set
MAX_TRAIN_SAMPLES below to something small (e.g. 500) and NUM_EPOCHS to 1, then set
them back for your real training run.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models/bert_fraud_model")
TARGET = "fraudulent"

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
NUM_EPOCHS = 1
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_TRAIN_SAMPLES = None  # set to an int (e.g. 500) for a fast smoke test


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    if name == "train" and MAX_TRAIN_SAMPLES:
        df = df.sample(n=min(MAX_TRAIN_SAMPLES, len(df)), random_state=42)
    return df


class WeightedTrainer(Trainer):
    """Trainer that applies class weights to the loss, so the ~5% fraud class
    doesn't get drowned out by the 95% real class during fine-tuning."""

    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score(labels, probs),
        "pr_auc": average_precision_score(labels, probs),
    }


def main():
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH
        )

    def to_dataset(df: pd.DataFrame) -> Dataset:
        ds = Dataset.from_pandas(
            df[["text", TARGET]].rename(columns={TARGET: "labels"}),
            preserve_index=False,
        )
        return ds.map(tokenize, batched=True)

    train_ds = to_dataset(train_df)
    val_ds = to_dataset(val_df)
    test_ds = to_dataset(test_df)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "real", 1: "fraudulent"},
        label2id={"real": 0, "fraudulent": 1},
    )

    # Inverse-frequency class weights: rare "fraudulent" class gets weighted up
    class_counts = train_df[TARGET].value_counts().sort_index()
    class_weights = torch.tensor(
        [len(train_df) / (2 * class_counts[0]), len(train_df) / (2 * class_counts[1])],
        dtype=torch.float,
    )
    print(f"Class weights [real, fraudulent]: {class_weights.tolist()}")

    training_args = TrainingArguments(
        output_dir="models/bert_checkpoints",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="pr_auc",
        logging_steps=50,
        report_to="none",
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\n=== Validation metrics (best checkpoint) ===")
    print(trainer.evaluate(val_ds))

    print("\n=== Test metrics ===")
    print(trainer.evaluate(test_ds))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"\nSaved fine-tuned model to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
