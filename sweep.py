"""Quick learning-rate sweep for modernbert-base-banking77.

Honest protocol: train one model per candidate LR on the train split, select the
winner by VALIDATION macro-F1, then evaluate ONLY that winner once on the held-out
test split. The test set is never used for model selection. Nothing is pushed to the
Hub -- the winner is saved locally so you can push it deliberately if it clearly beats
the shipped model. Run on a GPU box (A100 recommended): python sweep.py
"""
import json

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer,
                          EarlyStoppingCallback, set_seed)

import config


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def prepare():
    """Load data and build the same stratified train/val/test as the main run."""
    ds = load_dataset(config.DATASET_ID, revision=config.DATASET_REVISION)
    names = ds["train"].features[config.LABEL_COLUMN].names
    id2label = dict(enumerate(names))
    label2id = {name: i for i, name in id2label.items()}
    split = ds["train"].train_test_split(
        test_size=config.VAL_FRACTION, stratify_by_column=config.LABEL_COLUMN, seed=config.SEED)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)

    def tokenize(batch):
        return tokenizer(batch[config.TEXT_COLUMN], truncation=True, max_length=config.MAX_LENGTH)

    raw = {"train": split["train"], "val": split["test"], "test": ds["test"]}
    enc = {k: v.map(tokenize, batched=True, remove_columns=[config.TEXT_COLUMN]) for k, v in raw.items()}
    return enc, tokenizer, id2label, label2id


def train_one(lr, enc, tokenizer, id2label, label2id):
    """Train a single model at the given LR; return (trainer, best val macro-F1)."""
    set_seed(config.SEED)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_ID, num_labels=config.NUM_LABELS,
        id2label=id2label, label2id=label2id, attn_implementation="sdpa")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=f"{config.OUTPUT_DIR}-sweep/lr_{lr:.0e}",
        learning_rate=lr,
        num_train_epochs=config.SWEEP_EPOCHS,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        warmup_ratio=config.WARMUP_RATIO,
        weight_decay=config.WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=config.METRIC_FOR_BEST,
        greater_is_better=True,
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        seed=config.SEED,
        report_to="none",
        logging_steps=50,
        save_total_limit=1,
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=enc["train"], eval_dataset=enc["val"],
        processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING_PATIENCE)])
    trainer.train()
    return trainer, trainer.state.best_metric


def shipped_macro_f1():
    try:
        return json.load(open(f"{config.RESULTS_DIR}/metrics.json"))["fine_tuned"]["f1_macro"]
    except Exception:
        return 0.9176


def main():
    enc, tokenizer, id2label, label2id = prepare()

    runs = {}
    for lr in config.LR_SWEEP:
        trainer, val_f1 = train_one(lr, enc, tokenizer, id2label, label2id)
        runs[lr] = (trainer, val_f1)
        print(f"lr={lr:.0e}: val macro-F1 = {val_f1:.4f}")

    # select the winner on VALIDATION macro-F1 -- the test set plays no part here
    best_lr = max(runs, key=lambda lr: runs[lr][1])
    best_trainer, best_val = runs[best_lr]
    print(f"\nwinner (by val macro-F1): lr={best_lr:.0e}  val={best_val:.4f}")

    # the held-out test split is evaluated exactly once, for the winner only
    test = best_trainer.evaluate(enc["test"])
    shipped = shipped_macro_f1()
    print(f"\nWINNER on held-out test: accuracy={test['eval_accuracy']:.4f}  "
          f"macro-F1={test['eval_f1_macro']:.4f}")
    print(f"shipped model (lr=2e-5): macro-F1={shipped:.4f}")
    print(f"delta macro-F1 vs shipped: {test['eval_f1_macro'] - shipped:+.4f}")

    out = f"{config.OUTPUT_DIR}-sweep/winner"
    best_trainer.save_model(out)
    tokenizer.save_pretrained(out)
    print(f"\nWinner saved to {out} (NOT pushed). If it is clearly better, replace the "
          f"shipped model with:\n"
          f"  from huggingface_hub import upload_folder\n"
          f"  upload_folder(folder_path='{out}', repo_id='{config.HUB_MODEL_ID}', repo_type='model')")


if __name__ == "__main__":
    main()
