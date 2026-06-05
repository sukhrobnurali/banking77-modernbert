"""Standalone reproducible eval for modernbert-base-banking77.

Loads the published fine-tuned model and the base model, recomputes held-out
test metrics, per-class report, confusion matrix, and the majority + frozen
linear-probe baselines, and writes results/. Run on a GPU box (Colab Pro).
"""
import os
import json
from collections import Counter

import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)
from transformers import (AutoTokenizer, AutoModel,
                          AutoModelForSequenceClassification, set_seed)

import config


def load_data():
    ds = load_dataset(config.DATASET_ID, revision=config.DATASET_REVISION)
    names = ds["train"].features[config.LABEL_COLUMN].names
    return ds["train"], ds["test"], names


@torch.no_grad()
def predict_finetuned(test_ds, tokenizer, device):
    clf = AutoModelForSequenceClassification.from_pretrained(config.HUB_MODEL_ID).to(device).eval()
    preds = []
    for i in range(0, len(test_ds), config.EVAL_BATCH_SIZE):
        batch = test_ds[i:i + config.EVAL_BATCH_SIZE][config.TEXT_COLUMN]
        enc = tokenizer(batch, truncation=True, max_length=config.MAX_LENGTH,
                        padding=True, return_tensors="pt").to(device)
        preds.append(clf(**enc).logits.argmax(-1).cpu().numpy())
    return np.concatenate(preds)


@torch.no_grad()
def embed(texts, base, tokenizer, device):
    feats = []
    for i in range(0, len(texts), config.EVAL_BATCH_SIZE):
        enc = tokenizer(texts[i:i + config.EVAL_BATCH_SIZE], truncation=True,
                        max_length=config.MAX_LENGTH, padding=True, return_tensors="pt").to(device)
        out = base(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        feats.append(pooled.float().cpu().numpy())
    return np.concatenate(feats)


def scores(y_true, y_pred, **kw):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", **kw),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", **kw),
    }


def main():
    set_seed(config.SEED)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, test_ds, names = load_data()
    y_true = np.array(test_ds[config.LABEL_COLUMN])
    tokenizer = AutoTokenizer.from_pretrained(config.HUB_MODEL_ID)

    # fine-tuned model on the held-out test split
    y_pred = predict_finetuned(test_ds, tokenizer, device)
    ft = scores(y_true, y_pred)

    # per-class report + confusion matrix
    with open(f"{config.RESULTS_DIR}/classification_report.txt", "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=names, digits=4))
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(f"{config.RESULTS_DIR}/confusion_matrix.png", dpi=120)

    # baselines
    majority = Counter(train_ds[config.LABEL_COLUMN]).most_common(1)[0][0]
    maj = scores(y_true, np.full_like(y_true, majority), zero_division=0)
    base = AutoModel.from_pretrained(config.MODEL_ID, attn_implementation="sdpa").to(device).eval()
    probe = LogisticRegression(max_iter=2000, n_jobs=-1).fit(
        embed(train_ds[config.TEXT_COLUMN], base, tokenizer, device),
        np.array(train_ds[config.LABEL_COLUMN]))
    frozen = scores(y_true, probe.predict(embed(test_ds[config.TEXT_COLUMN], base, tokenizer, device)))

    metrics = {"dataset": config.DATASET_ID, "base_model": config.MODEL_ID,
               "test_size": int(len(y_true)),
               "majority_class": {"name": names[majority], **maj},
               "frozen_linear_probe": frozen, "fine_tuned": ft}
    with open(f"{config.RESULTS_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
