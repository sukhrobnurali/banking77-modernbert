# banking77-modernbert Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune `answerdotai/ModernBERT-base` on `PolyAI/banking77` into a 77-way intent classifier, evaluate it rigorously (accuracy + macro-F1 + per-class + confusion matrix, against majority-class and frozen-encoder baselines), and publish the model + a professional card to `hf.co/sukhrobnurali/modernbert-base-banking77`.

**Architecture:** One end-to-end Colab notebook does data -> train -> push; a standalone `eval.py` reproduces the held-out test metrics + confusion matrix + baselines against the published model; `config.py` is the single source of truth for ids/hyperparameters/paths. The test split is never touched until final evaluation; a stratified 10% carve-out of train drives early stopping on eval macro-F1.

**Tech Stack:** transformers 5.10.1 (v5 API: `eval_strategy`, `processing_class`), datasets 4.8.5, torch 2.12.0, scikit-learn 1.9.0, accelerate 1.13.0, matplotlib. HF Trainer + EarlyStoppingCallback.

**Execution boundary:**
- **[Local]** tasks produce committable code/docs and are verifiable on this machine (imports, syntax, static checks).
- **[Colab]** tasks run on Colab Pro (GPU + `HF_TOKEN`). They are authored locally but their *runtime verification* (training, metrics, Hub push) happens on Colab. Numbers flow back into the card/results from `results/metrics.json`.

**Reference standard:** the model card must match `c:/projects/portfolio/uz-sentance-embedding/model_cards/uzbek-e5-small.md` (frontmatter + model-index + base-vs-fine-tuned tables + provenance + reproducibility + citation + author).

**Source of truth for design decisions:** `docs/superpowers/specs/2026-06-04-banking77-modernbert-classifier-design.md`.

---

## File structure

| File | Responsibility |
|---|---|
| `config.py` | Constants only: ids, hyperparameters, paths, seed, split fraction, smoke flags. Imported by notebook + `eval.py`. |
| `requirements.txt` | Pinned dependency versions. |
| `notebooks/banking77_modernbert.ipynb` | End-to-end Colab flow: clone+install, load, split, tokenize, train, test-eval, confusion matrix, baselines, push. |
| `eval.py` | Standalone reproducible eval: loads published model + base model, recomputes test metrics + confusion matrix + baselines, writes `results/`. Self-contained (no import from the notebook) by design. |
| `MODEL_CARD.md` | The Hub card; numeric cells filled from `results/metrics.json`. |
| `README.md` | Project overview + how to run (Colab + local eval) + results summary + links. |
| `results/` | `metrics.json`, `confusion_matrix.png`, `classification_report.txt` (committed deliverables). |

A deliberate note on DRY: `eval.py` re-implements the tiny metric/pooling helpers rather than importing them from the notebook, because the spec requires it to be *standalone and reproducible* against the published model. The duplicated surface is a ~4-line metric function and a mean-pool helper; both files import all constants from `config.py`, so there are no duplicated magic numbers.

---

## Task 1: Pinned dependencies + config [Local]

**Files:**
- Create: `requirements.txt`
- Create: `config.py`

- [ ] **Step 1: Write `requirements.txt`**

```
transformers==5.10.1
datasets==4.8.5
accelerate==1.13.0
torch==2.12.0
scikit-learn==1.9.0
matplotlib==3.10.3
huggingface-hub==0.36.0
```

(Note: on Colab, torch is preinstalled with the correct CUDA build — the notebook installs the rest without forcing the torch wheel. `requirements.txt` pins the contract for local/reproducible eval. If `huggingface-hub==0.36.0` conflicts with the resolved `transformers` extra, let pip resolve it and record the resolved version — do not invent a version.)

- [ ] **Step 2: Write `config.py`**

```python
"""Single source of truth for the banking77-modernbert classifier."""

# --- Identities ---
MODEL_ID = "answerdotai/ModernBERT-base"
DATASET_ID = "PolyAI/banking77"
HUB_MODEL_ID = "sukhrobnurali/modernbert-base-banking77"

# --- Reproducibility ---
SEED = 42

# --- Data ---
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
NUM_LABELS = 77
VAL_FRACTION = 0.10          # stratified carve-out from train, for model selection
MAX_LENGTH = 64              # token cap; verified against the 99th-pctl query length

# --- Training hyperparameters ---
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3               # early stopping usually ends it sooner
TRAIN_BATCH_SIZE = 32        # drop to 16 on a T4
EVAL_BATCH_SIZE = 64
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 2
METRIC_FOR_BEST = "f1_macro"

# --- Paths ---
OUTPUT_DIR = "outputs/modernbert-base-banking77"   # Trainer checkpoints (gitignored)
RESULTS_DIR = "results"

# --- Smoke test: cheap end-to-end validation before the full run ---
SMOKE = False
SMOKE_TRAIN_SIZE = 200
SMOKE_MAX_STEPS = 4
```

- [ ] **Step 3: Verify config imports cleanly (no heavy deps)**

Run: `python -c "import config; print(config.MODEL_ID, config.HUB_MODEL_ID, config.SEED, config.NUM_LABELS)"`
Expected: `answerdotai/ModernBERT-base sukhrobnurali/modernbert-base-banking77 42 77`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt config.py
git commit -m "Add pinned requirements and central config"
```

---

## Task 2: Notebook — setup, data load, EDA [Colab]

**Files:**
- Create: `notebooks/banking77_modernbert.ipynb` (cells added here; more in Tasks 3-5)

Author these as ordered notebook cells (use NotebookEdit). Each `# CELL` below is one notebook cell.

- [ ] **Step 1: Setup cell — clone repo, install, import, seed**

```python
# CELL 1 — setup
!git clone https://github.com/sukhrobnurali/banking77-modernbert.git
%cd banking77-modernbert
!pip install -q -r requirements.txt

import numpy as np, torch
from datasets import load_dataset
from transformers import set_seed
import config

set_seed(config.SEED)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
print("bf16 supported:", torch.cuda.is_available() and torch.cuda.is_bf16_supported())
```

- [ ] **Step 2: Data-load cell — load Banking77, build label maps**

```python
# CELL 2 — load data + label maps
# PolyAI/banking77 is script-based; datasets 4.x reads its parquet files directly.
ds = load_dataset(config.DATASET_ID)
print(ds)

label_feature = ds["train"].features[config.LABEL_COLUMN]
id2label = {i: name for i, name in enumerate(label_feature.names)}
label2id = {name: i for i, name in id2label.items()}
assert len(id2label) == config.NUM_LABELS, f"expected {config.NUM_LABELS} labels, got {len(id2label)}"
print("labels:", config.NUM_LABELS, "| example:", id2label[0])
print("train/test sizes:", len(ds["train"]), len(ds["test"]))
```

Note: if `load_dataset(config.DATASET_ID)` errors on the loading script under datasets 4.x, load the repo's parquet files directly and KEEP provenance with PolyAI:
```python
ds = load_dataset("parquet", data_files={
    "train": "hf://datasets/PolyAI/banking77/data/train-*.parquet",
    "test":  "hf://datasets/PolyAI/banking77/data/test-*.parquet",
})
```
Lock whichever call actually returns the 77-label `ClassLabel` feature; verify by running the cell.

- [ ] **Step 3: EDA cell — class balance + token-length check (validates MAX_LENGTH)**

```python
# CELL 3 — EDA: class counts + token length distribution
from collections import Counter
from transformers import AutoTokenizer

counts = Counter(ds["train"][config.LABEL_COLUMN])
print("train per-class min/max:", min(counts.values()), max(counts.values()))

tok_probe = AutoTokenizer.from_pretrained(config.MODEL_ID)
lengths = [len(x) for x in tok_probe(ds["train"][config.TEXT_COLUMN])["input_ids"]]
p99 = int(np.percentile(lengths, 99))
print(f"token length: mean={np.mean(lengths):.1f} p99={p99} max={max(lengths)}")
assert p99 <= config.MAX_LENGTH, f"raise MAX_LENGTH: p99={p99} > {config.MAX_LENGTH}"
```

Expected: 77 classes present, p99 well under 64 (banking queries are short). If the assert trips, raise `config.MAX_LENGTH` to the next power of two and re-run.

- [ ] **Step 4: Commit the notebook**

```bash
git add notebooks/banking77_modernbert.ipynb
git commit -m "Notebook: setup, Banking77 load, label maps, token-length check"
```

---

## Task 3: Notebook — stratified validation split + tokenization [Colab]

**Files:**
- Modify: `notebooks/banking77_modernbert.ipynb`

- [ ] **Step 1: Split cell — stratified 10% val carve-out from train**

```python
# CELL 4 — stratified train/val split (test stays untouched)
split = ds["train"].train_test_split(
    test_size=config.VAL_FRACTION,
    stratify_by_column=config.LABEL_COLUMN,
    seed=config.SEED,
)
train_ds, val_ds, test_ds = split["train"], split["test"], ds["test"]
print("train/val/test:", len(train_ds), len(val_ds), len(test_ds))

# sanity: val label set == full label set, and stratification roughly preserved
assert set(val_ds[config.LABEL_COLUMN]) == set(range(config.NUM_LABELS))

if config.SMOKE:   # cheap pipeline validation
    train_ds = train_ds.select(range(config.SMOKE_TRAIN_SIZE))
    val_ds = val_ds.select(range(min(len(val_ds), config.SMOKE_TRAIN_SIZE)))
```

- [ ] **Step 2: Tokenize cell — dynamic padding collator**

```python
# CELL 5 — tokenize + collator
from transformers import DataCollatorWithPadding
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)

def tokenize(batch):
    return tokenizer(batch[config.TEXT_COLUMN], truncation=True, max_length=config.MAX_LENGTH)

tok_train = train_ds.map(tokenize, batched=True, remove_columns=[config.TEXT_COLUMN])
tok_val   = val_ds.map(tokenize,   batched=True, remove_columns=[config.TEXT_COLUMN])
tok_test  = test_ds.map(tokenize,  batched=True, remove_columns=[config.TEXT_COLUMN])
collator = DataCollatorWithPadding(tokenizer=tokenizer)
print(tok_train)
```

Expected: tokenized datasets retain `label` + `input_ids` + `attention_mask`; no `text` column.

- [ ] **Step 3: Commit**

```bash
git add notebooks/banking77_modernbert.ipynb
git commit -m "Notebook: stratified val split and tokenization"
```

---

## Task 4: Notebook — model, metrics, Trainer, train [Colab]

**Files:**
- Modify: `notebooks/banking77_modernbert.ipynb`

- [ ] **Step 1: Metrics cell — accuracy + macro-F1 + weighted-F1**

```python
# CELL 6 — compute_metrics
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }
```

- [ ] **Step 2: Model cell — sequence-classification head with label maps**

```python
# CELL 7 — model
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    config.MODEL_ID,
    num_labels=config.NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
    attn_implementation="sdpa",   # safe on T4; switch to "flash_attention_2" on A100/L4
)
```

- [ ] **Step 3: Trainer cell — v5 TrainingArguments + early stopping (train)**

```python
# CELL 8 — TrainingArguments + Trainer + train
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
args = TrainingArguments(
    output_dir=config.OUTPUT_DIR,
    learning_rate=config.LEARNING_RATE,
    num_train_epochs=config.NUM_EPOCHS,
    per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
    warmup_ratio=config.WARMUP_RATIO,
    weight_decay=config.WEIGHT_DECAY,
    eval_strategy="epoch",            # v5 name (was evaluation_strategy)
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model=config.METRIC_FOR_BEST,
    greater_is_better=True,
    logging_steps=50,
    bf16=use_bf16,
    fp16=torch.cuda.is_available() and not use_bf16,
    seed=config.SEED,
    push_to_hub=True,
    hub_model_id=config.HUB_MODEL_ID,
    report_to="none",                 # W&B off by default
    max_steps=config.SMOKE_MAX_STEPS if config.SMOKE else -1,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tok_train,
    eval_dataset=tok_val,
    processing_class=tokenizer,       # v5 name (was tokenizer=)
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING_PATIENCE)],
)
trainer.train()
```

- [ ] **Step 2 (gate): Run a SMOKE pass first**

Set `SMOKE = True` in `config.py`, run Cells 1-8. Expected: training completes in seconds, per-epoch eval logs show `eval_f1_macro`, no API errors (confirms the v5 kwargs `eval_strategy`/`processing_class` are correct). Then set `SMOKE = False` and run the full train. Expected (full): eval macro-F1 climbs across epochs to roughly 0.92-0.94; early stopping restores the best checkpoint.

- [ ] **Step 3: Commit**

```bash
git add notebooks/banking77_modernbert.ipynb
git commit -m "Notebook: model, metrics, v5 Trainer with early stopping"
```

---

## Task 5: Notebook — test eval, confusion matrix, baselines, push [Colab]

**Files:**
- Modify: `notebooks/banking77_modernbert.ipynb`
- Create (at runtime): `results/metrics.json`, `results/confusion_matrix.png`, `results/classification_report.txt`

- [ ] **Step 1: Test-eval cell — held-out predictions + headline metrics**

```python
# CELL 9 — held-out test predictions
import os, json
from sklearn.metrics import classification_report, confusion_matrix

pred = trainer.predict(tok_test)
y_true = pred.label_ids
y_pred = pred.predictions.argmax(-1)

ft = {
    "accuracy": accuracy_score(y_true, y_pred),
    "f1_macro": f1_score(y_true, y_pred, average="macro"),
    "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
}
print("fine-tuned (test):", ft)
```

- [ ] **Step 2: Confusion matrix + classification report cell**

```python
# CELL 10 — confusion matrix + per-class report
import matplotlib.pyplot as plt
os.makedirs(config.RESULTS_DIR, exist_ok=True)
target_names = [id2label[i] for i in range(config.NUM_LABELS)]

report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
with open(f"{config.RESULTS_DIR}/classification_report.txt", "w") as f:
    f.write(report)

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(cm, cmap="Blues")
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Banking77 confusion matrix")
fig.colorbar(im); fig.tight_layout()
fig.savefig(f"{config.RESULTS_DIR}/confusion_matrix.png", dpi=120)

# top confusable off-diagonal pairs for the card narrative
import numpy as np
cm_off = cm.copy(); np.fill_diagonal(cm_off, 0)
pairs = np.dstack(np.unravel_index(np.argsort(cm_off.ravel())[::-1][:8], cm.shape))[0]
print("top confusions:", [(id2label[i], id2label[j], int(cm[i, j])) for i, j in pairs])
```

- [ ] **Step 3: Baselines cell — majority class + frozen-encoder linear probe**

```python
# CELL 11 — baselines
from collections import Counter
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel

# (a) majority-class baseline
majority = Counter(train_ds[config.LABEL_COLUMN]).most_common(1)[0][0]
maj_pred = np.full_like(y_true, majority)
maj = {
    "accuracy": accuracy_score(y_true, maj_pred),
    "f1_macro": f1_score(y_true, maj_pred, average="macro", zero_division=0),
    "f1_weighted": f1_score(y_true, maj_pred, average="weighted", zero_division=0),
}

# (b) frozen-encoder linear probe: mean-pooled base embeddings -> logistic regression
base = AutoModel.from_pretrained(config.MODEL_ID, attn_implementation="sdpa").to(model.device).eval()

@torch.no_grad()
def embed(dataset):
    feats = []
    loader = trainer.get_eval_dataloader(dataset)   # reuse collator/batching
    for batch in loader:
        batch = {k: v.to(base.device) for k, v in batch.items() if k != "labels"}
        out = base(**batch).last_hidden_state                      # (B, T, H)
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # mean pool
        feats.append(pooled.float().cpu().numpy())
    return np.concatenate(feats)

Xtr, ytr = embed(tok_train), np.array(train_ds[config.LABEL_COLUMN])
Xte = embed(tok_test)
probe = LogisticRegression(max_iter=2000, n_jobs=-1).fit(Xtr, ytr)
pp = probe.predict(Xte)
frozen = {
    "accuracy": accuracy_score(y_true, pp),
    "f1_macro": f1_score(y_true, pp, average="macro"),
    "f1_weighted": f1_score(y_true, pp, average="weighted"),
}
print("majority:", maj); print("frozen probe:", frozen); print("fine-tuned:", ft)
```

- [ ] **Step 4: Persist metrics cell — write `results/metrics.json`**

```python
# CELL 12 — write metrics.json (source of truth for the card)
metrics = {
    "dataset": config.DATASET_ID,
    "base_model": config.MODEL_ID,
    "test_size": int(len(y_true)),
    "majority_class": {"name": id2label[majority], **maj},
    "frozen_linear_probe": frozen,
    "fine_tuned": ft,
}
with open(f"{config.RESULTS_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2))
```

- [ ] **Step 5: Push cell — model + tokenizer to the Hub**

```python
# CELL 13 — push model + tokenizer to the Hub
trainer.push_to_hub(commit_message="Fine-tuned ModernBERT-base on Banking77")
```

Expected: `hf.co/sukhrobnurali/modernbert-base-banking77` exists with model weights + tokenizer; widget shows intent names (not `LABEL_n`) thanks to the id2label maps.

- [ ] **Step 6: Commit notebook + results**

```bash
git add notebooks/banking77_modernbert.ipynb results/
git commit -m "Notebook: test eval, confusion matrix, baselines, Hub push; record results"
```

---

## Task 6: Standalone reproducible eval script [Local author / Colab or GPU run]

**Files:**
- Create: `eval.py`

- [ ] **Step 1: Write `eval.py`**

```python
"""Standalone reproducible eval for modernbert-base-banking77.

Loads the published fine-tuned model and the base model, recomputes held-out
test metrics, per-class report, confusion matrix, and the majority + frozen
linear-probe baselines, and writes results/. Run on a GPU box (Colab Pro).
"""
import os, json
import numpy as np, torch
from collections import Counter
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from transformers import (AutoTokenizer, AutoModel,
                          AutoModelForSequenceClassification, set_seed)
import config


def load_data():
    ds = load_dataset(config.DATASET_ID)
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

    # fine-tuned
    y_pred = predict_finetuned(test_ds, tokenizer, device)
    ft = scores(y_true, y_pred)

    # per-class report + confusion matrix
    with open(f"{config.RESULTS_DIR}/classification_report.txt", "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=names, digits=4))
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.imshow(cm, cmap="Blues"); ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.tight_layout(); fig.savefig(f"{config.RESULTS_DIR}/confusion_matrix.png", dpi=120)

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
```

- [ ] **Step 2: Static check (no GPU needed)**

Run: `python -c "import ast; ast.parse(open('eval.py').read()); print('ok')"`
Expected: `ok` (syntax valid). Full execution is a [Colab]/GPU step that must reproduce the notebook's `metrics.json` within float tolerance.

- [ ] **Step 3: Commit**

```bash
git add eval.py
git commit -m "Add standalone reproducible eval script"
```

---

## Task 7: Model card [Local — numbers from results/metrics.json]

**Files:**
- Create: `MODEL_CARD.md`

This task runs AFTER the Colab run produced `results/metrics.json`. Fill the numeric cells from that file — do not invent numbers. The structure mirrors `uzbek-e5-small.md`.

- [ ] **Step 1: Write `MODEL_CARD.md` (prose complete; numbers read from metrics.json)**

````markdown
---
language:
- en
license: apache-2.0
library_name: transformers
pipeline_tag: text-classification
base_model: answerdotai/ModernBERT-base
datasets:
- PolyAI/banking77
tags:
- text-classification
- intent-classification
- banking77
- modernbert
metrics:
- accuracy
- f1
model-index:
- name: modernbert-base-banking77
  results:
  - task:
      type: text-classification
      name: Intent classification
    dataset:
      type: PolyAI/banking77
      name: Banking77 (test, 3080)
    metrics:
    - type: accuracy
      value: <fine_tuned.accuracy from metrics.json>
      name: Accuracy
    - type: f1
      value: <fine_tuned.f1_macro from metrics.json>
      name: Macro-F1
---

# modernbert-base-banking77

A fine-tune of [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base)
for **77-way English banking-intent classification**, trained on
[`PolyAI/banking77`](https://huggingface.co/datasets/PolyAI/banking77).

- **Base model:** `answerdotai/ModernBERT-base` (149M params, Apache-2.0)
- **Task:** single-label intent detection over 77 fine-grained banking intents
- **Held-out test:** the official Banking77 `test` split (3,080 queries, 40/intent), disjoint from training
- **Training code:** https://github.com/sukhrobnurali/banking77-modernbert

## Intended use

Routing short English banking/customer-support queries to one of 77 intents
(e.g. `card_arrival`, `lost_or_stolen_card`, `exchange_rate`) for task-oriented
dialog and triage.

## Out of scope

- Non-English text and non-banking domains (single-domain training data).
- Out-of-scope / rejection detection: every input is forced into one of 77 intents.
- Long documents — trained at `max_length=64` on short single-sentence queries.

## Evaluation

Same protocol for every row of the table; all numbers are on the **untouched test split**.
Baselines contextualize the fine-tune's gain: a majority-class floor and a
frozen-encoder linear probe (mean-pooled base-model embeddings -> logistic regression).

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| Majority class | <majority_class.accuracy> | <majority_class.f1_macro> | <majority_class.f1_weighted> |
| Frozen ModernBERT + linear probe | <frozen_linear_probe.accuracy> | <frozen_linear_probe.f1_macro> | <frozen_linear_probe.f1_weighted> |
| **This model (fine-tuned)** | **<fine_tuned.accuracy>** | **<fine_tuned.f1_macro>** | **<fine_tuned.f1_weighted>** |

Per-class precision/recall/F1 are in
[`results/classification_report.txt`](https://github.com/sukhrobnurali/banking77-modernbert/blob/main/results/classification_report.txt);
the 77x77 confusion matrix is
[`results/confusion_matrix.png`](https://github.com/sukhrobnurali/banking77-modernbert/blob/main/results/confusion_matrix.png).
The remaining errors cluster among semantically adjacent intents
(insert the top confusable pairs printed by Cell 10, e.g. `card_arrival` vs
`card_delivery_estimate`).

## Usage

```python
from transformers import pipeline

clf = pipeline("text-classification", model="sukhrobnurali/modernbert-base-banking77")
clf("My card still hasn't arrived, when will I get it?")
# -> [{'label': 'card_arrival', 'score': 0.99}]
```

## Training data

[`PolyAI/banking77`](https://huggingface.co/datasets/PolyAI/banking77) (Casanueva
et al. 2020, arXiv:2003.04807): 10,003 train / 3,080 test online-banking queries
over 77 intents. License **CC-BY-4.0**. A stratified 10% of train was held out as
a validation set for early stopping; the official test split was used only for the
final numbers above.

## Reproducibility

Fixed `seed=42`. Fine-tuned with the HF Trainer, `lr=2e-5`, up to 3 epochs with
early stopping (patience 2) on validation macro-F1, `max_length=64`, batch size 32,
warmup ratio 0.1, weight decay 0.01. Pinned versions in `requirements.txt`
(transformers 5.10.1). `eval.py` reproduces the test numbers from the published
model.

## License

Apache-2.0, inherited from the base model `answerdotai/ModernBERT-base`. Training
data `PolyAI/banking77` is CC-BY-4.0 (attribution: Casanueva et al. 2020).

## Citation

```bibtex
@misc{nurali_modernbert_banking77_2026,
  author       = {Sukhrob Nurali},
  title        = {modernbert-base-banking77: a ModernBERT intent classifier for Banking77},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/sukhrobnurali/modernbert-base-banking77}}
}
```

## Author

- Sukhrob Nurali
- Hugging Face: [sukhrobnurali](https://huggingface.co/sukhrobnurali)
- GitHub: [sukhrobnurali](https://github.com/sukhrobnurali)
````

- [ ] **Step 2: Verify no angle-bracket placeholders remain**

Run: `python -c "import sys; t=open('MODEL_CARD.md',encoding='utf-8').read(); sys.exit('PLACEHOLDERS LEFT' if '<' in t and 'metrics.json' in t else 'ok')"`
Expected: exits `ok` — every `<...from metrics.json>` cell has been replaced with a real number and the confusable-pairs sentence filled from Cell 10's output.

- [ ] **Step 3: Commit**

```bash
git add MODEL_CARD.md
git commit -m "Add model card with eval results and base-vs-fine-tuned table"
```

---

## Task 8: README + push card to Hub [Local author / Colab for Hub upload]

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# banking77-modernbert

Fine-tune of [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base)
into a 77-way English banking-intent classifier on
[Banking77](https://huggingface.co/datasets/PolyAI/banking77).

- **Model:** https://huggingface.co/sukhrobnurali/modernbert-base-banking77
- **Card:** [MODEL_CARD.md](MODEL_CARD.md)
- **Design + plan:** [docs/superpowers/](docs/superpowers/)

## Results (held-out test, 3,080 queries)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Majority class | ... | ... |
| Frozen encoder + linear probe | ... | ... |
| **Fine-tuned** | **...** | **...** |

(Fill from `results/metrics.json`.)

## Reproduce

```bash
pip install -r requirements.txt
# Train + push: open notebooks/banking77_modernbert.ipynb on Colab Pro (set HF token).
# Reproduce eval against the published model:
python eval.py
```

Fixed seed 42; hyperparameters in `config.py`. See `MODEL_CARD.md` for the full
write-up, per-class breakdown, and confusion matrix.
```

- [ ] **Step 2: Verify links resolve locally**

Run: `python -c "import os; assert all(os.path.exists(p) for p in ['MODEL_CARD.md','eval.py','config.py','requirements.txt']); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Push the card to the Hub as the model README [Colab]**

```python
from huggingface_hub import upload_file
upload_file(path_or_fileobj="MODEL_CARD.md", path_in_repo="README.md",
            repo_id=config.HUB_MODEL_ID, repo_type="model",
            commit_message="Add model card")
```

Expected: the Hub model page renders the card; the `model-index` populates the metrics widget.

- [ ] **Step 4: Commit + push repo**

```bash
git add README.md
git commit -m "Add project README with results summary and run instructions"
git push
```

---

## Self-review

**Spec coverage:**
- Held-out disjoint test split -> Task 3 (test untouched) + Task 5 (eval on test). ✓
- Accuracy + macro-F1 + per-class + confusion matrix -> Task 5 Steps 1-2, Task 6. ✓
- Majority baseline + frozen-encoder probe -> Task 5 Step 3, Task 6. ✓
- Professional card matching uzbek-e5-small -> Task 7. ✓
- Reproducible: fixed seed, documented hyperparameters, committed eval script -> config.py, Task 4, Task 6. ✓
- HF Trainer, 2-3 epochs, lr 2e-5, early stopping on macro-F1 -> Task 4. ✓
- Auto-push model + card to Hub -> Task 5 Step 5, Task 8 Step 3. ✓
- Latest stable versions -> Task 1. ✓
- Notebook + eval script + card deliverables -> Tasks 2-5, 6, 7. ✓

**Placeholder scan:** The only intentional `<...>` markers are the metrics cells in Task 7 / README, each bound to a named `metrics.json` field and gated by a verification step (Task 7 Step 2). No "TBD/implement later" anywhere.

**Type consistency:** `compute_metrics` returns `accuracy`/`f1_macro`/`f1_weighted`; `config.METRIC_FOR_BEST="f1_macro"` matches the returned key (Trainer prepends `eval_`). `config.TEXT_COLUMN`/`LABEL_COLUMN` used consistently across notebook + eval.py. `scores()` in eval.py returns the same three keys written by the notebook's `metrics` dict, so `metrics.json` schema is identical from both paths.
