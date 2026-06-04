# Design: `modernbert-base-banking77` — English intent classifier

- **Date:** 2026-06-04
- **Status:** Approved (design); spec under review
- **Owner:** Sukhrob Nurali (`hf.co/sukhrobnurali`)

## Goal

Fine-tune a small modern encoder into a rigorously evaluated English intent
classifier and publish it (model + card) to the Hub. This is a deliberately
mainstream English piece to show range alongside the Uzbek NLP work; it must be
small, fast on Colab Pro, and held to the same evaluation/model-card standard as
`uzbek-e5-small`.

## Locked decisions

| Decision | Choice | Notes (verified 2026-06-04 against live HF/PyPI) |
|---|---|---|
| Task | 77-way single-label intent classification | Multi-class, non-trivial, useful (task-oriented dialog / chatbots) |
| Dataset | `PolyAI/banking77` | **CC-BY-4.0**, ungated; train 10,003 / test 3,080; no val split |
| Base model | `answerdotai/ModernBERT-base` | **apache-2.0**, 149M params, GLUE 88.4 (> BERT 84.7 / RoBERTa 86.4) |
| Hub repo | `sukhrobnurali/modernbert-base-banking77` | |
| Logging | W&B off by default, opt-in flag | Keeps the notebook simple |

### Pinned stack (latest stable, mid-2026)

`transformers==5.10.1`, `datasets==4.8.5`, `evaluate==0.4.6` (optional — sklearn
is the primary metric path), `accelerate==1.13.0`, `torch==2.12.0`,
`scikit-learn==1.9.0`, plus `pandas`, `matplotlib` (confusion-matrix plot).

**transformers v5 API notes (this is a v5 major line):**
- Eval cadence kwarg is `eval_strategy` — the old `evaluation_strategy` was
  removed at 4.46 and raises `TypeError` on v5.
- `Trainer` takes `processing_class=tokenizer`, not `tokenizer=`.
- scikit-learn 1.9.0 requires Python >= 3.11 (Colab is fine).

## 1. Data pipeline

- **Load** `PolyAI/banking77`. The repo is script-based and `datasets` 4.x does
  not auto-run loading scripts; at implementation, lock the exact call that works
  on 4.8.5 by loading the repo's own parquet files (not a third-party mirror, so
  provenance stays with PolyAI). Verify by running, not assuming.
- **Held-out test (requirement #1):** the official `test` split (3,080 rows,
  exactly 40 per intent) is untouched until final evaluation.
- **Validation:** Banking77 has no val split, so carve **10% stratified out of
  `train`** with `seed=42` for early stopping and model selection. Training fits
  on the remaining ~90% of train only; macro-F1 on the val carve-out drives
  selection.
- **Labels:** build `id2label`/`label2id` from the dataset features (77 intent
  names) so the Hub widget shows names, not `LABEL_n`.
- **Tokenization:** ModernBERT tokenizer, `max_length=64` (short single-sentence
  queries), dynamic padding via `DataCollatorWithPadding`. Confirm the 99th-pctl
  token length is < 64 against the actual data; raise only if needed.

## 2. Model & training (HF Trainer)

`AutoModelForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base",
num_labels=77, id2label=..., label2id=...)`.

**Hyperparameters** (in `config.py`, single source of truth — no scattered
magic numbers):

| Knob | Value |
|---|---|
| `epochs` | 3 (early-stopped) |
| `learning_rate` | 2e-5 |
| `per_device_train_batch_size` | 32 (16 on T4) |
| `per_device_eval_batch_size` | 64 |
| `warmup_ratio` | 0.1 |
| `weight_decay` | 0.01 |
| `max_length` | 64 |
| precision | `bf16` on Ampere+, `fp16` on T4 |
| `attn_implementation` | `sdpa` default; FA2 opt-in on A100/L4 |
| `seed` | 42 |

**Trainer / selection:** `eval_strategy="epoch"`, `save_strategy="epoch"`,
`load_best_model_at_end=True`, `metric_for_best_model="f1_macro"`,
`greater_is_better=True`, `EarlyStoppingCallback(early_stopping_patience=2)`,
`processing_class=tokenizer`, `push_to_hub=True`,
`hub_model_id="sukhrobnurali/modernbert-base-banking77"`.

**`compute_metrics`** returns `{"accuracy", "f1_macro", "f1_weighted"}` via
sklearn (`f1_score(..., average="macro"/"weighted")`).

## 3. Evaluation & baselines

Final evaluation runs on the **held-out test split** (requirement #2):

- **Headline metrics:** accuracy, macro-F1, weighted-F1.
- **Per-class breakdown:** full `classification_report` (precision/recall/F1 per
  intent).
- **Confusion matrix:** 77x77, saved as `results/confusion_matrix.png`; the top
  confusable intent pairs are extracted and called out in the card
  (e.g. `card_arrival` vs `card_not_working`). This is the analysis centerpiece.

**Baselines for context (requirement #3):**

1. **Majority-class** — always predict the most frequent train intent. Anchors
   the floor (~1.3% accuracy, near-zero macro-F1 across 77 classes).
2. **Frozen-encoder linear probe** — extract frozen ModernBERT embeddings
   (mean-pooled last hidden state) over train, fit a sklearn `LogisticRegression`
   (`max_iter` high enough to converge), evaluate on test. Deterministic, cheap;
   isolates the gain from fine-tuning vs the pretrained representation.

The card presents a **base/frozen vs fine-tuned** delta table in the same format
as the `uzbek-e5-small` card. Published Banking77 reference (~0.93 accuracy) is
cited for external context.

## 4. Deliverables & repo layout

```
2-classifier-model/
  README.md            # project overview + how to run
  MODEL_CARD.md        # Hub card (matches uzbek-e5-small standard)
  config.py            # all hyperparams / paths / ids
  requirements.txt     # pinned versions (see stack above)
  notebooks/
    banking77_modernbert.ipynb   # end-to-end Colab: data -> train -> push
  eval.py              # standalone reproducible eval: Hub model -> metrics + confusion matrix
  results/
    metrics.json
    confusion_matrix.png
    classification_report.txt
```

- The **notebook** is the primary one-file Colab artifact (end-to-end).
- **`eval.py`** is a standalone deliverable (per the brief's separate eval-script
  requirement) that reproduces test metrics + confusion matrix against the
  published model. It imports constants from `config.py`; no further plumbing.

## 5. Model card (matches `uzbek-e5-small` standard)

- **YAML frontmatter:** `license: apache-2.0` (from base), `base_model:
  answerdotai/ModernBERT-base`, `datasets: [PolyAI/banking77]`, `language: [en]`,
  `pipeline_tag: text-classification`, tags, and a `model-index` with test
  accuracy + macro-F1.
- **Body:** TL;DR; intended use; out-of-scope; limitations; dataset provenance &
  license (Casanueva et al. 2020, arXiv:2003.04807, CC-BY-4.0); base/frozen vs
  fine-tuned eval table + per-class/confusable-cluster note; usage snippet;
  reproducibility (seed 42, pinned versions, hyperparameter table); citation;
  author block.

## Reproducibility

Fixed `seed=42` (set via `transformers.set_seed`, also seeds the stratified
train/val split). All hyperparameters and ids live in `config.py`. Pinned
versions in `requirements.txt`. `eval.py` reproduces the reported test numbers
from the published model.

## Out of scope (YAGNI)

- No second full fine-tune (DistilBERT/DeBERTa comparison) — the frozen linear
  probe + majority baseline provide the contextualizing comparison.
- No out-of-scope / rejection handling (that was the CLINC150 path).
- No hyperparameter search or LR/seed ablations beyond the documented defaults.
- No multilingual or cross-domain generalization claims.

## Open items to resolve during implementation

1. Exact `datasets` 4.8.5 load call for `PolyAI/banking77` (parquet path) —
   verify by running.
2. Confirm 99th-pctl token length < 64 on the real data; raise `max_length`
   only if exceeded.
3. Confirm Colab Pro GPU tier at run time (L4/A100 enables bf16 + FA2; T4 forces
   fp16 + sdpa and `train_bs=16`).
