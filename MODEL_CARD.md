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
      value: 0.9175
      name: Accuracy
    - type: f1
      value: 0.9176
      name: Macro-F1
---

# modernbert-base-banking77

A fine-tune of [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base)
for **77-way English banking-intent classification**, trained on
[`PolyAI/banking77`](https://huggingface.co/datasets/PolyAI/banking77).

- **Base model:** `answerdotai/ModernBERT-base` (149M params, Apache-2.0)
- **Task:** single-label intent detection over 77 fine-grained banking intents
- **Held-out test:** the official Banking77 `test` split (3,080 queries, 40/intent), disjoint from training
- **Result:** 0.9175 accuracy / 0.9176 macro-F1 on the test split
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

The same protocol is applied to every row of the table, all on the **untouched test split**.
Two baselines contextualize the fine-tune's gain: a majority-class floor, and a
frozen-encoder linear probe (mean-pooled base-model embeddings → logistic regression).

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| Majority class | 0.0130 | 0.0003 | 0.0003 |
| Frozen ModernBERT + linear probe | 0.8906 | 0.8906 | 0.8906 |
| **This model (fine-tuned)** | **0.9175** | **0.9176** | **0.9176** |

ModernBERT's pretrained representations are already strong on this task — a linear probe
on frozen embeddings reaches 0.8906 macro-F1 — so fine-tuning adds a consistent **+2.7
points** of macro-F1 (0.8906 → 0.9176) on top of that, well clear of the 0.0003
majority-class floor. At ~0.92 accuracy the residual errors fall among semantically
adjacent intents; the per-class breakdown
([`results/classification_report.txt`](https://github.com/sukhrobnurali/banking77-modernbert/blob/main/results/classification_report.txt))
and the 77×77 confusion matrix
([`results/confusion_matrix.png`](https://github.com/sukhrobnurali/banking77-modernbert/blob/main/results/confusion_matrix.png))
pinpoint the specific clusters.

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
