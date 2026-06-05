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

(Filled from `results/metrics.json` after the training run.)

## Reproduce

**Colab (train + push):** open `notebooks/banking77_modernbert.ipynb` on Colab Pro,
select a GPU runtime, and run top to bottom. Cell 1 installs the pinned HF libraries on
top of Colab's preinstalled torch — it does **not** reinstall torch (doing so would break
Colab's bundled torchvision). CELL 1b prompts you to paste an HF write token for the push.

**Local (eval against the published model):**

```bash
pip install -r requirements.txt
python eval.py
```

Fixed seed 42; hyperparameters in `config.py`. See `MODEL_CARD.md` for the full
write-up, per-class breakdown, and confusion matrix.
