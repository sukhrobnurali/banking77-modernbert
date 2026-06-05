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
| Majority class | 0.0130 | 0.0003 |
| Frozen encoder + linear probe | 0.8945 | 0.8948 |
| **Fine-tuned** | **0.9399** | **0.9401** |

Fine-tuning adds +4.5 points of macro-F1 over the frozen-encoder probe. Full numbers in
[`results/metrics.json`](results/metrics.json).

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

Optional: `python sweep.py` runs a quick learning-rate sweep that selects the winner on
the validation split and evaluates it once on the held-out test (no Hub push).
