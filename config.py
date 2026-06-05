"""Single source of truth for the banking77-modernbert classifier."""

# --- Identities ---
MODEL_ID = "answerdotai/ModernBERT-base"
DATASET_ID = "PolyAI/banking77"
# PolyAI/banking77 ships a loader script that datasets 4.x rejects. Load HF's
# auto-converted parquet branch of the same first-party repo (no script executed).
DATASET_REVISION = "refs/convert/parquet"
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

# --- Quick LR sweep (sweep.py): select on val macro-F1, eval winner once on test ---
LR_SWEEP = [3e-5, 5e-5, 8e-5]
SWEEP_EPOCHS = 4

# --- Paths ---
OUTPUT_DIR = "outputs/modernbert-base-banking77"   # Trainer checkpoints (gitignored)
RESULTS_DIR = "results"

# --- Smoke test: cheap end-to-end validation before the full run ---
SMOKE = False
SMOKE_TRAIN_SIZE = 200
SMOKE_MAX_STEPS = 4
