# =========================
# Model Configuration
# =========================

EMBEDDING_DIM = 64

NUM_HEADS = 4

NUM_LAYERS = 2

HIDDEN_DIM = 256

CONTEXT_LENGTH = 64


# =========================
# Training Configuration
# =========================

BATCH_SIZE = 16

LEARNING_RATE = 0.0003

EPOCHS = 30


# =========================
# Paths
# =========================

TRAINING_DATA = "data/raw/corpus.txt"

TOKENIZER_PATH = "tokenizer/tokenizer.json"

CHECKPOINT_PATH = "checkpoints/gpt_model.pth"
