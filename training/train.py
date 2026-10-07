import torch
from torch.utils.data import DataLoader

from tokenizer.tokenizer import BPETokenizer
from training.dataset import LanguageModelDataset
from model.gpt import GPT

import config


# -------------------------
# Tokenizer
# -------------------------

tokenizer = BPETokenizer(
    config.TOKENIZER_PATH
)

vocab_size = tokenizer.get_vocab_size()

print("Vocabulary size:", vocab_size)


# -------------------------
# Load token data
# -------------------------

with open("data/raw/train_tokens.txt", "r") as file:
    train_ids = list(map(int, file.read().split()))

with open("data/raw/val_tokens.txt", "r") as file:
    val_ids = list(map(int, file.read().split()))


print("Training tokens:", len(train_ids))
print("Validation tokens:", len(val_ids))


# -------------------------
# Dataset
# -------------------------

train_dataset = LanguageModelDataset(
    train_ids,
    context_length=config.CONTEXT_LENGTH
)

val_dataset = LanguageModelDataset(
    val_ids,
    context_length=config.CONTEXT_LENGTH
)


print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# -------------------------
# DataLoader
# -------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False
)


# -------------------------
# Model
# -------------------------

model = GPT(
    vocab_size=vocab_size,
    embedding_dim=config.EMBEDDING_DIM,
    num_heads=config.NUM_HEADS,
    num_layers=config.NUM_LAYERS,
    hidden_dim=config.HIDDEN_DIM,
    max_context_length=config.CONTEXT_LENGTH
)


# -------------------------
# Loss
# -------------------------

loss_function = torch.nn.CrossEntropyLoss()


# -------------------------
# Optimizer
# -------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.LEARNING_RATE
)


# -------------------------
# Learning rate scheduler
# -------------------------

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config.EPOCHS
)


# -------------------------
# Early stopping
# -------------------------

best_val_loss = float("inf")

patience = 3

epochs_without_improvement = 0


# -------------------------
# Training
# -------------------------

for epoch in range(config.EPOCHS):

    # =====================
    # Training
    # =====================

    model.train()

    total_train_loss = 0

    for inputs, targets in train_loader:

        optimizer.zero_grad()

        logits = model(inputs)

        loss = loss_function(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1)
        )

        loss.backward()

        optimizer.step()

        total_train_loss += loss.item()


    average_train_loss = (
        total_train_loss / len(train_loader)
    )


    # =====================
    # Validation
    # =====================

    model.eval()

    total_val_loss = 0

    with torch.no_grad():

        for inputs, targets in val_loader:

            logits = model(inputs)

            loss = loss_function(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1)
            )

            total_val_loss += loss.item()


    average_val_loss = (
        total_val_loss / len(val_loader)
    )


    # =====================
    # Print metrics
    # =====================

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch + 1}/{config.EPOCHS} "
        f"| Train Loss: {average_train_loss:.4f} "
        f"| Val Loss: {average_val_loss:.4f} "
        f"| LR: {current_lr:.6f}"
    )


    # =====================
    # Best model
    # =====================

    if average_val_loss < best_val_loss:

        best_val_loss = average_val_loss

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            config.CHECKPOINT_PATH
        )

        print(
            f"✓ Best model saved! "
            f"Validation Loss: {best_val_loss:.4f}"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement "
            f"({epochs_without_improvement}/{patience})"
        )


    # =====================
    # Scheduler
    # =====================

    scheduler.step()


    # =====================
    # Early stopping
    # =====================

    if epochs_without_improvement >= patience:

        print("\nEarly stopping triggered.")

        break


# -------------------------
# Training complete
# -------------------------

print("\nTraining complete!")

print(
    f"Best Validation Loss: {best_val_loss:.4f}"
)

print(
    f"Best model saved to: {config.CHECKPOINT_PATH}"
)
