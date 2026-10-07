from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


# =========================
# Create BPE Tokenizer
# =========================

tokenizer = Tokenizer(
    BPE(unk_token="<UNK>")
)


# =========================
# Byte-Level Pre-tokenizer
# =========================

tokenizer.pre_tokenizer = ByteLevel(
    add_prefix_space=False
)


# =========================
# Decoder
# =========================

tokenizer.decoder = ByteLevelDecoder()


# =========================
# Trainer
# =========================

trainer = BpeTrainer(
    vocab_size=1000,
    min_frequency=2,
    special_tokens=[
        "<PAD>",
        "<UNK>",
        "<BOS>",
        "<EOS>"
    ]
)


# =========================
# Train
# =========================

tokenizer.train(
    ["data/raw/corpus.txt"],
    trainer
)


# =========================
# Save
# =========================

tokenizer.save(
    "tokenizer/tokenizer.json"
)


print("Tokenizer trained successfully!")

print(
    "Vocabulary size:",
    tokenizer.get_vocab_size()
)


# =========================
# Test
# =========================

test_text = "Data engineering uses machine learning."

output = tokenizer.encode(test_text)

print("\nTest text:")
print(test_text)

print("\nTokens:")
print(output.tokens)

print("\nToken IDs:")
print(output.ids)

print("\nDecoded:")
print(tokenizer.decode(output.ids))
