from tokenizer.tokenizer import BPETokenizer

TOKENIZER_PATH = "tokenizer/tokenizer.json"
CORPUS_PATH = "data/raw/corpus.txt"

TRAIN_PATH = "data/raw/train_tokens.txt"
VAL_PATH = "data/raw/val_tokens.txt"

tokenizer = BPETokenizer(TOKENIZER_PATH)

with open(CORPUS_PATH, "r") as file:
    text = file.read()

token_ids = tokenizer.encode(text)

split_index = int(len(token_ids) * 0.9)

train_ids = token_ids[:split_index]
val_ids = token_ids[split_index:]

with open(TRAIN_PATH, "w") as file:
    file.write(" ".join(map(str, train_ids)))

with open(VAL_PATH, "w") as file:
    file.write(" ".join(map(str, val_ids)))

print("Dataset split complete!")
print("Total tokens:", len(token_ids))
print("Training tokens:", len(train_ids))
print("Validation tokens:", len(val_ids))