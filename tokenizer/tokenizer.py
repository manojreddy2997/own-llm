from tokenizers import Tokenizer


class BPETokenizer:

    def __init__(self, tokenizer_path):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()


if __name__ == "__main__":

    tokenizer = BPETokenizer(
        "tokenizer/tokenizer.json"
    )

    text = "Data engineering is powerful."

    token_ids = tokenizer.encode(text)

    print("Text:")
    print(text)

    print("\nToken IDs:")
    print(token_ids)

    print("\nDecoded:")
    print(tokenizer.decode(token_ids))

    print("\nVocabulary size:")
    print(tokenizer.get_vocab_size())
