import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

    def forward(self, token_ids):
        return self.embedding(token_ids)


if __name__ == "__main__":

    vocab_size = 20
    embedding_dim = 16

    model = TokenEmbedding(
        vocab_size,
        embedding_dim
    )

    token_ids = torch.tensor([
        [1, 6, 9, 18, 16]
    ])

    embeddings = model(token_ids)

    print("Token IDs:")
    print(token_ids)

    print("\nEmbedding shape:")
    print(embeddings.shape)

    print("\nEmbeddings:")
    print(embeddings)