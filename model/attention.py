import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.query = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

    def forward(self, x):

        # Create Q, K and V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Attention scores
        scores = Q @ K.transpose(-2, -1)

        # Scale scores
        scores = scores / math.sqrt(self.embedding_dim)

        # Causal mask
        sequence_length = x.size(1)

        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device
            )
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # Convert scores to probabilities
        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # Weighted values
        output = attention_weights @ V

        return output


if __name__ == "__main__":

    batch_size = 1
    sequence_length = 5
    embedding_dim = 16

    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim
    )

    attention = SelfAttention(
        embedding_dim
    )

    output = attention(x)

    print("Input shape:")
    print(x.shape)

    print("\nOutput shape:")
    print(output.shape)

    print("\nAttention output:")
    print(output)