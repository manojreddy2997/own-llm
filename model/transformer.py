import torch
import torch.nn as nn

from model. multi_head_attention import MultiHeadAttention


class FeedForward(nn.Module):

    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        hidden_dim
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            embedding_dim,
            num_heads
        )

        self.feed_forward = FeedForward(
            embedding_dim,
            hidden_dim
        )

        self.layer_norm1 = nn.LayerNorm(
            embedding_dim
        )

        self.layer_norm2 = nn.LayerNorm(
            embedding_dim
        )

    def forward(self, x):

        # Attention + residual connection
        attention_output = self.attention(
            self.layer_norm1(x)
        )

        x = x + attention_output

        # Feed Forward + residual connection
        feed_forward_output = self.feed_forward(
            self.layer_norm2(x)
        )

        x = x + feed_forward_output

        return x


if __name__ == "__main__":

    batch_size = 1
    sequence_length = 5
    embedding_dim = 16
    num_heads = 4
    hidden_dim = 64

    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim
    )

    block = TransformerBlock(
        embedding_dim,
        num_heads,
        hidden_dim
    )

    output = block(x)

    print("Input shape:")
    print(x.shape)

    print("\nOutput shape:")
    print(output.shape)
