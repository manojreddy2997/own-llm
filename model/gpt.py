import torch
import torch.nn as nn

from model.multi_head_attention import MultiHeadAttention


class FeedForward(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, dropout=0.1):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        return self.network(x)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        hidden_dim,
        dropout=0.1
    ):

        super().__init__()

        self.layer_norm1 = nn.LayerNorm(embedding_dim)

        self.attention = MultiHeadAttention(
            embedding_dim,
            num_heads,
        )

        self.dropout1 = nn.Dropout(dropout)

        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.feed_forward = FeedForward(
            embedding_dim,
            hidden_dim,
            dropout
        )

    def forward(self, x):

        # Self-attention
        attention_output = self.attention(
            self.layer_norm1(x)
        )

        x = x + self.dropout1(
            attention_output
        )

        # Feed-forward network
        x = x + self.feed_forward(
            self.layer_norm2(x)
        )

        return x


class GPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        hidden_dim=256,
        max_context_length=64,
        dropout=0.1
    ):

        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.position_embedding = nn.Embedding(
            max_context_length,
            embedding_dim
        )

        self.dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList(

            [
                TransformerBlock(
                    embedding_dim,
                    num_heads,
                    hidden_dim,
                    dropout
                )

                for _ in range(num_layers)
            ]

        )

        self.final_layer_norm = nn.LayerNorm(
            embedding_dim
        )

        self.output_head = nn.Linear(
            embedding_dim,
            vocab_size,
            bias=False
        )

    def forward(self, token_ids):

        batch_size, sequence_length = token_ids.shape

        # Token embeddings
        token_embeddings = self.token_embedding(
            token_ids
        )

        # Position embeddings
        positions = torch.arange(
            sequence_length,
            device=token_ids.device
        )

        position_embeddings = self.position_embedding(
            positions
        )

        # Combine token + position information
        x = token_embeddings + position_embeddings

        x = self.dropout(x)

        # Transformer blocks
        for block in self.transformer_blocks:

            x = block(x)

        # Final normalization
        x = self.final_layer_norm(x)

        # Convert hidden states to vocabulary logits
        logits = self.output_head(x)

        return logits
