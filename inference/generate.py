import torch

from tokenizer.tokenizer import BPETokenizer
from model.gpt import GPT

import config


# =========================
# Load tokenizer
# =========================

tokenizer = BPETokenizer(
    config.TOKENIZER_PATH
)

vocab_size = tokenizer.get_vocab_size()


# =========================
# Load model
# =========================

model = GPT(
    vocab_size=vocab_size,
    embedding_dim=config.EMBEDDING_DIM,
    num_heads=config.NUM_HEADS,
    num_layers=config.NUM_LAYERS,
    hidden_dim=config.HIDDEN_DIM,
    max_context_length=config.CONTEXT_LENGTH
)

model.load_state_dict(
    torch.load(
        config.CHECKPOINT_PATH,
        map_location="cpu"
    )
)

model.eval()


# =========================
# Top-K Sampling
# =========================

def top_k_sampling(logits, k=20):

    k = min(k, logits.size(-1))

    values, indices = torch.topk(
        logits,
        k
    )

    probabilities = torch.softmax(
        values,
        dim=-1
    )

    sampled_index = torch.multinomial(
        probabilities,
        num_samples=1
    )

    next_token = torch.gather(
        indices,
        dim=-1,
        index=sampled_index
    )

    return next_token


# =========================
# Text Generation
# =========================

def generate_text(
    prompt,
    max_new_tokens=50,
    temperature=0.8,
    top_k=20
):

    token_ids = tokenizer.encode(prompt)

    tokens = torch.tensor(
        [token_ids],
        dtype=torch.long
    )

    for _ in range(max_new_tokens):

        tokens_for_model = tokens[
            :, -config.CONTEXT_LENGTH:
        ]

        with torch.no_grad():

            logits = model(
                tokens_for_model
            )

        next_token_logits = logits[
            :, -1, :
        ]

        next_token_logits = (
            next_token_logits / temperature
        )

        next_token = top_k_sampling(
            next_token_logits,
            k=top_k
        )

        tokens = torch.cat(
            [tokens, next_token],
            dim=1
        )

    generated_text = tokenizer.decode(
        tokens[0].tolist()
    )

    print("\n" + "=" * 60)
    print("Prompt:")
    print(prompt)

    print("\nGenerated text:")
    print(generated_text)


# =========================
# Tests
# =========================

prompts = [
    "Data engineering",
    "Machine learning",
    "A transformer",
    "Retrieval augmented generation"
]

for prompt in prompts:

    generate_text(
        prompt,
        max_new_tokens=50,
        temperature=0.8,
        top_k=20
    )
