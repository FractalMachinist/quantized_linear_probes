"""Last-token hidden-state extraction, copied and trimmed from
chapter1_transformer_interp/exercises/part31_linear_probes/solutions.py.
"""

import torch as t
from jaxtyping import Float
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def extract_activations(
    statements: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    layers: list[int],
    batch_size: int = 32,
) -> dict[int, Float[Tensor, "n_statements d_model"]]:
    """
    Extract last-token hidden state activations from specified layers for a list of statements.

    Args:
        statements: List of text statements to process (each must end with a period).
        model: A HuggingFace causal language model.
        tokenizer: The corresponding tokenizer.
        layers: List of layer indices (0-indexed; layer i = output of transformer block i).
        batch_size: Number of statements to process at once.

    Returns:
        Dictionary mapping layer index to tensor of activations, shape [n_statements, d_model].
    """
    all_acts = {layer: [] for layer in layers}

    for i in range(0, len(statements), batch_size):
        batch = statements[i : i + batch_size]

        for stmt in batch:
            assert stmt.rstrip().endswith("."), f"Statement doesn't end with period: {stmt!r}"

        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)

        with t.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        last_token_idx = inputs["attention_mask"].sum(dim=1) - 1  # [batch]

        for layer in layers:
            # hidden_states[0] is embedding, hidden_states[layer+1] is output of block `layer`
            hidden = outputs.hidden_states[layer + 1]  # [batch, seq_len, d_model]
            batch_indices = t.arange(hidden.shape[0], device=hidden.device)
            acts = hidden[batch_indices, last_token_idx]  # [batch, d_model]
            all_acts[layer].append(acts.cpu().float())

    return {layer: t.cat(acts_list, dim=0) for layer, acts_list in all_acts.items()}
