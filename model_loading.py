"""Loads GPT-2-small in two variants: full-bit (float32) and 8-bit quantized (bitsandbytes)."""

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizerBase

MODEL_NAME = "gpt2"
NUM_LAYERS = 12
D_MODEL = 768

FULL_BIT_DTYPE = t.float32


def load_tokenizer() -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_full_bit_model(device: str = "cuda") -> PreTrainedModel:
    """Load GPT-2-small at full precision (float32) as the unquantized baseline."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=FULL_BIT_DTYPE)
    model = model.to(device)
    model.eval()
    return model


def load_quantized_model() -> PreTrainedModel:
    """Load GPT-2-small with bitsandbytes 8-bit quantization (Linear8bitLt layers)."""
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, device_map="auto")
    model.eval()
    return model
