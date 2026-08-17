"""Loads a base model (GPT-2-small or Llama-2-13b) in two variants: full-bit (float32) and
8-bit quantized (bitsandbytes)."""

from pathlib import Path

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizerBase

MODEL_CONFIGS = {
    "gpt2": {"hf_name": "gpt2", "num_layers": 12, "d_model": 768, "gated": False},
    "llama2-13b": {"hf_name": "meta-llama/Llama-2-13b-hf", "num_layers": 40, "d_model": 5120, "gated": True},
}

FULL_BIT_DTYPE = t.float32


def _load_hf_token() -> str | None:
    """Read HF_TOKEN from the .env file in this directory, needed for gated repos like Llama 2."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip()
    return None


HF_TOKEN = _load_hf_token()


def _resolve(model: str) -> tuple[str, str | None]:
    config = MODEL_CONFIGS[model]
    return config["hf_name"], (HF_TOKEN if config["gated"] else None)


def load_tokenizer(model: str) -> PreTrainedTokenizerBase:
    hf_name, token = _resolve(model)
    tokenizer = AutoTokenizer.from_pretrained(hf_name, token=token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_full_bit_model(model: str, device: str = "cuda") -> PreTrainedModel:
    """Load the selected model at full precision (float32) as the unquantized baseline."""
    hf_name, token = _resolve(model)
    loaded = AutoModelForCausalLM.from_pretrained(hf_name, dtype=FULL_BIT_DTYPE, token=token)
    loaded = loaded.to(device)
    loaded.eval()
    return loaded


def load_quantized_model(model: str) -> PreTrainedModel:
    """Load the selected model with bitsandbytes 8-bit quantization (Linear8bitLt layers)."""
    hf_name, token = _resolve(model)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    loaded = AutoModelForCausalLM.from_pretrained(
        hf_name, quantization_config=bnb_config, device_map="auto", token=token
    )
    loaded.eval()
    return loaded
