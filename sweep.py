"""Orchestrates the layer x dataset x probe-type x model-variant sweep.

Produces a tidy long-format DataFrame; all plotting/interpretation happens in the notebook.
"""

import json
from pathlib import Path

import pandas as pd
import torch as t
from torch import Tensor
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from activations import extract_activations
from probes import PROBE_CLASSES


def make_split(n: int, train_frac: float = 0.8, seed: int = 42) -> tuple[Tensor, Tensor]:
    """Shared train/test index split, used identically across model variants for a fair comparison."""
    g = t.Generator().manual_seed(seed)
    perm = t.randperm(n, generator=g)
    n_train = int(n * train_frac)
    return perm[:n_train], perm[n_train:]


def run_layer_sweep(
    datasets: dict[str, pd.DataFrame],
    model_variants: dict[str, tuple[PreTrainedModel, PreTrainedTokenizerBase]],
    layers: list[int],
    train_frac: float = 0.8,
    seed: int = 42,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    For every (dataset, model variant, probe type, layer), train a probe and record train/test accuracy.

    Args:
        datasets: {dataset_name: DataFrame} with "statement" and "label" columns.
        model_variants: {variant_name: (model, tokenizer)}, e.g. {"full_bit": (...), "quantized": (...)}.
        layers: Layer indices to sweep over (0-indexed transformer blocks).
        train_frac: Fraction of each dataset used for training.
        seed: Seed for the train/test split (shared across model variants).
        batch_size: Batch size for activation extraction.

    Returns:
        Long-format DataFrame with columns: dataset, model_variant, probe_type, layer,
        n_train, n_test, train_acc, test_acc.
    """
    rows = []

    total_probes = len(datasets) * len(model_variants) * len(PROBE_CLASSES) * len(layers)
    pbar = tqdm(total=total_probes, desc="Training probes")

    for dataset_name, df in datasets.items():
        statements = df["statement"].tolist()
        labels = t.tensor(df["label"].values, dtype=t.float32)
        train_idx, test_idx = make_split(len(statements), train_frac, seed)
        train_statements = [statements[i] for i in train_idx]
        test_statements = [statements[i] for i in test_idx]
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        for variant_name, (model, tokenizer) in model_variants.items():
            train_acts = extract_activations(train_statements, model, tokenizer, layers, batch_size)
            test_acts = extract_activations(test_statements, model, tokenizer, layers, batch_size)

            for probe_name, probe_cls in PROBE_CLASSES.items():
                for layer in layers:
                    probe = probe_cls.from_data(train_acts[layer], train_labels)

                    train_preds = probe.pred(train_acts[layer])
                    test_preds = probe.pred(test_acts[layer])
                    train_acc = (train_preds == train_labels).float().mean().item()
                    test_acc = (test_preds == test_labels).float().mean().item()

                    rows.append(
                        {
                            "dataset": dataset_name,
                            "model_variant": variant_name,
                            "probe_type": probe_name,
                            "layer": layer,
                            "n_train": len(train_statements),
                            "n_test": len(test_statements),
                            "train_acc": train_acc,
                            "test_acc": test_acc,
                        }
                    )
                    pbar.update(1)

    pbar.close()
    return pd.DataFrame(rows)


def load_results(
    model: str, size: str, n_statements: int | None = None, results_dir: Path | str = "results"
) -> tuple[pd.DataFrame, dict]:
    """
    Load a sweep previously saved to results/{model}/{size}-{n_statements}/.

    Args:
        model: Model name, e.g. "llama2-13b" (matches the notebook's MODEL constant).
        size: Sweep size, "quick" or "full" (matches the notebook's SIZE constant).
        n_statements: Total statement count suffixing the run's folder name. If None,
            the single "{size}-*" folder under results/{model}/ is used, raising if
            there isn't exactly one match.
        results_dir: Root results directory.

    Returns:
        (results, metadata): the long-format DataFrame produced by run_layer_sweep,
        and the run's metadata.json contents.
    """
    model_dir = Path(results_dir) / model
    if n_statements is not None:
        run_dir = model_dir / f"{size}-{n_statements}"
    else:
        matches = sorted(model_dir.glob(f"{size}-*"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one '{size}-*' run under {model_dir}, found {len(matches)}: {matches}"
            )
        run_dir = matches[0]

    results = pd.read_csv(run_dir / "results.csv")
    with open(run_dir / "metadata.json") as f:
        metadata = json.load(f)
    return results, metadata
