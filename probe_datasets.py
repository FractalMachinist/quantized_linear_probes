"""Loads the 3 classification challenges used by this experiment.

Statements + binary labels copied from the geometry-of-truth repo (see
chapter1_transformer_interp/exercises/part31_linear_probes/solutions.py for the
original usage against Llama-2-13B). Every statement ends with a period, since
probes are read off the last (period) token's hidden state.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DATASET_NAMES = ["cities", "sp_en_trans", "larger_than"]


def load_challenge_datasets(size: str | None = None, seed: int = 42) -> dict[str, pd.DataFrame]:
    """
    Load the 3 challenge datasets as {name: DataFrame} with "statement" and "label" columns.

    Args:
        size: If None, use the full dataset. If an int, subsample each dataset down to
            that many rows (stratified by label) for fast iteration.
        seed: Random seed for subsampling.
    """
    datasets = {}
    for name in DATASET_NAMES:
        df = pd.read_csv(DATA_DIR / f"{name}.csv")[["statement", "label"]]
        if size is not None and size < len(df):
            df = (
                df.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(n=min(len(g), size // 2), random_state=seed))
                .reset_index(drop=True)
            )
        datasets[name] = df
    return datasets
