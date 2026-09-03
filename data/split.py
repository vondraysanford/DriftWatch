"""Hold out whole engines for evaluation.

Rows within one engine are strongly correlated, so a row-level split would put near-copies of the
test rows into training and inflate every metric. Engines are assigned to a side as a whole, with
a fixed seed so the split is reproducible and the same engines stay held out for the life of the
project (they are the benchmark for challenger vs champion later).

Usage as a DVC stage:

    python -m data.split --input data/processed/fd001_features.parquet \\
        --train-out data/processed/train.parquet --test-out data/processed/test.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data.schema import LABEL, UNIT

log = logging.getLogger("driftwatch.split")


def split_by_unit(table: pd.DataFrame, test_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) with no engine on both sides."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    units = np.sort(table[UNIT].unique())
    n_test = max(1, round(len(units) * test_fraction))
    test_units = np.random.default_rng(seed).choice(units, size=n_test, replace=False)
    is_test = table[UNIT].isin(test_units)
    train, test = table[~is_test], table[is_test]
    if set(train[UNIT]) & set(test[UNIT]):
        raise AssertionError("an engine landed on both sides of the split")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def describe(name: str, table: pd.DataFrame) -> None:
    log.info("%s: %d engines, %d rows, positive rate %.3f", name, table[UNIT].nunique(), len(table), table[LABEL].mean())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="feature table from data.features")
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--test-out", type=Path, required=True)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    train, test = split_by_unit(pd.read_parquet(args.input), args.test_fraction, args.seed)
    for path, table in ((args.train_out, train), (args.test_out, test)):
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(path, index=False)
    describe("train", train)
    describe("test", test)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
