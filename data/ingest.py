"""Load one raw C-MAPSS subset into a typed, validated parquet file with remaining useful life.

Usage from the repo root (these are the DVC stages in dvc.yaml):

    python -m data.ingest --subset FD001 --kind train --out data/interim/fd001_train.parquet
    python -m data.ingest --subset FD001 --kind test  --out data/interim/fd001_test.parquet

Training engines run to failure, so RUL is the distance to each engine's last cycle. Test engines
are cut off early and the RUL file gives the true remaining life at the cut, so RUL at every
earlier cycle is that value plus the distance to the cut.

Only FD001 is allowed by default. FD002 and FD004 are quarantined as production replay traffic for
the drift phase, and FD003 is unused. Phase 5 lifts the quarantine on purpose with ``--replay``,
which also offsets the unit numbers by REPLAY_UNIT_OFFSET so the regime is unambiguous everywhere
downstream. Nothing lifts it silently.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from data.schema import CYCLE, RAW_COLUMNS, REPLAY_UNIT_OFFSET, RUL, UNIT

log = logging.getLogger("driftwatch.ingest")

ALLOWED_SUBSETS: tuple[str, ...] = ("FD001",)
QUARANTINED_SUBSETS: tuple[str, ...] = ("FD002", "FD004")
DEFAULT_RAW_DIR = Path("data/raw")


def guard_subset(subset: str, replay: bool = False) -> None:
    """Refuse the quarantined subsets unless the caller says, explicitly, that this is the replay."""
    if subset in ALLOWED_SUBSETS:
        return
    if subset in QUARANTINED_SUBSETS:
        if replay:
            log.warning("%s quarantine lifted for the drift replay; unit numbers will be offset by %d", subset, REPLAY_UNIT_OFFSET)
            return
        raise SystemExit(f"{subset} is quarantined as production replay traffic; pass --replay only for the drift phase")
    raise SystemExit(f"{subset!r} is not part of the pipeline; allowed subsets: {', '.join(ALLOWED_SUBSETS)}")


def read_cycles(path: Path) -> pd.DataFrame:
    """Parse a space-separated cycle file. Lines carry trailing whitespace, hence the regex separator."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=RAW_COLUMNS)
    df[UNIT] = df[UNIT].astype("int32")
    df[CYCLE] = df[CYCLE].astype("int32")
    return df


def validate(df: pd.DataFrame) -> None:
    """Fail loudly on anything the feature code assumes: no gaps, no nulls, cycles start at 1."""
    if df.isna().any().any():
        raise ValueError("raw cycle data contains nulls")
    if (df[UNIT] < 1).any() or (df[CYCLE] < 1).any():
        raise ValueError("unit and cycle numbers must be positive")
    by_unit = df.groupby(UNIT)[CYCLE]
    if (by_unit.min() != 1).any():
        raise ValueError("every engine must start at cycle 1")
    contiguous = by_unit.apply(lambda s: s.is_monotonic_increasing and s.diff().dropna().eq(1).all())
    if not contiguous.all():
        raise ValueError(f"cycles are not contiguous for units: {contiguous.index[~contiguous].tolist()}")


def add_rul_train(df: pd.DataFrame) -> pd.DataFrame:
    """Every training engine fails on its last recorded cycle."""
    df[RUL] = (df.groupby(UNIT)[CYCLE].transform("max") - df[CYCLE]).astype("int32")
    return df


def add_rul_test(df: pd.DataFrame, rul_path: Path) -> pd.DataFrame:
    """Test engines stop before failure; the RUL file lists remaining life at the cut, one line per unit."""
    rul_at_end = pd.read_csv(rul_path, header=None, names=["rul_at_end"])["rul_at_end"]
    rul_at_end.index = rul_at_end.index + 1  # unit numbers are 1-based
    units = df[UNIT].unique()
    if len(rul_at_end) != len(units):
        raise ValueError(f"RUL file has {len(rul_at_end)} rows but data has {len(units)} engines")
    cycles_to_cut = df.groupby(UNIT)[CYCLE].transform("max") - df[CYCLE]
    df[RUL] = (df[UNIT].map(rul_at_end) + cycles_to_cut).astype("int32")
    return df


def ingest(subset: str, kind: str, raw_dir: Path, replay: bool = False) -> pd.DataFrame:
    guard_subset(subset, replay)
    df = read_cycles(raw_dir / f"{kind}_{subset}.txt")
    validate(df)
    df = add_rul_train(df) if kind == "train" else add_rul_test(df, raw_dir / f"RUL_{subset}.txt")
    if replay:
        df[UNIT] = (df[UNIT] + REPLAY_UNIT_OFFSET).astype("int32")  # after RUL, which maps by original unit
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--kind", choices=("train", "test"), default="train")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replay", action="store_true",
                        help="lift the quarantine on FD002/FD004 for the drift replay (Phase 5) and offset unit numbers")
    args = parser.parse_args(argv)

    df = ingest(args.subset, args.kind, args.raw_dir, replay=args.replay)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    log.info(
        "wrote %s: %d rows, %d engines, rul %d..%d",
        args.out, len(df), df[UNIT].nunique(), df[RUL].min(), df[RUL].max(),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
