import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_COLUMN = "species"


def _inject_missing_values(df: pd.DataFrame, missing_rate: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    mask = rng.random((len(out), len(FEATURE_COLUMNS))) < missing_rate
    for i, col in enumerate(FEATURE_COLUMNS):
        out.loc[mask[:, i], col] = np.nan
    return out


SPECIES_TO_ID = {
    "setosa": 0,
    "iris-setosa": 0,
    "versicolor": 1,
    "iris-versicolor": 1,
    "virginica": 2,
    "iris-virginica": 2,
}


def _normalize_species(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if TARGET_COLUMN not in out.columns:
        raise ValueError("Input must include 'species' column.")
    raw = out[TARGET_COLUMN].astype(str).str.strip().str.lower()
    mapped = raw.map(SPECIES_TO_ID)
    numeric = pd.to_numeric(out[TARGET_COLUMN], errors="coerce")
    out[TARGET_COLUMN] = mapped.fillna(numeric).astype(int)
    return out


def prepare_version(
    input_path: Path, output_path: Path, missing_rate: float = 0.1, seed: int = 42
) -> None:
    df = pd.read_csv(input_path)
    df = _normalize_species(df)
    df = _inject_missing_values(df, missing_rate=missing_rate, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def prepare_test_set(
    v0_path: Path,
    v1_path: Path,
    output_path: Path,
    test_size: float = 0.2,
    missing_rate: float = 0.1,
    seed: int = 42,
) -> None:
    v0 = _normalize_species(pd.read_csv(v0_path))
    v1 = _normalize_species(pd.read_csv(v1_path))
    merged = pd.concat([v0, v1], ignore_index=True)
    _, test_df = train_test_split(
        merged, test_size=test_size, stratify=merged[TARGET_COLUMN], random_state=seed
    )
    test_df = _inject_missing_values(test_df.reset_index(drop=True), missing_rate=missing_rate, seed=seed + 100)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Iris dataset versions.")
    sub = parser.add_subparsers(dest="command", required=True)

    version_cmd = sub.add_parser("prepare-version")
    version_cmd.add_argument("--input", required=True)
    version_cmd.add_argument("--output", required=True)
    version_cmd.add_argument("--missing-rate", type=float, default=0.1)
    version_cmd.add_argument("--seed", type=int, default=42)

    test_cmd = sub.add_parser("prepare-test")
    test_cmd.add_argument("--v0", required=True)
    test_cmd.add_argument("--v1", required=True)
    test_cmd.add_argument("--output", required=True)
    test_cmd.add_argument("--test-size", type=float, default=0.2)
    test_cmd.add_argument("--missing-rate", type=float, default=0.1)
    test_cmd.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.command == "prepare-version":
        prepare_version(Path(args.input), Path(args.output), args.missing_rate, args.seed)
    else:
        prepare_test_set(
            Path(args.v0),
            Path(args.v1),
            Path(args.output),
            args.test_size,
            args.missing_rate,
            args.seed,
        )


if __name__ == "__main__":
    main()
