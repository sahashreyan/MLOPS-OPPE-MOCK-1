import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
SPECIES_MAP = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}


def _parse_row(csv_row: str):
    raw = [x.strip() for x in csv_row.split(",")]
    if len(raw) != 4:
        raise ValueError("csv-row must have 4 comma-separated values.")
    values = [float(x) if x.lower() != "nan" else np.nan for x in raw]
    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


def _parse_json_row(json_row: str):
    payload = json.loads(json_row)
    values = [payload.get(col, np.nan) for col in FEATURE_COLUMNS]
    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


def predict(model_path: Path, csv_row: str | None = None, json_row: str | None = None):
    if not csv_row and not json_row:
        raise ValueError("Provide one of --csv-row or --json-row.")
    model = joblib.load(model_path)
    frame = _parse_row(csv_row) if csv_row else _parse_json_row(json_row)
    pred = int(model.predict(frame)[0])
    return {"species_id": pred, "species_name": SPECIES_MAP[pred]}


def main():
    parser = argparse.ArgumentParser(description="Run Iris model inference.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--csv-row", default=None, help="Comma separated 4 features; use 'nan' for missing")
    parser.add_argument("--json-row", default=None, help="JSON dict with feature names")
    args = parser.parse_args()

    output = predict(Path(args.model), csv_row=args.csv_row, json_row=args.json_row)
    print(json.dumps(output))


if __name__ == "__main__":
    main()
