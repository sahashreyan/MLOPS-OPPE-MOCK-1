import argparse
import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from src.imputers import LastNMeanImputer

FEATURE_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_COLUMN = "species"
SPECIES_TO_ID = {
    "setosa": 0,
    "iris-setosa": 0,
    "versicolor": 1,
    "iris-versicolor": 1,
    "virginica": 2,
    "iris-virginica": 2,
}


def _normalize_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    raw = out[TARGET_COLUMN].astype(str).str.strip().str.lower()
    mapped = raw.map(SPECIES_TO_ID)
    numeric = pd.to_numeric(out[TARGET_COLUMN], errors="coerce")
    out[TARGET_COLUMN] = mapped.fillna(numeric).astype(int)
    return out


def _load_iteration_data(data_dir: Path, iteration: str) -> pd.DataFrame:
    v0 = _normalize_target(pd.read_csv(data_dir / "iris_v0_prepared.csv"))
    if iteration == "v0":
        return v0
    if iteration == "v1":
        v1 = _normalize_target(pd.read_csv(data_dir / "iris_v1_prepared.csv"))
        return pd.concat([v0, v1], ignore_index=True)
    raise ValueError("iteration must be 'v0' or 'v1'")


def _train_and_eval(df: pd.DataFrame, seed: int):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    pipeline = Pipeline(
        steps=[("imputer", LastNMeanImputer(n_last=10)), ("clf", KNeighborsClassifier())]
    )
    grid = {
        "clf__n_neighbors": [3, 5, 7, 9],
        "clf__weights": ["uniform", "distance"],
    }
    search = GridSearchCV(
        estimator=pipeline, param_grid=grid, cv=3, scoring="accuracy", n_jobs=-1
    )
    search.fit(X_train, y_train)

    y_pred = search.best_estimator_.predict(X_val)
    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision_weighted": float(
            precision_score(y_val, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(recall_score(y_val, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_val, y_pred, average="weighted", zero_division=0)),
    }
    return search, metrics, X_train, X_val


def run_training(iteration: str, data_dir: Path, model_path: Path, metrics_path: Path, seed: int):
    df = _load_iteration_data(data_dir, iteration)
    search, metrics, X_train, X_val = _train_and_eval(df, seed=seed)

    mlflow.set_experiment("iris_incremental")
    with mlflow.start_run(run_name=f"train_{iteration}"):
        mlflow.log_param("iteration", iteration)
        mlflow.log_param("dataset_rows", len(df))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("val_rows", len(X_val))
        for k, v in search.best_params_.items():
            mlflow.log_param(k, v)
        mlflow.log_metrics(metrics)

        model_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(search.best_estimator_, model_path)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"iteration": iteration, "metrics": metrics}, f, indent=2)
        mlflow.log_artifact(str(metrics_path))


def main():
    parser = argparse.ArgumentParser(description="Train Iris model for v0 or v1 iteration.")
    parser.add_argument("--iteration", choices=["v0", "v1"], required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_training(
        iteration=args.iteration,
        data_dir=Path(args.data_dir),
        model_path=Path(args.model_path),
        metrics_path=Path(args.metrics_path),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
