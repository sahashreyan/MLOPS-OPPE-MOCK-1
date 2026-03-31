# Iris MLOps Pipeline (DVC + MLflow)

This project builds an Iris flower species classifier with an incremental two-iteration training process:

1. Train on `iris_v0`.
2. Retrain on merged `iris_v0 + iris_v1`.

It uses:
- **DVC** for data/model versioning and reproducible stages
- **MLflow** for experiment tracking
- **pytest** for inference integrity checks

## Setup

```bash
pip install -r requirements.txt
```

## Run pipeline

```bash
dvc repro
```

## Run tests

```bash
pytest -q
```

## Run inference

```bash
python -m src.predict --model models/model_v1.joblib --csv-row "5.1,3.5,1.4,0.2"
```
