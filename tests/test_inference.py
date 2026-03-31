from pathlib import Path

import joblib
import pandas as pd

FEATURE_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
VALID_IDS = {0, 1, 2}


def _pick_model() -> Path:
    v1 = Path("models/model_v1.joblib")
    v0 = Path("models/model_v0.joblib")
    if v1.exists():
        return v1
    if v0.exists():
        return v0
    raise FileNotFoundError("No model artifact found. Expected model_v1 or model_v0.")


def test_inference_on_dvc_test_data():
    model = joblib.load(_pick_model())
    test_df = pd.read_csv("data/iris_test.csv")
    X = test_df[FEATURE_COLUMNS]
    preds = model.predict(X)

    assert len(preds) == len(test_df)
    assert set(int(p) for p in preds).issubset(VALID_IDS)
