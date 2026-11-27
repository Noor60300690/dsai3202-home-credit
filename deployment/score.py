
import os
import json
import logging
import numpy as np
import pandas as pd
import mlflow.pyfunc

model = None
model_inputs = None


def init():
    global model, model_inputs

    model_dir = os.environ["AZUREML_MODEL_DIR"]

    model_paths = []

    for root, dirs, files in os.walk(model_dir):
        if "MLmodel" in files and "model.pkl" in files:
            model_paths.append(root)

    if len(model_paths) != 1:
        raise RuntimeError(
            "Expected exactly one MLflow model directory"
        )

    model = mlflow.pyfunc.load_model(model_paths[0])

    model_inputs = (
        model.metadata.get_input_schema().inputs
    )

    logging.info("Home Credit model loaded successfully")


def run(raw_data):
    payload = json.loads(raw_data)

    records = payload.get("inputs")

    if not isinstance(records, list) or not records:
        raise ValueError(
            "inputs must be a non-empty list of records"
        )

    if not all(isinstance(row, dict) for row in records):
        raise ValueError("Each input must be an object")

    required = [
        col.name for col in model_inputs
        if col.required
    ]

    for row in records:
        missing = [
            name for name in required
            if name not in row
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

    df = pd.DataFrame(records)

    for col in model_inputs:
        if col.name not in df.columns:
            df[col.name] = None

        if col.type.name == "double":
            df[col.name] = pd.to_numeric(
                df[col.name], errors="raise"
            ).astype("float64")

    features = df[
        [col.name for col in model_inputs]
    ].copy()

    predictions = np.asarray(model.predict(features))

    if predictions.shape == (len(records), 2):
        probabilities = predictions[:, 1]

    elif predictions.shape == (len(records),):
        probabilities = predictions

    else:
        raise ValueError("Unexpected prediction shape")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Non-finite model prediction")

    return {
        "predictions": [
            {
                "SK_ID_CURR": row.get("SK_ID_CURR"),
                "probability": float(probability)
            }
            for row, probability in zip(
                records, probabilities
            )
        ]
    }
