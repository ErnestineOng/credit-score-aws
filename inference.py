import os
import json
import joblib
import pandas as pd
import numpy as np


def model_fn(model_dir):
    model_path = os.path.join(model_dir, "best_model.pkl")
    model = joblib.load(model_path)
    return model


def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)

        if "instances" in data:
            df = pd.DataFrame(data["instances"])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])

        return df
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    return {
        "predictions": predictions,
        "probabilities": probabilities,
    }


def output_fn(prediction_result, content_type):
    if content_type == "application/json":
        preds = prediction_result["predictions"].tolist()
        probs = prediction_result["probabilities"].tolist()
        labels = [str(p) for p in preds]

        response = {
            "predictions": labels,
            "probabilities": probs,
        }
        return json.dumps(response)
    raise ValueError(f"Unsupported content type: {content_type}")
