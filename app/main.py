from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import json
import os
from app.model_loader import load_model
from app.schemas import InferenceRequest, InferenceResponse
from monitoring.data_logger import log_prediction

# ==== FastAPI App ====
app = FastAPI(title="SECOM Fault Classifier")

# ==== Load Model ====
model = load_model()

# ==== Load Metadata ====
METADATA_PATH = "models/model_metadata.json"
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, "r") as f:
        model_metadata = json.load(f)
else:
    model_metadata = {"model_version": "unknown", "n_features": None, "accuracy": None}

# ==== Schemas ====
class PredictionInput(BaseModel):
    features: List[float]

# ==== Routes ====
@app.get("/")
def root():
    return {"message": "ML Inference API is running."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def model_version():
    return {"model_version": model_metadata.get("model_version", "unknown")}

@app.get("/model-info")
def model_info():
    return {
        "model_version": model_metadata.get("model_version", "unknown"),
        "trained_on": model_metadata.get("timestamp", "unknown"),
        "accuracy": model_metadata.get("accuracy", "unknown"),
        "n_features": model_metadata.get("n_features", "unknown"),
    }

@app.post("/predict", response_model=InferenceResponse)
def predict(input: PredictionInput):
    try:
        features = input.features
        features_array = np.array(features).reshape(1, -1)

        print(f"[DEBUG] Received {len(features)} features.")
        print(f"[DEBUG] Feature array shape: {features_array.shape}")
        print(f"[DEBUG] Model expects {model.n_features_in_} features.")

        if hasattr(model, "n_features_in_") and len(features) != model.n_features_in_:
            raise ValueError(f"Model expects {model.n_features_in_} features, got {len(features)}")

        # Call model
        proba = model.predict_proba(features_array)[0]
        probability = proba[1] if len(proba) == 2 else proba[0]
        prediction = int(probability >= 0.5)

        print(f"[DEBUG] Probability: {probability}, Prediction: {prediction}")

        # Optional: log prediction
        log_prediction(features_array.tolist(), prediction, probability, model_metadata.get("model_version", "unknown"))

        return {"prediction": prediction, "probability": probability}

    except Exception as e:
        import traceback
        print("[ERROR] Prediction failed:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))