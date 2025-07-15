from fastapi import FastAPI
from fastapi import HTTPException
from app.model_loader import load_model
from app.schemas import InferenceRequest, InferenceResponse
from monitoring.data_logger import log_prediction
import numpy as np
import datetime

app = FastAPI(title="SECOM Fault Classifier")

model = load_model()

@app.get("/")
def root():
    return {"message": "ML Inference API is running."}

@app.post("/predict")
def predict(data: InferenceRequest):
    try:
        features = np.array(data.features).reshape(1, -1)
        prediction = model.predict(features)[0]

        # Defensive: check if predict_proba is available
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(features)[0]
            if 1 in model.classes_:
                class_1_index = list(model.classes_).index(1)
                prob = probas[class_1_index]
            else:
                prob = probas[0]  # fallback to whatever class exists
        else:
            prob = 0.0  # fallback if predict_proba not available

        # Log the prediction (optional)
        log_prediction(
            features=features.tolist(),
            prediction=prediction,
            probability=prob
        )

        return {"prediction": int(prediction), "probability": float(prob)}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Prediction failed.")