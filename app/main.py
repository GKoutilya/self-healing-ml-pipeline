from fastapi import FastAPI
from app.model_loader import load_model
from app.schemas import InferenceRequest, InferenceResponse
import numpy as np

app = FastAPI(title="SECOM Fault Classifier")

model = load_model()

@app.get("/")
def root():
    return {"message": "ML Inference API is running."}

@app.post("/predict", response_model=InferenceResponse)
def predict(request: InferenceRequest):
    features = np.array(request.features).reshape(1,-1)
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][prediction]
    return InferenceResponse(prediction=int(prediction), probability=round(float(prob), 4))