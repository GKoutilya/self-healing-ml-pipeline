from pydantic import BaseModel
from typing import List

class InferenceRequest(BaseModel):
    features: List[float]

class InferenceResponse(BaseModel):
    prediction: int
    probability: float
