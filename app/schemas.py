from pydantic import BaseModel
from typing import List

class InferenceRequest(BaseModel):
    """
    Request schema for the /predict endpoint.

    Attributes:
        features (List[float]): A list of numerical input features to be fed to the ML model.
    """
    features: List[float]

class InferenceResponse(BaseModel):
    """
    Response schema returned by the /predict endpoint.

    Attributes:
        prediction (int): The predicted class label (0 or 1).
        probability (float): The model's confidence score for the predicted class.
    """
    prediction: int
    probability: float
