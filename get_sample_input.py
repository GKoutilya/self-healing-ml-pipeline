# get_sample_input.py

import joblib
import numpy as np

# Load your training data
X, y = joblib.load("data/processed/secom_scaled.pkl")

# Pick the first row as an example
sample = X[0].tolist()

# Print it in JSON format for Swagger
import json
print(json.dumps({"features": sample}))
