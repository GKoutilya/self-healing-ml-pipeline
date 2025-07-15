import os
import csv
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler

# Paths
MODEL_PATH = "models/model_latest.pkl"
DATA_PATH = "data/processed/secom_scaled.pkl"
LOG_FILE = "monitoring/inference_log.csv"

# Load latest model
model = joblib.load(MODEL_PATH)

# Extract version from actual model file if it's a symlink or alias
def extract_model_version(path):
    for fname in os.listdir("models"):
        if fname.startswith("model_v") and fname.endswith(".pkl"):
            full_path = os.path.join("models", fname)
            try:
                if os.path.samefile(full_path, path):
                    return fname.replace("model_v", "").replace(".pkl", "")
            except:
                continue
    return "unknown"

model_version = extract_model_version(MODEL_PATH)

# Load SECOM features
X, _ = joblib.load(DATA_PATH)
df = pd.DataFrame(X)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Ensure output directory exists
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def log_prediction(features, prediction, probability, model_version, is_new_file):
    """
    Logs a prediction into inference_log.csv, including the model version used.
    Features are saved as a JSON string (single list).
    """
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        if is_new_file:
            writer.writerow(["timestamp", "prediction", "probability", "features", "model_version"])

        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            prediction,
            probability,
            json.dumps(features),  # ✅ changed from json.dumps([features])
            model_version
        ])

# Delete the old file to prevent malformed lines from persisting (optional)
is_new_file = not os.path.exists(LOG_FILE)

# Run 100 predictions
for i in range(100):
    features = df.iloc[i].tolist()
    features_array = np.array(features).reshape(1, -1)

    # Safe probability extraction
    proba = model.predict_proba(features_array)[0]
    if len(proba) == 2:
        probability = proba[1]  # probability of class 1
    else:
        probability = proba[0]  # fallback for one-class model

    prediction = int(probability >= 0.5)
    log_prediction(features, prediction, probability, model_version, is_new_file)
    is_new_file = False  # Only write header once
