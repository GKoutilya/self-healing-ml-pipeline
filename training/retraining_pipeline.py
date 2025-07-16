import pandas as pd
import numpy as np
import os
import json
import joblib
import shutil
import time
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ==== Paths ====
INFERENCE_LOG = "monitoring/inference_log.csv"
LATEST_MODEL_PATH = "models/model_v1.pkl"
MODEL_METADATA_PATH = "models/model_metadata.json"

# ==== Step 1: Load Inference Data ====
df = pd.read_csv(INFERENCE_LOG)

# Parse the features column (stored as JSON strings)
df['parsed_features'] = df['features'].apply(lambda x: json.loads(x)[0])

# Convert to numerical DataFrame
features = pd.DataFrame(df['parsed_features'].tolist())
features.columns = [f"f{i}" for i in range(features.shape[1])]

X = features.values
y = df['prediction'].values  # Assuming predictions are correct or manually labeled

# ==== Step 2: Train/Test Split ====
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== Step 3: Train New Model ====
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==== Step 4: Evaluate Model ====
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Classification Report on Validation Set:")
print(classification_report(y_val, y_pred))
print(f"Validation Accuracy: {accuracy:.4f}")

# ==== Step 5: Save Versioned and Latest Models ====
timestamp = int(time.time())
versioned_model_path = f"models/model_v{timestamp}.pkl"
joblib.dump(model, versioned_model_path)

# Save as latest
shutil.copy(versioned_model_path, LATEST_MODEL_PATH)

# ==== Step 6: Update Metadata ====
metadata = {
    "model_path": versioned_model_path,
    "model_version": f"v{timestamp}",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "accuracy": round(accuracy, 4),
    "n_features": X.shape[1]
}

with open(MODEL_METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"New model saved to: {versioned_model_path}")
print("Latest model updated at:", LATEST_MODEL_PATH)
print("Metadata logged to:", MODEL_METADATA_PATH)
