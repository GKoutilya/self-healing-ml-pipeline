import pandas as pd
import numpy as np
import os
import json
import joblib
import shutil
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ==== Paths ====
INFERENCE_LOG = "monitoring/inference_log.csv"
LATEST_MODEL_PATH = "models/model_latest.pkl"

# ==== Step 1: Load Inference Data ====
df = pd.read_csv(INFERENCE_LOG)

# Parse the features column (stored as JSON strings)
df['parsed_features'] = df['features'].apply(lambda x: json.loads(x)[0])

# Convert to numerical DataFrame
features = pd.DataFrame(df['parsed_features'].tolist())
features.columns = [f"f{i}" for i in range(features.shape[1])]

X = features.values
y = df['prediction'].values  # Assuming your predictions are correct (or from human labels)

# ==== Step 2: Train/Test Split ====
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== Step 3: Train New Model ====
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==== Step 4: Evaluate Model ====
y_pred = model.predict(X_val)
print("📊 Classification Report on Validation Set:")
print(classification_report(y_val, y_pred))

# ==== Step 5: Save Versioned and Latest Models ====
# Create versioned path (timestamp-based)
version = int(time.time())
versioned_model_path = f"models/model_v{version}.pkl"
joblib.dump(model, versioned_model_path)

# Save as latest
shutil.copy(versioned_model_path, LATEST_MODEL_PATH)

print(f"✅ New model saved to: {versioned_model_path}")
print("🔄 model_latest.pkl updated successfully.")