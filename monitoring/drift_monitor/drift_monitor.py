import pandas as pd
import numpy as np
import ast
import os
from scipy.stats import wasserstein_distance
import subprocess

# Config
DATA_PATH = os.path.join("monitoring", "inference_log.csv")
N_FEATURES = 10
REFERENCE_SIZE = 5
CURRENT_SIZE = 5
DRIFT_THRESHOLD = 0.3  # example threshold

def load_data():
    df = pd.read_csv(DATA_PATH)
    
    # Parse JSON-formatted feature list
    try:
        df['parsed_features'] = df['features'].apply(lambda x: ast.literal_eval(x))
    except Exception as e:
        raise ValueError(f"Failed to parse 'features': {e}")

    # Convert list of features into dataframe
    feature_array = df['parsed_features'].tolist()
    feature_df = pd.DataFrame(feature_array)
    feature_df.columns = [f"f{i}" for i in range(feature_df.shape[1])]

    df = pd.concat([df, feature_df], axis=1)
    return df

def compute_drift(df, n_features=N_FEATURES):
    recent = df.tail(CURRENT_SIZE)
    reference = df.head(REFERENCE_SIZE)
    drift_scores = {}

    for i in range(n_features):
        col = f"f{i}"
        if col in df.columns:
            try:
                score = wasserstein_distance(reference[col], recent[col])
                drift_scores[col] = round(score, 4)
            except Exception as e:
                print(f"Drift computation failed for {col}: {e}")
                drift_scores[col] = None
        else:
            print(f"Missing column: {col}")
            drift_scores[col] = None

    return drift_scores

def check_drift_trigger(drift_scores):
    for feat, score in drift_scores.items():
        if score is not None and score > DRIFT_THRESHOLD:
            print(f"Drift detected in {feat} (score = {score})")
            return True
    return False

if __name__ == "__main__":
    df = load_data()
    drift = compute_drift(df)

    print("Feature Drift (Wasserstein Distance): ")
    for feat, val in drift.items():
        print(f"{feat}: {val}")

    if check_drift_trigger(drift):
        print("Retraining recommended.")
        
        # Automatically trigger retraining script
        print("Launching retraining pipeline...")
        subprocess.run([
            "python", 
            "training/retraining_pipeline.py"
        ])
    else:
        print("No significant drift detected.")