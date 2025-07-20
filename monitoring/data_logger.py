import csv
import json
import os
from datetime import datetime, timezone

# Path to the CSV file where inferences will be logged
LOG_FILE = "monitoring/inference_log.csv"

def log_prediction(features, prediction, probability, model_version="unknown"):
    """
        Appends a new inference record to the inference_log.csv file.

        Parameters:
            features (list): Input feature values used in the prediction.
            prediction (int): Predicted class label (0 or 1).
            probability (float): Probability/confidence score from the model.
            model_version (str): Version of the model that made the prediction
        
        Behavior:
            Creates the log directory if it doesn't exist
            Creates the CSV file with headers if it's the first time.
            Appends each prediction result as a new row in the log.
    """

    is_new_file = not os.path.exists(LOG_FILE)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    flat_features = features[0] if isinstance(features[0], list) else features

    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Writes headers if the file is new
        if is_new_file:
            writer.writerow(["timestamp", "prediction", "probability", "features", "model_version"])

        # Append current prediction data
        writer.writerow([
            datetime.now(timezone.utc).isoformat(), # UTC Timestamp
            prediction,
            probability,
            json.dumps(flat_features), # Serialize list of features
            model_version
        ])
