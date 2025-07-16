import csv
import json
import os
from datetime import datetime, timezone

LOG_FILE = "monitoring/inference_log.csv"

def log_prediction(features, prediction, probability, model_version="unknown"):
    is_new_file = not os.path.exists(LOG_FILE)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if is_new_file:
            writer.writerow(["timestamp", "prediction", "probability", "features", "model_version"])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            prediction,
            probability,
            json.dumps(features),
            model_version
        ])
