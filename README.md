# Scalable ML Deployment + Drift Monitoring + Auto-Retraining MLOps Pipeline

## Overview

This project implements a scalable Machine Learning Operations (MLOps) pipeline for a fault classification problem using the SECOM dataset. The pipeline includes:

- **FastAPI-based inference API** for real-time predictions.
- **Prediction logging** to track incoming inference data.
- **Data drift monitoring** using the Wasserstein distance to detect feature distribution shifts.
- **Automated model retraining** triggered when significant drift is detected.
- **Model versioning and metadata management** for reproducibility and auditability.

The system is designed for continuous learning in production environments, ensuring model robustness against evolving data distributions.

---

## Features

- **FastAPI Inference Service**: Serve your ML model with an easy-to-use REST API.
- **Logging**: Persist inference data, predictions, probabilities, and timestamps for monitoring.
- **Drift Detection**: Monitor input data for distribution shifts using statistical distance metrics.
- **Auto-Retraining Pipeline**: Retrain and evaluate the model automatically upon drift detection.
- **Version Control**: Save new model versions and update metadata seamlessly.

---

## Getting Started

### Prerequisites

- Python 3.10+
- Pip

### Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the FastAPI server:

   ```bash
   uvicorn app.main:app --reload
   ```

4. The API will be accessible at: `http://127.0.0.1:8000`

---

## Usage

### Inference API

* **Health Check**

  ```
  GET /
  ```

  Returns a basic message indicating the API is running.

* **Model Version**

  ```
  GET /version
  ```

  Returns the current model version.

* **Make Prediction**

  ```
  POST /predict
  Content-Type: application/json

  {
    "features": [0.1, 0.2, 0.3, ..., 0.n]
  }
  ```

  Response:

  ```json
  {
    "prediction": 0,
    "probability": 0.123
  }
  ```

---

### Drift Monitoring

Run the drift monitor script to analyze feature distribution changes against baseline data:

```bash
python monitoring/drift_monitor/drift_monitor.py
```

If drift is detected, the retraining pipeline will automatically be triggered.

---

### Retraining Pipeline

The retraining pipeline:

* Loads the latest training data.
* Retrains the model.
* Evaluates and prints classification metrics.
* Saves the updated model with a new version number.
* Updates model metadata in JSON.

Run manually or trigger automatically by the drift monitor.

```bash
python training/retraining_pipeline.py
```

---

## Logging

Inference requests and predictions are logged in:

```
monitoring/inference_log.csv
```

Logged data includes:

* Timestamp (UTC)
* Input features
* Model prediction
* Prediction probability
* Model version used

---

## Model Versioning

Model files and metadata are stored in the `models/` directory. The metadata JSON tracks:

* Model version
* Training timestamp
* Model accuracy
* Number of features

---

## Future Improvements

* Integrate advanced drift detection metrics (e.g., KL divergence, population stability index).
* Add alerting via email/SMS when drift is detected.
* Containerize the API and pipeline for cloud deployment.
* Add automated testing and CI/CD integration.
* Support multi-model deployment with blue/green model switching.

---

## License

This project is licensed under the MIT License.

---

## Contact

Created by Koutilya Ganapathiraju. Feel free to reach out:

- **Email:** [gkoutilyaraju@gmail.com](gkoutilyaraju@gmail.com)  
- **GitHub:** [GitHub](https://github.com/GKoutilya)  
- **LinkedIn:** [LinkedIn](https://linkedin.com/in/koutilya-ganapathiraju-0a3350182)
