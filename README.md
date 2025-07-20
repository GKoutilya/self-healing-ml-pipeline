# 🚀 Scalable MLOps Pipeline: Real-Time Inference, Drift Monitoring & Auto-Retraining

## 🔍 Overview

This project is a full-stack **MLOps pipeline** built around the SECOM dataset for fault classification. It supports continuous model improvement and reliable deployment through:

* 🧠 Real-time inference via a **FastAPI** service
* 📈 **Streamlit dashboard** for drift monitoring, retraining, and model metadata visualization
* 📦 Drift detection using **Wasserstein Distance**
* 🔁 Automated **model retraining** pipeline upon drift detection
* 🗃️ Full **model versioning** and metadata logging

> Ideal for projects that demand scalable ML deployment with continuous learning and observability.

---

## 🧩 Features

- ✅ FastAPI server for serving predictions
- ✅ Streamlit dashboard for model insights and manual retraining
- ✅ Drift detection with customizable thresholds
- ✅ Auto-triggered or manual retraining based on logged inference data
- ✅ Model versioning and metadata stored in `models/`
- ✅ Modular, extensible, and production-ready codebase

---

## ⚙️ Setup

### 🔧 Prerequisites

* Python 3.10+
* Pip

### 🛠️ Installation

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

---

## 🚀 Run the Services

### ✅ FastAPI Inference API

```bash
uvicorn app.main:app --reload
```

📍 Access at: `http://127.0.0.1:8000`

---

### 📊 Streamlit Monitoring Dashboard

```bash
streamlit run dashboard.py
```

Use this dashboard to:

* Monitor data drift visually
* Inspect inference logs
* Trigger model retraining
* View current model metadata

---

## 🔮 Inference API Usage

### 🔎 Health Check

```http
GET /
```

### 📦 Model Version Info

```http
GET /version
```

### 📈 Make Prediction

```http
POST /predict
Content-Type: application/json

{
  "features": [0.1, 0.2, 0.3, ..., 0.n]
}
```

**Response**:

```json
{
  "prediction": 0,
  "probability": 0.913
}
```

---

## 📉 Drift Detection

Run the drift monitor to compare the most recent data distribution with baseline training data:

```bash
python monitoring/drift_monitor/drift_monitor.py
```

If drift exceeds a threshold, a message will indicate that retraining is needed.

---

## 🔁 Model Retraining

Automatically triggered by drift monitor or manually via Streamlit UI.

```bash
python training/retraining_pipeline.py
```

Performs:

* Preprocessing & validation
* Model training & evaluation
* Versioned model saving
* Metadata update

Model artifacts and metadata are saved to `models/`.

---

## 🧾 Inference Logging

All incoming inference data is logged to:

```
monitoring/inference_log.csv
```

Each log includes:

* Timestamp
* Input features
* Prediction and probability
* Model version

---

## 🧠 Model Metadata

Saved in:

```
models/model_metadata.json
```

Tracks:

* Model version
* Accuracy on validation set
* Training timestamp (UTC)
* Number of features

---

## 🚧 Future Enhancements

* 📬 Alerting via email/SMS on drift
* 🐳 Dockerize entire pipeline
* ☁️ Deploy with autoscaling (Render, Hugging Face Spaces, or AWS)
* 🔁 Schedule retraining jobs with CRON or Airflow
* ✅ CI/CD for model validation and deployment

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👤 Author

Created by **Koutilya Ganapathiraju**

* 📧 Email: [gkoutilyaraju@gmail.com](mailto:gkoutilyaraju@gmail.com)
* 🧑‍💻 GitHub: [@GKoutilya](https://github.com/GKoutilya)
* 💼 LinkedIn: [Koutilya Ganapathiraju](https://linkedin.com/in/koutilya-ganapathiraju-0a3350182)

---