import streamlit as st
import pandas as pd
import requests
import subprocess
import sys
import os
import time
import json

st.set_page_config(page_title="ML Deployment Dashboard", layout="wide")

st.title("ML Deployment & Drift Monitoring Dashboard")
st.markdown("This dashboard enables real-time predictions, drift monitoring, and on-demand retraining of your deployed model.")

st.header("Live Inference")

# Input features
features = st.text_input("Enter comma-separated feature values (e.g. 5.1, 3.5, 1.4, 0.2):")

if st.button("Run Inference"):
    st.info("Button clicked. Processing input...")
    
    if not features.strip():
        st.warning("Please enter some feature values.")
    else:
        try:
            # Parse and validate input
            features_list = [float(f.strip()) for f in features.split(",") if f.strip() != ""]

            if not features_list:
                st.warning("No valid numeric values found. Please enter numbers only.")
            else:
                st.write("Sending request to FastAPI backend with payload:", features_list)

                try:
                    response = requests.post(
                        "http://localhost:8000/predict",
                        json={"features": features_list},
                        timeout=5  # seconds
                    )

                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Prediction: **{data['prediction']}**, Probability: **{data['probability']:.2f}**")
                    else:
                        st.error(f"Inference request failed. Status code: {response.status_code}")
                        st.code(response.text)

                except requests.exceptions.ConnectionError:
                    st.error("ConnectionError: Could not connect to FastAPI server. Is it running on port 8000?")
                except requests.exceptions.Timeout:
                    st.error("TimeoutError: The request to the FastAPI server took too long.")
                except Exception as e:
                    st.error(f"Unexpected request error: {e}")

        except ValueError:
            st.error("Invalid input: make sure all feature values are numeric.")
        except Exception as e:
            st.error(f"Unexpected input processing error: {e}")

python_executable = sys.executable

# Drift Detection
st.header("Drift Detection")
if st.button("Run Drift Detection"):
    with st.spinner("Checking for data drift..."):
        result = subprocess.run([python_executable, "monitoring/drift_monitor/drift_monitor.py"], capture_output=True, text=True)
        st.code(result.stdout or result.stderr)

# Retraining
st.header("Model Retraining")
if st.button("Run Model Retraining"):
    with st.spinner("Retraining in progress..."):
        result = subprocess.run([sys.executable, "training/retraining_pipeline.py"], capture_output=True, text=True)

        st.code(result.stdout)
        if result.stderr:
            st.error(result.stderr)


st.header("Model Info")

try:
    with open("models/model_metadata.json", "r") as f:
        metadata = json.load(f)
    model_info = pd.DataFrame([metadata])
    st.subheader("Current Model Metadata")
    st.dataframe(model_info)
except FileNotFoundError:
    st.warning("Model metadata not available. Retrain to generate it.")
except Exception as e:
    st.error(f"Failed to load model metadata: {e}")
