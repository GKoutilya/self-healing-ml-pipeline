import joblib

def load_model(path="models/model_v1.pkl"):
    return joblib.load(path)
