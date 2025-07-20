import joblib

def load_model(path="models/model_v1.pkl"):
    """
    Load a trained machine learning model from the specified file path using joblib.

    Args:
        path (str): Path to the saved model file. Defaults to "models/model_v1.pkl".

    Returns:
        sklearn.base.BaseEstimator: Loaded scikit-learn model.
    """
    return joblib.load(path)
