"""
    preprocess.py

    Loads, imputes, and scales the SECOM manufacturing dataset for model training.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():
    """
    Loads and preprocesses SECOM data:
    - Drops rows with missing or invalid labels
    - Imputes missing values in features
    - Scales features

    Returns:
        X_scaled (np.ndarray): Feature matrix
        y (np.ndarray): Binary label vector
    """
    # Load data
    features = pd.read_csv('data/raw/secom.data', sep='\\s+', header=None)
    labels = pd.read_csv('data/raw/secom_labels.data', sep='\\s+', header=None)[0]

    # Combine for alignment
    df = features.copy()
    df['label'] = labels

    # Drop rows where label is NaN or not 0/1
    df = df[df['label'].isin([0, 1])]
    df = df.groupby('label').apply(lambda x: x.sample(n=min(len(x), 100), random_state=42)).reset_index(drop=True)
    df = df.dropna(subset=['label'])

    # Extract features and labels
    X = df.drop(columns=['label']).values
    y = df['label'].astype(float).values  # use float so np.isnan() works

    # Debug info
    print(f"[DEBUG] After cleaning: {len(y)} samples")
    print(f"[DEBUG] Unique labels: {np.unique(y)}")
    print(f"[DEBUG] NaNs in y: {np.isnan(y).sum()}")

    # Final defensive filtering (remove any NaNs just in case)
    non_nan_mask = ~np.isnan(y)
    X = X[non_nan_mask]
    y = y[non_nan_mask].astype(int)  # convert to int for classification

    # Impute features
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, y