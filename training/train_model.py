"""
    train_model.py

    Trains a Random Frest classifier on the SECOM dataset, evaluates its performance, and saves the trained model.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from training.preprocess import load_and_preprocess_data
import joblib
import os

def main():
    """
        Main training routine:
            Loads and preprocesses data
            Trains a Random Forest model
            Evaluates performance
            Saves the trained model to disk
    """
    print("Loading and preprocessing data...")
    x, y = load_and_preprocess_data()

    print("Splitting data into train/test sets...")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Training Random Forest model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(x_train, y_train)

    print("Evaluating model performance...")
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))

    print("Saving model to disk...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/model_v1.pkl")

    print("Training compltete. Model saves to models/model_v1.pkl")


if __name__ == "__main__":
    main()