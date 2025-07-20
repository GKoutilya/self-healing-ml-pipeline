import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from evidently import Report
from evidently.presets import DataDriftPreset
import evidently

# Constants
MIN_ROWS_REQUIRED = 20
DEFAULT_REF_SIZE = 200
DEFAULT_CUR_SIZE = 200
LOG_FILE = os.path.join("monitoring", "inference_log.csv")

def run_drift_monitor():
    try:
        df = pd.read_csv(LOG_FILE)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        print(f"Loaded data shape: {df.shape}")
        print("Column names:", df.columns.tolist())
        print("First 5 rows:\n", df.head())
        print("Data types:\n", df.dtypes)

        total_rows = df.shape[0]
        if total_rows < MIN_ROWS_REQUIRED:
            raise ValueError(f"Drift monitoring failed: Not enough total rows ({total_rows}). Need at least {MIN_ROWS_REQUIRED}.")

        max_window = total_rows // 2
        ref_size = min(DEFAULT_REF_SIZE, max_window)
        cur_size = min(DEFAULT_CUR_SIZE, max_window)

        if ref_size + cur_size > total_rows:
            raise ValueError("Not enough rows in logged data to compute drift. Add more data.")

        reference_data = df.iloc[:ref_size].drop(columns=["timestamp"], errors="ignore")
        current_data = df.iloc[-cur_size:].drop(columns=["timestamp"], errors="ignore")

        # Drop constant columns
        constant_cols = reference_data.columns[reference_data.nunique() <= 1]
        if len(constant_cols) > 0:
            print("Dropping constant columns:", constant_cols.tolist())
            reference_data = reference_data.drop(columns=constant_cols)
            current_data = current_data.drop(columns=constant_cols)

        print("NaNs in reference:\n", reference_data.isna().sum())
        print("NaNs in current:\n", current_data.isna().sum())

        report = Report(metrics=[DataDriftPreset()])

        try:
            my_report = report.run(reference_data=reference_data, current_data=current_data)
        except Exception as ee:
            print("Evidently failed with EvidentlyException:", str(ee))
            raise
        except Exception as ex:
            print("Unexpected failure during drift computation:", str(ex))
            raise

        drift_dir = os.path.join("monitoring", "drift_monitor", "results")
        os.makedirs(drift_dir, exist_ok=True)
        my_report.save_html("drift_report.html")
        print(f"Drift report saved to drift_report.html")

    except Exception as e:
        print(f"Drift monitoring failed: {e}")

if __name__ == "__main__":
    run_drift_monitor()