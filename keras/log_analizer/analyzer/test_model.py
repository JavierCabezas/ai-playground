import re
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Debug flag for evaluation
DEBUG_WITH_ANOMALY_FLAG = True

# File paths
LOG_FILE_PATH = "/app/logs/server_logs.txt"
DEBUG_LOG_FILE_PATH = "/app/logs/debug_logs.txt"
MODEL_FILE_PATH = "/app/models/autoencoder_model.keras"
SCALER_FILE_PATH = "/app/models/scaler.pkl"

# Directory for saving charts
CHARTS_DIR = "/app/charts"


def ensure_charts_dir():
    """Ensure the charts directory exists."""
    os.makedirs(CHARTS_DIR, exist_ok=True)


def parse_logs(file_path):
    """Load and preprocess logs."""
    logs = []
    with open(file_path, "r") as file:
        for line in file:
            match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(.*?)\] \[(.*?)\]: .*?([\d.]+)%", line)
            if match:
                timestamp, server, component, value = match.groups()
                logs.append({
                    "timestamp": pd.to_datetime(timestamp),
                    "server": server,
                    "component": component,
                    "value": float(value),
                })
    return pd.DataFrame(logs)


def load_debug_logs(debug_file_path, log_file_length):
    """Load debug logs and create an 'anomalous' column based on anomaly line numbers."""
    anomalous_indices = []
    with open(debug_file_path, "r") as file:
        for line in file:
            anomalous_indices.append(int(line.strip()))

    # Create a column indicating whether each log entry is anomalous
    anomalous_column = np.zeros(log_file_length, dtype=bool)
    anomalous_column[anomalous_indices] = True

    return anomalous_column


def preprocess_unseen_data(df, scaler):
    """Normalize unseen data using the trained scaler."""
    df.loc[:, "scaled_value"] = scaler.transform(df[["value"]])
    return df


def evaluate_model(df):
    """Evaluate the model using the 'anomalous' field as ground truth."""
    if "anomalous" not in df.columns:
        print("Ground truth ('anomalous') field not found. Skipping evaluation.")
        return

    # Compare ground truth to predictions
    ground_truth = df["anomalous"].astype(bool)
    predictions = df["is_anomaly"].astype(bool)

    # Generate confusion matrix
    tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP): {tp}")

    # Calculate and print metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\nEvaluation Metrics:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1_score:.2f}")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(ground_truth, predictions, target_names=["Normal", "Anomalous"]))


def test_model(unseen_data, autoencoder, scaler, threshold):
    """Test the trained model on unseen data."""
    print("Testing on unseen data...")

    # Preprocess unseen data
    unseen_data = preprocess_unseen_data(unseen_data, scaler)

    # Prepare input
    X_unseen = unseen_data["scaled_value"].values.reshape(-1, 1)

    # Calculate reconstruction errors
    reconstructions = autoencoder.predict(X_unseen)
    reconstruction_errors = np.mean(np.square(reconstructions - X_unseen), axis=1)

    # Identify anomalies
    unseen_data.loc[:, "reconstruction_error"] = reconstruction_errors
    unseen_data.loc[:, "is_anomaly"] = reconstruction_errors > threshold

    # Save anomalies to a file for manual inspection
    anomalies_df = unseen_data[unseen_data["is_anomaly"]]
    anomalies_df.to_csv("/app/logs/anomalies.csv", index=False)
    print(f"Saved anomalies to /app/logs/anomalies.csv")

    # Debugging: Evaluate model performance
    if DEBUG_WITH_ANOMALY_FLAG:
        evaluate_model(unseen_data)

    # Metrics
    total = len(reconstruction_errors)
    num_anomalies = np.sum(unseen_data["is_anomaly"])
    print(f"Total entries: {total}")
    print(f"Anomalies detected: {num_anomalies}")
    print(f"Percentage of anomalies: {100 * num_anomalies / total:.2f}%")

    # Save reconstruction error histogram
    ensure_charts_dir()
    plt.figure(figsize=(10, 6))
    plt.hist(reconstruction_errors, bins=50, alpha=0.6, label='Unseen Data')
    plt.axvline(x=threshold, color='red', linestyle='--', label='Threshold')
    plt.title('Reconstruction Errors on Unseen Data')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.legend()
    chart_path = os.path.join(CHARTS_DIR, "unseen_data_errors.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved reconstruction error histogram to {chart_path}")


def main():
    # Load logs
    print("Loading logs...")
    df = parse_logs(LOG_FILE_PATH)
    print(f"Loaded {len(df)} logs.")

    # Load the trained model and scaler
    print("Loading trained model and scaler...")
    autoencoder = load_model(MODEL_FILE_PATH)
    scaler = joblib.load(SCALER_FILE_PATH)

    # Load debug logs for ground truth
    if DEBUG_WITH_ANOMALY_FLAG:
        print("Loading debug logs for ground truth...")
        df["anomalous"] = load_debug_logs(DEBUG_LOG_FILE_PATH, len(df))

    # Split unseen data (last 2 weeks)
    test_cutoff = df["timestamp"].min() + pd.Timedelta(weeks=4)
    unseen_data = df[df["timestamp"] >= test_cutoff]

    # Determine threshold (95th percentile of training errors)
    print("Calculating threshold...")
    training_cutoff = df["timestamp"].min() + pd.Timedelta(weeks=4)
    training_data = df[df["timestamp"] < training_cutoff]
    training_data = preprocess_unseen_data(training_data, scaler)

    X_train = training_data["scaled_value"].values.reshape(-1, 1)
    reconstructions = autoencoder.predict(X_train)
    train_errors = np.mean(np.square(reconstructions - X_train), axis=1)
    threshold = np.percentile(train_errors, 95)

    print(f"Threshold for anomaly detection: {threshold}")

    # Test on unseen data
    test_model(unseen_data, autoencoder, scaler, threshold)


if __name__ == "__main__":
    main()
