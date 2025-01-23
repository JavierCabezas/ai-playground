import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt

# Paths
LOG_FILE_PATH = "/app/logs/server_logs.txt"
DEBUG_LOG_FILE_PATH = "/app/logs/debug_logs.txt"
SCALER_PATH = "/app/models/scaler.pkl"
MODEL_PATH = "/app/models/lstm_autoencoder.keras"
CHARTS_OUTPUT_DIR = "/app/charts/"
ANOMALIES_OUTPUT_PATH = "/app/logs/anomalies.csv"
ERROR_HISTOGRAM_PATH = "/app/charts/unseen_data_errors.png"

# Parameters
TIME_STEPS = 30

# Function to load logs
def load_logs(file_path):
    logs = []
    timestamps = []
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split(" | ")
            timestamps.append(pd.to_datetime(parts[0]))
            cpu = float(parts[1].split(": ")[1][:-1])
            ram = float(parts[2].split(": ")[1][:-1])
            disk = float(parts[3].split(": ")[1][:-1])
            logs.append([cpu, ram, disk])
    return pd.DataFrame(logs, columns=["cpu", "ram", "disk"]).assign(timestamp=timestamps)

# Function to create sequences
def create_sequences(data, time_steps):
    sequences = []
    for i in range(len(data) - time_steps + 1):
        sequences.append(data[i : i + time_steps])
    return np.array(sequences)

# Function to evaluate the model
def evaluate_model(ground_truth_anomalies, detected_anomalies, total_samples, timestamps):
    y_true = np.zeros(total_samples)
    y_pred = np.zeros(total_samples)

    # Map datetime anomalies to indices
    ground_truth_indices = [
        i for i, timestamp in enumerate(timestamps) if timestamp in ground_truth_anomalies
    ]

    y_true[ground_truth_indices] = 1
    y_pred[detected_anomalies] = 1

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return precision, recall, f1, cm

# Function to plot reconstruction error histogram
def plot_error_histogram(errors, threshold, output_path):
    plt.figure()
    plt.hist(errors, bins=50, alpha=0.7, label="Reconstruction Error")
    plt.axvline(x=threshold, color="red", linestyle="--", label="Threshold")
    plt.title("Reconstruction Error Histogram")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(output_path)
    plt.close()

# Main testing function
def main():
    print("Loading logs...")
    df = load_logs(LOG_FILE_PATH)
    print(f"Loaded {len(df)} logs.")

    print("Loading scaler...")
    scaler = joblib.load(SCALER_PATH)

    print("Creating sequences...")
    scaled_data = scaler.transform(df[["cpu", "ram", "disk"]])
    sequences = create_sequences(scaled_data, TIME_STEPS)
    print(f"Prepared {len(sequences)} sequences for testing.")

    print("Loading trained model...")
    model = load_model(MODEL_PATH)

    print("Calculating reconstruction errors...")
    reconstructed = model.predict(sequences)
    reconstruction_errors = np.mean(np.power(sequences - reconstructed, 2), axis=(1, 2))

    print("Optimizing threshold...")
    threshold = np.percentile(reconstruction_errors, 99)  # Use 99th percentile as threshold
    print(f"Optimal Threshold: {threshold}")

    print("Identifying anomalies...")
    anomaly_indices = np.where(reconstruction_errors > threshold)[0]
    print(f"Detected {len(anomaly_indices)} anomalies.")

    print("Loading ground truth anomalies...")
    with open(DEBUG_LOG_FILE_PATH, "r") as file:
        ground_truth_anomalies = [pd.to_datetime(line.split(" | ")[0]) for line in file.readlines()]
    print(f"Loaded {len(ground_truth_anomalies)} ground truth anomalies.")

    print("Evaluating model...")
    precision, recall, f1, cm = evaluate_model(
        ground_truth_anomalies, anomaly_indices, len(df) - TIME_STEPS + 1, df["timestamp"][TIME_STEPS - 1:].values
    )
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print("Confusion Matrix:")
    print(cm)

    print("Saving reconstruction error histogram...")
    plot_error_histogram(reconstruction_errors, threshold, ERROR_HISTOGRAM_PATH)
    print(f"Saved anomalies to {ANOMALIES_OUTPUT_PATH}")
    print(f"Saved reconstruction error histogram to {ERROR_HISTOGRAM_PATH}")

if __name__ == "__main__":
    main()
