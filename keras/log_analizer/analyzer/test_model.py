import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib
import os

# Paths
LOG_FILE_PATH = "/app/logs/server_logs.txt"
SCALER_PATH = "/app/models/scaler.pkl"
MODEL_PATH = "/app/models/lstm_autoencoder_v2.keras"
CHARTS_OUTPUT_DIR = "/app/charts/"
CHART_PATH = "/app/charts/loss_curves_lstm_v2.png"

# Parameters
TIME_STEPS = 50  # Increased number of timesteps in each sequence
BATCH_SIZE = 64
EPOCHS = 100
VALIDATION_SPLIT = 0.2

# Function to load logs
def load_logs(file_path):
    logs = []
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split(" | ")
            cpu = float(parts[1].split(": ")[1][:-1])
            ram = float(parts[2].split(": ")[1][:-1])
            disk = float(parts[3].split(": ")[1][:-1])
            logs.append([cpu, ram, disk])
    return pd.DataFrame(logs, columns=["cpu", "ram", "disk"])

# Function to create statistical features
def augment_features(df):
    df['cpu_mean'] = df['cpu'].rolling(window=10, min_periods=1).mean()
    df['cpu_std'] = df['cpu'].rolling(window=10, min_periods=1).std()
    df['ram_mean'] = df['ram'].rolling(window=10, min_periods=1).mean()
    df['ram_std'] = df['ram'].rolling(window=10, min_periods=1).std()
    df['disk_mean'] = df['disk'].rolling(window=10, min_periods=1).mean()
    df['disk_std'] = df['disk'].rolling(window=10, min_periods=1).std()
    return df.fillna(0)  # Fill NaN values with 0

# Function to prepare sequences for LSTM
def create_sequences(data, time_steps):
    sequences = []
    for i in range(len(data) - time_steps + 1):
        sequences.append(data[i : i + time_steps])
    return np.array(sequences)

# Function to preprocess data
def preprocess_data(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    sequences = create_sequences(scaled_data, TIME_STEPS)
    return sequences, scaler

# Build Bidirectional LSTM Autoencoder
def build_lstm_autoencoder(input_shape):
    inputs = Input(shape=input_shape)
    encoded = Bidirectional(LSTM(128, activation="relu", return_sequences=True))(inputs)
    encoded = LSTM(64, activation="relu")(encoded)
    decoded = RepeatVector(input_shape[0])(encoded)
    decoded = LSTM(64, activation="relu", return_sequences=True)(decoded)
    decoded = Bidirectional(LSTM(128, activation="relu", return_sequences=True))(decoded)
    outputs = TimeDistributed(Dense(input_shape[1]))(decoded)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model

# Plot training loss
def plot_loss(history, output_path):
    plt.figure()
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(output_path)
    plt.close()

# Main training function
def main():
    print("Loading logs...")
    df = load_logs(LOG_FILE_PATH)
    print(f"Loaded {len(df)} logs.")

    print("Augmenting features...")
    df = augment_features(df)

    print("Preprocessing data...")
    sequences, scaler = preprocess_data(df)
    print(f"Prepared {len(sequences)} sequences for LSTM.")

    print("Saving scaler...")
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    print("Building the Bidirectional LSTM autoencoder...")
    model = build_lstm_autoencoder(sequences.shape[1:])
    model.summary()

    print("Training the Bidirectional LSTM autoencoder...")
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        sequences,
        sequences,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        shuffle=True,
        callbacks=[early_stopping]
    )

    print("Saving the model...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)

    print("Plotting loss curves...")
    os.makedirs(CHARTS_OUTPUT_DIR, exist_ok=True)
    plot_loss(history, CHART_PATH)

    print("Training complete.")

if __name__ == "__main__":
    main()
