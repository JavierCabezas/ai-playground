import re
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os
from analysis_tools import (
    plot_loss_curves,
    plot_error_distribution,
    calculate_confusion_matrix,
    plot_roc_pr_curves
)

# File paths
LOG_FILE_PATH = "/app/logs/server_logs.txt"
MODEL_FILE_PATH = "/app/models/autoencoder_model.keras"  # Updated to .keras format
SCALER_FILE_PATH = "/app/models/scaler.pkl"

# Training configuration
TRAINING_WEEKS = 4
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
LATENT_DIM = 4  # Dimensionality of the bottleneck (compressed representation)

# Flags to enable/disable specific features
PLOT_LOSS_CURVES = True
PLOT_ERROR_DISTRIBUTION = True
CALCULATE_CONFUSION_MATRIX = True
PLOT_ROC_PR_CURVES = True

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

def preprocess_data(df, training_weeks):
    """Split and normalize the data."""
    split_point = df["timestamp"].min() + pd.Timedelta(weeks=training_weeks)
    training_data = df[df["timestamp"] < split_point]
    validation_data = df[df["timestamp"] >= split_point]

    # Normalize the values using Min-Max Scaling
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    training_data["scaled_value"] = scaler.fit_transform(training_data[["value"]])
    validation_data["scaled_value"] = scaler.transform(validation_data[["value"]])

    return training_data, validation_data, scaler

def build_autoencoder(input_dim, latent_dim):
    """Define the autoencoder model."""
    inputs = Input(shape=(input_dim,))
    encoded = Dense(16, activation="relu")(inputs)
    bottleneck = Dense(latent_dim, activation="relu")(encoded)
    decoded = Dense(16, activation="relu")(bottleneck)
    outputs = Dense(input_dim, activation="sigmoid")(decoded)

    autoencoder = Model(inputs, outputs)
    autoencoder.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse")
    return autoencoder

def main():
    # Load logs
    print("Loading logs...")
    df = parse_logs(LOG_FILE_PATH)
    print(f"Loaded {len(df)} logs.")

    # Preprocess data
    print("Preprocessing data...")
    training_data, validation_data, scaler = preprocess_data(df, TRAINING_WEEKS)

    # Prepare input for the autoencoder
    X_train = training_data["scaled_value"].values.reshape(-1, 1)
    X_val = validation_data["scaled_value"].values.reshape(-1, 1)

    # Build and train the autoencoder
    print("Building autoencoder...")
    autoencoder = build_autoencoder(input_dim=1, latent_dim=LATENT_DIM)

    early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    print("Training autoencoder...")
    history = autoencoder.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=EPOCHS,
                              batch_size=BATCH_SIZE, shuffle=True, callbacks=[early_stopping], verbose=1)

    # Save the model and scaler
    print("Saving model and scaler...")
    os.makedirs(os.path.dirname(MODEL_FILE_PATH), exist_ok=True)
    autoencoder.save(MODEL_FILE_PATH)  # Save as .keras format
    joblib.dump(scaler, SCALER_FILE_PATH)
    print(f"Model saved to {MODEL_FILE_PATH}.")
    print(f"Scaler saved to {SCALER_FILE_PATH}.")

    # Optional evaluations
    if PLOT_LOSS_CURVES:
        plot_loss_curves(history)

    # Calculate reconstruction errors
    train_reconstructions = autoencoder.predict(X_train)
    train_errors = np.mean(np.square(train_reconstructions - X_train), axis=1)

    val_reconstructions = autoencoder.predict(X_val)
    val_errors = np.mean(np.square(val_reconstructions - X_val), axis=1)

    if PLOT_ERROR_DISTRIBUTION:
        plot_error_distribution(train_errors, val_errors)

    threshold = np.percentile(train_errors, 95)

    if CALCULATE_CONFUSION_MATRIX:
        calculate_confusion_matrix(val_errors, threshold)

    if PLOT_ROC_PR_CURVES:
        plot_roc_pr_curves(val_errors, threshold)

if __name__ == "__main__":
    main()
