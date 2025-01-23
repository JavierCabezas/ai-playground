import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import os

# Constants
LOG_FILE_PATH = "/app/logs/server_logs.txt"
DEBUG_LOG_FILE_PATH = "/app/logs/debug_logs.txt"
CHARTS_OUTPUT_DIR = "/app/charts/"
TIME_PERIOD = 4  # Number of months to generate data
LOG_INTERVAL = 30  # Log interval in seconds
ANOMALY_PROBABILITY = 0.0005  # 0.05% chance of anomaly

# Amplified anomaly effects
CPU_ANOMALY_RANGE = (80, 100)  # CPU spikes to 80-100%
RAM_ANOMALY_RANGE = (90, 100)  # RAM spikes to 90-100%
DISK_ANOMALY_INCREASE = 20     # Disk increases by 20%


def generate_logs():
    """Generates synthetic logs with anomalies."""
    print(f"Starting log generation for {TIME_PERIOD} months...")

    start_time = datetime(2024, 1, 1)  # Example start date
    end_time = start_time + timedelta(weeks=4 * TIME_PERIOD)
    current_time = start_time

    logs = []
    anomalies = []

    while current_time < end_time:
        # Generate normal system metrics
        cpu = round(random.uniform(5, 50), 1)  # Normal CPU usage
        ram = round(random.uniform(10, 70), 1)  # Normal RAM usage
        disk = round(random.uniform(2, 30), 1)  # Normal Disk usage

        is_anomaly = False

        # Check if this log entry will be an anomaly
        if random.random() < ANOMALY_PROBABILITY:
            cpu = round(random.uniform(*CPU_ANOMALY_RANGE), 1)
            ram = round(random.uniform(*RAM_ANOMALY_RANGE), 1)
            disk = min(round(disk + DISK_ANOMALY_INCREASE, 1), 100)  # Cap disk at 100%
            anomalies.append({"timestamp": current_time, "event": "Anomaly Detected"})
            is_anomaly = True

        logs.append({
            "timestamp": current_time,
            "cpu": cpu,
            "ram": ram,
            "disk": disk,
            "is_anomaly": is_anomaly,
        })

        current_time += timedelta(seconds=LOG_INTERVAL)

    print(f"Generated {len(logs)} log entries for {TIME_PERIOD} months.")
    print(f"Anomalies detected: {len(anomalies)}")

    # Save logs to file
    save_logs_to_files(logs, anomalies)

    # Plot charts
    df = pd.DataFrame(logs)
    plot_distributions(df)
    plot_time_series_with_anomalies(df, CHARTS_OUTPUT_DIR)


def save_logs_to_files(logs, anomalies):
    """Saves logs and anomalies to files."""
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

    with open(LOG_FILE_PATH, "w") as log_file:
        for log in logs:
            log_file.write(f"{log['timestamp']} | CPU: {log['cpu']}% | RAM: {log['ram']}% | Disk: {log['disk']}%\n")

    with open(DEBUG_LOG_FILE_PATH, "w") as debug_file:
        for anomaly in anomalies:
            debug_file.write(f"{anomaly['timestamp']} | EVENT: {anomaly['event']}\n")


def plot_distributions(df):
    """Plots distributions of CPU, RAM, and Disk usage."""
    for metric, color in zip(["cpu", "ram", "disk"], ["blue", "green", "orange"]):
        plt.figure()
        plt.hist(df[metric], bins=50, color=color, alpha=0.7)
        plt.title(f"{metric.upper()} Usage Distribution ({TIME_PERIOD} months)")
        plt.xlabel(f"{metric.upper()} Usage (%)")
        plt.ylabel("Frequency")
        plt.savefig(f"{CHARTS_OUTPUT_DIR}logs_{metric}_usage_distribution.png")
        plt.close()

    print("Saved usage distribution charts.")


def plot_time_series_with_anomalies(df, output_dir):
    """Plots each system metric over time with anomalies marked, split by month."""
    df["month"] = df["timestamp"].dt.to_period("M")  # Extract the month from the timestamp

    metrics = ["cpu", "ram", "disk"]
    colors = {"cpu": "blue", "ram": "green", "disk": "orange"}

    for metric in metrics:
        for month, month_data in df.groupby("month"):
            plt.figure(figsize=(12, 6))
            plt.plot(
                month_data["timestamp"],
                month_data[metric],
                label=f"{metric.upper()} Usage (%)",
                color=colors[metric],
                alpha=0.7,
            )

            # Mark anomalies
            anomalies = month_data[month_data["is_anomaly"]]
            if not anomalies.empty:
                plt.scatter(
                    anomalies["timestamp"],
                    anomalies[metric],
                    color="red",
                    label="Anomaly",
                    marker="x",
                    alpha=0.8,
                )

            plt.title(f"{metric.upper()} Usage Over Time ({month})")
            plt.xlabel("Time")
            plt.ylabel("Usage (%)")
            plt.legend()
            plt.tight_layout()

            # Save the chart
            output_path = os.path.join(output_dir, f"logs_{metric}_usage_{month}.png")
            plt.savefig(output_path)
            plt.close()

            print(f"Saved {metric.upper()} chart for {month} to {output_path}")


if __name__ == "__main__":
    generate_logs()
