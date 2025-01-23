import json
import random
from datetime import datetime, timedelta
import os

LOG_FILE_PATH = "/app/logs/server_logs.txt"
DEBUG_FILE_PATH = "/app/logs/debug_logs.txt"
LOG_DURATION_WEEKS = 6  # Generate logs for 6 weeks
LOG_FREQUENCY_SECONDS = 30  # Interval between logs

# Load configuration
with open("config.json", "r") as config_file:
    config = json.load(config_file)

server_name = config["server_name"]
components = config["components"]
value_range = config["value_range"]
patterns = config["patterns"]
anomaly_config = config["anomalies"]

MESSAGE_FORMATS = {
    "CPU": ["CPU usage at {value}%", "Processor utilization: {value}%"],
    "Memory": ["Memory usage: {value}%", "RAM utilization: {value}%"],
    "Disk": ["Disk utilization: {value}%", "Storage usage at {value}%"]
}


def generate_value(component, current_time):
    """Generate a value based on patterns and random noise."""
    pattern = patterns.get(component, {})
    base_value = random.uniform(value_range[component][0], value_range[component][1])

    # Apply time-based patterns
    if "peak_hours" in pattern:
        start, end = pattern["peak_hours"]
        if start <= current_time.hour < end:
            base_value *= pattern.get("peak_multiplier", 1.5)
    if "off_hours" in pattern:
        start, end = pattern["off_hours"]
        if start <= current_time.hour < end:
            base_value *= pattern.get("off_multiplier", 0.5)

    # Add random fluctuation
    noise = random.uniform(-pattern.get("noise_level", 1), pattern.get("noise_level", 1))
    return round(base_value + noise, 2)

def is_anomalous(component, value):
    """Determine if a value is atypical based on anomaly configuration."""
    if component in anomaly_config:
        for anomaly_type, range_values in anomaly_config[component].items():
            if range_values[0] <= value <= range_values[1]:
                return True
    return False

def generate_log(timestamp, component, line_number):
    """Generate a single log entry."""
    value = generate_value(component, timestamp)
    message_format = random.choice(MESSAGE_FORMATS[component])
    message = message_format.format(value=value)
    log_entry = f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} [{server_name}] [{component}]: {message}"

    # If the log is atypical, write the line number to the debug file
    if is_anomalous(component, value):
        with open(DEBUG_FILE_PATH, "a") as debug_file:
            debug_file.write(f"{line_number}\n")

    return log_entry

def main():
    start_time = datetime.now() - timedelta(weeks=LOG_DURATION_WEEKS)  # Start 3 weeks ago
    end_time = datetime.now()  # End at the current time
    current_time = start_time
    line_number = 1  # Line number starts at 1

    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    with open(LOG_FILE_PATH, "w") as log_file, open(DEBUG_FILE_PATH, "w") as debug_file:
        # Clear the debug file at the start of each run
        debug_file.write("")

        while current_time <= end_time:
            for component in components:
                # Add random time noise
                time_noise = random.randint(-10, 10)
                adjusted_time = current_time + timedelta(seconds=time_noise)

                # Generate log and increment line number
                log_entry = generate_log(adjusted_time, component, line_number)
                log_file.write(log_entry + "\n")
                print(f"Generated log: {log_entry}")
                line_number += 1

            current_time += timedelta(seconds=LOG_FREQUENCY_SECONDS)

if __name__ == "__main__":
    main()
