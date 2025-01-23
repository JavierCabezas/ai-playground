# Log Analyzer Project

This project generates, processes, and analyzes server logs to identify anomalies using a deep learning-based autoencoder model. The system includes modules for log generation, preprocessing, model training, and anomaly detection. It provides visual insights into system metrics and anomalies.

---

## Project Structure

```
log_analizer/
├── analyzer/
│   ├── train_model.py        # Script to train the autoencoder
│   ├── test_model.py         # Script to test the trained autoencoder
│   ├── analysis_tools.py     # Contains utilities for analysis and chart generation
├── generator/
│   ├── generator.py          # Log generation script
│   ├── processes.py          # Defines system processes
│   ├── base_process.py       # Base process class
│   ├── events.py             # Defines anomalies and their probabilities
│   ├── logs/                 # Folder for generated logs
│   ├── charts/               # Folder for generated charts
├── docker/
│   ├── train_model.Dockerfile # Dockerfile for training
│   ├── test_model.Dockerfile  # Dockerfile for testing
│   ├── generator.Dockerfile   # Dockerfile for log generation
│   ├── docker-compose.yml     # Compose file for all services
├── README.md                # Project documentation
```

---

## Requirements

- Python 3.8+
- Docker & Docker Compose
- Python packages: TensorFlow, Pandas, Matplotlib, Scikit-learn, NumPy, Joblib

---

## Usage Instructions

### **1. Log Generation**

Generate synthetic logs with anomalies:

```bash
docker-compose run --rm generator
```

**Details:**
- Generates logs for 6 months.
- Stores logs in `generator/logs/server_logs.txt`.
- Anomalies are stored in `generator/logs/debug_logs.txt`.
- Charts of CPU, RAM, and Disk usage are saved in `generator/charts/`.

---

### **2. Train the Model**

Train an autoencoder model using generated logs:

```bash
docker-compose run --rm train_model
```

**Details:**
- Reads `server_logs.txt`.
- Splits data into training (80%) and validation (20%).
- Saves the trained model to `analyzer/models/autoencoder_model.keras`.
- Saves the scaler to `analyzer/models/scaler.pkl`.
- Loss curves are saved in `analyzer/charts/loss_curves.png`.

---

### **3. Test the Model**

Test the model and detect anomalies:

```bash
docker-compose run --rm test_model
```

**Details:**
- Reads `server_logs.txt` and `debug_logs.txt`.
- Loads the trained model and scaler.
- Calculates reconstruction errors and identifies anomalies.
- Saves detected anomalies to `analyzer/logs/anomalies.csv`.
- Generates reconstruction error histogram in `analyzer/charts/unseen_data_errors.png`.

---

### **4. Visualize Logs and Anomalies**

Generated charts include:
1. **Distributions** of CPU, RAM, and Disk usage.
2. **Time-series charts** with anomalies per system metric (e.g., CPU, RAM, Disk) split by month.

Charts are saved in `generator/charts/` and updated during log generation.

---

## Features

1. **Log Generator**
   - Simulates realistic system metrics.
   - Introduces anomalies with configurable probabilities and impacts.

2. **Autoencoder**
   - Learns normal system behavior.
   - Flags anomalies based on reconstruction errors.

3. **Visualization**
   - CPU, RAM, and Disk usage distributions.
   - Time-series visualizations of metrics and anomalies.

---

## Configuration Options

### **Log Generator (generator.py)**
- `TIME_PERIOD`: Number of months to generate logs (default: 6).
- `LOG_INTERVAL`: Interval between logs in seconds (default: 30).
- `ANOMALY_PROBABILITY`: Probability of anomalies (default: 0.05%).

### **Anomalies (events.py)**
Define anomaly types, probabilities, and impacts.

```python
EVENTS = {
    "Unexpected Spike": {
        "probability": 0.0002,  # 0.02%
        "cpu_range": (80, 100),
        "ram_range": (90, 100),
        "disk_increase": 10,
    },
    "Server Restart": {
        "probability": 0.0001,  # 0.01%
        "cpu_range": (0, 5),
        "ram_range": (0, 10),
        "disk_increase": 0,
    },
}
```

---

## Future Enhancements

1. Add dependencies between processes (e.g., cascading failures).
2. Extend anomaly detection with contextual features (e.g., time of day).
3. Incorporate real-world data for model evaluation.
4. Add a web interface for real-time log analysis.

---

For questions or improvements, feel free to reach out!

