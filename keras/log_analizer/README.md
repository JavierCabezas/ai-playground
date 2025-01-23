
# Log Analyzer Project

This project aims to analyze server logs using an autoencoder to detect anomalies in system behavior.

## Project Structure

```
log_analyzer/
├── analyzer/
│   ├── train_model.py        # Script to train the autoencoder
│   ├── test_model.py         # Script to test the trained autoencoder
│   ├── analysis_tools.py     # Contains utilities for analysis and chart generation
├── docker/
│   ├── train_model.Dockerfile  # Dockerfile for training the model
│   ├── generator.Dockerfile    # Dockerfile for generating logs
│   ├── docker-compose.yml      # Compose file to manage services
├── generator/
│   ├── generator.py          # Log generation script
├── logs/
│   ├── debug_logs.txt        # Debug logs for anomalies
│   ├── server_logs.txt       # Generated server logs
│   ├── anomalies.csv         # Detected anomalies (created during testing)
├── models/
│   ├── autoencoder_model.keras  # Trained model in the new Keras format
│   ├── scaler.pkl             # Scaler used for normalizing input values
├── charts/
│   ├── loss_curves.png        # Training loss curves
│   ├── error_distribution.png # Reconstruction error histogram for training data
│   ├── unseen_data_errors.png # Reconstruction error histogram for unseen data
└── README.md                  # Project documentation
```

## Features

1. **Log Generator**:
   - Generates synthetic server logs with fields: `timestamp`, `server`, `component`, `value`, and `anomalous`.
   - Logs are designed to simulate real-world scenarios, including anomalies.

2. **Autoencoder for Anomaly Detection**:
   - Trained to reconstruct patterns from normal behavior and identify deviations.
   - Uses reconstruction error to classify anomalies.

3. **Testing Pipeline**:
   - Tests the trained model on unseen data.
   - Flags entries as anomalous based on a dynamically calculated threshold.
   - Saves anomalies to a CSV file for manual inspection.

4. **Metrics and Evaluation** (Enabled by `DEBUG_WITH_ANOMALY_FLAG`):
   - Compares detected anomalies against the ground truth (`anomalous` field in generated logs).
   - Outputs precision, recall, F1-score, and a classification report.

5. **Chart Generation**:
   - Generates histograms and visualizations for training, validation, and testing phases.

6. **Dockerized Architecture**:
   - Each component (log generator, model training, and testing) is fully containerized for easy setup and execution.

## Installation

1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd log_analyzer
   ```

2. Build the Docker images:
   ```bash
   cd docker
   docker-compose build
   ```

## Usage

### 1. Generate Logs
Run the log generator to create synthetic server logs:
```bash
docker-compose run --rm generator
```

Generated logs are saved to the `logs/server_logs.txt` file.

---

### 2. Train the Model
Train the autoencoder on the generated logs:
```bash
docker-compose run --rm train_model
```

Outputs:
- **Trained model**: `models/autoencoder_model.keras`
- **Scaler**: `models/scaler.pkl`
- **Training charts**: `charts/`

---

### 3. Test the Model
Test the trained model on unseen data:
```bash
docker-compose run --rm test_model
```

Outputs:
- **Detected anomalies**: `logs/anomalies.csv`
- **Reconstruction error chart**: `charts/unseen_data_errors.png`

---

### Debugging and Validation

#### **Enabling the Debug Flag**
To compare detected anomalies with ground truth (`anomalous` field in logs), enable the debug mode:
1. Open `test_model.py`.
2. Set `DEBUG_WITH_ANOMALY_FLAG = True`.

This will:
- Output a confusion matrix and metrics (precision, recall, F1-score).
- Print a classification report.

Example output:
```
Confusion Matrix:
True Negatives (TN): 11234
False Positives (FP): 300
False Negatives (FN): 100
True Positives (TP): 6060

Evaluation Metrics:
Precision: 0.95
Recall: 0.98
F1-Score: 0.96
```

#### **Disabling the Debug Flag**
For real-world usage, set:
```python
DEBUG_WITH_ANOMALY_FLAG = False
```
This will skip the evaluation and only save the detected anomalies.

---

## File Outputs

| File                          | Description                                      |
|-------------------------------|--------------------------------------------------|
| `logs/server_logs.txt`        | Generated server logs.                          |
| `logs/anomalies.csv`          | Detected anomalies from testing.                |
| `charts/loss_curves.png`      | Training loss curves.                           |
| `charts/error_distribution.png` | Reconstruction error histogram (training data). |
| `charts/unseen_data_errors.png` | Reconstruction error histogram (unseen data).   |

---

## Future Enhancements

1. **Periodic Retraining**:
   - Implement automated retraining pipelines to adapt to evolving behavior.

2. **Dashboard**:
   - Develop a web-based dashboard for interactive visualization of logs, anomalies, and charts.

3. **Multi-Server Support**:
   - Extend log generation and analysis for multiple servers.

---

## Troubleshooting

1. **Docker Cache Issues**:
   - If changes in `test_model.py` or `train_model.py` aren't reflected, rebuild with:
     ```bash
     docker-compose build --no-cache
     ```

2. **Missing Timestamps in Logs**:
   - Ensure logs are correctly generated by inspecting `server_logs.txt`.

3. **SettingWithCopyWarning**:
   - This warning is benign and occurs during DataFrame updates. It has been resolved using `.loc[]` for safety.

---

Let me know if there’s anything else you'd like to include or refine! 🚀
