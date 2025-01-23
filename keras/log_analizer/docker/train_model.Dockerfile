# Use a lightweight Python image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy necessary scripts
COPY ../analyzer/train_model.py .
COPY ../analyzer/test_model.py .
COPY ../analyzer/analysis_tools.py .

# Install dependencies directly
RUN pip install --no-cache-dir tensorflow pandas numpy scikit-learn joblib matplotlib

# Create models directory for saving the trained model
RUN mkdir -p models

# Command to run the training script
CMD ["python", "train_model.py"]