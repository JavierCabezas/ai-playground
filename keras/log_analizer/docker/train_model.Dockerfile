# Use a lightweight Python image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install dependencies directly
RUN pip install --no-cache-dir tensorflow pandas numpy scikit-learn joblib matplotlib

# Command to run the training script
CMD ["python", "train_model.py"]
