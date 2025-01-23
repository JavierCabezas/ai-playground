# Use a lightweight Python image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy necessary scripts
COPY ../analyzer/test_model.py .
COPY ../analyzer/analysis_tools.py .
COPY ../analyzer/train_model.py .

# Install dependencies
RUN pip install --no-cache-dir tensorflow pandas numpy scikit-learn joblib matplotlib

# Command to run the test script
CMD ["python", "test_model.py"]
