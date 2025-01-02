# Use a TensorFlow base image with Python 3.x
FROM tensorflow/tensorflow:latest-gpu

# Set working directory
WORKDIR /app

# Copy all project files
COPY . .

# Install required Python libraries
RUN pip install --no-cache-dir matplotlib scikit-learn seaborn kaggle

# Expose a port if needed (optional for visualization tools)
EXPOSE 5000

# Default command
CMD ["python"]
