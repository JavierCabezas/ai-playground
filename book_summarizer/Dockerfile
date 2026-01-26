# syntax=docker/dockerfile:1.4
FROM python:3.10-slim


# Install system deps (needed for transformers & torch)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libssl-dev \
        git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip first – newer versions handle the “+” in the version string better.
RUN pip install --upgrade pip

# Install everything, pointing pip to the ROCm wheel index
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/rocm6.1

# Copy the source code
COPY . .

# Expose port for a simple FastAPI demo (optional)
EXPOSE 8000

CMD ["python", "app.py"]
