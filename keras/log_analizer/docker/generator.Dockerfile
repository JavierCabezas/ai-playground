# Use a lightweight Python image
FROM python:3.8-slim

# Set working directory inside the container
WORKDIR /app

# Copy the entire `generator` folder into the container
COPY ../generator /app

# Install dependencies
RUN pip install --no-cache-dir pandas numpy matplotlib

# Command to run the generator script
CMD ["python", "generator.py"]