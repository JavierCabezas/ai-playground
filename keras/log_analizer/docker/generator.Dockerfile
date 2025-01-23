# Use a lightweight Python image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install any dependencies if required (e.g., pip install commands)

# Run the generator script
CMD ["python", "generator.py"]
