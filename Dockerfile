# Use the official Python image from the Docker Hub
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install ffmpeg and any needed dependencies
RUN apt-get update && apt-get install -y ffmpeg

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container (optional)
# This is not needed if you're using bind mounts for development
# COPY . .

# Run app.py when the container launches
CMD ["python", "app.py"]
