# Lightweight Python image for running the qPCA demo
FROM python:3.11-slim

# Install system packages needed for science stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency list and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Default command runs the CLI demo
CMD ["python", "app/main.py"]
