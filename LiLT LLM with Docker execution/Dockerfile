# Dockerfile
FROM --platform=linux/amd64 python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*
    # The '&& rm -rf /var/lib/apt/lists/*' should be on the same RUN line as apt-get install,
    # and the preceding line should NOT have a backslash for line continuation.

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Create necessary directories (input/output will be bind-mounted from host)
RUN mkdir -p /app/input /app/output /app/models

# Download and prepare model (ensure <200MB total)
COPY download_model.py .
RUN python download_model.py

# Command to run the application
CMD ["python", "main.py"]