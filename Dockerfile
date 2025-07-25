# Dockerfile
FROM --platform=linux/amd64 python:3.10

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*
    # The '&& rm -rf /var/lib/apt/lists/*' should be on the same RUN line as apt-get install,
    # and the preceding line should NOT have a backslash for line continuation.


# --- Setup Application Environment ---
# Set the main working directory for the application
WORKDIR /app

# --- Install Python Dependencies ---
# Copy only the requirements file first to take advantage of Docker's layer caching.
# This layer is only rebuilt if requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy Application Code ---
# Copy the rest of your application code into the container
COPY . .

# --- Configure Environment Variables ---
# Set the environment variables that your Python script will use for I/O directories.
# The script will look for PDFs in /app/input and write JSONs to /app/output.
ENV INPUT_DIR=/app/input
ENV OUTPUT_DIR=/app/output

# --- Set Final Working Directory ---
# Change the working directory to where your main script is located.
# This allows you to run 'python model.py' directly, just like you do locally,
# and ensures any relative imports within your script work correctly.
WORKDIR /app/src

# --- Set the Default Command to Run ---
# This command is executed when the container starts.
CMD ["python", "model.py"]