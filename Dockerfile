FROM python:3.10-slim

# System dependencies for PyMuPDF, Pillow, poppler-utils (for pdf2image), etc.
RUN apt-get update && \
    apt-get install -y gcc build-essential libglib2.0-0 libsm6 libxrender1 libxext6 poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements (if you have one)
COPY requirements.txt .

# Install Python dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Set environment variables for input/output
ENV INPUT_DIR=/app/input
ENV OUTPUT_DIR=/app/output

# Default command: process all PDFs from /app/input to /app/output
CMD ["python", "extract_structure/model.py"]