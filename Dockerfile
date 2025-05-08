# Use slim Python 3.10 image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install required system dependencies (optional for things like git, curl, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Default run command
CMD ["python", "main.py"]
