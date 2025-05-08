# Base image with Python
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies if needed (e.g., curl, build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application
COPY . .

# Run the main script by default
CMD ["python", "main.py"]
