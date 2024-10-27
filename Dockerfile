# Use official Python 3.11 image from Docker Hub
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment in the /venv directory
RUN python -m venv /venv

# Activate the virtual environment and upgrade pip inside it
RUN /venv/bin/pip install --upgrade pip

# Copy the requirements.txt file into the container
COPY requirements_dev.txt .

# Install Python dependencies inside the virtual environment
RUN /venv/bin/pip install --no-cache-dir -r requirements_dev.txt

# Copy the rest of the project files into the container
# COPY . .

# Set environment variables to use the virtual environment's Python
ENV VIRTUAL_ENV=/venv
ENV PATH="/venv/bin:$PATH"

# Set environment variable for unbuffered Python output (optional)
ENV PYTHONUNBUFFERED=1

# Define the default command to run when the container starts
CMD ["bash"]
