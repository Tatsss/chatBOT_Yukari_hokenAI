# Dockerfile for deploying FastAPI LINE Bot to Cloud Run

# Use official Python slim image as a parent image
FROM python:3.11-slim

# Set working directory in the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Ensure Python outputs logs straight to the container logs (no buffering)
ENV PYTHONUNBUFFERED=1

# Cloud Run expects the container to listen on $PORT
ENV PORT=8080
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
