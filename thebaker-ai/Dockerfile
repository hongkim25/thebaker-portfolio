# Use a slim Python 3.11 image to keep the container lightweight
FROM python:3.11-slim

# Set the working directory strictly to /app
WORKDIR /app

# Copy dependency definitions and install early to cache the docker layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend source code
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# The standard deterministic command to boot the FastAPI application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
