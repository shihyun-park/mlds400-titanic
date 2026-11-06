FROM python:3.11-slim

WORKDIR /app

# Install pinned dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the script
COPY titanic.py .

# Data is mounted at runtime to /app/data (not baked into the image)
CMD ["python", "titanic.py"]