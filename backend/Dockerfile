# ============================================================
# Dockerfile — TruthLens Backend for Hugging Face Spaces
#
# Hugging Face Spaces runs this container automatically.
# It must listen on port 7860 (HF requirement).
# ============================================================

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker caches this layer)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all backend files
COPY . .

# Hugging Face Spaces requires port 7860
EXPOSE 7860

# Set environment variables
ENV FLASK_ENV=production
ENV PORT=7860

# Start the Flask server
CMD ["python", "app.py"]