# Use official lightweight Python image
FROM python:3.11-slim

# Install system dependencies (like zbar for pyzbar)
RUN apt-get update && apt-get install -y \
    libzbar0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# === ✅ Speed Optimizer: cache dependencies ===
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r backend/requirements.txt

# Copy entire project contents AFTER installing deps
COPY . .

# Expose port (same as in start command)
EXPOSE 10000

# Run FastAPI with uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "10000"]
