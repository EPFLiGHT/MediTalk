# services/webui/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY . .

# Expose FastAPI port
EXPOSE 8080

# Start FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]