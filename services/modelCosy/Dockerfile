FROM python:3.11

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    git git-lfs ffmpeg build-essential libsndfile1 \
    python3-dev libffi-dev libssl-dev curl wget unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt --no-cache-dir

# Copy service code and pretrained models
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]