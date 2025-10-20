FROM python:3.11-slim
FROM python:3.11

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git git-lfs ffmpeg build-essential libsndfile1 \
    python3-dev libffi-dev libssl-dev curl wget unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Copy app code
COPY . .