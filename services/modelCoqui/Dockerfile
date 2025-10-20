FROM python:3.11

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system dependencies (incl. build tools and sound libs)
RUN apt-get update && apt-get install -y \
    git git-lfs ffmpeg build-essential libsndfile1 \
    python3-dev libffi-dev libssl-dev curl wget unzip pkg-config \
    && rm -rf /var/lib/apt/lists/*

# --- Install Rust (needed for SudachiPy) ---
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# --- Python deps ---
COPY requirements.txt .
RUN pip install --upgrade pip==24.0 setuptools==69.5 wheel
RUN pip install -r requirements.txt --no-cache-dir

# Copy app code
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]