FROM python:3.11-slim

# System deps (optional but useful)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency list first (leverages Docker layer cache)
COPY requirements.txt requirements-ml.txt ./
# Install CPU-only PyTorch wheels first to avoid CUDA payloads,
# then install the rest. Keep cache off to minimize layer size.
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.8.0 torchvision==0.23.0 \
    && python -m pip install --no-cache-dir -r requirements.txt -r requirements-ml.txt

# Copy app code
COPY . .

# Create default data dirs (can be mounted as volumes)
RUN mkdir -p Source_PDF Ground_Truth

# Flask env
ENV FLASK_APP=app.py \
    FLASK_RUN_HOST=0.0.0.0 \
    FLASK_RUN_PORT=5000 \
    PYTHONUNBUFFERED=1

EXPOSE 5000

# Default command: run the web app
CMD ["flask", "run"]
