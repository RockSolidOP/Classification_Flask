FROM python:3.11-slim

# System deps
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Helpful envs for leaner/faster installs & cleaner logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copy dependency lists first (cache-friendly)
COPY requirements.txt requirements-ml.txt ./

# Install: upgrade pip -> CPU-only torch -> normal deps -> certs fixes
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu \
         torch==2.8.0 torchvision==0.23.0 \
    && python -m pip install -r requirements.txt -r requirements-ml.txt \
    # trust system cert store (useful behind corp proxies)
    && python -m pip install pip-system-certs certifi --upgrade

# App code
COPY . .

# Default data dirs (mount as volumes if you like)
RUN mkdir -p Source_PDF Ground_Truth

# Flask env
ENV FLASK_APP=app.py \
    FLASK_RUN_HOST=0.0.0.0 \
    FLASK_RUN_PORT=5000

EXPOSE 5000

# Run the web app
CMD ["flask", "run"]
