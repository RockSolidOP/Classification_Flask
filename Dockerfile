FROM python:3.11-slim

# System deps
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Helpful envs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copy dependency lists first (cache-friendly)
COPY requirements.txt requirements-ml.txt ./

# Upgrade pip and tools
RUN python -m pip install --upgrade pip && \
    python -m pip install --upgrade pip-system-certs certifi

# Install CPU-only PyTorch using trusted-host for download.pytorch.org
RUN python -m pip install \
    --trusted-host download.pytorch.org \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.8.0 torchvision==0.23.0

# Install the rest of your dependencies
# If you hit similar TLS issues with PyPI, uncomment the two --trusted-host lines below.
RUN python -m pip install \
    -r requirements.txt -r requirements-ml.txt
    # --trusted-host pypi.org \
    # --trusted-host files.pythonhosted.org

# App code last for better layer caching
COPY . .

# Default data dirs
RUN mkdir -p Source_PDF Ground_Truth

# Flask env
ENV FLASK_APP=app.py \
    FLASK_RUN_HOST=0.0.0.0 \
    FLASK_RUN_PORT=5000

EXPOSE 5000
CMD ["flask", "run"]