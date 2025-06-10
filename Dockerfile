# Version simplifiée avec permissions corrigées
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    build-essential \
    ffmpeg \
    libsndfile1 \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip and install wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support first
RUN pip install --no-cache-dir \
    torch==2.1.0+cu121 \
    torchaudio==2.1.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install ctranslate2 first
RUN pip install --no-cache-dir ctranslate2==3.24.0

# Install other core dependencies
RUN pip install --no-cache-dir \
    faster-whisper==1.0.1 \
    transformers==4.36.2 \
    pandas \
    setuptools>=65 \
    nltk \
    runpod \
    fastapi \
    uvicorn \
    aiohttp \
    aiofiles \
    pydantic

RUN pip install --no-cache-dir nemo_toolkit[ALL]
# Install WhisperX from source
RUN pip install --no-cache-dir git+https://github.com/m-bain/whisperx.git

# Create cache directories accessible par tous
RUN mkdir -p /models/hf-cache \
    && mkdir -p /models/transformers-cache \
    && mkdir -p /models/whisperx-cache \
    && chmod -R 777 /models

# Set environment variables (defaults qui peuvent être overridés)
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV OMP_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default cache locations (seront overridés par RunPod si nécessaire)
ENV HF_HOME=/models/hf-cache
ENV TRANSFORMERS_CACHE=/models/transformers-cache

# Copy main.py
COPY main.py /app/main.py
COPY main.py /app/handler.py

# Test ctranslate2 installation
RUN python -c "import ctranslate2; print('ctranslate2 version:', ctranslate2.__version__)"

# Script de démarrage simple pour gérer les permissions
RUN echo '#!/bin/bash\n\
# Créer les répertoires de cache si ils n'\''existent pas\n\
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" 2>/dev/null || true\n\
\n\
# Essayer de fixer les permissions si possible\n\
chmod -R 777 "${HF_HOME}" 2>/dev/null || true\n\
chmod -R 777 "${TRANSFORMERS_CACHE}" 2>/dev/null || true\n\
\n\
# Démarrer l'\''application\n\
exec python -u main.py' > /app/start.sh

RUN chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import whisperx, ctranslate2; print('WhisperX ready')" || exit 1

# Start the application
CMD ["/app/start.sh"]
