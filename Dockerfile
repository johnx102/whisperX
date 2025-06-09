# Version corrigée avec gestion des permissions RunPod
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

# Install PyTorch with CUDA support first (compatible avec CUDA 12.2)
RUN pip install --no-cache-dir \
    torch==2.1.0+cu121 \
    torchaudio==2.1.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install ctranslate2 first (version spécifique stable)
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

# Install WhisperX from source
RUN pip install --no-cache-dir git+https://github.com/m-bain/whisperx.git

# Set environment variables for GPU and optimization
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV OMP_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create cache directories with proper permissions
RUN mkdir -p /app/models/hf-cache \
    && mkdir -p /app/models/transformers-cache \
    && mkdir -p /app/models/whisperx-cache \
    && mkdir -p /app/tmp

# Create non-root user for security
RUN useradd -m -u 1001 whisperx

# Set default cache locations (will be overridden by env vars if provided)
ENV HF_HOME=/app/models/hf-cache
ENV TRANSFORMERS_CACHE=/app/models/transformers-cache
ENV WHISPERX_CACHE=/app/models/whisperx-cache

# Copy main.py as both main.py and handler.py for RunPod compatibility
COPY --chown=whisperx:whisperx main.py /app/main.py
COPY --chown=whisperx:whisperx main.py /app/handler.py

# Create init script to handle RunPod volumes
RUN echo '#!/bin/bash\n\
# Create cache directories if they dont exist and fix permissions\n\
mkdir -p ${HF_HOME} ${TRANSFORMERS_CACHE} ${WHISPERX_CACHE} 2>/dev/null || true\n\
\n\
# Try to fix permissions on cache directories\n\
if [ -w "$(dirname ${HF_HOME})" ]; then\n\
    chown -R whisperx:whisperx ${HF_HOME} 2>/dev/null || true\n\
    chmod -R 755 ${HF_HOME} 2>/dev/null || true\n\
fi\n\
\n\
if [ -w "$(dirname ${TRANSFORMERS_CACHE})" ]; then\n\
    chown -R whisperx:whisperx ${TRANSFORMERS_CACHE} 2>/dev/null || true\n\
    chmod -R 755 ${TRANSFORMERS_CACHE} 2>/dev/null || true\n\
fi\n\
\n\
if [ -w "$(dirname ${WHISPERX_CACHE})" ]; then\n\
    chown -R whisperx:whisperx ${WHISPERX_CACHE} 2>/dev/null || true\n\
    chmod -R 755 ${WHISPERX_CACHE} 2>/dev/null || true\n\
fi\n\
\n\
# Switch to whisperx user and start app\n\
exec gosu whisperx "$@"' > /app/entrypoint.sh

# Install gosu for user switching
RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Give ownership of /app to whisperx user
RUN chown -R whisperx:whisperx /app

# Test ctranslate2 installation
RUN python -c "import ctranslate2; print('ctranslate2 version:', ctranslate2.__version__)"

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import whisperx, ctranslate2; print('WhisperX ready')" || exit 1

# Use custom entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "-u", "main.py"]
