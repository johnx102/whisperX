FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 AS base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
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

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA support first
RUN pip install --no-cache-dir \
    torch==2.1.0+cu121 \
    torchaudio==2.1.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install other core dependencies
RUN pip install --no-cache-dir \
    faster-whisper==1.1.0 \
    "ctranslate2<4.5.0" \
    transformers \
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
ENV TRANSFORMERS_CACHE=/models/cache
ENV HF_HOME=/models/cache
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create cache directory
RUN mkdir -p /models/cache

# Create non-root user for security
RUN useradd -m -u 1001 whisperx
RUN chown -R whisperx:whisperx /app /models

# Copy application files
COPY --chown=whisperx:whisperx main.py /app/
COPY --chown=whisperx:whisperx download_models.py /app/

# Pre-download models during build (optional - reduces cold start time)
RUN python download_models.py || echo "Model download failed - will download at runtime"

# Switch to non-root user
USER whisperx

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import whisperx; print('WhisperX ready')" || exit 1

# Start the serverless handler
CMD ["python", "-u", "main.py"]
