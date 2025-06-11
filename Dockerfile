# Dockerfile ultra-simplifié sans tests complexes
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Installer dépendances système
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    build-essential cmake pkg-config \
    ffmpeg libsndfile1 libsox-dev sox \
    git wget curl cython3 \
    && rm -rf /var/lib/apt/lists/*

# Python par défaut
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Pip
RUN python -m pip install --upgrade pip setuptools wheel cython

WORKDIR /app

# Cloner projet
RUN git clone https://github.com/MahmoudAshraf97/whisper-diarization.git /tmp/whisper-diarization
RUN cp /tmp/whisper-diarization/*.py /app/ 2>/dev/null || true
RUN cp -r /tmp/whisper-diarization/config /app/ 2>/dev/null || mkdir -p /app/config

# Copier fichiers
COPY requirements.txt /app/requirements.txt
COPY constraints.txt /app/constraints.txt
COPY main.py /app/main.py

# PyTorch CUDA 12.1
RUN pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# NumPy/Numba
RUN pip install "numpy>=1.21.0,<2.0.0" "numba>=0.56.0,<0.60.0"

# Autres dépendances
RUN pip install -c constraints.txt -r requirements.txt

# Répertoires
RUN mkdir -p temp_outputs outputs /models/cache
RUN chmod -R 777 temp_outputs outputs /models/cache

# Variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV HF_HOME=/models/cache
ENV TRANSFORMERS_CACHE=/models/cache
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV OMP_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false

# Test simple séparé
RUN python -c "import torch"
RUN python -c "import faster_whisper"
RUN python -c "import runpod"

# Script simple
COPY <<EOF /app/start.sh
#!/bin/bash
echo "Starting service..."
mkdir -p temp_outputs outputs /models/cache
chmod -R 777 temp_outputs outputs /models/cache 2>/dev/null || true
exec python -u main.py
EOF

RUN chmod +x /app/start.sh

EXPOSE 8000
CMD ["/app/start.sh"]
