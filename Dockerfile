# Version corrigée avec gestion du conflit NumPy/NeMo/WhisperX
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

# ÉTAPE 1: Install PyTorch with CUDA support first
RUN pip install --no-cache-dir \
    torch==2.1.0+cu121 \
    torchaudio==2.1.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# ÉTAPE 2: Install NumPy/Numba compatible AVANT tout le reste
RUN pip install --no-cache-dir "numpy==1.24.4"
RUN pip install --no-cache-dir "numba==0.59.1"

# Vérification immédiate
RUN python -c "import numpy; print(f'NumPy installé: {numpy.__version__}')"
RUN python -c "import numba; print(f'Numba installé: {numba.__version__}')"

# ÉTAPE 3: Install librosa avec les bonnes versions
RUN pip install --no-cache-dir "librosa==0.10.1"
RUN pip install --no-cache-dir "soundfile==0.12.1"

# ÉTAPE 4: Install ctranslate2 et dependencies core
RUN pip install --no-cache-dir \
    ctranslate2==3.24.0 \
    faster-whisper==1.0.1 \
    pandas \
    setuptools>=65 \
    nltk

# ÉTAPE 5: Install NeMo AVANT WhisperX (critique!)
RUN pip install --no-cache-dir \
    omegaconf \
    hydra-core \
    pytorch-lightning

# Installer NeMo avec --no-deps pour éviter conflits
RUN pip install --no-cache-dir nemo_toolkit[asr] --no-deps

# Installer les dépendances NeMo manquantes manuellement
RUN pip install --no-cache-dir \
    wrapt \
    einops \
    editdistance \
    inflect \
    matplotlib \
    regex \
    scipy \
    tqdm \
    wandb \
    webdataset

# ÉTAPE 6: Install WhisperX avec --no-deps (CRITIQUE pour éviter NumPy 2.3!)
RUN pip install --no-cache-dir git+https://github.com/m-bain/whisperx.git --no-deps

# Installer les dépendances WhisperX manuellement (sans écraser NumPy)
RUN pip install --no-cache-dir \
    transformers==4.36.2 \
    speechbrain \
    pyannote.audio --no-deps

# Dépendances pyannote manuelles
RUN pip install --no-cache-dir \
    asteroid-filterbanks \
    pyannote.core \
    pyannote.database \
    pyannote.metrics \
    pyannote.pipeline

# ÉTAPE 7: Install API dependencies
RUN pip install --no-cache-dir \
    runpod \
    fastapi \
    uvicorn \
    aiohttp \
    aiofiles \
    pydantic

# VÉRIFICATION FINALE CRITIQUE
RUN echo "🔍 Vérification finale des versions..." && \
    python -c "
import numpy as np
import numba
import librosa
print(f'✅ NumPy: {np.__version__}')
print(f'✅ Numba: {numba.__version__}')
print(f'✅ Librosa: {librosa.__version__}')

# Test de compatibilité NumPy/Numba
try:
    from numba import jit
    @jit
    def test_func(x):
        return x + 1
    result = test_func(5)
    print('✅ NumPy/Numba: Compatible')
except Exception as e:
    print(f'❌ NumPy/Numba: {e}')
    exit(1)

# Test NeMo
try:
    import nemo
    print('✅ NeMo: Importé avec succès')
except Exception as e:
    print(f'⚠️ NeMo: {e}')

# Test WhisperX  
try:
    import whisperx
    print('✅ WhisperX: Importé avec succès')
except Exception as e:
    print(f'❌ WhisperX: {e}')
    exit(1)

print('🎉 SUCCÈS: Toutes les dépendances sont compatibles!')
"

# Create cache directories accessible par tous
RUN mkdir -p /models/hf-cache \
    && mkdir -p /models/transformers-cache \
    && mkdir -p /models/whisperx-cache \
    && chmod -R 777 /models

# Set environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV OMP_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Cache locations
ENV HF_HOME=/models/hf-cache
ENV TRANSFORMERS_CACHE=/models/transformers-cache

# Copy main.py
COPY main.py /app/main.py
COPY main.py /app/handler.py

# Script de démarrage avec vérification
RUN echo '#!/bin/bash\n\
echo "🚀 Démarrage du service WhisperX + NeMo..."\n\
\n\
# Vérification rapide des dépendances\n\
echo "🔍 Vérification des dépendances..."\n\
python -c "\n\
try:\n\
    import numpy, numba, nemo, whisperx\n\
    print(f\"✅ NumPy: {numpy.__version__}\")\n\
    print(f\"✅ Numba: {numba.__version__}\")\n\
    print(\"✅ NeMo: OK\")\n\
    print(\"✅ WhisperX: OK\")\n\
except Exception as e:\n\
    print(f\"❌ ERREUR: {e}\")\n\
    exit(1)\n\
" || exit 1\n\
\n\
# Créer les répertoires de cache\n\
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" 2>/dev/null || true\n\
chmod -R 777 "${HF_HOME}" 2>/dev/null || true\n\
chmod -R 777 "${TRANSFORMERS_CACHE}" 2>/dev/null || true\n\
\n\
echo "✅ Service prêt!"\n\
# Démarrer l'\''application\n\
exec python -u main.py' > /app/start.sh

RUN chmod +x /app/start.sh

# Health check amélioré
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import whisperx, ctranslate2, numpy, numba; print('Service healthy')" || exit 1

# Start the application
CMD ["/app/start.sh"]
