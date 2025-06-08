FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    git \
    libsndfile1 \
    && apt-get clean

# Mise à jour pip
RUN pip install --upgrade pip

# Installer les dépendances Python
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install transformers==4.40.1
RUN pip install tensorflow==2.15.0
RUN pip install flask requests soundfile

# (Optionnel) précharger Whisper Large-v2 TF (converti PyTorch)
RUN python3 -c "from transformers import WhisperForConditionalGeneration; WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v2', from_tf=True, cache_dir='/models/whisper-large-v2')"

# Copier ton code dans le conteneur
WORKDIR /app
COPY . /app

# Commande par défaut
CMD ["python3", "serverless_main.py"]
