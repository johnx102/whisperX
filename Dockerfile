FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Installer dépendances système
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    git \
    libsndfile1 \
    && apt-get clean

# Mise à jour de pip
RUN pip install --upgrade pip

# Installer les paquets Python requis AVANT d'exécuter des scripts Python
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install transformers==4.40.1

# Installer tensorflow pour pouvoir utiliser from_tf=True
RUN pip install tensorflow==2.15.0

# Précharger le modèle Whisper large-v2 en version TF (converti en torch)
RUN python3 -c "from transformers import WhisperForConditionalGeneration; WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v2', from_tf=True, cache_dir='/models/whisper-large-v2')"

# Créer le dossier de l'app
WORKDIR /app
COPY . /app

# Commande par défaut
CMD ["python3", "serverless_main.py"]
