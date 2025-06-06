FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git ffmpeg libsndfile1 python3 python3-pip && \
    apt-get clean

# Python dependencies
RUN pip install --upgrade pip

# Install WhisperX with dependencies
RUN pip install git+https://github.com/m-bain/whisperx.git
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install soundfile

# (Optionnel) Si tu veux activer la diarisation
# RUN pip install pyannote-audio
# RUN pyannote-audio download pretrained --force

# Créer dossier pour app
WORKDIR /app
COPY serverless_main.py .

CMD ["python3", "serverless_main.py"]
