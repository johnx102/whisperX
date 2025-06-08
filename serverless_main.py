from flask import Flask, request, jsonify
import tempfile
import requests
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio

# Config modèle
MODEL_NAME = "openai/whisper-large-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"📦 Chargement du modèle {MODEL_NAME} sur {DEVICE}...")

# Chargement modèle et processeur avec conversion depuis TensorFlow si nécessaire
processor = WhisperProcessor.from_pretrained(MODEL_NAME, cache_dir="/models")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, from_tf=True, cache_dir="/models").to(DEVICE)

app = Flask(__name__)

@app.route("/hello", methods=["GET"])
def hello():
    return jsonify({"message": "👋 Hello from Whisper Serverless!"})

@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        audio_url = request.json.get("url")
        if not audio_url:
            return jsonify({"error": "No audio URL provided."}), 400

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            response = requests.get(audio_url)
            tmp.write(response.content)
            audio_path = tmp.name

        # Charger et prétraiter audio
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        input_features = processor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(DEVICE)

        # Génération
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        return jsonify({"text": transcription})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
