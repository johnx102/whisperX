import os
import whisperx
import tempfile
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# Charger modèle au démarrage (important pour éviter rechargement à chaque appel)
model = whisperx.load_model("large-v2", device="cuda")

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    audio_url = request.json.get("url")
    if not audio_url:
        return jsonify({"error": "Missing audio URL"}), 400

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = tmp.name
            response = requests.get(audio_url)
            tmp.write(response.content)

        print(f"🔍 Transcribing {audio_path}")
        result = model.transcribe(audio_path)

        return jsonify({
            "transcript": result["segments"],
            "text": result["text"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
