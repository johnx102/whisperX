import os
import json
import tempfile
import asyncio
import gc
import traceback
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime
from urllib.parse import urlparse

import torch
import torchaudio
import pandas as pd
import aiohttp
import aiofiles
import runpod
import whisperx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl, validator
import uvicorn

# Initialize FastAPI app for health checks and development
app = FastAPI(title="WhisperX Transcription Service", version="1.0.0")

# Configuration
MAX_FILE_SIZE = 300 * 1024 * 1024  # 300MB
DOWNLOAD_TIMEOUT = 300  # 5 minutes
PROCESSING_TIMEOUT = 900  # 15 minutes
SUPPORTED_FORMATS = {
    'audio/wav', 'audio/mpeg', 'audio/mp4', 'audio/m4a', 
    'audio/ogg', 'audio/flac', 'video/mp4', 'video/avi'
}

class SileroVADHelper:
    """
    Helper pour Silero VAD - amélioration de la diarisation
    """
    def __init__(self):
        self.model = None
        self.utils = None
        self._load_model()
    
    def _load_model(self):
        """Charge le modèle Silero VAD une seule fois"""
        try:
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=True  # Version optimisée
            )
            print("✅ Silero VAD loaded successfully")
        except Exception as e:
            print(f"❌ Error loading Silero VAD: {e}")
            self.model = None
    
    def clean_audio_for_diarization(self, audio_numpy: np.ndarray, threshold: float = 0.5):
        """
        Nettoie l'audio avec VAD avant diarisation
        
        Args:
            audio_numpy: Audio numpy array (16kHz)
            threshold: Seuil de confiance VAD (0.3=sensible, 0.7=strict)
        
        Returns:
            tuple: (clean_audio_tensor, success)
        """
        if self.model is None:
            return torch.from_numpy(audio_numpy), False
        
        try:
            # Extraction des fonctions
            (get_speech_timestamps, save_audio, read_audio, 
             VADIterator, collect_chunks) = self.utils
            
            # Conversion audio pour Silero (besoin d'un tensor)
            if isinstance(audio_numpy, np.ndarray):
                wav_tensor = torch.from_numpy(audio_numpy)
            else:
                wav_tensor = audio_numpy
                
            # Détection segments de parole
            speech_timestamps = get_speech_timestamps(
                wav_tensor, 
                self.model,
                threshold=threshold,
                sampling_rate=16000,
                min_speech_duration_ms=250,
                min_silence_duration_ms=100,
                speech_pad_ms=30
            )
            
            if not speech_timestamps:
                print("⚠️ No speech detected by Silero VAD")
                return torch.from_numpy(audio_numpy), False
            
            # Collecter les chunks de parole uniquement
            clean_speech = collect_chunks(speech_timestamps, wav_tensor)
            
            print(f"✅ VAD: {len(speech_timestamps)} speech segments extracted")
            print(f"   Original audio: {len(audio_numpy)/16000:.1f}s")
            print(f"   Clean speech: {len(clean_speech)/16000:.1f}s")
            
            return clean_speech, True
                
        except Exception as e:
            print(f"❌ VAD processing error: {e}")
            return torch.from_numpy(audio_numpy), False

# Global model storage
models = {
    'whisper': None,
    'align_model': None,
    'align_metadata': None,
    'diarize_model': None,
    'vad_helper': None  # NOUVEAU
}

class TranscriptionRequest(BaseModel):
    audio_url: HttpUrl
    language: Optional[str] = "auto"
    model_size: Optional[str] = "large-v2"
    compute_type: Optional[str] = "float16"
    batch_size: Optional[int] = 16
    enable_diarization: Optional[bool] = False
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    num_speakers: Optional[int] = None
    diarization_model: Optional[str] = "pyannote/speaker-diarization-3.1"
    return_char_alignments: Optional[bool] = False
    hf_token: Optional[str] = None
    
    # NOUVEAUX PARAMÈTRES VAD
    use_vad: Optional[bool] = True
    vad_threshold: Optional[float] = 0.3
    
    @validator('model_size')
    def validate_model_size(cls, v):
        valid_sizes = ['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3']
        if v not in valid_sizes:
            raise ValueError(f'Model size must be one of: {", ".join(valid_sizes)}')
        return v
    
    @validator('compute_type')
    def validate_compute_type(cls, v):
        valid_types = ['float16', 'int8', 'float32']
        if v not in valid_types:
            raise ValueError(f'Compute type must be one of: {", ".join(valid_types)}')
        return v

class TranscriptionResponse(BaseModel):
    text: str
    segments: list
    language: str
    processing_time: float
    model_info: Dict[str, Any]
    error: Optional[str] = None

def setup_models():
    """Initialize models globally to avoid reloading"""
    global models
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"
    
    try:
        print(f"Setting up models on device: {device}")
        
        # 🚀 Optimisations GPU si disponible
        if device == "cuda":
            print(f"🎯 GPU Optimizations enabled:")
            print(f"   - GPU Name: {torch.cuda.get_device_name()}")
            print(f"   - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            print(f"   - CUDA Version: {torch.version.cuda}")
            
            # Activer les optimisations GPU
            torch.backends.cudnn.benchmark = True  # Optimise les convolutions
            torch.backends.cuda.matmul.allow_tf32 = True  # TF32 pour plus de vitesse
            torch.backends.cudnn.allow_tf32 = True
            
            # Nettoyer la mémoire GPU au démarrage
            torch.cuda.empty_cache()
        
        # NOUVEAU: Initialiser Silero VAD
        if models['vad_helper'] is None:
            print("🎤 Initializing Silero VAD...")
            models['vad_helper'] = SileroVADHelper()
            
        return device, compute_type
    except Exception as e:
        print(f"Error setting up models: {str(e)}")
        return "cpu", "float32"

async def validate_url_security(url: str) -> bool:
    """Validate URL for security (prevent SSRF)"""
    try:
        parsed = urlparse(str(url))
        
        # Check scheme
        if parsed.scheme not in ['http', 'https']:
            raise ValueError("Invalid URL scheme. Only HTTP and HTTPS are allowed.")
        
        # Basic domain validation
        if not parsed.hostname:
            raise ValueError("Invalid hostname in URL")
            
        # Block localhost and private IPs (basic check)
        if parsed.hostname.lower() in ['localhost', '127.0.0.1', '0.0.0.0']:
            raise ValueError("Access to localhost is forbidden")
            
        return True
    except Exception as e:
        raise ValueError(f"URL validation failed: {str(e)}")

async def download_audio_file(url: str) -> bytes:
    """Download audio file from URL with security checks"""
    await validate_url_security(url)
    
    timeout = aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.get(str(url)) as response:
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if not any(fmt in content_type for fmt in SUPPORTED_FORMATS):
                    print(f"Warning: Unrecognized content-type: {content_type}")
                
                # Check file size
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > MAX_FILE_SIZE:
                    raise ValueError(f"File too large: {content_length} bytes (max: {MAX_FILE_SIZE})")
                
                # Download file
                content = await response.read()
                
                if len(content) > MAX_FILE_SIZE:
                    raise ValueError(f"Downloaded file too large: {len(content)} bytes")
                
                return content
                
        except asyncio.TimeoutError:
            raise ValueError("Download timeout - file took too long to download")
        except aiohttp.ClientError as e:
            raise ValueError(f"Download failed: {str(e)}")

def load_whisper_model(model_size: str, device: str, compute_type: str):
    """Load WhisperX model with specified parameters"""
    global models
    
    try:
        print(f"Loading WhisperX model: {model_size} on {device} with {compute_type}")
        model = whisperx.load_model(
            model_size, 
            device=device, 
            compute_type=compute_type,
            download_root="/models/cache"
        )
        models['whisper'] = model
        return model
    except Exception as e:
        print(f"Error loading model {model_size}: {str(e)}")
        raise

def cleanup_gpu_memory():
    """Clean up GPU memory aggressively"""
    try:
        # 🚀 Nettoyage GPU optimisé
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            # Vider tous les caches GPU
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Stats mémoire pour debug
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            print(f"🧹 GPU Memory: {allocated:.1f}GB allocated, {cached:.1f}GB cached")
            
    except Exception as e:
        print(f"Error during GPU cleanup: {str(e)}")

async def process_transcription(
    audio_data: bytes, 
    request: TranscriptionRequest,
    device: str,
    compute_type: str
) -> TranscriptionResponse:
    """Process audio transcription with WhisperX"""
    
    start_time = datetime.now()
    temp_file_path = None
    
    # Récupérer le token HF
    hf_token = request.hf_token or os.getenv('HF_TOKEN')
    print(f"🔧 DEBUG: Processing transcription with diarization={request.enable_diarization}, token_present={bool(hf_token)}, use_vad={request.use_vad}")
    
    try:
        # Save audio data to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.audio') as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        # Load audio
        print("Loading audio file...")
        audio = whisperx.load_audio(temp_file_path)
        
        # Load or reuse Whisper model
        if models['whisper'] is None:
            models['whisper'] = load_whisper_model(request.model_size, device, compute_type)
        
        # Transcribe
        print("Starting transcription...")
        
        # 🚀 Optimiser batch_size selon GPU disponible
        if device == "cuda":
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory_gb >= 24:  # A100/H100/A5000
                optimal_batch = min(request.batch_size, 32)
            elif gpu_memory_gb >= 16:  # V100/A10
                optimal_batch = min(request.batch_size, 24)
            else:  # T4 et autres
                optimal_batch = min(request.batch_size, 16)
            
            # 🔥 Suggestion d'optimisation si batch_size trop petit
            if request.batch_size < optimal_batch:
                print(f"💡 Tip: You could use batch_size up to {32 if gpu_memory_gb >= 24 else 24} for better performance on your {gpu_memory_gb:.1f}GB GPU")
            
            print(f"🎯 Optimized batch size: {optimal_batch} (GPU: {gpu_memory_gb:.1f}GB)")
        else:
            optimal_batch = 8  # CPU fallback
        
        transcribe_params = {
            'audio': audio,
            'batch_size': optimal_batch
        }
        
        # Add language parameter if specified (and not auto)
        if request.language and request.language.lower() != "auto":
            print(f"Using specified language: {request.language}")
            transcribe_params['language'] = request.language
        else:
            print("Using automatic language detection")
            
        result = models['whisper'].transcribe(**transcribe_params)
        
        detected_language = result.get("language", "unknown")
        print(f"Detected language: {detected_language}")
        
        # Load alignment model if needed
        if detected_language and detected_language != "unknown":
            try:
                print("Loading alignment model...")
                if models['align_model'] is None or models.get('align_language') != detected_language:
                    align_model, metadata = whisperx.load_align_model(
                        language_code=detected_language, 
                        device=device
                    )
                    models['align_model'] = align_model
                    models['align_metadata'] = metadata
                    models['align_language'] = detected_language
                
                # Align transcription
                print("Aligning transcription...")
                result = whisperx.align(
                    result["segments"], 
                    models['align_model'], 
                    models['align_metadata'], 
                    audio, 
                    device, 
                    return_char_alignments=request.return_char_alignments
                )
            except Exception as e:
                print(f"Alignment failed: {str(e)}")
                # Continue without alignment
        
        # Speaker diarization with Silero VAD preprocessing
        if request.enable_diarization and hf_token:
            try:
                print("Starting speaker diarization with VAD preprocessing...")
                
                # Load diarization model
                diarization_model = request.diarization_model or "pyannote/speaker-diarization-3.1"
                
                if models['diarize_model'] is None or models.get('diarize_model_name') != diarization_model:
                    from pyannote.audio import Pipeline
                    print(f"Loading diarization model: {diarization_model}")
                    
                    diarize_model = Pipeline.from_pretrained(
                        diarization_model, 
                        use_auth_token=hf_token
                    )
                    # 🚀 IMPORTANT: Forcer l'utilisation du GPU
                    diarize_model.to(torch.device(device))
                    models['diarize_model'] = diarize_model
                    models['diarize_model_name'] = diarization_model
                    print(f"✅ Diarization model loaded on {device}")
                
                # NOUVEAU: Préprocessing VAD
                if request.use_vad and models['vad_helper'] and models['vad_helper'].model:
                    print(f"🎤 Applying Silero VAD preprocessing (threshold: {request.vad_threshold})...")
                    clean_audio_tensor, vad_success = models['vad_helper'].clean_audio_for_diarization(
                        audio, 
                        threshold=request.vad_threshold
                    )
                    
                    if vad_success:
                        print("✅ Using VAD-cleaned audio for diarization")
                        waveform = clean_audio_tensor.unsqueeze(0)
                    else:
                        print("⚠️ VAD preprocessing failed, using original audio")
                        waveform = torch.from_numpy(audio).unsqueeze(0)
                else:
                    print("ℹ️ Using original audio (VAD disabled or unavailable)")
                    waveform = torch.from_numpy(audio).unsqueeze(0)
                
                # Prepare parameters for diarization
                diarization_kwargs = {}
                if request.min_speakers is not None:
                    diarization_kwargs['min_speakers'] = request.min_speakers
                if request.max_speakers is not None:
                    diarization_kwargs['max_speakers'] = request.max_speakers
                if request.num_speakers is not None:
                    diarization_kwargs['num_speakers'] = request.num_speakers
                
                print(f"Diarization parameters: {diarization_kwargs}")
                
                # 🚀 S'assurer que l'audio est sur le bon device
                if device == "cuda":
                    waveform = waveform.to(device)
                    print(f"✅ Audio tensor moved to {device}")
                    
                    # 🔥 Re-forcer TF32 car PyAnnote le désactive parfois
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                
                # Perform diarization 
                diarize_segments = models['diarize_model'](
                    {"waveform": waveform, "sample_rate": 16000},
                    **diarization_kwargs
                )
                
                print("Diarization completed. Converting format for WhisperX...")
                
                # Convert PyAnnote output to format expected by WhisperX
                # Créer un DataFrame pandas comme attendu par assign_word_speakers
                diarize_data = []
                for segment, _, speaker in diarize_segments.itertracks(yield_label=True):
                    diarize_data.append({
                        'start': segment.start,
                        'end': segment.end,
                        'speaker': speaker
                    })
                
                if diarize_data:
                    diarize_df = pd.DataFrame(diarize_data)
                    print(f"Created diarization DataFrame with {len(diarize_df)} segments")
                    
                    # Assigner les speakers aux segments/mots
                    print("Assigning speakers to words...")
                    result = whisperx.assign_word_speakers(diarize_df, result)
                    
                    print("Speaker diarization completed successfully!")
                else:
                    print("⚠️  No diarization segments found")
                
            except Exception as e:
                print(f"Diarization failed: {str(e)}")
                print(f"Diarization error details: {traceback.format_exc()}")
                # Continue without diarization
        elif request.enable_diarization and not hf_token:
            print("⚠️  Diarization requested but no HF_TOKEN found in environment or request")
        
        # Extract full text
        if isinstance(result, dict) and 'segments' in result:
            full_text = ' '.join([segment.get('text', '') for segment in result['segments']])
            segments = result['segments']
        else:
            full_text = ' '.join([segment.get('text', '') for segment in result])
            segments = result
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return TranscriptionResponse(
            text=full_text.strip(),
            segments=segments,
            language=detected_language,
            processing_time=processing_time,
            model_info={
                "model_size": request.model_size,
                "compute_type": compute_type,
                "device": device,
                "batch_size": request.batch_size,
                "vad_enabled": request.use_vad,
                "vad_threshold": request.vad_threshold
            }
        )
        
    except Exception as e:
        error_msg = f"Transcription failed: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return TranscriptionResponse(
            text="",
            segments=[],
            language="unknown",
            processing_time=processing_time,
            model_info={
                "model_size": request.model_size,
                "compute_type": compute_type,
                "device": device
            },
            error=error_msg
        )
    
    finally:
        # Cleanup
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Failed to delete temp file: {str(e)}")
        
        cleanup_gpu_memory()

# RunPod handler function
async def handler(job):
    """Main RunPod handler function"""
    job_input = job.get("input", {})
    
    try:
        # Validate input
        request = TranscriptionRequest(**job_input)
        
        # Download audio file
        print(f"Downloading audio from: {request.audio_url}")
        audio_data = await download_audio_file(request.audio_url)
        print(f"Downloaded {len(audio_data)} bytes")
        
        # Setup device and compute type
        device, compute_type_default = setup_models()
        compute_type = request.compute_type if request.compute_type else compute_type_default
        
        # Process transcription
        result = await process_transcription(audio_data, request, device, compute_type)
        
        # Return result
        if result.error:
            return {"error": result.error}
        else:
            return {
                "transcription": result.text,
                "segments": result.segments,
                "language": result.language,
                "processing_time": result.processing_time,
                "model_info": result.model_info
            }
            
    except Exception as e:
        error_msg = f"Handler error: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return {"error": error_msg}

# FastAPI endpoints for health checks
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gpu_info = {}
        
        if torch.cuda.is_available():
            gpu_info = {
                "gpu_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_memory": {
                    "allocated": torch.cuda.memory_allocated(),
                    "cached": torch.cuda.memory_reserved()
                }
            }
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "device": device,
            "gpu_info": gpu_info,
            "models_loaded": {
                "whisper": models['whisper'] is not None,
                "align": models['align_model'] is not None,
                "diarize": models['diarize_model'] is not None,
                "vad": models['vad_helper'] is not None and models['vad_helper'].model is not None
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.post("/transcribe")
async def transcribe_endpoint(request: TranscriptionRequest):
    """Direct transcription endpoint for testing"""
    try:
        # Download audio
        audio_data = await download_audio_file(request.audio_url)
        
        # Setup processing
        device, compute_type_default = setup_models()
        compute_type = request.compute_type if request.compute_type else compute_type_default
        
        # Process
        result = await process_transcription(audio_data, request, device, compute_type)
        
        if result.error:
            raise HTTPException(status_code=500, detail=result.error)
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/compare_vad")
async def compare_vad_endpoint(request: TranscriptionRequest):
    """
    Endpoint pour comparer avec/sans VAD - utile pour tester l'amélioration
    """
    try:
        # Download audio
        audio_data = await download_audio_file(request.audio_url)
        
        # Setup processing
        device, compute_type_default = setup_models()
        compute_type = request.compute_type if request.compute_type else compute_type_default
        
        # Test sans VAD
        request_no_vad = request.copy()
        request_no_vad.use_vad = False
        result_no_vad = await process_transcription(audio_data, request_no_vad, device, compute_type)
        
        # Test avec VAD
        request_with_vad = request.copy()
        request_with_vad.use_vad = True
        result_with_vad = await process_transcription(audio_data, request_with_vad, device, compute_type)
        
        return {
            "without_vad": {
                "speakers_detected": len(set([seg.get('speaker') for seg in result_no_vad.segments if seg.get('speaker')])),
                "segments_count": len(result_no_vad.segments),
                "processing_time": result_no_vad.processing_time,
                "text_sample": result_no_vad.text[:200] + "..." if len(result_no_vad.text) > 200 else result_no_vad.text
            },
            "with_vad": {
                "speakers_detected": len(set([seg.get('speaker') for seg in result_with_vad.segments if seg.get('speaker')])),
                "segments_count": len(result_with_vad.segments),
                "processing_time": result_with_vad.processing_time,
                "text_sample": result_with_vad.text[:200] + "..." if len(result_with_vad.text) > 200 else result_with_vad.text
            },
            "improvement": {
                "vad_enabled": request_with_vad.use_vad,
                "vad_threshold": request_with_vad.vad_threshold,
                "processing_time_diff": result_with_vad.processing_time - result_no_vad.processing_time,
                "speakers_diff": len(set([seg.get('speaker') for seg in result_with_vad.segments if seg.get('speaker')])) - len(set([seg.get('speaker') for seg in result_no_vad.segments if seg.get('speaker')]))
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

if __name__ == "__main__":
    # Initialize models on startup
    device, compute_type = setup_models()
    print(f"WhisperX service with Silero VAD initialized on {device}")
    
    # Start RunPod serverless handler
    print("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
