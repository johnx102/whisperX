import os
import json
import tempfile
import asyncio
import gc
import traceback
from typing import Optional, Dict, Any
from datetime import datetime
from urllib.parse import urlparse

import torch
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

# Global model storage
models = {
    'whisper': None,
    'align_model': None,
    'align_metadata': None,
    'diarize_model': None
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
        # Load main Whisper model (will be loaded per request with different sizes)
        print(f"Setting up models on device: {device}")
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
        
        # Basic domain validation (you may want to add more restrictions)
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
                    # Still proceed - WhisperX can handle many formats via ffmpeg
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
    
    # Check if we already have this model loaded
    model_key = f"{model_size}_{device}_{compute_type}"
    
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
    """Clean up GPU memory"""
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
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
    
    # Debug log pour vérifier que le code est à jour
    hf_token = request.hf_token or os.getenv('HF_TOKEN')
    print(f"🔧 DEBUG: Processing transcription with diarization={request.enable_diarization}, token_present={bool(hf_token)}")
    
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
        transcribe_params = {
            'audio': audio,
            'batch_size': request.batch_size
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
        
        # Optional speaker diarization
        # Récupérer le token HF depuis l'env si pas fourni dans la requête
        hf_token = request.hf_token or os.getenv('HF_TOKEN')
        
        if request.enable_diarization and hf_token:
            try:
                print("Performing speaker diarization...")
                if models['diarize_model'] is None:
                    # Essayer différents modèles de diarisation
                    diarization_model = request.diarization_model or "pyannote/speaker-diarization-3.1"
                    
                    from pyannote.audio import Pipeline
                    diarize_model = Pipeline.from_pretrained(
                        diarization_model,
                        use_auth_token=hf_token
                    )
                    diarize_model.to(torch.device(device))
                    models['diarize_model'] = diarize_model
                    print(f"Loaded diarization model: {diarization_model}")
                
                # Prepare diarization parameters
                diarization_params = {}
                if request.min_speakers is not None:
                    diarization_params['min_speakers'] = request.min_speakers
                    print(f"Min speakers set to: {request.min_speakers}")
                if request.max_speakers is not None:
                    diarization_params['max_speakers'] = request.max_speakers
                    print(f"Max speakers set to: {request.max_speakers}")
                if request.num_speakers is not None:
                    diarization_params['num_speakers'] = request.num_speakers
                    print(f"Num speakers set to: {request.num_speakers}")
                
                print(f"Diarization parameters: {diarization_params}")
                
                # Effectuer la diarisation avec le bon format
                waveform = torch.from_numpy(audio).unsqueeze(0)
                diarization_result = models['diarize_model'](
                    {"waveform": waveform, "sample_rate": 16000},
                    **diarization_params
                )
                
                # Convertir en segments simples
                speaker_segments = []
                for segment, _, speaker in diarization_result.itertracks(yield_label=True):
                    speaker_segments.append({
                        'start': float(segment.start),
                        'end': float(segment.end),
                        'speaker': str(speaker)
                    })
                
                print(f"Found {len(speaker_segments)} speaker segments")
                
                # Post-processing : fusionner les segments courts du même speaker
                def merge_short_segments(segments, min_duration=0.5):
                    """Fusionne les segments très courts avec les segments adjacents du même speaker"""
                    if not segments:
                        return segments
                        
                    merged = [segments[0]]
                    
                    for current in segments[1:]:
                        last = merged[-1]
                        duration = current['end'] - current['start']
                        
                        # Si le segment est très court, l'attribuer au speaker majoritaire autour
                        if duration < min_duration:
                            # Vérifier si c'est le même speaker que le précédent
                            if last['speaker'] == current['speaker']:
                                # Fusionner avec le segment précédent
                                last['end'] = current['end']
                                continue
                            # Sinon vérifier le segment suivant dans la liste originale
                            elif len(segments) > segments.index(current) + 1:
                                next_seg = segments[segments.index(current) + 1]
                                if next_seg['speaker'] == last['speaker']:
                                    # Attribuer au speaker précédent
                                    current['speaker'] = last['speaker']
                        
                        merged.append(current)
                    
                    return merged
                
                # Appliquer le post-processing
                speaker_segments = merge_short_segments(speaker_segments)
                print(f"After post-processing: {len(speaker_segments)} speaker segments")
                
                # Assignation manuelle améliorée des speakers aux mots
                if isinstance(result, dict) and 'segments' in result:
                    segments = result['segments']
                else:
                    segments = result if isinstance(result, list) else []
                
                # Pour chaque segment de transcription
                for i, seg in enumerate(segments):
                    seg_start = seg.get('start', 0)
                    seg_end = seg.get('end', 0)
                    seg_duration = seg_end - seg_start
                    
                    # Trouver le speaker majoritaire pour ce segment
                    speaker_overlaps = {}
                    
                    for spk_seg in speaker_segments:
                        # Calculer le chevauchement
                        overlap_start = max(seg_start, spk_seg['start'])
                        overlap_end = min(seg_end, spk_seg['end'])
                        overlap = max(0, overlap_end - overlap_start)
                        
                        if overlap > 0:
                            speaker = spk_seg['speaker']
                            speaker_overlaps[speaker] = speaker_overlaps.get(speaker, 0) + overlap
                    
                    # Sélectionner le speaker avec le plus grand chevauchement
                    if speaker_overlaps:
                        best_speaker = max(speaker_overlaps.items(), key=lambda x: x[1])[0]
                        
                        # Vérification de cohérence : si le segment est très court (< 0.5s)
                        # et que le speaker est différent des segments adjacents, corriger
                        if seg_duration < 0.5 and len(segments) > 1:
                            prev_speaker = segments[i-1].get('speaker') if i > 0 else None
                            next_speaker = segments[i+1].get('speaker') if i < len(segments)-1 else None
                            
                            # Si les segments adjacents ont le même speaker, utiliser celui-ci
                            if prev_speaker and next_speaker and prev_speaker == next_speaker and prev_speaker != best_speaker:
                                best_speaker = prev_speaker
                                print(f"Corrected short segment {i}: {best_speaker}")
                        
                        # Assigner le speaker au segment
                        seg['speaker'] = best_speaker
                        
                        # Assigner aussi aux mots si disponibles
                        if 'words' in seg:
                            for word in seg['words']:
                                word['speaker'] = best_speaker
                
                print("Diarization completed successfully (manual assignment)")
                
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
                "batch_size": request.batch_size
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
                "diarize": models['diarize_model'] is not None
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

if __name__ == "__main__":
    # Initialize models on startup
    device, compute_type = setup_models()
    print(f"WhisperX service initialized on {device}")
    
    # Start RunPod serverless handler
    print("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
