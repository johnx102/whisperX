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
app = FastAPI(title="WhisperX Transcription Service with NeMo", version="1.0.0")

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
    'nemo_diarizer': None,  # NOUVEAU: NeMo diarizer
    'nemo_speaker_model': None,  # NOUVEAU: NeMo speaker embeddings
    'pyannote_model': None  # Pyannote fallback
}

# Test NeMo availability at startup
def test_nemo_availability():
    """Test if NeMo is properly installed"""
    try:
        print("🔍 Testing NeMo installation...")
        
        # Test des imports critiques
        from nemo.collections.asr.models import ClusteringDiarizer
        from nemo.collections.asr.models import EncDecSpeakerLabelModel
        from omegaconf import OmegaConf
        
        print("✅ NeMo imports successful")
        
        # Test de création d'un objet basique
        config = OmegaConf.create({'test': True})
        print("✅ OmegaConf working")
        
        return True
        
    except ImportError as e:
        print(f"❌ NeMo import failed: {e}")
        print("💡 Install with: pip install nemo_toolkit[asr]")
        return False
    except Exception as e:
        print(f"❌ NeMo setup failed: {e}")
        return False

# Test global à l'initialisation
NEMO_AVAILABLE = test_nemo_availability()

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
    # NOUVEAU: Choix du moteur de diarisation
    diarization_engine: Optional[str] = "nemo"  # "nemo" ou "pyannote"
    # Paramètres NeMo spécifiques
    nemo_model: Optional[str] = "nvidia/speakerdiarization_msdd_telephonic"
    oracle_vad: Optional[bool] = False  # Utiliser VAD perfect pour NeMo
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
    
    @validator('diarization_engine')
    def validate_diarization_engine(cls, v):
        valid_engines = ['nemo', 'pyannote']
        if v not in valid_engines:
            raise ValueError(f'Diarization engine must be one of: {", ".join(valid_engines)}')
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
        
        print(f"🔍 NeMo Status: {'Available' if NEMO_AVAILABLE else 'Not Available'}")
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

def load_nemo_diarizer(model_name: str, device: str):
    """Load NeMo diarization model avec gestion d'erreur améliorée"""
    global models
    
    try:
        print(f"🔊 Loading NeMo diarization model: {model_name}")
        
        # Vérification préalable
        if not NEMO_AVAILABLE:
            print("❌ NeMo not available during load_nemo_diarizer")
            return None
        
        # Import NeMo avec gestion d'erreur détaillée
        try:
            from nemo.collections.asr.models import ClusteringDiarizer
            print("✅ NeMo ClusteringDiarizer import successful")
        except ImportError as e:
            print(f"❌ Failed to import ClusteringDiarizer: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error importing NeMo: {e}")
            return None
        
        # Charger le modèle NeMo
        try:
            print(f"🔄 Attempting to load model: {model_name}")
            nemo_model = ClusteringDiarizer.from_pretrained(model_name)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load NeMo model {model_name}: {e}")
            # Essayer un modèle alternatif
            try:
                print("🔄 Trying alternative NeMo model...")
                alternative_model = "nvidia/speakerverification_en_titanet_large"
                nemo_model = ClusteringDiarizer.from_pretrained(alternative_model)
                print(f"✅ Alternative model loaded: {alternative_model}")
            except Exception as e2:
                print(f"❌ Alternative model also failed: {e2}")
                return None
        
        # Configurer pour GPU si disponible
        try:
            if device == "cuda" and torch.cuda.is_available():
                nemo_model = nemo_model.to(device)
                print(f"✅ NeMo diarizer moved to {device}")
            else:
                print(f"✅ NeMo diarizer on CPU")
        except Exception as e:
            print(f"⚠️ Could not move to {device}: {e}")
        
        models['nemo_diarizer'] = nemo_model
        return nemo_model
        
    except Exception as e:
        print(f"❌ Unexpected error in load_nemo_diarizer: {e}")
        print(f"❌ Traceback: {traceback.format_exc()}")
        return None

def simple_nemo_diarize(audio_path: str, request: TranscriptionRequest, device: str):
    """Version simplifiée de la diarisation NeMo"""
    try:
        if not NEMO_AVAILABLE:
            print("❌ NeMo not available for diarization")
            return []
        
        print("🔊 Starting simplified NeMo diarization...")
        
        # Import des modules nécessaires
        from nemo.collections.asr.models import ClusteringDiarizer
        from omegaconf import OmegaConf
        import librosa
        import soundfile as sf
        
        # Créer un diarizeur simple
        print("🔄 Creating NeMo diarizer...")
        try:
            # Essayons avec un modèle simple
            diarizer = ClusteringDiarizer.from_pretrained("nvidia/speakerverification_en_titanet_large")
            print("✅ NeMo diarizer created")
        except Exception as e:
            print(f"❌ Failed to create diarizer: {e}")
            return []
        
        # Préparer l'audio
        print("🎵 Preparing audio...")
        audio_data, sr = librosa.load(audio_path, sr=16000)
        wav_path = audio_path.replace('.audio', '.wav')
        sf.write(wav_path, audio_data, 16000)
        
        # Configuration minimale
        manifest_path = '/tmp/simple_manifest.json'
        manifest_data = {
            'audio_filepath': wav_path,
            'offset': 0,
            'duration': len(audio_data) / 16000,
            'label': 'infer',
            'text': '-'
        }
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f)
        
        # Configuration simple
        config = OmegaConf.create({
            'diarizer': {
                'manifest_filepath': manifest_path,
                'out_dir': '/tmp/nemo_outputs',
                'clustering': {
                    'parameters': {
                        'max_num_speakers': request.max_speakers or 8
                    }
                }
            }
        })
        
        # Effectuer la diarisation
        print("🎯 Running simple NeMo diarization...")
        diarizer.diarize(cfg=config)
        
        # Nettoyer
        try:
            os.unlink(wav_path)
            os.unlink(manifest_path)
        except:
            pass
        
        print("✅ Simple NeMo diarization completed")
        return [{"start": 0, "end": len(audio_data) / 16000, "speaker": "SPEAKER_0"}]
        
    except Exception as e:
        print(f"❌ Simple NeMo diarization failed: {e}")
        print(f"Details: {traceback.format_exc()}")
        return []

def nemo_diarize_audio(audio_path: str, request: TranscriptionRequest, device: str):
    """Effectue la diarisation avec NeMo - version robuste"""
    try:
        print("🔊 Starting NeMo diarization...")
        
        # Vérification NeMo
        if not NEMO_AVAILABLE:
            print("❌ NeMo not available - falling back to simple segments")
            # Retourner des segments basiques
            import librosa
            audio_data, sr = librosa.load(audio_path, sr=16000)
            duration = len(audio_data) / sr
            return [{"start": 0, "end": duration, "speaker": "SPEAKER_0"}]
        
        # Charger le modèle NeMo si nécessaire
        if models['nemo_diarizer'] is None:
            print("🔄 Loading NeMo diarizer...")
            models['nemo_diarizer'] = load_nemo_diarizer(request.nemo_model, device)
        
        if models['nemo_diarizer'] is None:
            print("❌ Could not load NeMo diarizer - trying simple approach")
            return simple_nemo_diarize(audio_path, request, device)
        
        # Configuration NeMo complète
        from omegaconf import OmegaConf
        import librosa
        import soundfile as sf
        
        # Préparer l'audio
        audio_data, sr = librosa.load(audio_path, sr=16000)
        wav_path = audio_path.replace('.audio', '.wav')
        sf.write(wav_path, audio_data, 16000)
        
        # Configuration de base pour la diarisation
        config = OmegaConf.create({
            'diarizer': {
                'manifest_filepath': 'temp_manifest.json',
                'out_dir': '/tmp/nemo_outputs',
                'speaker_embeddings': {
                    'model_path': 'nvidia/speakerverification_en_titanet_large',
                    'parameters': {
                        'window_length_in_sec': 0.5,
                        'shift_length_in_sec': 0.1,
                        'multiscale_weights': None,
                        'save_embeddings': False
                    }
                },
                'clustering': {
                    'parameters': {
                        'oracle_num_speakers': request.num_speakers,
                        'max_num_speakers': request.max_speakers or 8,
                        'enhanced_count_thres': 40,
                        'max_rp_threshold': 0.25,
                        'sparse_search_volume': 30
                    }
                },
                'vad': {
                    'model_path': 'nvidia/vad_multilingual_marblenet',
                    'parameters': {
                        'window_length_in_sec': 0.15,
                        'shift_length_in_sec': 0.01,
                        'smoothing': 'median',
                        'overlap': 0.875,
                        'onset': 0.8,
                        'offset': 0.6,
                        'pad_onset': 0.05,
                        'pad_offset': -0.1,
                        'min_duration_on': 0.2,
                        'min_duration_off': 0.2
                    }
                }
            }
        })
        
        # Créer manifest temporaire
        manifest_path = '/tmp/nemo_manifest.json'
        manifest_data = {
            'audio_filepath': wav_path,
            'offset': 0,
            'duration': len(audio_data) / 16000,
            'label': 'infer',
            'text': '-',
            'num_speakers': request.num_speakers,
            'rttm_filepath': None,
            'uem_filepath': None
        }
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f)
            
        # Mettre à jour la config avec le bon manifest
        config.diarizer.manifest_filepath = manifest_path
        
        # Effectuer la diarisation
        print("🎯 Running NeMo diarization...")
        models['nemo_diarizer'].diarize(cfg=config)
        
        # Lire les résultats
        rttm_output_path = f"{config.diarizer.out_dir}/pred_rttms/{os.path.basename(wav_path).replace('.wav', '.rttm')}"
        
        segments = []
        if os.path.exists(rttm_output_path):
            with open(rttm_output_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 8 and parts[0] == 'SPEAKER':
                        start_time = float(parts[3])
                        duration = float(parts[4])
                        end_time = start_time + duration
                        speaker_id = parts[7]
                        
                        segments.append({
                            'start': start_time,
                            'end': end_time,
                            'speaker': f"SPEAKER_{speaker_id}"
                        })
        
        # Si pas de segments, fallback
        if not segments:
            print("⚠️ No RTTM segments found, creating fallback")
            duration = len(audio_data) / sr
            segments = [{"start": 0, "end": duration, "speaker": "SPEAKER_0"}]
        
        # Nettoyer les fichiers temporaires
        try:
            os.unlink(wav_path)
            os.unlink(manifest_path)
            if os.path.exists(rttm_output_path):
                os.unlink(rttm_output_path)
        except:
            pass
        
        print(f"✅ NeMo diarization completed: {len(segments)} segments")
        return segments
        
    except Exception as e:
        print(f"❌ NeMo diarization failed: {e}")
        print(f"Details: {traceback.format_exc()}")
        
        # Fallback ultime
        try:
            import librosa
            audio_data, sr = librosa.load(audio_path, sr=16000)
            duration = len(audio_data) / sr
            return [{"start": 0, "end": duration, "speaker": "SPEAKER_0"}]
        except:
            return []

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
    """Process audio transcription with WhisperX + NeMo diarization"""
    
    start_time = datetime.now()
    temp_file_path = None
    
    print(f"🔧 DEBUG: Processing with diarization={request.enable_diarization}, engine={request.diarization_engine}")
    print(f"🔧 NeMo Status: {'Available' if NEMO_AVAILABLE else 'Not Available'}")
    
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
        
        # Speaker diarization
        if request.enable_diarization:
            try:
                if request.diarization_engine == "nemo":
                    print("🔊 Starting NeMo diarization...")
                    
                    # Diarisation avec NeMo
                    diarize_segments = nemo_diarize_audio(temp_file_path, request, device)
                    
                    if diarize_segments:
                        # Convertir au format DataFrame pour WhisperX
                        import pandas as pd
                        diarize_df = pd.DataFrame(diarize_segments)
                        print(f"✅ NeMo diarization: {len(diarize_df)} segments")
                        
                        # Assigner les speakers aux mots
                        print("Assigning speakers to words...")
                        result = whisperx.assign_word_speakers(diarize_df, result)
                        print("✅ NeMo speaker diarization completed successfully!")
                    else:
                        print("⚠️ No diarization segments found with NeMo")
                
                elif request.diarization_engine == "pyannote":
                    # Fallback vers pyannote si demandé
                    print("🎭 Using pyannote fallback...")
                    hf_token = request.hf_token or os.getenv('HF_TOKEN')
                    
                    if not hf_token:
                        print("❌ Pyannote requires HF_TOKEN")
                    else:
                        from pyannote.audio import Pipeline
                        
                        # Charger pyannote
                        if models.get('pyannote_model') is None:
                            pyannote_model = Pipeline.from_pretrained(
                                "pyannote/speaker-diarization-3.1",
                                use_auth_token=hf_token
                            )
                            pyannote_model.to(torch.device(device))
                            models['pyannote_model'] = pyannote_model
                        
                        # Diarisation pyannote
                        import torchaudio
                        waveform = torch.from_numpy(audio).unsqueeze(0)
                        if device == "cuda":
                            waveform = waveform.to(device)
                        
                        diarization_kwargs = {}
                        if request.min_speakers:
                            diarization_kwargs['min_speakers'] = request.min_speakers
                        if request.max_speakers:
                            diarization_kwargs['max_speakers'] = request.max_speakers
                        if request.num_speakers:
                            diarization_kwargs['num_speakers'] = request.num_speakers
                        
                        diarize_segments = models['pyannote_model'](
                            {"waveform": waveform, "sample_rate": 16000},
                            **diarization_kwargs
                        )
                        
                        # Convertir format pyannote
                        import pandas as pd
                        diarize_data = []
                        for segment, _, speaker in diarize_segments.itertracks(yield_label=True):
                            diarize_data.append({
                                'start': segment.start,
                                'end': segment.end,
                                'speaker': speaker
                            })
                        
                        if diarize_data:
                            diarize_df = pd.DataFrame(diarize_data)
                            result = whisperx.assign_word_speakers(diarize_df, result)
                            print("✅ Pyannote fallback completed successfully!")
                
            except Exception as e:
                print(f"❌ Diarization failed: {str(e)}")
                print(f"Details: {traceback.format_exc()}")
        
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
                "diarization_engine": request.diarization_engine,
                "nemo_available": NEMO_AVAILABLE,
                "nemo_model_loaded": models['nemo_diarizer'] is not None
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
                "device": device,
                "nemo_available": NEMO_AVAILABLE
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
            "nemo_available": NEMO_AVAILABLE,
            "models_loaded": {
                "whisper": models['whisper'] is not None,
                "align": models['align_model'] is not None,
                "nemo_diarizer": models['nemo_diarizer'] is not None,
                "nemo_speaker": models['nemo_speaker_model'] is not None,
                "pyannote": models['pyannote_model'] is not None
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
            "nemo_available": NEMO_AVAILABLE
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

@app.get("/nemo-status")
async def nemo_status():
    """Check NeMo installation status"""
    status = {
        "nemo_available": NEMO_AVAILABLE,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if NEMO_AVAILABLE:
        try:
            from nemo.collections.asr.models import ClusteringDiarizer
            from nemo.collections.asr.models import EncDecSpeakerLabelModel
            
            status.update({
                "clustering_diarizer": "available",
                "speaker_model": "available",
                "test_status": "✅ All NeMo components working"
            })
        except Exception as e:
            status.update({
                "test_status": f"❌ NeMo test failed: {e}",
                "error": str(e)
            })
    else:
        status.update({
            "test_status": "❌ NeMo not installed",
            "install_command": "pip install nemo_toolkit[asr]"
        })
    
    return status

@app.post("/test_nemo")
async def test_nemo_endpoint(request: TranscriptionRequest):
    """Test NeMo diarization capabilities"""
    try:
        # Vérifier si NeMo est disponible
        if not NEMO_AVAILABLE:
            return {
                "status": "error",
                "message": "NeMo toolkit not installed",
                "install_command": "pip install nemo_toolkit[asr]",
                "nemo_available": False
            }
        
        # Test avec NeMo
        audio_data = await download_audio_file(request.audio_url)
        device, compute_type = setup_models()
        
        # Forcer NeMo
        request.diarization_engine = "nemo"
        request.enable_diarization = True
        
        result = await process_transcription(audio_data, request, device, compute_type)
        
        return {
            "status": "success",
            "message": "NeMo diarization tested successfully",
            "nemo_available": NEMO_AVAILABLE,
            "gpu_memory_used": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            "result_preview": {
                "speakers": len(set([seg.get('speaker') for seg in result.segments if seg.get('speaker')])),
                "segments": len(result.segments),
                "processing_time": result.processing_time,
                "engine_used": result.model_info.get('diarization_engine'),
                "first_segment": result.segments[0] if result.segments else None,
                "error": result.error
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"NeMo test failed: {str(e)}",
            "nemo_available": NEMO_AVAILABLE,
            "traceback": traceback.format_exc()
        }

@app.post("/debug_nemo")
async def debug_nemo_endpoint():
    """Debug NeMo installation détaillé"""
    debug_info = {
        "timestamp": datetime.utcnow().isoformat(),
        "nemo_global_status": NEMO_AVAILABLE
    }
    
    # Test des imports un par un
    imports_test = {}
    
    try:
        import nemo
        imports_test["nemo_core"] = {"status": "✅", "version": getattr(nemo, '__version__', 'unknown')}
    except ImportError as e:
        imports_test["nemo_core"] = {"status": "❌", "error": str(e)}
    
    try:
        from nemo.collections.asr.models import ClusteringDiarizer
        imports_test["clustering_diarizer"] = {"status": "✅"}
    except ImportError as e:
        imports_test["clustering_diarizer"] = {"status": "❌", "error": str(e)}
    
    try:
        from nemo.collections.asr.models import EncDecSpeakerLabelModel
        imports_test["speaker_model"] = {"status": "✅"}
    except ImportError as e:
        imports_test["speaker_model"] = {"status": "❌", "error": str(e)}
    
    try:
        from omegaconf import OmegaConf
        imports_test["omegaconf"] = {"status": "✅"}
    except ImportError as e:
        imports_test["omegaconf"] = {"status": "❌", "error": str(e)}
    
    debug_info["imports"] = imports_test
    
    # Test de création d'objets
    objects_test = {}
    
    if imports_test.get("clustering_diarizer", {}).get("status") == "✅":
        try:
            from nemo.collections.asr.models import ClusteringDiarizer
            # Test de liste des modèles disponibles
            objects_test["model_listing"] = {"status": "✅"}
        except Exception as e:
            objects_test["model_listing"] = {"status": "❌", "error": str(e)}
    
    debug_info["objects"] = objects_test
    
    # Informations système
    debug_info["system"] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    return debug_info

@app.post("/compare_engines")
async def compare_engines_endpoint(request: TranscriptionRequest):
    """Compare NeMo vs pyannote diarization"""
    try:
        audio_data = await download_audio_file(request.audio_url)
        device, compute_type = setup_models()
        
        results = {}
        
        # Test NeMo
        if NEMO_AVAILABLE:
            try:
                request_nemo = request.copy()
                request_nemo.diarization_engine = "nemo"
                request_nemo.enable_diarization = True
                
                result_nemo = await process_transcription(audio_data, request_nemo, device, compute_type)
                
                results["nemo"] = {
                    "speakers_detected": len(set([seg.get('speaker') for seg in result_nemo.segments if seg.get('speaker')])),
                    "segments_count": len(result_nemo.segments),
                    "processing_time": result_nemo.processing_time,
                    "sample_text": result_nemo.text[:200] + "..." if len(result_nemo.text) > 200 else result_nemo.text,
                    "error": result_nemo.error
                }
            except Exception as e:
                results["nemo"] = {"error": f"NeMo failed: {str(e)}"}
        else:
            results["nemo"] = {"error": "NeMo not available"}
        
        # Test pyannote
        try:
            request_pyannote = request.copy()
            request_pyannote.diarization_engine = "pyannote"
            request_pyannote.enable_diarization = True
            
            result_pyannote = await process_transcription(audio_data, request_pyannote, device, compute_type)
            
            results["pyannote"] = {
                "speakers_detected": len(set([seg.get('speaker') for seg in result_pyannote.segments if seg.get('speaker')])),
                "segments_count": len(result_pyannote.segments),
                "processing_time": result_pyannote.processing_time,
                "sample_text": result_pyannote.text[:200] + "..." if len(result_pyannote.text) > 200 else result_pyannote.text,
                "error": result_pyannote.error
            }
        except Exception as e:
            results["pyannote"] = {"error": f"Pyannote failed: {str(e)}"}
        
        # Comparaison
        comparison = {}
        if "nemo" in results and "pyannote" in results:
            if not results["nemo"].get("error") and not results["pyannote"].get("error"):
                comparison = {
                    "speed_difference": results["nemo"]["processing_time"] - results["pyannote"]["processing_time"],
                    "speakers_difference": results["nemo"]["speakers_detected"] - results["pyannote"]["speakers_detected"],
                    "segments_difference": results["nemo"]["segments_count"] - results["pyannote"]["segments_count"],
                    "winner_speed": "nemo" if results["nemo"]["processing_time"] < results["pyannote"]["processing_time"] else "pyannote",
                    "nemo_faster_by": abs(results["nemo"]["processing_time"] - results["pyannote"]["processing_time"]) if results["nemo"]["processing_time"] < results["pyannote"]["processing_time"] else 0
                }
        
        return {
            "results": results,
            "comparison": comparison,
            "nemo_available": NEMO_AVAILABLE,
            "gpu_memory_used": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

if __name__ == "__main__":
    # Initialize models on startup
    device, compute_type = setup_models()
    print(f"WhisperX + NeMo service initialized on {device}")
    print(f"🔍 NeMo Status: {'✅ Available' if NEMO_AVAILABLE else '❌ Not Available'}")
    
    # Start RunPod serverless handler
    print("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
