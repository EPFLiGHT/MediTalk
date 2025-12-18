"""
TTS Client - Unified interface for all TTS services
"""

import requests
import logging
import time
import json
import tempfile
import librosa
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def get_audio_duration(audio_path: str) -> float:
    """
    Get duration of audio file in seconds.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Duration in seconds, or 0.0 if error
    """
    try:
        duration = librosa.get_duration(path=audio_path)
        return float(duration)
    except Exception as e:
        logger.warning(f"Failed to get audio duration for {audio_path}: {e}")
        return 0.0


class TTSClient:
    """Simple client for TTS services."""
    
    def __init__(self, service_name: str, service_url: str, timeout: int = 300):
        """
        Initialize TTS client.
        
        Args:
            service_name: Name of TTS model (e.g., 'bark', 'csm')
            service_url: Base URL of service (e.g., 'http://localhost:5008')
            timeout: Request timeout in seconds (default: 300s = 5min)
        """
        self.service_name = service_name
        self.service_url = service_url.rstrip('/')
        self.timeout = timeout
        self.temp_dir = Path(tempfile.gettempdir()) / "tts_benchmark_conversations"
        self.temp_dir.mkdir(exist_ok=True)
        logger.info(f"Initialized TTS client for {service_name} at {service_url}")
        
    def _create_conversation_file(self, text: str) -> str:
        """
        Create a temporary conversation JSON file for TTS synthesis.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Path to the created conversation file
        """
        # Create conversation structure matching the expected format
        conversation = {
            "id": "bench",
            "messages": [
                {
                    "id": "msg1",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "data": text,
                            "metadata": None
                        }
                    ],
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "service_metadata": None
                }
            ],
            "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f"),
            "updated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f"),
            "metadata": None
        }
        
        # Save to temporary file
        temp_file = self.temp_dir / f"conversation_{int(time.time() * 1000000)}.json"
        with open(temp_file, 'w') as f:
            json.dump(conversation, f, indent=2)
        
        return str(temp_file)
        
    def health_check(self) -> bool:
        """
        Check if TTS service is healthy.
        
        Returns:
            True if service is responding, False otherwise
        """
        try:
            response = requests.get(f"{self.service_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"✓ {self.service_name} service is healthy")
                return True
        except Exception as e:
            logger.error(f"✗ {self.service_name} service health check failed: {e}")
        return False
    
    def synthesize(self, text: str, output_path: str) -> Dict:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            output_path: Where to save the generated audio file
            
        Returns:
            {
                'audio_path': str,      # Path to saved audio
                'duration': float,      # Audio duration in seconds
                'generation_time': float,  # Time taken to generate
                'success': bool,
                'error': str or None
            }
        """
        start_time = time.time()
        conversation_file = None
        
        try:
            # Create temporary conversation file
            conversation_file = self._create_conversation_file(text)
            
            # Call TTS service with conversation_path
            response = requests.post(
                f"{self.service_url}/synthesize",
                json={
                    'conversation_path': conversation_file,
                    'output_filename': output_path
                },
                timeout=self.timeout
            )
            
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                audio_path = result.get('audio_path', output_path)
                
                # Calculate duration from the audio file if not provided
                duration = result.get('duration', 0.0)
                if duration == 0.0 and Path(audio_path).exists():
                    duration = get_audio_duration(audio_path)
                
                return {
                    'audio_path': audio_path,
                    'duration': duration,
                    'generation_time': generation_time,
                    'success': True,
                    'error': None
                }
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"TTS synthesis failed: {error_msg}")
                return {
                    'audio_path': None,
                    'duration': 0.0,
                    'generation_time': generation_time,
                    'success': False,
                    'error': error_msg
                }
                
        except requests.Timeout:
            generation_time = time.time() - start_time
            error_msg = f"Request timeout after {self.timeout}s"
            logger.error(f"TTS synthesis timeout: {error_msg}")
            return {
                'audio_path': None,
                'duration': 0.0,
                'generation_time': generation_time,
                'success': False,
                'error': error_msg
            }
            
        except Exception as e:
            generation_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"TTS synthesis error: {error_msg}")
            return {
                'audio_path': None,
                'duration': 0.0,
                'generation_time': generation_time,
                'success': False,
                'error': error_msg
            }
        finally:
            # Clean up temporary conversation file
            if conversation_file and Path(conversation_file).exists():
                try:
                    Path(conversation_file).unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {conversation_file}: {e}")
