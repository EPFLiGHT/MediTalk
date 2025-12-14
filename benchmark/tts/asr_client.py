"""
ASR Client - Whisper service wrapper for round-trip evaluation

Connects to Whisper ASR service (http://localhost:5007) to transcribe
generated audio files for intelligibility measurement.
"""

import requests
import logging
import time
from typing import Dict

logger = logging.getLogger(__name__)


class ASRClient:
    """Simple client for Whisper ASR service."""
    
    def __init__(self, whisper_url: str, timeout: int = 600):
        """
        Initialize ASR client.
        
        Args:
            whisper_url: Whisper service URL (e.g., 'http://localhost:5007')
            timeout: Request timeout in seconds (default: 600s)
        """
        self.whisper_url = whisper_url.rstrip('/')
        self.timeout = timeout
        logger.info(f"Initialized ASR client at {whisper_url}")
        
    def health_check(self) -> bool:
        """
        Check if Whisper service is healthy.
        
        Returns:
            True if service is responding, False otherwise
        """
        try:
            response = requests.get(f"{self.whisper_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info(" ✅ Whisper ASR service is healthy")
                return True
        except Exception as e:
            logger.error(f" ❌ Whisper ASR service health check failed: {e}")
        return False
    
    def transcribe_file(self, audio_path: str) -> Dict:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Absolute path to audio file
            
        Returns:
            {
                'text': str,           # Transcribed text
                'language': str,       # Detected language
                'latency': float,      # Processing time
                'success': bool,
                'error': str or None
            }
        """
        start_time = time.time()
        
        try:
            # Send audio file to Whisper service
            with open(audio_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(
                    f"{self.whisper_url}/transcribe",
                    files=files,
                    timeout=self.timeout
                )
            
            latency = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'text': result.get('text', ''),
                    'language': result.get('language', 'unknown'),
                    'latency': latency,
                    'success': True,
                    'error': None
                }
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"ASR transcription failed: {error_msg}")
                return {
                    'text': '',
                    'language': 'unknown',
                    'latency': latency,
                    'success': False,
                    'error': error_msg
                }
                
        except requests.Timeout:
            latency = time.time() - start_time
            error_msg = f"Request timeout after {self.timeout}s"
            logger.error(f"ASR transcription timeout: {error_msg}")
            return {
                'text': '',
                'language': 'unknown',
                'latency': latency,
                'success': False,
                'error': error_msg
            }
            
        except Exception as e:
            latency = time.time() - start_time
            error_msg = str(e)
            logger.error(f"ASR transcription error: {error_msg}")
            return {
                'text': '',
                'language': 'unknown',
                'latency': latency,
                'success': False,
                'error': error_msg
            }
