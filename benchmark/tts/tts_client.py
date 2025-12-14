"""
TTS Client - Unified interface for all TTS services

Provides a simple wrapper around TTS service HTTP APIs:
- Bark (http://localhost:5008)
- CSM (http://localhost:5010)
- Orpheus (http://localhost:5005)

Each service has /synthesize endpoint that takes text and returns audio.
"""

import requests
import logging
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)


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
        logger.info(f"Initialized TTS client for {service_name} at {service_url}")
        
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
        
        try:
            # Call TTS service
            response = requests.post(
                f"{self.service_url}/synthesize",
                json={'text': text, 'output_path': output_path},
                timeout=self.timeout
            )
            
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'audio_path': result.get('audio_path', output_path),
                    'duration': result.get('duration', 0.0),
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
