"""
NISQA Client - NISQA-TTS service wrapper for MOS prediction
"""

import requests
import logging
import time
from typing import Dict

logger = logging.getLogger(__name__)


class NISQAClient:
    """Simple client for NISQA-TTS quality prediction service."""
    
    def __init__(self, nisqa_url: str, timeout: int = 60):
        """
        Initialize NISQA client.
        
        Args:
            nisqa_url: NISQA service URL (e.g., 'http://localhost:8006')
            timeout: Request timeout in seconds (default: 60s)
        """
        self.nisqa_url = nisqa_url.rstrip('/')
        self.timeout = timeout
        logger.info(f"Initialized NISQA client at {nisqa_url}")
        
    def health_check(self) -> bool:
        """
        Check if NISQA service is healthy.
        
        Returns:
            True if service is responding, False otherwise
        """
        try:
            response = requests.get(f"{self.nisqa_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✓ NISQA service is healthy")
                return True
        except Exception as e:
            logger.error(f"✗ NISQA service health check failed: {e}")
        return False
    
    def predict_quality(self, audio_path: str) -> Dict:
        """
        Predict MOS quality score for audio file.
        
        Args:
            audio_path: Absolute path to audio file
            
        Returns:
            {
                'mos': float,          # Overall MOS score (1-5)
                'latency': float,      # Processing time
                'success': bool,
                'error': str or None
            }
        """
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.nisqa_url}/predict_from_path",
                json={'audio_path': audio_path},
                timeout=self.timeout
            )
            
            latency = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'mos': result.get('mos', 0.0),
                    'latency': latency,
                    'success': True,
                    'error': None
                }
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"NISQA prediction failed: {error_msg}")
                return {
                    'mos': 0.0,
                    'latency': latency,
                    'success': False,
                    'error': error_msg
                }
                
        except requests.Timeout:
            latency = time.time() - start_time
            error_msg = f"Request timeout after {self.timeout}s"
            logger.error(f"NISQA prediction timeout: {error_msg}")
            return {
                'mos': 0.0,
                'latency': latency,
                'success': False,
                'error': error_msg
            }
            
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"NISQA prediction failed: {error_msg}")
            return {
                'mos': 0.0,
                'latency': latency,
                'success': False,
                'error': error_msg
            }
