"""
TTS-Specific Metrics - Additional evaluation metrics for TTS quality

Core metrics (WER, CER, RTF) imported from shared utilities.

TTS-specific metrics:
- ASR Round-Trip - intelligibility via Whisper transcription
- NISQA-TTS - naturalness/quality (MOS prediction)
"""

import numpy as np
from typing import Dict, List
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared import metrics as shared_metrics

calculate_wer = shared_metrics.calculate_wer
calculate_cer = shared_metrics.calculate_cer
calculate_rtf = shared_metrics.calculate_rtf
normalize_text = shared_metrics.normalize_text

logger = logging.getLogger(__name__)


# ============================================================================
# TTS-Specific Metrics
# ============================================================================

def calculate_asr_roundtrip_wer(text: str, audio_path: str, asr_client) -> Dict:
    """
    ASR Round-Trip Evaluation.
    
    Generate audio -> transcribe with Whisper -> compute WER/CER
    """
    raise NotImplementedError("calculate_asr_roundtrip_wer not implemented yet")
    pass


# ============================================================================
# Metrics Aggregator
# ============================================================================

class TTSMetricsAggregator:
    """Aggregates metrics across multiple samples."""
    
    def __init__(self):
        self.asr_wer_scores = []
        self.asr_cer_scores = []
        self.rtf_scores = []
        self.generation_times = []
        self.audio_durations = []
        self.errors = []
        
    def add_sample(self, metrics_dict: Dict):
        """Add metrics for a single sample."""
        raise NotImplementedError("TTSMetricsAggregator.add_sample not implemented yet")
        pass
    
    def get_summary(self) -> Dict:
        """Get comprehensive summary statistics."""
        raise NotImplementedError("TTSMetricsAggregator.get_summary not implemented yet")
        pass
