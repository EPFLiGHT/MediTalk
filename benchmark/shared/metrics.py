"""
Shared Metrics - Common evaluation metrics for ASR/TTS benchmarks

Reusable functions:
- WER/CER calculation
- Text normalization
- RTF calculation
- Metrics aggregation

Used by:
- benchmark/whisper/ (ASR evaluation)
- benchmark/tts/ (TTS evaluation)
"""

import re
import numpy as np
from typing import Dict, List, Tuple
from jiwer import wer, cer


def normalize_text(text: str) -> str:
    """
    Normalize text for fair comparison.
    - Lowercase
    - Remove extra whitespace
    """
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def calculate_wer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """
    Calculate Word Error Rate (WER).
    
    Args:
        reference: Ground truth text
        hypothesis: Model prediction
        normalize: Whether to normalize text before comparison
        
    Returns:
        WER as a float (0.0 = perfect, 1.0 = completely wrong)
    """
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)
    
    if not reference:
        return 1.0 if hypothesis else 0.0
    
    return wer(reference, hypothesis)


def calculate_cer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """
    Calculate Character Error Rate (CER).
    
    Args:
        reference: Ground truth text
        hypothesis: Model prediction
        normalize: Whether to normalize text before comparison
        
    Returns:
        CER as a float
    """
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)
    
    if not reference:
        return 1.0 if hypothesis else 0.0
    
    return cer(reference, hypothesis)


def calculate_rtf(processing_time: float, audio_duration: float) -> float:
    """
    Calculate Real-Time Factor (RTF).
    RTF = processing_time / audio_duration
    
    RTF < 1.0 means faster than real-time
    RTF = 1.0 means real-time
    RTF > 1.0 means slower than real-time
    
    Args:
        processing_time: Time taken to process audio (seconds)
        audio_duration: Duration of audio file (seconds)
        
    Returns:
        RTF as a float
    """
    if audio_duration <= 0:
        return float('inf')
    return processing_time / audio_duration


class MetricsAggregator:
    """Aggregates metrics across multiple samples."""
    
    def __init__(self):
        self.wer_scores = []
        self.cer_scores = []
        self.latencies = []
        self.audio_durations = []
        self.rtf_scores = []
        self.errors = []
        
    def add_sample(
        self,
        wer_score: float,
        cer_score: float,
        latency: float,
        audio_duration: float,
        error: str = None
    ):
        """Add metrics for a single sample."""
        self.wer_scores.append(wer_score)
        self.cer_scores.append(cer_score)
        self.latencies.append(latency)
        self.audio_durations.append(audio_duration)
        self.rtf_scores.append(calculate_rtf(latency, audio_duration))
        if error:
            self.errors.append(error)
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.wer_scores:
            return {}
        
        return {
            # Accuracy metrics
            "wer": {
                "mean": float(np.mean(self.wer_scores)),
                "median": float(np.median(self.wer_scores)),
                "std": float(np.std(self.wer_scores)),
                "min": float(np.min(self.wer_scores)),
                "max": float(np.max(self.wer_scores)),
            },
            "cer": {
                "mean": float(np.mean(self.cer_scores)),
                "median": float(np.median(self.cer_scores)),
                "std": float(np.std(self.cer_scores)),
                "min": float(np.min(self.cer_scores)),
                "max": float(np.max(self.cer_scores)),
            },
            # Performance metrics
            "latency": {
                "mean": float(np.mean(self.latencies)),
                "median": float(np.median(self.latencies)),
                "p95": float(np.percentile(self.latencies, 95)),
                "p99": float(np.percentile(self.latencies, 99)),
                "min": float(np.min(self.latencies)),
                "max": float(np.max(self.latencies)),
            },
            "rtf": {
                "mean": float(np.mean(self.rtf_scores)),
                "median": float(np.median(self.rtf_scores)),
                "p95": float(np.percentile(self.rtf_scores, 95)),
                "p99": float(np.percentile(self.rtf_scores, 99)),
            },
            "audio_duration": {
                "total_seconds": float(np.sum(self.audio_durations)),
                "total_minutes": float(np.sum(self.audio_durations) / 60),
                "mean": float(np.mean(self.audio_durations)),
            },
            "sample_count": len(self.wer_scores),
            "error_count": len(self.errors),
        }
