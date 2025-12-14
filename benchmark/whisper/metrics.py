"""
Metrics calculation module for Whisper ASR benchmarking.
Computes WER, CER, and performance metrics.

Core metrics (WER, CER, RTF, MetricsAggregator) imported from shared utilities.
"""

from typing import Dict, List, Tuple
import numpy as np
import time
import sys
from pathlib import Path

# Import shared metrics (WER, CER, RTF, normalize_text, MetricsAggregator)
sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))
from metrics import (
    calculate_wer,
    calculate_cer,
    calculate_rtf,
    normalize_text,
    MetricsAggregator as BaseMetricsAggregator
)


# ============================================================================
# Whisper-Specific Extensions
# ============================================================================

class MetricsAggregator(BaseMetricsAggregator):
    """
    Whisper-specific metrics aggregator.
    Extends base aggregator with Whisper-specific features.
    """
    
    def get_summary(self) -> Dict:
        """
        Get summary statistics with Whisper-specific throughput metric.
        """
        base_summary = super().get_summary()
        
        # Add Whisper-specific throughput metric
        if base_summary and "audio_duration" in base_summary:
            base_summary["throughput"] = {
                "audio_minutes_per_hour": float(
                    (np.sum(self.audio_durations) / 60) / (np.sum(self.latencies) / 3600)
                ) if np.sum(self.latencies) > 0 else 0,
            }
        
        return base_summary
    
    def get_percentile_buckets(self, metric: str = 'wer', buckets: List[Tuple[float, float]] = None) -> Dict:
        """
        Get distribution of samples across performance buckets.
        
        Args:
            metric: 'wer' or 'cer'
            buckets: List of (min, max) tuples defining buckets
        
        Returns:
            Dictionary with bucket counts
        """
        if buckets is None:
            buckets = [
                (0.0, 0.05),   # Excellent
                (0.05, 0.10),  # Very Good
                (0.10, 0.20),  # Good
                (0.20, 0.30),  # Fair
                (0.30, 0.50),  # Poor
                (0.50, 1.0),   # Very Poor
            ]
        
        scores = self.wer_scores if metric == 'wer' else self.cer_scores
        
        distribution = {}
        for min_val, max_val in buckets:
            bucket_name = f"{min_val:.2f}-{max_val:.2f}"
            count = sum(1 for score in scores if min_val <= score < max_val)
            distribution[bucket_name] = count
        
        return distribution
