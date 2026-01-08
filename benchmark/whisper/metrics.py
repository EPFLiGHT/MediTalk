"""
Whisper ASR Metrics - Re-exports shared metrics for convenience

All metrics imported from shared module for consistency across benchmarks.
"""

import sys
from pathlib import Path

# Import all shared metrics
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.metrics import (
    calculate_wer,
    calculate_cer,
    calculate_rtf,
    normalize_text,
    MetricsAggregator
)

# Re-export for convenience
__all__ = ['calculate_wer', 'calculate_cer', 'calculate_rtf', 'normalize_text', 'MetricsAggregator']
