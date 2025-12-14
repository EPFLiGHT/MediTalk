"""
TTS Data Sampler - Wrapper for shared USM sampling utilities
"""

import pandas as pd
import logging
import sys
from pathlib import Path

# Import shared data sampler
sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))
import data_sampler as shared_sampler

logger = logging.getLogger(__name__)


def create_benchmark_sample(
    metadata_path: str,
    sample_size: int,
    output_path: str = None,
    strategy: str = "stratified",
    seed: int = 42
) -> pd.DataFrame:
    """
    Create TTS benchmark sample from USM dataset.
    
    Args:
        metadata_path: Path to USM metadata.csv
        sample_size: Number of samples to select
        output_path: Optional path to save sample
        strategy: 'random' or 'stratified'
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns: sample_id, text, text_length
    """
    # Use shared sampler
    df = shared_sampler.create_benchmark_sample(
        metadata_path=metadata_path,
        sample_size=sample_size,
        output_path=None,  # We'll handle saving ourselves
        strategy=strategy,
        seed=seed,
        text_column='transcription'
    )
    
    # Rename for TTS clarity
    if 'transcription' in df.columns:
        df = df.rename(columns={'transcription': 'text'})
    
    # Add sample IDs if not present
    if 'sample_id' not in df.columns:
        df.insert(0, 'sample_id', range(len(df)))
    
    # Add text length column for analysis
    if 'text_length' not in df.columns:
        df['text_length'] = df['text'].str.len()
    
    # Keep only necessary columns
    df = df[['sample_id', 'text', 'text_length']].copy()
    
    logger.info(f"TTS sample created: {len(df)} texts")
    logger.info(f"  Length range: {df['text_length'].min()}-{df['text_length'].max()} chars "
                f"(mean: {df['text_length'].mean():.1f})")
    
    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"Saved to {output_path}")
    
    return df
