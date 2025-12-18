"""
TTS Data Sampler - Wrapper for shared USM sampling utilities
"""

import pandas as pd
import logging
import sys
from pathlib import Path
import importlib.util

shared_path = Path(__file__).parent.parent / 'shared' / 'data_sampler.py'
spec = importlib.util.spec_from_file_location("shared_data_sampler", shared_path)
shared_sampler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared_sampler)

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

    df = shared_sampler.create_benchmark_sample(
        metadata_path=metadata_path,
        sample_size=sample_size,
        output_path=None, # saving handled here, not in shared
        strategy=strategy,
        seed=seed,
        text_column='transcription'
    )
    
    if 'transcription' in df.columns:
        df = df.rename(columns={'transcription': 'text'})
    
    if 'sample_id' not in df.columns:
        df.insert(0, 'sample_id', range(len(df)))
    
    if 'text_length' not in df.columns:
        df['text_length'] = df['text'].str.len()
    
    df = df[['sample_id', 'text', 'text_length']].copy()
    
    logger.info(f"TTS sample created: {len(df)} texts")
    logger.info(f"  Length range: {df['text_length'].min()}-{df['text_length'].max()} chars "
                f"(mean: {df['text_length'].mean():.1f})")
    
    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"Saved to {output_path}")
    
    return df
