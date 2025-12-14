"""
TTS Benchmark - Main Orchestrator

Runs the complete TTS benchmark pipeline:
1. Load dataset sample
2. For each TTS model (sequentially):
   - Prompt user to start service
   - Generate all audio files
   - Evaluate all metrics
   - Save results
3. Generate comparison reports and plots

Usage:
    python benchmark_tts.py --config config.yaml
"""

import argparse
import logging
import yaml
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Import our modules
from data_sampler import create_benchmark_sample
from tts_client import TTSClient
from asr_client import ASRClient
# Import shared metrics
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))
import metrics as shared_metrics

calculate_wer = shared_metrics.calculate_wer
calculate_cer = shared_metrics.calculate_cer
calculate_rtf = shared_metrics.calculate_rtf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('benchmark.log')
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def benchmark_single_model(
    model_config: dict,
    data_sample: pd.DataFrame,
    whisper_url: str,
    results_dir: Path,
    save_audio: bool = True
) -> dict:
    """
    Benchmark a single TTS model.
    
    Args:
        model_config: {'name': str, 'url': str}
        data_sample: DataFrame with 'text' column
        whisper_url: Whisper ASR service URL
        results_dir: Where to save results
        save_audio: Whether to keep generated audio files
        
    Returns:
        Dictionary with aggregated results
    """
    model_name = model_config['name']
    model_url = model_config['url']
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking: {model_name}")
    logger.info(f"{'='*60}")
    
    # Initialize clients
    tts_client = TTSClient(model_name, model_url)
    asr_client = ASRClient(whisper_url)
    
    # Health checks
    logger.info("Running health checks...")
    if not tts_client.health_check():
        logger.error(f"❌ {model_name} service is not responding!")
        return {'error': f'{model_name} service not available'}
    
    if not asr_client.health_check():
        logger.error("❌ Whisper ASR service is not responding!")
        return {'error': 'Whisper service not available'}
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_results_dir = results_dir / f"{timestamp}_{model_name}"
    audio_dir = model_results_dir / "audio_samples"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Results will be saved to: {model_results_dir}")
    
    # Metrics storage
    results = []
    
    # Process each sample
    logger.info(f"\nProcessing {len(data_sample)} samples...")
    for idx, row in tqdm(data_sample.iterrows(), total=len(data_sample), desc=f"{model_name}"):
        sample_id = row['sample_id']
        text = row['text']
        
        # Generate audio
        audio_path = audio_dir / f"sample_{sample_id:04d}.wav"
        tts_result = tts_client.synthesize(text, str(audio_path))
        
        if not tts_result['success']:
            results.append({
                'sample_id': sample_id,
                'text': text,
                'error': tts_result['error'],
                'success': False
            })
            continue
        
        # Transcribe audio (ASR round-trip)
        asr_result = asr_client.transcribe_file(str(audio_path))
        
        if not asr_result['success']:
            results.append({
                'sample_id': sample_id,
                'text': text,
                'audio_path': str(audio_path),
                'generation_time': tts_result['generation_time'],
                'audio_duration': tts_result['duration'],
                'error': asr_result['error'],
                'success': False
            })
            continue
        
        # Calculate metrics
        transcription = asr_result['text']
        wer_score = calculate_wer(text, transcription)
        cer_score = calculate_cer(text, transcription)
        rtf = calculate_rtf(tts_result['generation_time'], tts_result['duration'])
        
        # Store results
        results.append({
            'sample_id': sample_id,
            'text': text,
            'transcription': transcription,
            'audio_path': str(audio_path),
            'generation_time': tts_result['generation_time'],
            'audio_duration': tts_result['duration'],
            'rtf': rtf,
            'wer': wer_score,
            'cer': cer_score,
            'asr_latency': asr_result['latency'],
            'success': True
        })
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_csv = model_results_dir / "detailed_results.csv"
    results_df.to_csv(results_csv, index=False)
    logger.info(f"✓ Saved detailed results to {results_csv}")
    
    # Calculate summary statistics
    successful = results_df[results_df['success'] == True]
    summary = {
        'model': model_name,
        'timestamp': timestamp,
        'total_samples': len(results_df),
        'successful_samples': len(successful),
        'failed_samples': len(results_df) - len(successful),
        'success_rate': len(successful) / len(results_df) if len(results_df) > 0 else 0,
    }
    
    if len(successful) > 0:
        summary.update({
            'wer_mean': float(successful['wer'].mean()),
            'wer_median': float(successful['wer'].median()),
            'wer_std': float(successful['wer'].std()),
            'cer_mean': float(successful['cer'].mean()),
            'cer_median': float(successful['cer'].median()),
            'rtf_mean': float(successful['rtf'].mean()),
            'rtf_median': float(successful['rtf'].median()),
            'avg_generation_time': float(successful['generation_time'].mean()),
            'avg_audio_duration': float(successful['audio_duration'].mean()),
            'total_audio_minutes': float(successful['audio_duration'].sum() / 60),
        })
    
    # Save summary
    summary_json = model_results_dir / "summary.json"
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✓ Saved summary to {summary_json}")
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS SUMMARY - {model_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Success Rate: {summary['success_rate']:.1%} ({summary['successful_samples']}/{summary['total_samples']})")
    if len(successful) > 0:
        logger.info(f"WER: {summary['wer_mean']:.3f} ± {summary['wer_std']:.3f} (median: {summary['wer_median']:.3f})")
        logger.info(f"CER: {summary['cer_mean']:.3f} (median: {summary['cer_median']:.3f})")
        logger.info(f"RTF: {summary['rtf_mean']:.3f} (median: {summary['rtf_median']:.3f})")
        logger.info(f"Avg Generation Time: {summary['avg_generation_time']:.2f}s")
        logger.info(f"Total Audio: {summary['total_audio_minutes']:.1f} minutes")
    
    # Delete audio if not saving
    if not save_audio:
        import shutil
        shutil.rmtree(audio_dir)
        logger.info(f"✓ Cleaned up audio files (save_audio=False)")
    
    return summary


def main():
    """Main entry point for TTS benchmark."""
    parser = argparse.ArgumentParser(description='TTS Benchmark for Medical Speech')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("TTS BENCHMARK - Medical Conversational Speech")
    logger.info("="*60)
    
    # Load configuration
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # Create results directory
    results_dir = Path(config['results_dir'])
    results_dir.mkdir(exist_ok=True)
    
    # Load or create data sample
    dataset_config = config['dataset']
    sample_csv = results_dir / "benchmark_sample.csv"
    
    if sample_csv.exists():
        logger.info(f"Loading existing sample from {sample_csv}")
        data_sample = pd.read_csv(sample_csv)
    else:
        logger.info("Creating new benchmark sample...")
        data_sample = create_benchmark_sample(
            metadata_path=dataset_config['metadata_path'],
            sample_size=dataset_config['sample_size'],
            output_path=str(sample_csv),
            strategy=dataset_config['sampling_strategy'],
            seed=dataset_config['random_seed']
        )
    
    logger.info(f"Loaded {len(data_sample)} samples")
    
    # Benchmark each model sequentially
    all_summaries = []
    whisper_url = config['whisper_service']['url']
    
    for model_config in config['tts_models']:
        model_name = model_config['name']
        
        # Prompt user to start service
        logger.info(f"\n{'='*60}")
        logger.info(f"Ready to benchmark: {model_name}")
        logger.info(f"Service URL: {model_config['url']}")
        logger.info(f"{'='*60}")
        input(f"\n⚠️  Please ensure {model_name} service is running, then press ENTER to continue...")
        
        # Run benchmark
        summary = benchmark_single_model(
            model_config=model_config,
            data_sample=data_sample,
            whisper_url=whisper_url,
            results_dir=results_dir,
            save_audio=config.get('save_audio', True)
        )
        
        all_summaries.append(summary)
        
        # Prompt to stop service
        logger.info(f"\n✓ Completed {model_name} benchmark")
        input(f"You can now stop the {model_name} service. Press ENTER to continue...")
    
    # Save overall comparison
    comparison_json = results_dir / "model_comparison.json"
    with open(comparison_json, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    logger.info(f"\n✓ Saved model comparison to {comparison_json}")
    
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK COMPLETE!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
