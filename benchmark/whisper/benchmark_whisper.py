"""
Whisper ASR Benchmark Script.
Run via: ./run_benchmark.sh

Loads config from config.yaml, benchmarks all models, generates results + plots.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
import requests
import pandas as pd
from tqdm import tqdm
import yaml

from metrics import MetricsAggregator, calculate_wer, calculate_cer
from data_sampler import create_benchmark_sample

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('benchmark.log')]
)
logger = logging.getLogger(__name__)


class WhisperBenchmark:
    """Simple benchmark runner for Whisper ASR."""
    
    def __init__(self, config_path='config.yaml'):
        """Load configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.whisper_url = self.config['whisper_service']['url']
        self.results_dir = Path(self.config['output']['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model_sizes = self.config['models']['sizes']
        
    def transcribe_audio(self, audio_path, language=None):
        """Transcribe a single audio file."""
        endpoint = f"{self.whisper_url}/transcribe_from_path"
        payload = {"audio_path": audio_path, "language": language}
        
        start_time = time.time()
        try:
            response = requests.post(endpoint, json=payload, timeout=300)
            latency = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "text": data.get("text", ""),
                    "detected_language": data.get("detected_language", ""),
                    "latency": latency,
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "text": "",
                    "latency": latency,
                    "error": f"HTTP {response.status_code}"
                }
        except Exception as e:
            return {
                "success": False,
                "text": "",
                "latency": time.time() - start_time,
                "error": str(e)
            }
    
    def run_benchmark_for_model(self, model_size, sample_df):
        """Run benchmark for a specific model size."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking model: {model_size}")
        logger.info(f"{'='*60}\n")
        
        metrics = MetricsAggregator()
        results_list = []
        base_path = self.config['data']['base_path']
        
        for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc=f"{model_size}"):
            audio_path = row['audio_path']
            reference_text = row['transcription']
            
            # Make path absolute
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(base_path, audio_path)
            
            # Transcribe
            result = self.transcribe_audio(audio_path)
            
            if result['success']:
                wer_score = calculate_wer(reference_text, result['text'])
                cer_score = calculate_cer(reference_text, result['text'])
                audio_duration = 5.0  # Estimate
                
                metrics.add_sample(wer_score, cer_score, result['latency'], audio_duration)
                
                results_list.append({
                    'audio_path': audio_path,
                    'reference': reference_text,
                    'hypothesis': result['text'],
                    'wer': wer_score,
                    'cer': cer_score,
                    'latency': result['latency'],
                    'detected_language': result['detected_language'],
                    'error': None
                })
            else:
                metrics.add_sample(1.0, 1.0, result['latency'], 1.0, error=result['error'])
                results_list.append({
                    'audio_path': audio_path,
                    'reference': reference_text,
                    'hypothesis': "",
                    'wer': 1.0,
                    'cer': 1.0,
                    'latency': result['latency'],
                    'detected_language': "",
                    'error': result['error']
                })
        
        # Save results
        summary = metrics.get_summary()
        summary['model_size'] = model_size
        summary['timestamp'] = datetime.now().isoformat()
        
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(self.results_dir / f"{model_size}_detailed_results.csv", index=False)
        
        with open(self.results_dir / f"{model_size}_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Whisper {model_size} complete - WER: {summary['wer']['mean']:.4f}")
        return summary
    
    def run(self):
        """Run full benchmark for all models."""
        logger.info("="*60)
        logger.info("WHISPER ASR BENCHMARK")
        logger.info("="*60)
        
        # Create sample
        logger.info("\nCreating benchmark sample...")
        sample_df = create_benchmark_sample(
            metadata_path=self.config['data']['metadata_path'],
            sample_size=self.config['data']['sample_size'],
            output_path=str(self.results_dir / 'benchmark_sample.csv'),
            strategy=self.config['data']['sampling_strategy'],
            seed=self.config['data']['random_seed']
        )
        logger.info(f"Sample created: {len(sample_df)} files\n")
        
        # Benchmark each model
        all_results = {}
        for model_size in self.model_sizes:
            logger.info(f"\nPlease prepare Whisper service with model '{model_size}'")
            logger.info("   Then press Enter to continue...")
            input()
            
            # Check service health
            try:
                response = requests.get(f"{self.whisper_url}/health", timeout=5)
                if response.status_code != 200:
                    logger.error(f"Service not healthy, skipping {model_size}")
                    continue
                logger.info(f"Service ready")
            except:
                logger.error(f"Service not responding, skipping {model_size}")
                continue
            
            # Run benchmark
            results = self.run_benchmark_for_model(model_size, sample_df)
            all_results[model_size] = results
        
        # Generate reports
        self._generate_reports(all_results)
        
        logger.info("\n" + "="*60)
        logger.info("BENCHMARK COMPLETE")
        logger.info(f"Results: {self.results_dir}")
        logger.info("="*60)
    
    def _generate_reports(self, all_results):
        """Generate comparison reports and visualizations."""
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'models_tested': list(all_results.keys()),
            'comparison': all_results
        }
        with open(self.results_dir / "model_comparison.json", 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info("\n" + "="*60)
        logger.info("RESULTS SUMMARY")
        logger.info("="*60)
        print(f"\n{'Model':<10} {'WER':<10} {'CER':<10} {'Latency':<12} {'RTF':<10}")
        print("-" * 52)
        for model, results in all_results.items():
            wer = results['wer']['mean']
            cer = results['cer']['mean']
            lat = results['latency']['p95']
            rtf = results['rtf']['mean']
            print(f"{model:<10} {wer:<10.4f} {cer:<10.4f} {lat:<12.3f} {rtf:<10.3f}")
        print()
        
        # Visualizations
        try:
            from visualization.visualize import generate_all_plots
            generate_all_plots(self.results_dir)
            logger.info("Visualizations generated")
        except Exception as e:
            logger.error(f"Visualizations failed: {e}")


if __name__ == "__main__":
    benchmark = WhisperBenchmark()
    benchmark.run()
