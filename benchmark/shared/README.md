# Shared Benchmark Utilities

Common code used by both Whisper (ASR) and TTS benchmarks to avoid duplication.

## Structure

```
benchmark/
├── shared/              # Shared utilities (this folder)
│   ├── __init__.py
│   ├── metrics.py       # WER, CER, RTF, MetricsAggregator
│   ├── data_sampler.py  # USM dataset sampling
│   └── README.md        # This file
│
├── whisper/             # ASR benchmark
│   ├── metrics.py       # Imports from ../shared, adds Whisper extensions
│   ├── data_sampler.py  # Imports from ../shared
│   └── ...
│
└── tts/                 # TTS benchmark  
    ├── metrics.py       # Imports from ../shared, adds TTS-specific metrics
    ├── data_sampler.py  # Imports from ../shared
    └── ...
```