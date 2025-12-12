# Medical Speech Dataset Processing Instructions

How to process the medical conversational speech datasets **"United-Syn-Med"** from HuggingFace which is used for benchmarking in the MediTalk project. This is necessary in order to complete the benchmarking process successfully.

## Prerequisites

Have completed the data fetching process as described in the [data fetching instructions](../fetching/README.md) to obtain the raw datasets.

## Usage

```sh
# Run the processing script
chmod +x process_data.sh
./process_data.sh
```

Note: Processing may take several minutes (took up to 1 hour during development).

## Output Structure
```
../processed/
└── USM/                             # HuggingFace Dataset
    ├── audio/                       # Processed audio files
    └── metadata.csv                 # Processed metadata file
```

Note: Opening the 'audio' folder in your explorer on VSCode may cause your system to crash due to the large number of files (~800'000 .mp3 files). You don't need to access this folder directly.
