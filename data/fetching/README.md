# Medical Speech Dataset Fetching Instructions

How to download the medical conversational speech datasets **"United-Syn-Med"** from HuggingFace and **"Medical Speech, Transcription, and Intent"** from Kaggle, which are used for benchmarking in the MediTalk project.

## Prerequisites

1. **HuggingFace Token**
   - Go to https://huggingface.co/datasets/united-we-care/United-Syn-Med
   - Accept the dataset terms
   - Get your token from https://huggingface.co/settings/tokens
   - Copy the token (format: `hf_...`)

2. **Kaggle API Credentials**
   - Go to https://www.kaggle.com/settings
   - Click "Generate New Token"
   - Copy the displayed token (format: `KGAT_...`)

3. **.env file**
    - Create a file named `.env` in the parent directory of this script if not already present (follow .env.example template).
    - Add the following lines to the `.env` file:
      ```
      HF_DATA_TOKEN=your_huggingface_token_here
      KAGGLE_API_TOKEN=your_kaggle_api_token_here
      ```


## Usage

```sh
# Run the fetching script
chmod +x fetch_data.sh
./fetch_data.sh
```

## Output Structure
```
../raw/
├── United-Syn-Med/                             # HuggingFace Dataset
└── medical-speech-transcription-and-intent/    # Kaggle Dataset
```
