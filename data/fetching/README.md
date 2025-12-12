# Medical Speech Dataset Fetching Instructions

How to download the medical conversational speech datasets **"United-Syn-Med"** from HuggingFace, which is used for benchmarking in the MediTalk project. This is necessary in order to complete the data processing and benchmarking process successfully.

## Prerequisites

1. **HuggingFace Token**
   - Go to https://huggingface.co/datasets/united-we-care/United-Syn-Med
   - Accept the dataset terms
   - Get your token from https://huggingface.co/settings/tokens
   - Copy the token (format: `hf_...`)

2. **.env file**
    - Create a file named `.env` in the parent directory of this script if not already present (follow .env.example template).
    - Add the following lines to the `.env` file:
      ```
      HF_DATA_TOKEN=your_huggingface_token_here
      ```


## Usage

```sh
# Run the fetching script
chmod +x fetch_data.sh
./fetch_data.sh
```

Note: Downloading may take several minutes depending on your internet connection. Additional time may be required for unzipping the datasets after downloading.

## Output Structure
```
../raw/
└── United-Syn-Med/     # HuggingFace Dataset
```
