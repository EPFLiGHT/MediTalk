"""
Data fetching script for medical speech datasets.
"""

import os
from pathlib import Path
from abc import ABC, abstractmethod
from huggingface_hub import snapshot_download


def load_env_file(env_path: str = ".env"):
    """Load environment variables from .env file."""
    env_file = Path(env_path)
    if not env_file.exists():
        print(f"Warning: {env_path} not found")
        return
    
    print(f"Loading environment variables from {env_path}...")
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Remove inline comments
            if '#' in line:
                line = line.split('#')[0].strip()
            
            # Parse key=value
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                os.environ[key] = value
                print(f"  Loaded: {key}")


class DatasetFetcher(ABC):
    """Abstract base class for dataset fetchers."""
    
    def __init__(self, base_dir: str = "../raw"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def fetch(self) -> Path:
        """Download the dataset and return the local path."""
        pass


class HuggingFaceDatasetFetcher(DatasetFetcher):
    """Fetcher for Hugging Face datasets."""
    
    def __init__(
        self,
        repo_id: str,
        dataset_name: str,
        token: str = None,
        base_dir: str = "../raw"
    ):
        super().__init__(base_dir)
        self.repo_id = repo_id
        self.dataset_name = dataset_name
        self.token = token or os.getenv("HF_DATA_TOKEN")
        self.local_dir = self.base_dir / dataset_name
    
    def fetch(self) -> Path:
        """Download the dataset from HuggingFace."""
        if self.local_dir.exists() and any(self.local_dir.iterdir()):
            print(f"✓ Dataset {self.dataset_name} already exists at {self.local_dir}")
            return self.local_dir
        
        if not self.token:
            raise ValueError("HF_DATA_TOKEN not found in environment variables")
        
        print(f"Downloading {self.repo_id} to {self.local_dir}...")
        
        try:
            snapshot_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                local_dir=str(self.local_dir),
                token=self.token,
                resume_download=True
            )
            print(f"✓ Successfully downloaded {self.dataset_name}")
            return self.local_dir
        
        except Exception as e:
            print(f"✗ Error downloading {self.dataset_name}: {e}")
            raise


class KaggleDatasetFetcher(DatasetFetcher):
    """Fetcher for Kaggle datasets."""
    
    def __init__(
        self,
        dataset_id: str,
        dataset_name: str,
        base_dir: str = "../raw"
    ):
        super().__init__(base_dir)
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.local_dir = self.base_dir / dataset_name
    
    def fetch(self) -> Path:
        """Download the dataset from Kaggle."""
        # Check if dataset already exists
        if self.local_dir.exists() and any(self.local_dir.iterdir()):
            print(f"✓ Dataset {self.dataset_name} already exists at {self.local_dir}")
            return self.local_dir
        
        # IMPORTANT: Save token BEFORE importing kaggle
        kaggle_token = os.getenv("KAGGLE_API_TOKEN")
        if not kaggle_token:
            raise ValueError("KAGGLE_API_TOKEN not found in environment variables")
        
        import kaggle
        
        # Create kaggle.json temporarily for authentication
        kaggle_dir = Path.home() / ".config" / "kaggle"
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        kaggle_json_path = kaggle_dir / "kaggle.json"
        
        # Write temporary kaggle.json with the token
        import json
        kaggle_config = {
            "username": "",
            "key": kaggle_token
        }
        
        print(f"Downloading {self.dataset_id} to {self.local_dir}...")
        print("(Kaggle API does not show progress - please wait...)")
        
        try:
            # Write kaggle.json
            with open(kaggle_json_path, 'w') as f:
                json.dump(kaggle_config, f)
            kaggle_json_path.chmod(0o600)
            
            # Authenticate and download
            kaggle.api.authenticate()
            self.local_dir.mkdir(parents=True, exist_ok=True)
            
            # Download with quiet=False for some output
            kaggle.api.dataset_download_files(
                self.dataset_id,
                path=str(self.local_dir),
                unzip=True,
                quiet=False
            )
            
            print(f"✓ Successfully downloaded {self.dataset_name}")
            
            # Show what was downloaded
            files = list(self.local_dir.glob("*"))
            print(f"  Downloaded {len(files)} file(s):")
            for f in files[:5]:
                print(f"    - {f.name}")
            if len(files) > 5:
                print(f"    ... and {len(files) - 5} more")
            
            return self.local_dir
        
        except Exception as e:
            print(f"✗ Error downloading {self.dataset_name}: {e}")
            raise
        
        finally:
            # Clean up kaggle.json
            if kaggle_json_path.exists():
                kaggle_json_path.unlink()


def main():
    """Main function to fetch both datasets."""
    
    # Load .env file
    load_env_file(".env")
    
    # Check environment variables
    print("\nChecking environment variables...")
    hf_token = os.getenv("HF_DATA_TOKEN")
    kaggle_token = os.getenv("KAGGLE_API_TOKEN")
    
    if not hf_token:
        print("✗ HF_DATA_TOKEN not set!")
    else:
        print(f"✓ HF_DATA_TOKEN set: {hf_token[:10]}...")
    
    if not kaggle_token:
        print("✗ KAGGLE_API_TOKEN not set!")
    else:
        print(f"✓ KAGGLE_API_TOKEN set: {kaggle_token[:10]}...\n")
    
    # Fetch United-Syn-Med from HuggingFace
    hf_fetcher = HuggingFaceDatasetFetcher(
        repo_id="united-we-care/United-Syn-Med",
        dataset_name="United-Syn-Med"
    )
    
    # Fetch Medical Speech Transcription from Kaggle
    kaggle_fetcher = KaggleDatasetFetcher(
        dataset_id="paultimothymooney/medical-speech-transcription-and-intent",
        dataset_name="medical-speech-transcription-and-intent"
    )
    
    # Download datasets
    try:
        hf_path = hf_fetcher.fetch()
        print(f"HuggingFace dataset saved to: {hf_path}\n")
    except Exception as e:
        print(f"Failed to fetch HuggingFace dataset: {e}\n")
    
    try:
        kaggle_path = kaggle_fetcher.fetch()
        print(f"Kaggle dataset saved to: {kaggle_path}\n")
    except Exception as e:
        print(f"Failed to fetch Kaggle dataset: {e}\n")


if __name__ == "__main__":
    main()
