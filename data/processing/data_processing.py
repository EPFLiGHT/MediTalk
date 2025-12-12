"""
Data processing script for medical speech datasets.
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm


USM_DATA_DIR ='../raw/United-Syn-Med/data/'
OUTPUT_DIR = '../processed/USM/'

class DataProcessor:
    def __init__(self, usm_data_dir=USM_DATA_DIR, output_dir=OUTPUT_DIR):
        self.usm_data_dir = usm_data_dir
        self.output_dir = output_dir

    def process_USM_metadata(self, verbose: bool = True):
        """Process United Syn-Med metadata files."""
        if verbose:
            print(f"Processing USM metadata in {self.usm_data_dir}...")
        
        data_dir = Path(self.usm_data_dir)
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if metadata already exists
        output_metadata_path = output_dir / "metadata.csv"
        if output_metadata_path.exists():
            if verbose:
                print(f"✓ Metadata already exists at {output_metadata_path}")
            return output_metadata_path
        
        # Load split CSV files
        train_df = pd.read_csv(data_dir / "train.csv")
        validation_df = pd.read_csv(data_dir / "validation.csv")
        test_df = pd.read_csv(data_dir / "test.csv")
        
        if verbose:
            print(f"  Train: {len(train_df)} samples")
            print(f"  Validation: {len(validation_df)} samples")
            print(f"  Test: {len(test_df)} samples")
            print(f"  Columns: {list(train_df.columns)}")
        
        # Concatenate all splits
        all_metadata = pd.concat([train_df, validation_df, test_df], ignore_index=True)
        
        if verbose:
            print(f"  Total samples after concatenation: {len(all_metadata)}")
        
        # Update file paths to point to processed audio directory
        processed_audio_prefix = 'data/processed/USM/audio'
        all_metadata['file_name'] = all_metadata['file_name'].apply(
            lambda x: str(Path(processed_audio_prefix) / Path(x).name)
        )

        # Rename column to audio_path
        all_metadata.rename(columns={'file_name': 'audio_path'}, inplace=True)

        if verbose:
            print(f"  Updated audio file paths to {processed_audio_prefix} directory.")
        
        # Save processed metadata
        all_metadata.to_csv(output_metadata_path, index=False)
        
        if verbose:
            print(f"✓ Saved processed metadata to {output_metadata_path}")
            print(f"  Sample columns: {list(all_metadata.columns)}")
            print(f"  Sample data:\n{all_metadata.head(1)[['audio_path']].to_string(index=False)}")
        
        return output_metadata_path


    def process_USM_audios(self, verbose: bool = True):
        """Process United Syn-Med audio files."""
        if verbose:
            print(f"Processing USM audio files in {self.usm_data_dir}...")
        
        audio_dir = Path(self.usm_data_dir) / "audio"
        processed_dir = Path(self.output_dir) / "audio"
        processed_dir.mkdir(parents=True, exist_ok=True)

        for dir in audio_dir.iterdir():
            if dir.is_dir():
                audio_files = list(dir.rglob("*.mp3"))
                iterator = tqdm(audio_files, desc=f"Processing {dir.name}") if verbose else audio_files
                
                copied_count = 0
                skipped_count = 0
                
                for audio_file in iterator:
                    dest_file = processed_dir / audio_file.name
                    
                    # Skip if file already exists
                    if dest_file.exists():
                        skipped_count += 1
                        continue
                    
                    # Copy the file
                    with open(audio_file, 'rb') as src, open(dest_file, 'wb') as dst:
                        dst.write(src.read())
                    copied_count += 1
                
                if verbose:
                    print(f"  {dir.name}: Copied {copied_count} files, Skipped {skipped_count} existing files")

if __name__ == "__main__":
    processor = DataProcessor()
    processor.process_USM_metadata(verbose=True)
    # processor.process_USM_audios(verbose=True)
