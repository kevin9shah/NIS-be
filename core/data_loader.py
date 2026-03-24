import os
import glob
import logging
import pandas as pd
from typing import List

logger = logging.getLogger(__name__)

def load_directory_sampled(directory_path: str, sample_frac: float = 0.1) -> pd.DataFrame:
    """
    Reads all CSV files in a directory, randomly samples a fraction of rows 
    from each, and concatenates them into a single DataFrame.
    """
    if not os.path.exists(directory_path):
        raise ValueError(f"Path not found: {directory_path}")
        
    if os.path.isfile(directory_path) and directory_path.endswith('.csv'):
        csv_files = [directory_path]
    else:
        csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
        
    if not csv_files:
        raise ValueError(f"No CSV files found in: {directory_path}")
        
    logger.info(f"Found {len(csv_files)} CSV files in {directory_path}. Sampling {sample_frac*100}% of each.")
    
    dfs: List[pd.DataFrame] = []
    total_files_loaded = 0
    total_original_rows = 0
    
    for idx, file_path in enumerate(csv_files):
        try:
            # Read the CSV (low_memory=False prevents mixed-type inference chunks)
            df = pd.read_csv(file_path, low_memory=False)
            rows = len(df)
            total_original_rows += rows
            
            # Sample it
            sampled_df = df.sample(frac=sample_frac, random_state=42)
            dfs.append(sampled_df)
            total_files_loaded += 1
            
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(csv_files)} files...")
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            
    if not dfs:
        raise ValueError("Failed to load any valid CSV files from the directory.")
        
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Explicitly shuffle the entire concatenated dataset
    logger.info("Shuffling the globally merged dataset...")
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Successfully loaded {total_files_loaded} files. "
                f"Combined sampled rows: {len(merged_df)} (from {total_original_rows} total rows).")
    
    return merged_df
