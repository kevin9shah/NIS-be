import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.data_loader import load_directory_sampled

def main():
    directory = "/Users/kabra_jay/Downloads/archive/Edge-IIoTset dataset/Selected dataset for ML and DL"
    try:
        df = load_directory_sampled(directory, sample_frac=0.01)
        print(f"Success! Loaded DataFrame shape: {df.shape}")
        print(df.head())
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
