import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analysis.feature_engineering import RollingStatsCalculator

def process_dataset(input_file, output_file):
    """
    Main processing function.
    Reads input_file, applies RollingStatsCalculator, and saves to output_file.
    """
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run the scraper first.")
        return False
        
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Original shape: {df.shape}")
    
    # Initialize Calculator
    calculator = RollingStatsCalculator(window_size=5)
    
    print("Calculating rolling averages...")
    df_enhanced = calculator.calculate_rolling_averages(df)
    
    # print("Calculating H2H history locally (fixing look-ahead bias)...")
    # df_enhanced = calculator.calculate_h2h_history(df_enhanced)
    
    print(f"Enhanced shape: {df_enhanced.shape}")
    
    # Verify new columns
    new_cols = [c for c in df_enhanced.columns if 'avg_' in c]
    print(f"Added {len(new_cols)} rolling feature columns.")
    
    print(f"Saving to {output_file}...")
    df_enhanced.to_csv(output_file, index=False)
    print("Done.")
    return True

if __name__ == "__main__":
    process_dataset("tournament_17_season_61627_full.csv", "dataset_rolling_features.csv")
