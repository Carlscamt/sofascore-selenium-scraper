from sofascore_scraper import process_date
import pandas as pd
import sys

# Encoding fix
sys.stdout.reconfigure(encoding='utf-8')

# Test date known to have matches (based on previous logs)
date_str = "2025-10-15"

print(f"Testing scraper integration for date: {date_str}")
results = process_date(date_str)

if results:
    df = pd.DataFrame(results)
    
    # Check if new columns exist
    cols = [
        'home_total_market_value', 'away_total_market_value', 
        'home_avg_height', 'away_avg_height',
        'home_defenders', 'home_forwards'
    ]
    
    missing = [c for c in cols if c not in df.columns]
    
    if missing:
        print(f"FAILED: Missing columns: {missing}")
    else:
        print("\nSUCCESS! New columns found. Sample data:")
        print(df[['home', 'away'] + cols].head(3).to_string())
        
        # Check for non-null values
        non_null = df['home_total_market_value'].notna().sum()
        print(f"\nNon-null home_total_market_value: {non_null}/{len(df)}")
        
        output_file = "sofascore_sample_with_lineups.csv"
        df.to_csv(output_file, index=False)
        print(f"\n[SAVE] Saved sample data to {output_file}")
else:
    print("No results found for date.")
