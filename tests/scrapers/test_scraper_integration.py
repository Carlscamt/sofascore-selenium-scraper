import sys
import os
import time

# Add parent directory to path to find utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sofascore_scraper import process_date

def main():
    print("Testing refactored scraper integration...")
    # Test a date with known matches
    test_date = "2024-05-19" # Premier League final day 23/24
    
    results = process_date(test_date)
    
    if len(results) > 0:
        print(f"SUCCESS: Fetched {len(results)} matches.")
        first = results[0]
        print("Sample Data Keys:", first.keys())
        
        # Check if new stats are present (from utils)
        if any(k.startswith('stats_') for k in first.keys()):
             print("SUCCESS: Match statistics present.")
        else:
             print("WARNING: No match statistics found (might be missing for this specific match).")
             
    else:
        print("FAILURE: No matches fetched.")

if __name__ == "__main__":
    main()
