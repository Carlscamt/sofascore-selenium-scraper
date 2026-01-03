import sys
import os
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tournament_scraper import process_round

def main():
    print("Verifying Tournament Scraper (Round 1 Only)...")
    
    unique_tournament_id = 17 # EPL
    season_id = 61627 # 24/25
    round_id = 1
    
    results = process_round(unique_tournament_id, season_id, round_id)
    
    if results:
        print(f"SUCCESS: Fetched {len(results)} matches for Round {round_id}.")
        df = pd.DataFrame(results)
        print("Columns:", df.columns.tolist())
        print(df.head(2))
        
        # Verify specific stats
        if 'stats_home_ballPossession' in df.columns:
            print("Verfication PASSED: Stats columns found.")
        else:
            print("Verification WARN: Stats columns missing.")
            
        # Verify H2H
        if 'h2h_home_wins' in df.columns:
            print("Verification PASSED: H2H columns found.")
    else:
        print("FAILURE: No results returned.")

if __name__ == "__main__":
    main()
