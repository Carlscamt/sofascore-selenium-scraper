
import pandas as pd
import sys
import os
import time
from tqdm import tqdm

# Add parent directory to path to import scrapers
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scrapers.tournament_scraper import initialize_driver
from scrapers.utils import fetch_json_content, process_lineup_data

INPUT_FILE = "sofascore_combined.csv"
OUTPUT_FILE = "sofascore_combined_enriched.csv"
SAVE_INTERVAL = 20

def main():
    print(f">>> STARTING LINEUP ENRICHMENT")
    
    # Load Data
    if os.path.exists(OUTPUT_FILE):
        print(f"   [INFO] Resuming from {OUTPUT_FILE}")
        df = pd.read_csv(OUTPUT_FILE)
    elif os.path.exists(INPUT_FILE):
        print(f"   [INFO] Loading {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE)
        # Initialize new columns
        df['home_lineup_rating'] = None
        df['away_lineup_rating'] = None
        df['home_missing_count'] = None
        df['away_missing_count'] = None
    else:
        print(f"   [ERR] Input file {INPUT_FILE} not found!")
        return

    # Filter rows that need processing
    # We check if 'home_lineup_rating' is NaN
    # Note: 'pd.isna' works for None/NaN
    to_process_indices = df[df['home_lineup_rating'].isna()].index
    print(f"   [INFO] Matches to process: {len(to_process_indices)}")
    
    if len(to_process_indices) == 0:
        print("   [DONE] All matches already enriched.")
        return

    driver = initialize_driver()
    if not driver:
        print("   [ERR] Failed to initialize driver.")
        return

    processed_count = 0
    
    try:
        for idx in tqdm(to_process_indices, desc="Enriching Matches"):
            match_id = df.loc[idx, 'id']
            url = f"https://www.sofascore.com/api/v1/event/{match_id}/lineups"
            
            # Fetch Data
            try:
                lineup_json = fetch_json_content(driver, url)
                if lineup_json:
                    metrics = process_lineup_data(lineup_json)
                    
                    df.loc[idx, 'home_lineup_rating'] = metrics['home_lineup_rating']
                    df.loc[idx, 'away_lineup_rating'] = metrics['away_lineup_rating']
                    df.loc[idx, 'home_missing_count'] = metrics['home_missing_count']
                    df.loc[idx, 'away_missing_count'] = metrics['away_missing_count']
                else:
                    # Mark as processed but empty to avoid loop
                    df.loc[idx, 'home_lineup_rating'] = -1 
            except Exception as e:
                print(f"\n[ERR] Failed on match {match_id}: {e}")
                
            processed_count += 1
            
            # Periodic Save
            if processed_count % SAVE_INTERVAL == 0:
                df.to_csv(OUTPUT_FILE, index=False)
                
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user. Saving progress...")
    finally:
        df.to_csv(OUTPUT_FILE, index=False)
        driver.quit()
        print(f"\n[DONE] Saved progress to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
