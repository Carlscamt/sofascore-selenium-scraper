
import sys
import os
import argparse
import json

# Add parent directory to path to import scrapers
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scrapers.tournament_scraper import initialize_driver
from scrapers.utils import fetch_json_content

def main():
    parser = argparse.ArgumentParser(description="Debug Lineups Inspector")
    parser.add_argument("event_id", type=int, help="Event ID (e.g. 14025156)")
    args = parser.parse_args()
    
    print(f"Fetching lineups for Event {args.event_id}...")
    url = f"https://www.sofascore.com/api/v1/event/{args.event_id}/lineups"
    
    driver = initialize_driver()
    if not driver:
        print("Driver failed.")
        return

    try:
        data = fetch_json_content(driver, url)
        if data:
            print(">>> LINEUPS DATA RECEIVED")
            print(json.dumps(data, indent=2))
            

            # Quick Analysis Summary
            for side in ['home', 'away']:
                if side in data:
                    print(f"\n--- {side.upper()} TEAM ---")
                    print(f"Formation: {data[side].get('formation')}")
                    
                    players = data[side].get('players', [])
                    ratings = []
                    for p in players:
                        stats = p.get('statistics', {})
                        rating = stats.get('rating')
                        if rating:
                            ratings.append(rating)
                            
                    if ratings:
                        calc_avg = sum(ratings) / len(ratings)
                        print(f"Calculated Avg Rating (from {len(ratings)} players): {calc_avg:.2f}")
                    else:
                        print("No player ratings found in this lineup.")

                    missing = data[side].get('missingPlayers', [])
                    print(f"Missing Players: {len(missing)}")
                    for mp in missing:
                        print(f"  - {mp.get('player', {}).get('name')} ({mp.get('type')})")

        else:
            print("No data found or request blocked.")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
