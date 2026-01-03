
import sys
import os
import argparse

# Add parent directory to path to import scrapers
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scrapers.tournament_scraper import initialize_driver
from scrapers.utils import fetch_json_content

def main():
    parser = argparse.ArgumentParser(description="Debug Inspector")
    parser.add_argument("tournament_id", type=int, help="Unique Tournament ID (e.g. 17)")
    args = parser.parse_args()
    
    print(f"Fetching seasons for Tournament {args.tournament_id}...")
    url = f"https://www.sofascore.com/api/v1/unique-tournament/{args.tournament_id}/seasons"
    
    driver = initialize_driver()
    if not driver:
        print("Driver failed.")
        return

    try:
        data = fetch_json_content(driver, url)
        if data and 'seasons' in data:
            print(f"{'ID':<10} | {'Name':<20} | {'Year':<10}")
            print("-" * 50)
            for s in data['seasons']:
                print(f"{s.get('id'):<10} | {s.get('name'):<20} | {s.get('year'):<10}")
        else:
            print("No data found.")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
