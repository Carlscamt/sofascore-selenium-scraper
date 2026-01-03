
"""
Pipeline Orchestrator
=====================
Streamlines the Football Prediction Workflow into a single command.

Steps:
1. scrape: Runs the tournament scraper (optional).
2. process: Calculates rolling stats & cleans data.
3. model: Trains model and evaluates strategies.
"""

import sys
import os
import argparse

# Add subdirectories to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scrapers'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from scrapers.tournament_scraper import run_scraper
from analysis.process_data import process_dataset
from models.evaluate_strategies_8020 import evaluate_strategies
from models.generate_detailed_report import generate_report

def main():
    parser = argparse.ArgumentParser(description="Football Data Pipeline")
    parser.add_argument("--scrape", action="store_true", help="Run the web scraper first")
    parser.add_argument("--tournament", type=int, default=17, help="Unique Tournament ID (default 17 for EPL)")
    parser.add_argument("--season", type=int, default=61627, help="Season ID (default 61627 for 24/25)")
    parser.add_argument("--process", action="store_true", help="Run data processing")
    parser.add_argument("--model", action="store_true", help="Run model training and evaluation")
    parser.add_argument("--all", action="store_true", help="Run ALL steps")
    parser.add_argument("--year", type=str, help="Season Year (e.g., '24/25'). Overrides --season.")
    
    args = parser.parse_args()
    
    # 0. RESOLVE SEASON (If --year provided)
    if args.year:
        print(f"\n[SETUP] Resolving Season ID for Year: {args.year}")
        from scrapers.tournament_scraper import initialize_driver
        from scrapers.utils import find_season_id
        
        driver = initialize_driver()
        if driver:
            try:
                found_id = find_season_id(driver, args.tournament, args.year)
                if found_id:
                    args.season = found_id
                    print(f"   [SUCCESS] Using Season ID: {args.season}")
                else:
                    print(f"   [ERR] Could not resolve season. Using default: {args.season}")
            finally:
                driver.quit()
        else:
             print("   [ERR] Driver init failed. Using default.")

    # 1. SCRAPE
    scraped_file = f"tournament_{args.tournament}_season_{args.season}_full.csv"
    
    if args.scrape or args.all:
        print("\n" + "#"*40)
        print(">>> STEP 1: SCRAPING")
        print("#"*40)
        scraped_file = run_scraper(args.tournament, args.season)
        if not scraped_file:
            print("[STOP] Scraping failed or returned no data.")
            return

    # 2. PROCESS
    processed_file = "dataset_rolling_features.csv"
    
    if args.process or args.all:
        print("\n" + "#"*40)
        print(">>> STEP 2: PROCESSING")
        print("#"*40)
        # Verify input exists
        if not os.path.exists(scraped_file):
             # Try default name if not just scraped
             default_scraped = f"tournament_{args.tournament}_season_{args.season}_full.csv"
             if os.path.exists(default_scraped):
                 scraped_file = default_scraped
             else:
                 print(f"Error: Scraped file {scraped_file} not found.")
                 return

        success = process_dataset(scraped_file, processed_file)
        if not success:
            print("[STOP] Processing failed.")
            return

    # 3. MODEL
    if args.model or args.all:
        print("\n" + "#"*40)
        print(">>> STEP 3: MODELING & REPORTING")
        print("#"*40)
        
        if not os.path.exists(processed_file):
            print(f"Error: Processed file {processed_file} not found.")
            return
            
        print("\n--- Generating Detailed Metrics ---")
        generate_report(processed_file)
        
        print("\n--- Evaluating Strategies ---")
        evaluate_strategies(processed_file)
        
    print("\n[DONE] Pipeline execution finished.")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Default behavior if no args: Show help
        print("No arguments provided. Running --all as default or use --help")
        # Or uncomment to auto-run everything: # sys.argv.append('--all')
        # Let's show help
        sys.argv.append('--help')
        
    main()
