"""
Recalculate H2H with Proper Filtering (No Leakage)
===================================================
Uses the h2h/events endpoint and filters:
1. Only status.type == "finished"
2. Only event ID < current event ID (excludes current match)
Then retrains the model with corrected h2h features.
"""

import pandas as pd
import numpy as np
import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def get_driver():
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--blink-settings=imagesEnabled=false')
    options.add_argument('--log-level=3')
    prefs = {
        "profile.managed_default_content_settings.images": 2,
        "profile.managed_default_content_settings.stylesheets": 2,
    }
    options.add_experimental_option("prefs", prefs)
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)

def fetch_json(driver, url, retries=2):
    for _ in range(retries + 1):
        try:
            driver.get(url)
            elems = driver.find_elements(By.TAG_NAME, "pre")
            if elems:
                return json.loads(elems[0].text)
        except:
            pass
        time.sleep(0.5)
    return None

def calculate_clean_h2h(events, current_event_id, home_team_id, away_team_id):
    """
    Calculate h2h stats from events list, excluding:
    - Not finished matches
    - Current match (by ID)
    - Matches with ID >= current (future leakage)
    """
    home_wins = 0
    away_wins = 0
    draws = 0
    
    for event in events:
        # Skip if not finished
        if event.get('status', {}).get('type') != 'finished':
            continue
        
        # Skip if event ID >= current (includes current match)
        event_id = event.get('id', float('inf'))
        if event_id >= current_event_id:
            continue
        
        # Determine winner
        winner_code = event.get('winnerCode')
        event_home_id = event.get('homeTeam', {}).get('id')
        event_away_id = event.get('awayTeam', {}).get('id')
        
        # Match the winner to our home/away perspective
        if winner_code == 1:  # Home team won in this event
            if event_home_id == home_team_id:
                home_wins += 1
            elif event_home_id == away_team_id:
                away_wins += 1
        elif winner_code == 2:  # Away team won in this event
            if event_away_id == home_team_id:
                home_wins += 1
            elif event_away_id == away_team_id:
                away_wins += 1
        elif winner_code == 3:  # Draw
            draws += 1
    
    return home_wins, away_wins, draws

def recalculate_h2h_for_dataset(df, sample_size=None):
    """
    Recalculate h2h for matches in dataset using proper filtering.
    """
    driver = get_driver()
    
    results = []
    total = len(df) if sample_size is None else min(sample_size, len(df))
    
    print(f"Recalculating H2H for {total} matches...")
    
    for idx, row in df.head(total).iterrows():
        event_id = row['id']
        
        try:
            # Fetch h2h events
            url = f"https://www.sofascore.com/api/v1/event/{event_id}/h2h/events"
            data = fetch_json(driver, url)
            
            if data and 'events' in data:
                # We need team IDs - fetch event details
                event_url = f"https://www.sofascore.com/api/v1/event/{event_id}"
                event_data = fetch_json(driver, event_url)
                
                if event_data and 'event' in event_data:
                    home_team_id = event_data['event'].get('homeTeam', {}).get('id')
                    away_team_id = event_data['event'].get('awayTeam', {}).get('id')
                    
                    h_wins, a_wins, draws = calculate_clean_h2h(
                        data['events'], event_id, home_team_id, away_team_id
                    )
                    
                    results.append({
                        'id': event_id,
                        'h2h_home_wins_clean': h_wins,
                        'h2h_away_wins_clean': a_wins,
                        'h2h_draws_clean': draws,
                        'h2h_total_clean': h_wins + a_wins + draws
                    })
                    
                    if len(results) % 20 == 0:
                        print(f"  Processed {len(results)}/{total} matches...")
                else:
                    results.append({'id': event_id, 'h2h_home_wins_clean': None, 
                                   'h2h_away_wins_clean': None, 'h2h_draws_clean': None})
            else:
                results.append({'id': event_id, 'h2h_home_wins_clean': None,
                               'h2h_away_wins_clean': None, 'h2h_draws_clean': None})
                
        except Exception as e:
            print(f"  Error on event {event_id}: {e}")
            results.append({'id': event_id, 'h2h_home_wins_clean': None,
                           'h2h_away_wins_clean': None, 'h2h_draws_clean': None})
        
        time.sleep(0.3)  # Rate limiting
    
    driver.quit()
    
    return pd.DataFrame(results)

def verify_h2h_difference(df_original, df_clean_h2h):
    """Compare original h2h with cleaned version."""
    merged = df_original.merge(df_clean_h2h, on='id', how='inner')
    
    print("\n" + "="*50)
    print("H2H COMPARISON (Original vs Clean)")
    print("="*50)
    
    # Check for differences
    merged['home_diff'] = merged['h2h_home_wins'] - merged['h2h_home_wins_clean']
    merged['away_diff'] = merged['h2h_away_wins'] - merged['h2h_away_wins_clean']
    merged['draw_diff'] = merged['h2h_draws'] - merged['h2h_draws_clean']
    
    has_diff = merged[(merged['home_diff'] != 0) | (merged['away_diff'] != 0) | (merged['draw_diff'] != 0)]
    
    print(f"Matches with H2H difference: {len(has_diff)} / {len(merged)}")
    
    if len(has_diff) > 0:
        print(f"\nAverage difference in h2h_home_wins: {merged['home_diff'].mean():.2f}")
        print(f"Average difference in h2h_away_wins: {merged['away_diff'].mean():.2f}")
        print(f"Average difference in h2h_draws: {merged['draw_diff'].mean():.2f}")
        
        print("\n[!] LEAKAGE DETECTED: Original h2h included current/future matches!")
    else:
        print("\n[OK] No difference - original h2h was already clean!")
    
    return merged

if __name__ == "__main__":
    # Load original dataset
    df = pd.read_csv("sofascore_large_dataset.csv")
    print(f"Loaded {len(df)} matches")
    
    # Recalculate h2h for a sample (full dataset would take too long)
    sample_size = 50  # Adjust as needed
    print(f"\nRecalculating H2H for {sample_size} matches (sample)...")
    
    clean_h2h = recalculate_h2h_for_dataset(df, sample_size=sample_size)
    
    # Compare
    comparison = verify_h2h_difference(df.head(sample_size), clean_h2h)
    
    # Save cleaned h2h
    clean_h2h.to_csv("h2h_clean_sample.csv", index=False)
    print(f"\nSaved clean h2h to: h2h_clean_sample.csv")
