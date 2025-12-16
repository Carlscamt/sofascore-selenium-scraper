"""
Debug H2H Scraper - Small Batch Test
=====================================
Scrapes 10 matches from the last week with detailed debug output
to verify the h2h/events endpoint is being processed correctly.
"""

import pandas as pd
import json
import time
from datetime import datetime, timedelta
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
    options.add_argument('--log-level=3')
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)

def fetch_json(driver, url, retries=2):
    for attempt in range(retries + 1):
        try:
            driver.get(url)
            elems = driver.find_elements(By.TAG_NAME, "pre")
            if elems:
                text = elems[0].text
                return json.loads(text), text[:500]  # Return parsed and raw preview
        except Exception as e:
            print(f"      [Attempt {attempt+1}] Error: {e}")
        time.sleep(1)
    return None, None

def debug_scrape():
    print("="*70)
    print("DEBUG H2H SCRAPER - 3 MATCHES")
    print("="*70)
    
    driver = get_driver()
    results = []
    
    # Get matches from 3 days ago
    date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    print(f"\nFetching matches for: {date}")
    
    url = f"https://www.sofascore.com/api/v1/sport/football/scheduled-events/{date}"
    data, raw = fetch_json(driver, url)
    
    if not data:
        print("ERROR: Could not fetch schedule!")
        driver.quit()
        return
    
    events = data.get('events', [])
    finished = [e for e in events if e.get('status', {}).get('type') == 'finished']
    print(f"Found {len(finished)} finished matches, taking first 3...")
    
    for i, event in enumerate(finished[:3]):
        event_id = event['id']
        home = event.get('homeTeam', {}).get('name', 'Unknown')
        away = event.get('awayTeam', {}).get('name', 'Unknown')
        home_id = event.get('homeTeam', {}).get('id')
        away_id = event.get('awayTeam', {}).get('id')
        
        print(f"\n--- Match {i+1}/10: {home} vs {away} (ID: {event_id}) ---")
        print(f"    Home Team ID: {home_id}, Away Team ID: {away_id}")
        
        # Use customId for /h2h/events endpoint (this is the key!)
        custom_id = event.get('customId')
        
        h2h_data = None
        h2h_raw = None
        
        if custom_id:
            h2h_url = f"https://www.sofascore.com/api/v1/event/{custom_id}/h2h/events"
            print(f"    Fetching H2H with customId: {h2h_url}")
            h2h_data, h2h_raw = fetch_json(driver, h2h_url)
        else:
            print(f"    [!] No customId in event data!")
            # Fallback to numeric /h2h endpoint
            h2h_url = f"https://www.sofascore.com/api/v1/event/{event_id}/h2h"
            print(f"    Trying fallback: {h2h_url}")
            h2h_data, h2h_raw = fetch_json(driver, h2h_url)
        
        if h2h_data is None:
            print("    [!] H2H data is None - API returned no data")
            results.append({'match': f"{home} vs {away}", 'h2h_home': None, 'h2h_away': None, 'h2h_draws': None, 'error': 'No data'})
            continue
        
        if 'events' not in h2h_data:
            print(f"    [!] No 'events' key in response. Keys: {list(h2h_data.keys())}")
            print(f"    Raw preview: {h2h_raw}")
            results.append({'match': f"{home} vs {away}", 'h2h_home': None, 'h2h_away': None, 'h2h_draws': None, 'error': 'No events key'})
            continue
        
        h2h_events = h2h_data['events']
        print(f"    Found {len(h2h_events)} H2H events")
        
        # Process h2h with filtering
        h2h_home_wins = 0
        h2h_away_wins = 0
        h2h_draws = 0
        counted = 0
        skipped_not_finished = 0
        skipped_future_id = 0
        seen_ids = set()
        
        for h2h_event in h2h_events:
            status_type = h2h_event.get('status', {}).get('type')
            h2h_id = h2h_event.get('id', float('inf'))
            
            # CRITICAL: Filter using START TIMESTAMP if available (more reliable than ID)
            # We only want matches that strictly started BEFORE the current match
            current_start_time = event.get('startTimestamp')
            h2h_start_time = h2h_event.get('startTimestamp')
            
            if current_start_time and h2h_start_time:
                if h2h_start_time >= current_start_time:
                    print(f"      [FILTER] Skipped future match (ID: {h2h_id}). Start: {h2h_start_time} >= Current: {current_start_time}")
                    skipped_future_id += 1 
                    continue
            else:
                # Fallback
                if h2h_id >= event_id:
                    print(f"      [FILTER] Skipped by ID (ID: {h2h_id} >= Current: {event_id})")
                    skipped_future_id += 1
                    continue
            
            if status_type != 'finished':
                skipped_not_finished += 1
                continue
            
            # Skip duplicates
            if h2h_id in seen_ids:
                continue
            seen_ids.add(h2h_id)
            
            counted += 1
            winner_code = h2h_event.get('winnerCode')
            h2h_home_team_id = h2h_event.get('homeTeam', {}).get('id')
            h2h_away_team_id = h2h_event.get('awayTeam', {}).get('id')
            
            if winner_code == 1:  # Home won
                if h2h_home_team_id == home_id:
                    h2h_home_wins += 1
                elif h2h_home_team_id == away_id:
                    h2h_away_wins += 1
            elif winner_code == 2:  # Away won
                if h2h_away_team_id == home_id:
                    h2h_home_wins += 1
                elif h2h_away_team_id == away_id:
                    h2h_away_wins += 1
            elif winner_code == 3:  # Draw
                h2h_draws += 1
        
        print(f"    Filtered: {counted} counted, {skipped_not_finished} not finished, {skipped_future_id} future/same ID")
        print(f"    Result: H2H Home={h2h_home_wins}, Away={h2h_away_wins}, Draws={h2h_draws}")
        
        results.append({
            'match': f"{home} vs {away}",
            'event_id': event_id,
            'h2h_total_events': len(h2h_events),
            'h2h_counted': counted,
            'h2h_home': h2h_home_wins,
            'h2h_away': h2h_away_wins,
            'h2h_draws': h2h_draws
        })
        
        time.sleep(0.5)
    
    driver.quit()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Save
    df.to_csv("h2h_debug_results.csv", index=False)
    print(f"\nSaved to: h2h_debug_results.csv")

if __name__ == "__main__":
    debug_scrape()
