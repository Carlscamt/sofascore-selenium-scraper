import pandas as pd
import time
import concurrent.futures
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import sys
import os

# Ensure we can import from utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (
    fetch_json_content, 
    process_complex_odds, 
    process_match_statistics, 
    process_streaks,
    process_h2h_filtered
)

# --- Configuration ---
MAX_WORKERS = 3  # Reduced workers to avoid ratelimits on heavy scraping
# ---------------------

def get_optimized_options():
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--blink-settings=imagesEnabled=false')
    options.add_argument('--disable-extensions')
    options.add_argument('--log-level=3')
    
    prefs = {
        "profile.managed_default_content_settings.images": 2,
        "profile.managed_default_content_settings.stylesheets": 2,
        "profile.managed_default_content_settings.fonts": 2,
    }
    options.add_experimental_option("prefs", prefs)
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    return options

def initialize_driver():
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=get_optimized_options())
        driver.set_page_load_timeout(30)
        return driver
    except Exception as e:
        print(f"   [Error] Driver Init Failed: {e}")
        return None

def process_event(event, driver, unique_tournament_id, season_id, round_id):
    """
    Processes a single event to extract all details.
    """
    event_id = event['id']
    date_str = event.get('startTimestamp') # Could convert to date string if needed
    
    row = {
        'id': event_id,
        'tournament_id': unique_tournament_id,
        'season_id': season_id,
        'round': round_id,
        'date': pd.to_datetime(event.get('startTimestamp'), unit='s').strftime('%Y-%m-%d'),
        'timestamp': event.get('startTimestamp'),
        'league': event.get('tournament', {}).get('name'),
        'home': event.get('homeTeam', {}).get('name'),
        'away': event.get('awayTeam', {}).get('name'),
        'status': event.get('status', {}).get('description'),
        'score_home': event.get('homeScore', {}).get('current'),
        'score_away': event.get('awayScore', {}).get('current'),
        # Initialize columns
        'odd_1': None, 'odd_X': None, 'odd_2': None,
        'home_avg_rating': None, 'home_position': None, 'home_form': None,
        'away_avg_rating': None, 'away_position': None, 'away_form': None,
        'h2h_home_wins': None, 'h2h_away_wins': None, 'h2h_draws': None,
        'home_total_market_value': None, 'away_total_market_value': None,
        'home_avg_height': None, 'away_avg_height': None,
        'home_defenders': 0, 'home_midfielders': 0, 'home_forwards': 0,
        'away_defenders': 0, 'away_midfielders': 0, 'away_forwards': 0
    }

    # 1. ODDS
    odds_data = fetch_json_content(driver, f"https://www.sofascore.com/api/v1/event/{event_id}/odds/1/all")
    if odds_data:
        for market in odds_data.get('markets', []):
            if market.get('marketName') == 'Full time':
                for choice in market.get('choices', []):
                    frac = choice.get('fractionalValue', '')
                    try:
                        if '/' in str(frac):
                            n, d = map(int, frac.split('/'))
                            val = round(1 + n/d, 2)
                        else:
                            val = float(frac)
                        
                        if choice['name'] == '1': row['odd_1'] = val
                        elif choice['name'] == 'X': row['odd_X'] = val
                        elif choice['name'] == '2': row['odd_2'] = val
                    except: pass
                break

    # 2. PREGAME FORM
    form_data = fetch_json_content(driver, f"https://www.sofascore.com/api/v1/event/{event_id}/pregame-form")
    if form_data:
        ht = form_data.get('homeTeam', {})
        at = form_data.get('awayTeam', {})
        row['home_avg_rating'] = ht.get('avgRating')
        row['home_position'] = ht.get('position')
        row['home_form'] = ','.join(ht.get('form', []))
        row['away_avg_rating'] = at.get('avgRating')
        row['away_position'] = at.get('position')
        row['away_form'] = ','.join(at.get('form', []))

    # 3. H2H - (With Date Filter via utils)
    custom_id = event.get('customId')
    current_start_time = event.get('startTimestamp')
    home_id = event.get('homeTeam', {}).get('id')
    away_id = event.get('awayTeam', {}).get('id')
    
    try:
        h2h_row = process_h2h_filtered(driver, event_id, custom_id, current_start_time, home_id, away_id)
        if h2h_row:
            row['h2h_home_wins'] = h2h_row['h2h_home_wins']
            row['h2h_away_wins'] = h2h_row['h2h_away_wins']
            row['h2h_draws'] = h2h_row['h2h_draws']
        else:
             # Fallback to simple endpoint (only if utils failed)
            h2h_fallback = fetch_json_content(driver, f"https://www.sofascore.com/api/v1/event/{event_id}/h2h")
            if h2h_fallback and 'teamDuel' in h2h_fallback:
                td = h2h_fallback.get('teamDuel', {})
                row['h2h_home_wins'] = td.get('homeWins', 0)
                row['h2h_away_wins'] = td.get('awayWins', 0)
                row['h2h_draws'] = td.get('draws', 0)
    except:
        pass

    # 4. LINEUPS
    lineups_data = fetch_json_content(driver, f"https://www.sofascore.com/api/v1/event/{event_id}/lineups")
    if lineups_data:
        for side in ['home', 'away']:
            team_players = lineups_data.get(side, {}).get('players', [])
            market_values = []
            heights = []
            defenders = 0
            midfielders = 0
            forwards = 0
            
            for p in team_players:
                player = p.get('player', {})
                mv = player.get('marketValue')
                if mv is None:
                    raw_mv = player.get('proposedMarketValueRaw')
                    if isinstance(raw_mv, dict): mv = raw_mv.get('value')
                if mv: market_values.append(mv)
                h = player.get('height')
                if h: heights.append(h)
                pos = player.get('position')
                if pos == 'D': defenders += 1
                elif pos == 'M': midfielders += 1
                elif pos == 'F': forwards += 1
            
            total_mv = sum(market_values) if market_values else 0
            avg_height = sum(heights) / len(heights) if heights else 0
            
            if side == 'home':
                row['home_total_market_value'] = total_mv
                row['home_avg_height'] = round(avg_height, 2)
                row['home_defenders'] = defenders
                row['home_midfielders'] = midfielders
                row['home_forwards'] = forwards
            else:
                row['away_total_market_value'] = total_mv
                row['away_avg_height'] = round(avg_height, 2)
                row['away_defenders'] = defenders
                row['away_midfielders'] = midfielders
                row['away_forwards'] = forwards

    # 5. STREAKS
    try:
        streaks = process_streaks(driver, event_id)
        if streaks: row.update(streaks)
    except: pass

    # 6. MATCH STATISTICS
    try:
        stats = process_match_statistics(driver, event_id)
        if stats: row.update(stats)
    except: pass

    # 7. COMPLEX ODDS
    try:
        complex = process_complex_odds(driver, event_id)
        if complex: row.update(complex)
    except: pass
    
    return row

import queue

def fetch_tournament_rounds(unique_tournament_id, season_id, rounds_range=range(1, 40)):
    """
    PRODUCER: Quickly fetches all finished matches from the tournament rounds.
    Returns a list of event objects to be processed.
    Uses a single driver to query api/events endpoint.
    """
    print(">>> [PRODUCER] Fetching Match List...")
    driver = initialize_driver()
    if not driver: return []
    
    all_events = []
    
    try:
        # We can iterate rounds sequentially as this is fast (just one API call per round)
        for r in rounds_range:
            try:
                url = f"https://www.sofascore.com/api/v1/unique-tournament/{unique_tournament_id}/season/{season_id}/events/round/{r}"
                data = fetch_json_content(driver, url)
                
                if not data or 'events' not in data:
                    continue
                    
                events = data.get('events', [])
                finished = [e for e in events if e.get('status', {}).get('type') == 'finished']
                
                # Enrich with round info for processing
                for e in finished:
                    e['_meta_round'] = r
                    
                all_events.extend(finished)
                print(f"   [FOUND] Round {r}: {len(finished)} matches.")
                
            except Exception as e:
                print(f"   [ERR] Round {r}: {e}")
                
    finally:
        driver.quit()
        
    print(f">>> [PRODUCER] Total Matches Found: {len(all_events)}")
    return all_events

def scrape_worker(unique_tournament_id, season_id, task_queue, result_list):
    """
    CONSUMER: Dedicated Agent that runs a browser and processes matches from the queue.
    """
    agent_id = threading.get_ident()
    print(f"   [AGENT {agent_id}] Starting up...")
    
    driver = initialize_driver()
    if not driver:
        print(f"   [AGENT {agent_id}] Failed to init driver.")
        return

    try:
        while True:
            try:
                # Non-blocking get? No, blocking with timeout to allow graceful shutdown
                event = task_queue.get(timeout=5) 
            except queue.Empty:
                break # Queue is empty, work done
                
            try:
                round_id = event.get('_meta_round')
                home = event.get('homeTeam',{}).get('name')
                away = event.get('awayTeam',{}).get('name')
                
                # print(f"   [AGENT {agent_id}] Processing: {home} vs {away}")
                
                row = process_event(event, driver, unique_tournament_id, season_id, round_id)
                result_list.append(row)
                
                task_queue.task_done()
            except Exception as e:
                print(f"   [AGENT {agent_id}] Error on event {event.get('id')}: {e}")
                task_queue.task_done()
                
    finally:
        driver.quit()
        print(f"   [AGENT {agent_id}] Shutting down.")

import threading

def run_scraper(unique_tournament_id=17, season_id=61627, max_workers=5):
    """
    Orchestrates the Multi-Agent Scraping.
    """
    print(f">>> STARTING MULTI-AGENT SCRAPER")
    print(f"Tournament: {unique_tournament_id} | Season: {season_id} | Agents: {max_workers}")
    
    # 1. PRODUCER PHASE
    events = fetch_tournament_rounds(unique_tournament_id, season_id)
    if not events:
        print("[STOP] No events found to scrape.")
        return None
        
    # 2. POPULATE QUEUE
    task_queue = queue.Queue()
    for e in events:
        task_queue.put(e)
        
    # 3. CONSUMER PHASE (AGENTS)
    result_list = [] # Thread-safe list append? Yes, append is atomic in CPython, but for safety usually lock.
    # However, for simple append, it's rarely an issue. We can use a lock if we want to be 100% safe.
    # result_lock = threading.Lock() # Not strictly needed for append
    
    print(f">>> [CONSUMER] Starting {max_workers} Agents to process {task_queue.qsize()} matches...")
    
    threads = []
    for _ in range(max_workers):
        t = threading.Thread(target=scrape_worker, args=(unique_tournament_id, season_id, task_queue, result_list))
        t.start()
        threads.append(t)
        
    # Wait for completion
    for t in threads:
        t.join()
        
    print(f">>> Processing Complete.")

    output_file = None
    if result_list:
        df = pd.DataFrame(result_list)
        df = df.sort_values(by='timestamp')
        
        output_file = f"tournament_{unique_tournament_id}_season_{season_id}_full.csv"
        df.to_csv(output_file, index=False)
        print(f"\n[SAVE] Saved {len(df)} matches to {output_file}")
    else:
        print("No data collected.")
        
    return output_file

if __name__ == "__main__":
    run_scraper()
