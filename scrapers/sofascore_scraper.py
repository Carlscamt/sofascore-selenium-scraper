import pandas as pd
import json
import time
import random
import concurrent.futures
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import WebDriverException

# --- Configuration ---
MAX_WORKERS = 5          # Number of parallel browsers
TARGET_MATCHES = 1300    # Goal
DAYS_TO_CHECK = 200      # Look back approx 6 months
# ---------------------

def get_optimized_options():
    """Confirgure headless chrome for maximum speed."""
    options = Options()
    options.add_argument('--headless=new') # Modern headless mode
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--blink-settings=imagesEnabled=false') # Disable images
    options.add_argument('--disable-extensions')
    options.add_argument('--log-level=3') # Minimize logging
    
    # Block huge resource types to save bandwidth
    prefs = {
        "profile.managed_default_content_settings.images": 2,
        "profile.managed_default_content_settings.stylesheets": 2,  # Disable CSS
        "profile.managed_default_content_settings.fonts": 2,        # Disable Fonts
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

def fetch_json_content(driver, url, retries=2):
    """Robust JSON fetcher from <pre> tag."""
    for attempt in range(retries + 1):
        try:
            driver.get(url)
            # Fast check for pre tag
            elems = driver.find_elements(By.TAG_NAME, "pre")
            if elems:
                return json.loads(elems[0].text)
        except Exception:
            pass
        time.sleep(random.uniform(0.5, 1.5)) # Tiny randomized wait
    return None

def convert_fractional(frac_str):
    """Converts '8/13' to 1.615."""
    try:
        if '/' in frac_str:
            num, den = map(int, frac_str.split('/'))
            return round(1 + (num / den), 3)
        return float(frac_str)
    except:
        return None

def process_complex_odds(driver, event_id):
    """
    Fetches advanced odds (BTTS, etc.) from odds/1/all.
    Returns dict with flattened odds keys.
    """
    odds_data = {}
    url = f"https://www.sofascore.com/api/v1/event/{event_id}/odds/1/all"
    data = fetch_json_content(driver, url)
    if not data: return odds_data
    
    markets = data.get('markets', [])
    for market in markets:
        # Market ID 5 = Both teams to score
        if market.get('marketId') == 5:
            for choice in market.get('choices', []):
                name = choice.get('name', '').lower()
                frac = choice.get('fractionalValue')
                decimal = convert_fractional(frac)
                
                if name == 'yes':
                    odds_data['odd_btts_yes'] = decimal
                elif name == 'no':
                    odds_data['odd_btts_no'] = decimal
                    
    return odds_data


import re

def slugify(text):
    """Converts 'More than 2.5 goals' to 'more_than_2_5_goals'."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')

def parse_streak_value(val_str):
    """
    Parses streak value string (e.g. '3' or '6/7').
    Returns dict with count, sample, pct.
    """
    try:
        if '/' in str(val_str):
            parts = str(val_str).split('/')
            count = int(parts[0])
            sample = int(parts[1])
            pct = round(count / sample, 3) if sample > 0 else 0.0
        else:
            # Just a number like "3"
            count = int(val_str)
            sample = count # Implied sample is the streak itself
            pct = 1.0
            
        return {
            'count': count,
            'sample': sample,
            'pct': pct
        }
    except Exception as e:
        return None

def process_streaks(driver, event_id):
    """
    Fetches and processes team streaks data.
    Returns a dictionary of flattened columns.
    """
    streaks_data = {}
    url = f"https://www.sofascore.com/api/v1/event/{event_id}/team-streaks"
    
    data = fetch_json_content(driver, url)
    if not data:
        return streaks_data
        
    # Process General Streaks
    if 'general' in data:
        for item in data['general']:
            name = item.get('name', '')
            val_str = item.get('value', '')
            team = item.get('team', '') # home, away, both
            
            parsed = parse_streak_value(val_str)
            if not parsed:
                continue
                
            slug = slugify(name)
            base_col = f"streak_{team}_{slug}"
            
            streaks_data[f"{base_col}_count"] = parsed['count']
            streaks_data[f"{base_col}_len"] = parsed['sample'] # Rename sample to len for clarity? Or keep sample.
            streaks_data[f"{base_col}_pct"] = parsed['pct']
            
    # Process Head2Head Streaks
    if 'head2head' in data:
        for item in data['head2head']:
            name = item.get('name', '')
            val_str = item.get('value', '')
            team = item.get('team', '')
            
            parsed = parse_streak_value(val_str)
            if not parsed:
                continue
                
            slug = slugify(name)
            # Prefix with h2h explicitly
            base_col = f"streak_h2h_{team}_{slug}"
            
            streaks_data[f"{base_col}_count"] = parsed['count']
            streaks_data[f"{base_col}_len"] = parsed['sample']
            streaks_data[f"{base_col}_pct"] = parsed['pct']
            
    return streaks_data

def process_date(date_str):
    """
    Worker function: Scrapes ALL desired matches for a single date.
    Returns a list of match dictionaries.
    """
    results = []
    print(f"[DATE] Starting processing for: {date_str}")
    
    driver = initialize_driver()
    if not driver:
        return []

    try:
        # 1. Get Schedule
        url_matches = f"https://www.sofascore.com/api/v1/sport/football/scheduled-events/{date_str}"
        data = fetch_json_content(driver, url_matches)
        
        if not data:
            print(f"   [WARN] No schedule found for {date_str}")
            driver.quit()
            return []
            
        events = data.get('events', [])
        # Filter for "finished" matches only to have full data
        finished_events = [e for e in events if e.get('status', {}).get('type') == 'finished']
        
        # Limit matches per day to ensure diversity? Or take all?
        # Let's take top 10 most relevant ones per day to avoid cluttering with minor leagues if we want quality.
        # But user said "1000 matches", fetching EVERYTHING is faster than filtering. 
        # Let's take up to 15 finished matches per day to spread the 1000 matches over ~70-100 days.
        matches_to_scrape = finished_events[:15]
        
        print(f"   [FOUND] Found {len(finished_events)} finished matches for {date_str}. Scraping top {len(matches_to_scrape)}...")
        
        for event in matches_to_scrape:
            # Build Row
            row = {
                'id': event['id'],
                'date': date_str,
                'league': event.get('tournament', {}).get('name'),
                'home': event.get('homeTeam', {}).get('name'),
                'away': event.get('awayTeam', {}).get('name'),
                'status': event.get('status', {}).get('description'),
                'score_home': event.get('homeScore', {}).get('current'),
                'score_away': event.get('awayScore', {}).get('current'),
                'odd_1': None, 'odd_X': None, 'odd_2': None,
                'home_avg_rating': None, 'home_position': None, 'home_form': None,
                'away_avg_rating': None, 'away_position': None, 'away_form': None,
                'h2h_home_wins': None, 'h2h_away_wins': None, 'h2h_draws': None,
                'home_total_market_value': None, 'away_total_market_value': None,
                'home_avg_height': None, 'away_avg_height': None,
                'home_defenders': 0, 'home_midfielders': 0, 'home_forwards': 0,
                'away_defenders': 0, 'away_midfielders': 0, 'away_forwards': 0
            }
            
            event_id = event['id']
            
            # --- FETCH DETAILS (Sequential within the Thread) ---
            
            # 1. ODDS
            odds_data = fetch_json_content(driver, f"https://www.sofascore.com/api/v1/event/{event_id}/odds/1/all")
            if odds_data:
                for market in odds_data.get('markets', []):
                    if market.get('marketName') == 'Full time':
                        for choice in market.get('choices', []):
                            frac = choice.get('fractionalValue', '')
                            # Parse fractional if needed, usually '1.5' comes in string too as decimal sometimes? 
                            # Actually implementation showed frac like '3/2'.
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

            # 3. H2H - Use /h2h/events endpoint with customId (works with /h2h/events)
            # The customId field (e.g. 'OdbsMeb') is required for the /h2h/events endpoint
            custom_id = event.get('customId')
            h2h_data = None
            
            # Use customId with /h2h/events endpoint (the new API format)
            if custom_id:
                h2h_data = fetch_json_content(driver, f"https://www.sofascore.com/api/v1/event/{custom_id}/h2h/events")
            
            # Fallback to /h2h endpoint with numeric ID (returns aggregate counts only)
            if not h2h_data or 'events' not in h2h_data:
                # Try the aggregate h2h endpoint as fallback (less precise but works)
                h2h_fallback = fetch_json_content(driver, f"https://www.sofascore.com/api/v1/event/{event_id}/h2h")
                if h2h_fallback and 'teamDuel' in h2h_fallback:
                    td = h2h_fallback.get('teamDuel', {})
                    row['h2h_home_wins'] = td.get('homeWins', 0)
                    row['h2h_away_wins'] = td.get('awayWins', 0)
                    row['h2h_draws'] = td.get('draws', 0)
            
            if h2h_data and 'events' in h2h_data:
                # Get team IDs from current event
                home_team_id = event.get('homeTeam', {}).get('id')
                away_team_id = event.get('awayTeam', {}).get('id')
                
                h2h_home_wins = 0
                h2h_away_wins = 0
                h2h_draws = 0
                seen_ids = set()  # Track seen event IDs to avoid duplicates
                
                # Get current event timestamp
                current_start_time = event.get('startTimestamp')
                
                for h2h_event in h2h_data['events']:
                    h2h_event_id = h2h_event.get('id', float('inf'))
                    h2h_start_time = h2h_event.get('startTimestamp')
                    
                    # CRITICAL: Filter using START TIMESTAMP if available (more reliable than ID)
                    # We only want matches that strictly started BEFORE the current match
                    if current_start_time and h2h_start_time:
                        if h2h_start_time >= current_start_time:
                            continue
                    else:
                        # Fallback to ID filtering if timestamps are missing (less reliable)
                        if h2h_event_id >= event_id:
                            continue
                    
                    # Skip non-finished matches
                    if h2h_event.get('status', {}).get('type') != 'finished':
                        continue
                    
                    # Skip duplicates
                    if h2h_event_id in seen_ids:
                        continue
                    seen_ids.add(h2h_event_id)
                    
                    winner_code = h2h_event.get('winnerCode')
                    h2h_home_id = h2h_event.get('homeTeam', {}).get('id')
                    h2h_away_id = h2h_event.get('awayTeam', {}).get('id')
                    
                    if winner_code == 1:  # Home won in h2h event
                        if h2h_home_id == home_team_id:
                            h2h_home_wins += 1
                        elif h2h_home_id == away_team_id:
                            h2h_away_wins += 1
                    elif winner_code == 2:  # Away won in h2h event
                        if h2h_away_id == home_team_id:
                            h2h_home_wins += 1
                        elif h2h_away_id == away_team_id:
                            h2h_away_wins += 1
                    elif winner_code == 3:  # Draw
                        h2h_draws += 1
                
                row['h2h_home_wins'] = h2h_home_wins
                row['h2h_away_wins'] = h2h_away_wins
                row['h2h_draws'] = h2h_draws

            # 4. LINEUPS & PLAYERS
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
                        
                        # Market Value
                        mv = player.get('marketValue')
                        if mv is None:
                            # Try raw structure
                            raw_mv = player.get('proposedMarketValueRaw')
                            if isinstance(raw_mv, dict):
                                mv = raw_mv.get('value')
                        if mv:
                            market_values.append(mv)
                            
                        # Height
                        h = player.get('height')
                        if h:
                            heights.append(h)
                            
                        # Position
                        pos = player.get('position')
                        if pos == 'D': defenders += 1
                        elif pos == 'M': midfielders += 1
                        elif pos == 'F': forwards += 1
                    
                    # Aggregates
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

            # 5. TEAM STREAKS
            try:
                streaks_data = process_streaks(driver, event_id)
                if streaks_data:
                    row.update(streaks_data)
            except Exception as e:
                pass # Fail silently for streaks, not critical

            # 6. COMPLEX ODDS (BTTS, etc.)
            try:
                complex_odds = process_complex_odds(driver, event_id)
                if complex_odds:
                    row.update(complex_odds)
            except Exception as e:
                pass

            except Exception as e:
                pass # Fail silently for streaks, not critical

            results.append(row)
            
    except Exception as e:
        print(f"   [ERR] Error processing {date_str}: {e}")
    finally:
        driver.quit()
        print(f"   [DONE] Finished {date_str}: Collected {len(results)} matches.")
        
    return results

def main():
    print(f">>> STARTING HIGH-PERFORMANCE SCRAPER")
    print(f"Target: {TARGET_MATCHES} matches | Workers: {MAX_WORKERS}")
    
    # Generate Date List (Last 180 days)
    base_date = datetime.now().date()
    dates = [(base_date - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, DAYS_TO_CHECK)]
    
    all_data = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_date = {executor.submit(process_date, d): d for d in dates}
        
        for future in concurrent.futures.as_completed(future_to_date):
            date = future_to_date[future]
            try:
                data = future.result()
                all_data.extend(data)
                
                print(f"[STATS] TOTAL PROGRESS: {len(all_data)} / {TARGET_MATCHES} matches")
                
                # Check target
                if len(all_data) >= TARGET_MATCHES:
                    print("[DONE] TARGET REACHED! Cancelling remaining tasks...")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                    
            except Exception as exc:
                print(f"   [ERR] {date} generated an exception: {exc}")

    # Save
    if all_data:
        df = pd.DataFrame(all_data)
        # Drop duplicates just in case
        df = df.drop_duplicates(subset=['id'])
        filename = "sofascore_dataset_v2.csv"
        df.to_csv(filename, index=False)
        print(f"\n[SAVE] SAVED {len(df)} matches to {filename}")
    else:
        print("\n[WARN] No data collected.")

if __name__ == "__main__":
    main()
