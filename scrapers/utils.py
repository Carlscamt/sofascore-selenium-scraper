import json
import time
import random
import re
from selenium.webdriver.common.by import By

def fetch_json_content(driver, url, retries=2):
    """Robust JSON fetcher from <pre> tag."""
    # print(f"Fetching: {url}") # Optional verify
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
            streaks_data[f"{base_col}_len"] = parsed['sample'] 
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

def process_h2h_filtered(driver, event_id, custom_id, current_start_time, home_team_id, away_team_id):
    """
    Fetches H2H events and filters them to strictly include only matches
    that occurred BEFORE the current match (based on timestamp).
    Returns dictionary with h2h_home_wins, h2h_away_wins, h2h_draws.
    """
    row = {'h2h_home_wins': 0, 'h2h_away_wins': 0, 'h2h_draws': 0}
    
    # 1. Try URL with customId (preferred for new API)
    h2h_data = None
    if custom_id:
        h2h_data = fetch_json_content(driver, f"https://www.sofascore.com/api/v1/event/{custom_id}/h2h/events")
    
    # 2. Fallback if no customId or failed fetch
    if not h2h_data or 'events' not in h2h_data:
         return row

    # 3. Process Events
    seen_ids = set()
    h2h_home_wins = 0
    h2h_away_wins = 0
    h2h_draws = 0
    
    for h2h_event in h2h_data['events']:
        h2h_event_id = h2h_event.get('id')
        h2h_start_time = h2h_event.get('startTimestamp')
        
        # CRITICAL: Filter using START TIMESTAMP
        if current_start_time and h2h_start_time:
            if h2h_start_time >= current_start_time:
                continue # Future or current match
        else:
             # Fallback to ID 
            if h2h_event_id >= event_id:
                continue
                
        # Skip non-finished
        if h2h_event.get('status', {}).get('type') != 'finished':
            continue
            
        if h2h_event_id in seen_ids:
            continue
        seen_ids.add(h2h_event_id)
        
        winner_code = h2h_event.get('winnerCode')
        h2h_home = h2h_event.get('homeTeam', {}).get('id')
        h2h_away = h2h_event.get('awayTeam', {}).get('id')
        
        if winner_code == 1:  # Home won
            if h2h_home == home_team_id: h2h_home_wins += 1
            elif h2h_home == away_team_id: h2h_away_wins += 1
        elif winner_code == 2:  # Away won
            if h2h_away == home_team_id: h2h_home_wins += 1
            elif h2h_away == away_team_id: h2h_away_wins += 1
        elif winner_code == 3:  # Draw
            h2h_draws += 1
            
    row['h2h_home_wins'] = h2h_home_wins
    row['h2h_away_wins'] = h2h_away_wins
    row['h2h_draws'] = h2h_draws
    
    return row


def process_match_statistics(driver, event_id):
    """
    Fetches and processes detailed match statistics.
    Returns a dictionary of flattened columns (e.g., stats_home_ballPossession).
    """
    stats_data = {}
    url = f"https://www.sofascore.com/api/v1/event/{event_id}/statistics"
    
    data = fetch_json_content(driver, url)
    if not data or 'statistics' not in data:
        return stats_data

    # Iterate through periods, we mainly care about "ALL"
    for period in data.get('statistics', []):
        if period.get('period') == 'ALL':
            for group in period.get('groups', []):
                for item in group.get('statisticsItems', []):
                    key = item.get('key')
                    home_val = item.get('homeValue')
                    away_val = item.get('awayValue')
                    
                    if key:
                        stats_data[f"stats_home_{key}"] = home_val
                        stats_data[f"stats_away_{key}"] = away_val
            break # Found ALL, stop looking
            

def find_season_id(driver, unique_tournament_id, year_str):
    """
    Finds the season ID for a given tournament and year string (e.g., "24/25" or "2024/2025").
    Returns the integer season ID if found, else None.
    """
    print(f"   [LOOKUP] Searching for Season '{year_str}' in Tournament {unique_tournament_id}...")
    url = f"https://www.sofascore.com/api/v1/unique-tournament/{unique_tournament_id}/seasons"
    
    data = fetch_json_content(driver, url)
    if not data or 'seasons' not in data:
        print("   [ERR] Failed to fetch seasons list.")
        return None
        
    for s in data['seasons']:
        if s.get('year') == year_str or s.get('name', '').endswith(year_str):
            print(f"   [FOUND] Season ID: {s.get('id')} ({s.get('name')})")
            return int(s.get('id'))
            
    # If not found, try fuzzy match on year
    print("   [INFO] Exact match not found. Trying fuzzy check...")
    for s in data['seasons']:
        if year_str in s.get('year', '') or year_str in s.get('name', ''):
            print(f"   [FOUND] Season ID: {s.get('id')} ({s.get('name')}) [Fuzzy Match]")
            return int(s.get('id'))
            
    print(f"   [ERR] Season '{year_str}' not found.")
    return None

def process_lineup_data(lineup_data):
    """
    Parses the /lineups API response to extract:
    - home_lineup_rating: Average rating of all players in the lineup
    - away_lineup_rating: Average rating of all players in the lineup
    - home_missing_count: Number of missing players
    - away_missing_count: Number of missing players
    """
    result = {
        'home_lineup_rating': None,
        'away_lineup_rating': None,
        'home_missing_count': 0,
        'away_missing_count': 0
    }
    
    if not lineup_data:
        return result

    for side in ['home', 'away']:
        if side in lineup_data:
            # 1. Calculate Average Rating from Players
            players = lineup_data[side].get('players', [])
            ratings = []
            for p in players:
                stats = p.get('statistics', {})
                rating = stats.get('rating')
                if rating:
                    ratings.append(rating)
            
            if ratings:
                result[f'{side}_lineup_rating'] = sum(ratings) / len(ratings)

            # 2. Count Missing Players
            missing = lineup_data[side].get('missingPlayers', [])
            result[f'{side}_missing_count'] = len(missing)
            
    return result
