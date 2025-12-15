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

# --- Configuration ---
MAX_WORKERS = 4          # Number of parallel browsers
DAYS_AHEAD = 4           # Check Today + 3 days ahead
MATCHES_PER_DAY = 30     # Limit to top matches to avoid clutter
# ---------------------

def get_optimized_options():
    """Confirgure headless chrome for maximum speed."""
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

def fetch_json_content(driver, url, retries=2):
    """Robust JSON fetcher from <pre> tag."""
    for attempt in range(retries + 1):
        try:
            driver.get(url)
            elems = driver.find_elements(By.TAG_NAME, "pre")
            if elems:
                return json.loads(elems[0].text)
        except Exception:
            pass
        time.sleep(random.uniform(0.5, 1.5))
    return None

def process_date(date_str):
    """
    Worker function: Scrapes FUTURE matches for a single date.
    Returns a list of match dictionaries.
    """
    results = []
    print(f"[DATE] Checking schedule for: {date_str}")
    
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
        # Filter for "notstarted" matches
        future_events = [e for e in events if e.get('status', {}).get('type') == 'notstarted']
        
        # Sort by importance logic? (Sofascore roughly lists better leagues first usually)
        matches_to_scrape = future_events[:MATCHES_PER_DAY]
        
        print(f"   [FOUND] Found {len(future_events)} scheduled matches for {date_str}. Scraping top {len(matches_to_scrape)}...")
        
        for event in matches_to_scrape:
            # Build Row
            row = {
                'id': event['id'],
                'date': date_str,
                'league': event.get('tournament', {}).get('name'),
                'home': event.get('homeTeam', {}).get('name'),
                'away': event.get('awayTeam', {}).get('name'),
                'status': "Scheduled",
                'score_home': None, # Future
                'score_away': None, # Future
                'odd_1': None, 'odd_X': None, 'odd_2': None,
                'home_avg_rating': None, 'home_position': None, 'home_form': None,
                'away_avg_rating': None, 'away_position': None, 'away_form': None,
                'h2h_home_wins': None, 'h2h_away_wins': None, 'h2h_draws': None
            }
            
            event_id = event['id']
            
            # --- FETCH DETAILS ---
            
            # 1. ODDS (Crucial for prediction)
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
            
            # Only proceed if we have odds? No, let's keep even without odds just in case, but odds are needed for strategy.
            
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

            # 3. H2H
            h2h_data = fetch_json_content(driver, f"https://www.sofascore.com/api/v1/event/{event_id}/h2h")
            if h2h_data:
                td = h2h_data.get('teamDuel', {})
                row['h2h_home_wins'] = td.get('homeWins')
                row['h2h_away_wins'] = td.get('awayWins')
                row['h2h_draws'] = td.get('draws')

            results.append(row)
            
    except Exception as e:
        print(f"   [ERR] Error processing {date_str}: {e}")
    finally:
        driver.quit()
        print(f"   [DONE] Finished {date_str}: Collected {len(results)} matches.")
        
    return results

def main():
    print(f">>> STARTING FUTURE MATCH SCRAPER")
    print(f"Scope: Next {DAYS_AHEAD} days | Workers: {MAX_WORKERS}")
    
    # Generate Date List (Today + Next 3 days)
    base_date = datetime.now().date()
    dates = [(base_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(DAYS_AHEAD)]
    
    all_data = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_date = {executor.submit(process_date, d): d for d in dates}
        
        for future in concurrent.futures.as_completed(future_to_date):
            try:
                data = future.result()
                all_data.extend(data)
                print(f"[STATS] Total Progress: {len(all_data)} matches collected")
            except Exception as exc:
                print(f"   [ERR] Exception: {exc}")

    # Save
    if all_data:
        df = pd.DataFrame(all_data)
        # Filter rows where we at least have some odds or data (optional, but cleaner)
        filename = "sofascore_future_matches.csv"
        df.to_csv(filename, index=False)
        print(f"\n[SAVE] SAVED {len(df)} future matches to {filename}")
    else:
        print("\n[WARN] No future matches collected.")

if __name__ == "__main__":
    main()
