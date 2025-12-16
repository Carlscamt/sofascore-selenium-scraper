import time
import json
import random
import sys
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

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
    print(f"Fetching: {url}")
    for attempt in range(retries + 1):
        try:
            driver.get(url)
            # Fast check for pre tag
            elems = driver.find_elements(By.TAG_NAME, "pre")
            if elems:
                content = elems[0].text
                if not content:
                    print("   [WARN] Empty pre tag content")
                    continue
                return json.loads(content)
        except Exception as e:
            print(f"   [Retry {attempt}] Error: {e}")
            pass
        time.sleep(random.uniform(1.0, 2.0)) 
    return None

def analyze_player(player_data, side):
    """Deep dive into a single player's data structure"""
    player = player_data.get('player', {})
    print(f"\n--- {side} Player: {player.get('name')} ---")
    
    # Check for requested features
    market_value = player.get('marketValueCurrency') 
    market_val_amt = player.get('marketValue')
    
    if market_val_amt is None:
        market_val_amt = player.get('proposedMarketValueRaw')
        
    height = player.get('height')
    position = player.get('position')
    
    print(f"  Name: {player.get('name')}")
    print(f"  Position: {position}")
    print(f"  Height: {height}")
    print(f"  Market Value: {market_val_amt} {market_value}")
    
    # Print all keys to see what else is there
    print(f"  All Player Keys: {list(player.keys())}")
    
    return {
        'has_height': height is not None,
        'has_market_value': market_val_amt is not None,
        'has_position': position is not None
    }

def process_lineups(data):
    if not data:
        print("No data to process")
        return

    confirmed = data.get('confirmed', False)
    print(f"\nLineups Confirmed: {confirmed}")
    
    home_team = data.get('home', {}).get('players', [])
    away_team = data.get('away', {}).get('players', [])
    
    print(f"Home Players: {len(home_team)}")
    print(f"Away Players: {len(away_team)}")
    
    if home_team:
        print("\nAnalyzing first Home player...")
        analyze_player(home_team[0], "HOME")
        
        # Calculate totals using proposedMarketValueRaw if marketValue is missing
        home_mvs = []
        for p in home_team:
            pl = p.get('player', {})
            val = pl.get('marketValue') or pl.get('proposedMarketValueRaw')
            
            if isinstance(val, dict):
                val = val.get('value')
                
            if val is not None: 
                home_mvs.append(val)
            
        home_heights = [p.get('player', {}).get('height', 0) for p in home_team if p.get('player', {}).get('height')]
        
        total_market_value = sum(home_mvs)
        avg_market_value = sum(home_mvs) / len(home_mvs) if home_mvs else 0
        avg_height = sum(home_heights) / len(home_heights) if home_heights else 0
        
        print(f"\nHOME Stats:")
        print(f"  Total Market Value: {total_market_value}")
        print(f"  Avg Market Value: {avg_market_value:.2f}")
        print(f"  Avg Height: {avg_height:.2f}")

    if away_team:
        print("\nAnalyzing first Away player...")
        analyze_player(away_team[0], "AWAY")
        
        # Calculate totals
        away_mvs = []
        for p in away_team:
            pl = p.get('player', {})
            val = pl.get('marketValue') or pl.get('proposedMarketValueRaw')
            
            if isinstance(val, dict):
                val = val.get('value')
                
            if val is not None: 
                away_mvs.append(val)

        away_heights = [p.get('player', {}).get('height', 0) for p in away_team if p.get('player', {}).get('height')]
        
        total_market_value = sum(away_mvs)
        avg_market_value = sum(away_mvs) / len(away_mvs) if away_mvs else 0
        avg_height = sum(away_heights) / len(away_heights) if away_heights else 0
        
        print(f"\nAWAY Stats:")
        print(f"  Total Market Value: {total_market_value}")
        print(f"  Avg Market Value: {avg_market_value:.2f}")
        print(f"  Avg Height: {avg_height:.2f}")

if __name__ == "__main__":
    # Fix encoding for Windows console
    sys.stdout.reconfigure(encoding='utf-8')
    
    # User provided: 15035359 
    target_id = 15035359 
    
    if len(sys.argv) > 1:
        target_id = sys.argv[1]
        
    print(f"Debugging Lineups for Event: {target_id}")
    
    driver = initialize_driver()
    if driver:
        try:
            url = f"https://www.sofascore.com/api/v1/event/{target_id}/lineups"
            data = fetch_json_content(driver, url)
            
            if data:
                process_lineups(data)
            else:
                print("Failed to fetch data")
        finally:
            driver.quit()
