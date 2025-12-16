import time
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# User provided event ID: 15035359 (Bournemouth vs Man Utd)
# URL: https://www.sofascore.com/api/v1/event/15035359/odds/1/all

EVENT_ID = "15035359"
URL = f"https://www.sofascore.com/api/v1/event/{EVENT_ID}/odds/1/all"

def get_json_selenium(url):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    driver = webdriver.Chrome(options=options)
    try:
        print(f"Fetching {url}...")
        driver.get(url)
        time.sleep(2) # Wait for potential challenge/load
        
        # Sofascore JSON often appears in a <pre> tag when accessed directly in browser context
        content = driver.find_element(By.TAG_NAME, "body").text
        
        # Try to parse straight text first
        try:
            data = json.loads(content)
            return data
        except json.JSONDecodeError:
            # Try <pre> tag if body fail
            try:
                elem = driver.find_element(By.TAG_NAME, "pre")
                data = json.loads(elem.text)
                return data
            except:
                return None
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        driver.quit()

def analyze_odds(data):
    if not data:
        print("No data found.")
        return

    print("\n>>> MARKETS FOUND:")
    markets = data.get('markets', [])
    
    for market in markets:
        market_name = market.get('marketName', 'Unknown')
        market_id = market.get('marketId')
        is_live = market.get('isLive')
        choices = market.get('choices', [])
        
        print(f"\nMarket: {market_name} (ID: {market_id})")
        for choice in choices:
            name = choice.get('name') or choice.get('choiceName')
            fractional = choice.get('fractionalValue')
            decimal = choice.get('initialFractionalValue') # Sometimes useful check
            # Usually 'fractionalValue' is actually 'n/d' string in some APIs, 
            # but usually Sofascore returns decimal as direct value or we calculate it.
            # Let's inspect the raw choice keys first.
            print(f"  - Choice: {choice}")

def main():
    data = get_json_selenium(URL)
    if data:
        # Dump full structure to confirm keys
        print(json.dumps(data, indent=2)[:500] + "...") 
        analyze_odds(data)
    else:
        print("Failed to fetch data.")

if __name__ == "__main__":
    main()
