import json
import time
import random
import os
import sys
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Reuse configuration from main scraper
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

def fetch_json_content(driver, url, retries=2):
    print(f"Fetching: {url}")
    for attempt in range(retries + 1):
        try:
            driver.get(url)
            elems = driver.find_elements(By.TAG_NAME, "pre")
            if elems:
                content = elems[0].text
                return json.loads(content)
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            pass
        time.sleep(random.uniform(0.5, 1.5))
    return None

def main():
    # URL provided by user: Premier League (17), Season (61627), Round (37)
    # https://www.sofascore.com/api/v1/unique-tournament/17/season/61627/events/round/37
    
    unique_tournament_id = 17
    season_id = 61627
    round_id = 37
    
    url = f"https://www.sofascore.com/api/v1/unique-tournament/{unique_tournament_id}/season/{season_id}/events/round/{round_id}"
    
    driver = initialize_driver()
    if not driver:
        return

    try:
        data = fetch_json_content(driver, url)
        
        if data:
            print("Successfully fetched tournament round data.")
            
            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'debug')
            os.makedirs(output_dir, exist_ok=True)
            
            output_file = os.path.join(output_dir, f'tournament_{unique_tournament_id}_season_{season_id}_round_{round_id}.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            
            print(f"Data saved to: {output_file}")
            
            # Print analysis
            if 'events' in data:
                print(f"Found {len(data['events'])} events in this round.")
                if len(data['events']) > 0:
                    first_event = data['events'][0]
                    print(f"Sample Event: {first_event.get('homeTeam', {}).get('name')} vs {first_event.get('awayTeam', {}).get('name')}")
            else:
                print("No 'events' key found in response.")
                
        else:
            print("Failed to fetch data.")
            
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
