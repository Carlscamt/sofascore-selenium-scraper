"""
Test to find the customId/slug in event data structure
"""
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

def fetch_json(driver, url):
    try:
        driver.get(url)
        elems = driver.find_elements(By.TAG_NAME, "pre")
        if elems:
            return json.loads(elems[0].text)
    except Exception as e:
        print(f"Error: {e}")
    return None

def main():
    driver = get_driver()
    
    # Get events from a date
    date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    url = f"https://www.sofascore.com/api/v1/sport/football/scheduled-events/{date}"
    print(f"Fetching events for {date}...")
    
    data = fetch_json(driver, url)
    if not data:
        print("Failed to fetch events")
        driver.quit()
        return
    
    events = data.get('events', [])
    finished = [e for e in events if e.get('status', {}).get('type') == 'finished']
    
    if finished:
        event = finished[0]
        print(f"\n{'='*60}")
        print(f"Sample Event Keys: {list(event.keys())}")
        print('='*60)
        
        # Look for slug-related fields
        for key in ['customId', 'slug', 'id', 'startTimestamp']:
            if key in event:
                print(f"  {key}: {event[key]}")
        
        print(f"\nFull event (first 1500 chars):")
        print(json.dumps(event, indent=2)[:1500])
        
        # Now test if the customId works for H2H
        custom_id = event.get('customId')
        event_id = event.get('id')
        
        if custom_id:
            h2h_url = f"https://www.sofascore.com/api/v1/event/{custom_id}/h2h/events"
            print(f"\n\nTesting H2H with customId: {h2h_url}")
            h2h_data = fetch_json(driver, h2h_url)
            if h2h_data and 'events' in h2h_data:
                print(f"SUCCESS! Got {len(h2h_data['events'])} H2H events using customId!")
                # Print first event IDs to verify filtering logic
                print("\nH2H Event IDs (for filtering check):")
                for i, h2h_event in enumerate(h2h_data['events'][:5]):
                    print(f"  Event {i+1}: ID={h2h_event.get('id')}, Status={h2h_event.get('status', {}).get('type')}")
            else:
                print(f"Failed: {h2h_data}")
        else:
            print("No customId found in event!")
    
    driver.quit()

if __name__ == "__main__":
    main()
