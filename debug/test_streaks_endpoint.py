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
    options.add_argument('--log-level=3')
    # Use a realistic user agent
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)

def fetch_json(driver, url):
    try:
        print(f"Navigating to {url}...")
        driver.get(url)
        # Give it a moment to render
        time.sleep(2)
        # Sofascore API responses in browser often wrapped in pre tag
        elems = driver.find_elements(By.TAG_NAME, "pre")
        if elems:
            return json.loads(elems[0].text)
        else:
            # Fallback: try getting body text if it's raw JSON
            body = driver.find_element(By.TAG_NAME, "body").text
            return json.loads(body)
    except Exception as e:
        print(f"Error fetching JSON: {e}")
    return None

def test_streaks_endpoint():
    url = "https://www.sofascore.com/api/v1/event/12436470/team-streaks"
    
    driver = get_driver()
    try:
        data = fetch_json(driver, url)
        
        if data:
            print("Response JSON:")
            print(json.dumps(data, indent=2))
        else:
            print("Failed to get data.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    test_streaks_endpoint()
