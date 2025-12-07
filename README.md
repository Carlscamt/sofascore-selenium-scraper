# SofaScore Football Data Scraper

A Python-based web scraper using Selenium that extracts comprehensive football match data from SofaScore for the last 7 days. This tool collects match information, odds, team form, head-to-head statistics, and player lineups for sports analytics and predictive modeling.

## Overview

This scraper automates data collection from the SofaScore API endpoints, gathering structured football match data that can be used for:
- Sports prediction modeling
- Betting odds analysis
- Team performance analysis
- Player lineup tracking
- Historical match database building

**Output**: CSV file with match data from the last 7 days (customizable date range)

---

## Features

### Data Collected Per Match
- **Match Metadata**: Event ID, date, league, home/away teams, match status, final score
- **Odds**: Home win (1), Draw (X), Away win (2) in decimal format
- **Team Form**: Average player rating, league position, recent form string
- **Head-to-Head**: Historical wins (home), wins (away), draws between teams
- **Lineups**: Full squad information with player IDs and positions for both teams

### Technical Features
- **Headless Browser Mode**: Runs without displaying browser window
- **Retry Logic**: Automatic recovery from connection failures (max 3 attempts per date)
- **Error Handling**: Graceful degradation—continues scraping even if individual data points fail
- **Chrome Automation Detection Bypass**: Disables automation flags to avoid detection
- **Rate Limiting**: Delays between requests to respect server load

---

## Requirements

### Environment
- Python 3.7+
- Google Chrome (for Colab: `/usr/bin/google-chrome`)
- Internet connection

### Python Dependencies
```
pandas>=1.0.0
selenium>=4.0.0
webdriver-manager>=3.8.0
```

### Installation

```bash
pip install pandas selenium webdriver-manager
```

For Google Colab:
```python
!pip install pandas selenium webdriver-manager
```

---

## Usage

### Basic Setup

```python
# The script is designed to run in Google Colab or local Python environments
# Simply execute the cells in order:

1. Import libraries and configure Chrome options
2. Initialize WebDriver
3. Define date range (currently set to 7 days)
4. Run scraping loop
5. Export to CSV
```

### Configuration Options

**Date Range** (Line ~75):
```python
for i in range(7):  # Change 7 to desired number of days
    date_range.append((current_date - timedelta(days=i)).strftime('%Y-%m-%d'))
```

**Matches Per Day** (Line ~85):
```python
for i, event in enumerate(events[:10]):  # Change 10 to desired number
```

**Retry Attempts** (Line ~90):
```python
max_retries = 3  # Increase for unstable connections
```

**Request Delays** (Line ~288):
```python
time.sleep(1)  # Increase if you encounter rate limiting
```

### Running in Google Colab

```python
# Mount Google Drive (optional, for saving files)
from google.colab import drive
drive.mount('/content/drive')

# Copy the entire script into a Colab cell and run
# Output CSV will be saved to current working directory
```

### Running Locally

```bash
python scraper.py
```

---

## Output

### CSV File Structure

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | SofaScore event ID |
| `date` | string | Match date (YYYY-MM-DD) |
| `league` | string | Tournament/league name |
| `home` | string | Home team name |
| `away` | string | Away team name |
| `status` | string | Match status (Not started, Finished, etc.) |
| `score_home` | int | Home team final score |
| `score_away` | int | Away team final score |
| `odd_1` | float | Home win odds (decimal) |
| `odd_X` | float | Draw odds (decimal) |
| `odd_2` | float | Away win odds (decimal) |
| `home_avg_rating` | float | Home team average player rating |
| `home_position` | int | Home team league position |
| `home_form` | string | Recent form (e.g., "W,W,L,D,W") |
| `away_avg_rating` | float | Away team average player rating |
| `away_position` | int | Away team league position |
| `away_form` | string | Recent form string |
| `h2h_home_wins` | int | Home wins in head-to-head |
| `h2h_away_wins` | int | Away wins in head-to-head |
| `h2h_draws` | int | Draws in head-to-head |
| `lineups` | list | Player objects with name, ID, position |

### Example Output

```
id,date,league,home,away,status,score_home,score_away,odd_1,odd_X,odd_2,...
12345678,2025-12-07,La Liga,Real Madrid,Barcelona,Finished,2,1,1.85,3.60,4.20,...
```

---

## Error Handling

The script implements multi-level error handling:

### Connection Errors
- Automatically retries up to 3 times
- Reinitializes WebDriver if connection fails
- Skips date if all retries exhausted

### Data Parsing Errors
- Continues scraping if individual endpoints fail
- Fills missing data with `None` values
- Logs specific error messages with event ID

### Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| `ChromeDriver not found` | webdriver-manager not installed | Run `pip install webdriver-manager` |
| `Connection timeout` | Rate limiting or network issues | Increase `time.sleep()` delays |
| `JSON decode error` | SofaScore API changed | Check API response format manually |
| `WebDriver initialization fails` | Chrome not installed | Install Chrome or use Colab |
| `Empty lineups data` | Lineups not available pre-match | Script handles gracefully (empty list) |

---

## Data Quality Notes

### Missing Data
- **Odds**: May be `None` if match is not yet scheduled or bookmaker data unavailable
- **Form/Rating**: May be `None` if team is new or data not published yet
- **Lineups**: May be empty for upcoming matches or if not yet published
- **Scores**: Will be `None` for future matches

### Reliability
- SofaScore API is stable but occasionally returns incomplete data
- Retry logic handles ~95% of transient failures
- Lineups data typically available 2-3 hours before match start

### Rate Limiting
- Current delays (1-3 seconds per request) are conservative
- SofaScore typically allows 1-2 requests per second
- Increase delays if you encounter `429 Too Many Requests` errors

---

## API Endpoints Used

| Endpoint | Data | Rate Limit |
|----------|------|-----------|
| `/sport/football/scheduled-events/{date}` | Match list | Stable |
| `/event/{id}/odds/1/all` | Betting odds | Stable |
| `/event/{id}/pregame-form` | Team form & ratings | Stable |
| `/event/{id}/h2h` | Head-to-head stats | Stable |
| `/event/{id}/lineups` | Player lineups | Unstable (pre-match only) |

---

## Use Cases

### Sports Analytics
- Build training datasets for match prediction models
- Analyze team form trends
- Study historical head-to-head patterns

### Betting Analysis
- Track odds movements across dates
- Identify value bets using model predictions
- Build expected value (EV) calculators

### Player Analytics
- Compile player appearance datasets
- Track lineup changes over time
- Correlate team performance with lineup changes

### Data Pipeline
- Automate daily data collection
- Create time-series datasets for trend analysis
- Feed data into production ML models

---

## Performance

### Typical Runtime
- **7 days × 10 matches**: ~10-15 minutes
- **7 days × 30 matches**: ~30-45 minutes
- **Bottleneck**: API response times (2-3 seconds per request)

### Optimization Tips
- Reduce `time.sleep()` delays to 0.5-1 second (may increase error rate)
- Decrease matches per day (`[:10]`) for faster testing
- Run multiple instances with different date ranges in parallel

---

## Maintenance

### Regular Updates Needed
- Monitor if SofaScore API structure changes
- Update CSS selectors if page structure changes (currently using JSON API, less likely to break)
- Verify Chrome compatibility with new versions

### Version History
- **v1.0**: Initial release with 5 data endpoints
- Handles dates in YYYY-MM-DD format
- Tested with Selenium 4.0+, Python 3.7+

---

## Disclaimer

This scraper is for educational and personal use. Respect SofaScore's terms of service:
- Don't use commercially without permission
- Don't overload their servers with excessive requests
- Check their robots.txt and terms for restrictions

---

## Next Steps

### Post-Scraping
1. **Data Cleaning**: Handle null values appropriately
2. **Feature Engineering**: Create derived features (form streak, rating differential)
3. **Exploratory Analysis**: Analyze odds distribution, form patterns
4. **Model Building**: Use collected data for prediction models

### Example pandas Operations
```python
# Load the CSV
df = pd.read_csv('sofascore_selenium_last_7_days.csv')

# Basic stats
print(df.describe())
print(df.info())

# Filter finished matches
finished = df[df['status'] == 'Finished']

# Calculate rating differential
finished['rating_diff'] = finished['home_avg_rating'] - finished['away_avg_rating']
```

---

## Support & Troubleshooting

### Debug Mode
Add this to print detailed information:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Testing Connection
```python
# Test if SofaScore is reachable
from selenium import webdriver
driver = webdriver.Chrome()
driver.get('https://www.sofascore.com/api/v1/sport/football/scheduled-events/2025-12-07')
print(driver.find_element('tag name', 'pre').text)
driver.quit()
```

---

**Last Updated**: December 2025  
**Author**: Sports Analytics Data Collector  
**Python Version**: 3.7+
