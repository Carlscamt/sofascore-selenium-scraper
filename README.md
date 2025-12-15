# Football Match Predictor & Betting Bot 🤖⚽

A complete end-to-end machine learning system for predicting football matches and identifying value bets.

## 🚀 Overview
This project consists of 4 main components working together to beat the bookmakers:
1.  **Data Collector**: High-speed, multi-threaded Selenium scraper to build a large historical dataset.
2.  **ML Engine**: XGBoost model that learns from historical data to predict match outcomes.
3.  **Future Scraper**: Targeted scraper to fetch odds and stats for upcoming matches (next 3-5 days).
4.  **Prediction Dashboard**: Interactive Streamlit app to visualize predictions and find the best bets.

---

## 🛠️ Installation

### Prerequisites
*   Python 3.8+
*   Google Chrome (installed)

### Install Dependencies
```bash
pip install pandas selenium webdriver-manager xgboost scikit-learn matplotlib streamlit
```

---

## ⚡ Usage Guide

### Step 1: Build the Database
Run the high-performance scraper to collect historical data (last ~6 months).
> *Note: This opens 4 headless Chrome workers to speed up collection.*
```bash
python sofascore_scraper.py
```
*   **Output**: `sofascore_large_dataset.csv` (~600+ matches)

### Step 2: Update Upcoming Matches
Fetch the latest schedule, odds, and pre-match stats for the next few days.
```bash
python future_scraper.py
```
*   **Output**: `sofascore_future_matches.csv`

### Step 3: Launch the Dashboard
Open the interactive app to view predictions and betting advice.
```bash
python -m streamlit run app.py
```
*   **Features**:
    *   **Live Training**: Retrains the model on your latest data instantly.
    *   **Value Finder**: Highlights bets with positive Expected Value (EV).
    *   **Filters**: Filter by League, Confidence, or EV.

---

## 📊 File Structure

| File | Description |
| :--- | :--- |
| `sofascore_scraper.py` | **Historical Scraper**. Multi-threaded. Collects 1000+ past matches. |
| `future_scraper.py` | **Future Scraper**. Collects upcoming scheduled matches for prediction. |
| `ml_model.py` | **Analytics Core**. Contains the XGBoost model, Cross-Validation logic, and Strategy Backtesting. |
| `app.py` | **Frontend**. Streamlit dashboard for user interaction. |
| `check_leakage.py` | **Verification Tool**. Checks for data leakage to ensure model integrity. |

---

## 📈 Performance (Backtest)
Based on a 5-Fold Cross-Validation of 600+ matches:
*   **Base Strategy**: 55% ROI
*   **Conservative Strategy (>60% Prob)**: 66% ROI
*   **Longshot Strategy (>3.0 Odds)**: 121% ROI

*> Note: Past performance does not guarantee future results. Always gamble responsibly.*
