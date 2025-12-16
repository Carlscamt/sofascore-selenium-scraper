# Sofascore Football Prediction System

A comprehensive football match prediction and value betting system using machine learning.

## 📁 Project Structure

```
sofascore-selenium-scraper/
├── scrapers/              # Data collection scripts
│   ├── sofascore_scraper.py    # Main historical data scraper
│   └── future_scraper.py       # Upcoming matches scraper
│
├── models/                # Machine learning models
│   ├── ml_model.py            # Main XGBoost model
│   ├── ml_model_fixed.py      # Fixed version with H2H
│   ├── ml_no_h2h.py           # Baseline model without H2H
│   └── run_full_model.py      # Full model training script
│
├── analysis/              # Backtesting & analysis scripts
│   ├── backtest_split.py           # Train/test split backtest
│   ├── bankroll_backtest.py        # 2% bankroll management test
│   ├── all_strategies_bankroll.py  # All strategies with bankroll
│   ├── strategy_backtest.py        # Strategy comparison
│   ├── strategy_explorer.py        # Deep strategy analysis
│   ├── performance_report.py       # Performance dashboard
│   ├── feature_importance_report.py # Feature analysis
│   └── test_betting_strategies.py  # Strategy testing
│
├── data/                  # CSV datasets
│   ├── sofascore_dataset_v2.csv         # Latest with lineup features
│   ├── sofascore_large_dataset.csv      # Large historical dataset
│   ├── sofascore_future_matches.csv     # Upcoming matches
│   └── ...other CSV files
│
├── reports/               # Generated charts & visualizations
│   ├── all_strategies_bankroll.png
│   ├── strategy_backtest_report.png
│   ├── feature_importance_report.png
│   └── ...other PNG reports
│
├── debug/                 # Debug & testing scripts
│   ├── debug_h2h_scraper.py
│   ├── debug_lineups_scraper.py
│   ├── check_leakage.py
│   └── ...other test files
│
├── app.py                 # Streamlit web dashboard
└── README.md
```

## 🚀 Quick Start

### 1. Scrape Data
```bash
cd scrapers
python sofascore_scraper.py
```

### 2. Train Model
```bash
cd models
python run_full_model.py
```

### 3. Run Backtest
```bash
cd analysis
python all_strategies_bankroll.py
```

## 📊 Features

- **Lineup Features**: Market value, height, position counts
- **H2H Data**: Historical head-to-head records
- **Multiple Strategies**: Favorites, Value Hunter, Conservative, etc.
- **Bankroll Management**: 2% stake simulation

## 📈 Latest Results (80/20 Split, 2% Stakes)

| Strategy | ROI |
|----------|-----|
| Favorites (Odds < 1.5) | -1.33% |
| Value Hunter (EV > 10%) | -3.85% |
| Base Case (All +EV) | -7.19% |

*Note: More data needed to achieve profitability*

## 🔧 Requirements

- Python 3.11+
- selenium, pandas, numpy, xgboost, matplotlib
- Chrome/ChromeDriver
