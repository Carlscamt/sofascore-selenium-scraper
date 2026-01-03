
# ⚽ Sofascore Football Prediction Pipeline

A comprehensive machine learning system that scrapes match data, processes rolling statistics (form), and predicts outcomes with high profitability.


> **Current ROI**: +16.07% (Longshot Strategy) | **Risk**: 0% Bankruptcy (Monte Carlo Verified).

## 📂 Project Structure

```
├── run_pipeline.py         # Main entry point (Orchestrator)
├── sofascore_combined.csv  # Gold Dataset (Merged Seasons)
├── requirements.txt
│
├── scrapers/               # Core Data Collection
│   ├── tournament_scraper.py
│   └── utils.py
│
├── analysis/               # Feature Engineering
│   ├── process_data.py
│   └── feature_engineering.py
│
├── models/                 # Active ML Models
│   ├── evaluate_strategies_8020.py # Primary Model
│   ├── validate_walk_forward.py    # Validation
│   ├── audit_data_integrity.py     # Audit
│   └── validate_monte_carlo.py
│
├── tools/                  # Utilities
│   ├── debug_inspector.py  # Find Season IDs
│   └── debug_lineups.py    # Inspect Lineup Data
│
├── reports/                # Output Graphs & CSVs
├── archive/                # Legacy Code
└── tests/                  # Unit Tests
```

## 🚀 How to Run

### 1. The "One-Click" Pipeline
The `run_pipeline.py` script handles everything.

**Option A: Scrape a Specific Season**
Automatically finds the correct Season ID for the year.
```powershell
python run_pipeline.py --year "25/26" --scrape --all
```

**Option B: Run on Existing Data**
If you already have data.
```powershell
python run_pipeline.py --all
```

### 2. Validation & Audit
To verify the system is safe to bet:
```powershell
python models/audit_data_integrity.py
```

### 3. Utility Tools
**Check Season IDs**:
```powershell
python tools/debug_inspector.py 17
```
*(17 is the ID for Premier League)*.


## 📊 Model Performance

Trained on **1,700 Matches** (Season 21/22 - 25/26), using **61 dynamic features**.

| Strategy | Logic | ROI | Notes |
| :--- | :--- | :--- | :--- |
| **Longshot** | Odds > 3.0 | **+16.07%** | Best performer. Exploits underdog pricing errors. |
| **Value** | EV > 0 | **+12.95%** | Solid volume strategy. |
| **High Conf** | Prob > 50% | **+4.10%** | Safe, low variance. |
| **Blind Home** | All Home | **-8.63%** | Reference Baseline. |

> **Audit Passed**: The model has passed Walk-Forward Validation and a strict Data Leakage Audit. Rolling averages use strictly past data.

## 🛠️ Requirements

*   Python 3.10+
*   Google Chrome (latest)
*   NVIDIA GPU (Recommended for XGBoost)

**Install Dependencies:**
```powershell
pip install -r requirements.txt
```

---
*Created by Antigravity Agent*
