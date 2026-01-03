
"""
Walk-Forward Validation Script
==============================
Simulates "living" with the model by retraining every season.
Time-aware split: Train on past seasons, Test on the next season.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import sys
import os

# --- CONFIG ---
INPUT_FILE = "dataset_rolling_features_enriched.csv"
ODDS_COLS = ['odd_1', 'odd_X', 'odd_2', 'odd_btts_yes', 'odd_btts_no']

# Define Season Cutoffs (Approximate Aug 1st start)
SEASONS = {
    '22/23': ('2022-08-01', '2023-06-01'),
    '23/24': ('2023-08-01', '2024-06-01'),
    '24/25': ('2024-08-01', '2025-06-01'),
    '25/26': ('2025-08-01', '2026-06-01') # Current partial season
}

def load_data():
    if not os.path.exists(INPUT_FILE):
        print(f"File {INPUT_FILE} not found.")
        sys.exit(1)
    df = pd.read_csv(INPUT_FILE)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Target 1X2
    df['target_1x2'] = np.select(
        [df['score_home'] > df['score_away'], df['score_home'] == df['score_away']],
        [0, 1], default=2
    )
    return df

def feature_engineering(df):
    """
    Same robust feature engineering as evaluating_strategies_8020.py
    """
    # --- FEATURE SELECTION (WHITELIST APPROACH) ---
    # We must be extremely careful not to include raw match stats (e.g. 'stats_home_goals')
    
    # 1. Start with known safe features
    features = [
        'home_avg_rating', 'away_avg_rating',
        'home_position', 'away_position',
        'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
        'home_total_market_value', 'away_total_market_value',
        'home_avg_height', 'away_avg_height',
        'home_missing_count', 'away_missing_count',
        'xg_diff', 'possession_diff', 'market_value_log_diff'
    ]
    
    # 2. Add Rolling Averages (avg_*)
    rolling_cols = [c for c in df.columns if 'avg_' in c and c not in features]
    features.extend(rolling_cols)
    
    # 3. Add Streaks (streak_*)
    streak_cols = [c for c in df.columns if c.startswith('streak_')]
    features.extend(streak_cols)
    
    # 4. Filter strictly to what exists
    features = [f for f in features if f in df.columns]
    
    # 5. One-Hot or Label Encode League
    le = LabelEncoder()
    df['league_encoded'] = le.fit_transform(df['league'])
    features.append('league_encoded')
    
    # Handle NaNs and create X
    X = df[features].fillna(-1)
    y = df['target_1x2']
    
    return X, y, df

def get_bets(model, X_test, df_test, y_test):
    """Generates bets and calculates Edge."""
    probs = model.predict_proba(X_test)
    X_reset = X_test.reset_index(drop=True)
    df_reset = df_test.reset_index(drop=True)
    y_reset = y_test.reset_index(drop=True)
    
    bets_data = []
    
    for i in range(len(X_reset)):
        odd_1 = df_reset.loc[i, 'odd_1']
        odd_X = df_reset.loc[i, 'odd_X']
        odd_2 = df_reset.loc[i, 'odd_2']
        
        if pd.isna(odd_1) or odd_1 <= 1: continue
        
        p_1, p_X, p_2 = probs[i]
        actual = y_reset.iloc[i]
        
        # Calculate EVs
        evs = [
            (p_1 * odd_1) - 1,
            (p_X * odd_X) - 1,
            (p_2 * odd_2) - 1
        ]
        
        # Select Best EV
        options = [
            (evs[0], 0, odd_1, p_1),
            (evs[1], 1, odd_X, p_X),
            (evs[2], 2, odd_2, p_2)
        ]
        options.sort(key=lambda x: x[0], reverse=True)
        top_ev, top_choice, top_odd, top_prob = options[0]
        
        if top_ev > 0:
            profit = (top_odd - 1) if top_choice == actual else -1
            bets_data.append({
                'date': df_reset.loc[i, 'date'],
                'ev': top_ev,
                'prob': top_prob,
                'odd': top_odd,
                'choice': top_choice,
                'result': 'Win' if top_choice == actual else 'Loss',
                'profit': profit
            })
            
    return pd.DataFrame(bets_data)

def run_walk_forward():
    print(">>> STARTING WALK-FORWARD VALIDATION")
    df_full = load_data()
    X, y, df_all = feature_engineering(df_full)
    
    results = []
    
    for season_name, (start_date, end_date) in SEASONS.items():
        print(f"\n--- Processing Season: {season_name} ---")
        
        # TRAIN: All data BEFORE this season
        train_mask = df_all['date'] < start_date
        # TEST: Data WITHIN this season
        test_mask = (df_all['date'] >= start_date) & (df_all['date'] <= end_date)
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        df_test = df_all[test_mask]
        
        if len(X_train) < 100 or len(X_test) < 10:
            print(f"   Skipping {season_name}: Not enough data (Train: {len(X_train)}, Test: {len(X_test)})")
            continue
            
        print(f"   Train Size: {len(X_train)} | Test Size: {len(X_test)}")
        
        # Train
        model = xgb.XGBClassifier(
            objective='multi:softprob', n_estimators=100, max_depth=3, learning_rate=0.1, 
            random_state=42, verbosity=0, reg_alpha=0.5, reg_lambda=1.5, device="cuda"
        )
        model.fit(X_train, y_train)
        
        # Predict & Bet
        bets = get_bets(model, X_test, df_test, y_test)
        
        if len(bets) == 0:
            print("   No bets found.")
            continue
            
        # Analysis
        roi = (bets['profit'].sum() / len(bets)) * 100
        win_rate = (len(bets[bets['profit'] > 0]) / len(bets)) * 100
        profit = bets['profit'].sum()
        
        # Decile 9 Check
        try:
            bets['decile'] = pd.qcut(bets['ev'], 10, labels=False, duplicates='drop')
            d9_bets = bets[bets['decile'] == 9]
            d9_roi = (d9_bets['profit'].sum() / len(d9_bets) * 100) if len(d9_bets) > 0 else 0
            
            # Decile Filtered ROI (Exclude D9)
            filtered_bets = bets[bets['decile'] != 9]
            filtered_roi = (filtered_bets['profit'].sum() / len(filtered_bets) * 100) if len(filtered_bets) > 0 else 0
        except:
             d9_roi = 0
             filtered_roi = roi
        
        print(f"   [RESULT] ROI: {roi:.2f}% | Profit: {profit:.2f}u | Bets: {len(bets)}")
        print(f"   [TRAP CHECK] Decile 9 ROI: {d9_roi:.2f}%")
        print(f"   [OPTIMIZED] ROI (No D9): {filtered_roi:.2f}%")
        
        results.append({
            'Season': season_name,
            'Train_Size': len(X_train),
            'Test_Bets': len(bets),
            'ROI_Raw': roi,
            'ROI_NoTrap': filtered_roi,
            'ROI_Trap': d9_roi,
            'Win_Rate': win_rate
        })
        
    print("\n" + "="*30)
    print("WALK-FORWARD SUMMARY")
    print("="*30)
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        print(res_df.to_string(index=False))
        print("-" * 30)
        print(f"Mean ROI (Raw): {res_df['ROI_Raw'].mean():.2f}%")
        print(f"Mean ROI (Optimized): {res_df['ROI_NoTrap'].mean():.2f}%")
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(res_df['Season'], res_df['ROI_Raw'], marker='o', label='Raw ROI', linewidth=2)
        plt.plot(res_df['Season'], res_df['ROI_NoTrap'], marker='s', label='Optimized ROI (No Trap)', linestyle='--', linewidth=2)
        plt.axhline(0, color='black')
        plt.title("Walk-Forward Validation: Consistency Check")
        plt.ylabel("ROI %")
        plt.legend()
        plt.grid(True)
        plt.savefig('reports/walk_forward_validation.png')
        print("[GRAPH] Saved to reports/walk_forward_validation.png")
    else:
        print("No results generated.")

if __name__ == "__main__":
    run_walk_forward()
