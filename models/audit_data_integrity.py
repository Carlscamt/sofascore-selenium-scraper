
"""
Data Integrity & Leakage Audit Script
=====================================
Performs verified checks on Rolling Averages and Odds Distribution 
to ensure the model is "Safe for Live Betting".
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

INPUT_FILE = "dataset_rolling_features_enriched.csv"

def load_data():
    if not os.path.exists(INPUT_FILE):
        print(f"File {INPUT_FILE} not found.")
        sys.exit(1)
    df = pd.read_csv(INPUT_FILE)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Standardize 1X2 from score if not present
    if 'result' not in df.columns:
        df['result'] = np.select(
            [df['score_home'] > df['score_away'], df['score_home'] == df['score_away']],
            ['Home Win', 'Draw'], default='Away Win'
        )
    return df

def audit_rolling_leakage(df, sample_size=100):
    print("\n[AUDIT 1] Checking Rolling Averages for Time Travel...")
    
    # We will manually re-calculate 'home_avg_ballPossession' for a sample
    # Logic: For match M at date D, looking at matches < D
    
    target_metric = 'ballPossession'
    col_name = f'home_avg_{target_metric}'
    raw_col_name = f'stats_home_{target_metric}'
    
    if col_name not in df.columns or raw_col_name not in df.columns:
        print(f"   Skipping Rolling Audit: Columns for {target_metric} not found.")
        return

    # Filter for teams with enough history
    valid_teams = df['home'].value_counts()
    valid_teams = valid_teams[valid_teams > 5].index.tolist()
    
    sample = df[df['home'].isin(valid_teams)].sample(min(len(df), sample_size), random_state=42)
    
    errors = []
    
    for idx, row in sample.iterrows():
        team = row['home'] # Note: The rolling logic rolls by 'team' column which is standardized 'home' or 'away'
        # But in the processed DF, we merged it back. 
        # To strictly verify, we need the raw team history.
        # It's easier to verify "Does the current value match the PREVIOUS 5?"
        
        match_date = row['date']
        
        # Reconstruct history from the full DF
        # Find all matches where this team played (Home or Away) BEFORE this date
        history = df[
            ((df['home'] == team) | (df['away'] == team)) & 
            (df['date'] < match_date)
        ].sort_values('date').tail(5)
        
        if len(history) == 0: continue
        
        # Calculate mean of raw stat
        values = []
        for _, h_row in history.iterrows():
            if h_row['home'] == team:
                val = h_row.get(f'stats_home_{target_metric}')
            else:
                val = h_row.get(f'stats_away_{target_metric}')
            
            # Clean string percentages if needed
            if isinstance(val, str): val = float(val.replace('%', ''))
            values.append(val)
            
        if not values: continue
        
        recalc_mean = np.mean(values)
        stored_mean = row[col_name]
        
        if pd.isna(stored_mean): continue
        
        # Tolerance check
        if abs(recalc_mean - stored_mean) > 0.1:
            errors.append({
                'id': row['id'],
                'date': match_date,
                'team': team,
                'calc': recalc_mean,
                'stored': stored_mean,
                'diff': abs(recalc_mean - stored_mean)
            })
            
    if len(errors) > 0:
        print(f"   🚨 LEAKAGE DETECTED (or calc mismatch)! {len(errors)} errors found.")
        print(pd.DataFrame(errors).head())
        print("   Analysis: stored_mean SHOULD equal recalc_mean. If stored is notably different, check logic.")
    else:
        print("   [PASS]: Rolling averages strictly match previous games data.")

def audit_odds_timing(df):
    print("\n[AUDIT 2] Checking Odds vs Reality (Calibration)...")
    
    # Check "Too Good To Be True" accuracy on specific odds brackets
    audit_data = []
    
    for outcome, odd_col in [('Home Win', 'odd_1'), ('Draw', 'odd_X'), ('Away Win', 'odd_2')]:
        for bracket in [1.5, 2.0, 3.0, 5.0]:
            # Range +/- 0.2
            subset = df[(df[odd_col] >= bracket - 0.2) & (df[odd_col] < bracket + 0.2)]
            
            if len(subset) < 20: continue
            
            win_count = (subset['result'] == outcome).sum()
            actual_prob = win_count / len(subset)
            implied_prob = 1 / bracket
            
            diff = actual_prob - implied_prob
            
            # Identify suspicion
            status = "OK"
            if abs(diff) > 0.20: status = "🚨 SUSPICIOUS"
            elif abs(diff) > 0.10: status = "⚠️ WARN"
            
            audit_data.append({
                'Outcome': outcome,
                'Odds': bracket,
                'N': len(subset),
                'Implied': f"{implied_prob:.1%}",
                'Actual': f"{actual_prob:.1%}",
                'Diff': f"{diff:+.1%}",
                'Status': status
            })
            
    res = pd.DataFrame(audit_data)
    print(res.to_string(index=False))
    
    suspicious = res[res['Status'].str.contains("SUSPICIOUS")]
    if len(suspicious) > 0:
        print("\n   [WARN] Found buckets with >20% deviation. Could imply Post-Match Odds or just bad calibration.")
    else:
        print("\n   [PASS]: Odds look organic (no 100% win rates on high odds).")

def audit_lineup_timing(df):
    print("\n[AUDIT 3] Checking Lineup Data Signals...")
    # Analyzing missing_count impact across odds
    
    # Do favorites lose more when missing count is high?
    favorites = df[df['odd_1'] < 1.8] # Home favorites
    
    if 'home_missing_count' not in favorites.columns:
        print("   Skipping: missing_count column not found.")
        return

    print("   Analyzing Home Favorites (< 1.80):")
    
    # Group by missing count
    # Create numeric target if missing
    mapping = {'Home Win': 0, 'Draw': 1, 'Away Win': 2}
    favorites['target_1x2'] = favorites['result'].map(mapping)

    summary = favorites.groupby('home_missing_count').agg(
        n_matches=('id', 'count'),
        win_rate=('target_1x2', lambda x: (x==0).mean())
    )
    summary['implied_win_rate'] = 1 / favorites['odd_1'].mean()
    
    print(summary)
    print("   Interpretation: If Win Rate drops significantly as missing_count increases, the signal is VALID.")
    print("   If missing_count=5 has 0% win rate, it might be post-match retroactive tagging.")

if __name__ == "__main__":
    df = load_data()
    print(f"Auditing Dataset: {len(df)} matches")
    
    audit_rolling_leakage(df)
    audit_odds_timing(df)
    audit_lineup_timing(df)
