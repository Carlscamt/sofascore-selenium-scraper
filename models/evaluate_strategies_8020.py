
"""
Evaluate Strategies with 80/20 Split (Rolling Averages)
=======================================================
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import sys
import os

# Ensure clean output
if not os.path.exists('reports'):
    os.makedirs('reports')

def load_and_preprocess_data(filepath):
    """Loads and preprocesses the dataset."""
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        sys.exit(1)

    # Target 1X2: 0 (Home), 1 (Draw), 2 (Away)
    df['target_1x2'] = np.select(
        [df['score_home'] > df['score_away'], df['score_home'] == df['score_away']],
        [0, 1], default=2
    )
    
    # Target BTTS: 1 (Yes), 0 (No)
    df['target_btts'] = ((df['score_home'] > 0) & (df['score_away'] > 0)).astype(int)
    
    # Date sorting for chronological split
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

def process_form_string(form_str):
    if pd.isna(form_str):
        return [1] * 5 
    mapping = {'W': 2, 'D': 1, 'L': 0}
    parts = form_str.split(',')
    numeric_form = [mapping.get(x.strip(), 1) for x in parts]
    if len(numeric_form) > 5:
        numeric_form = numeric_form[:5]
    elif len(numeric_form) < 5:
        numeric_form += [1] * (5 - len(numeric_form))
    return numeric_form

def feature_engineering(df):
    """Generates rolling features + streaks."""
    
    # Process Form
    home_form_data = df['home_form'].apply(process_form_string).tolist()
    home_form_df = pd.DataFrame(home_form_data, columns=[f'h_form_{i+1}' for i in range(5)])
    
    away_form_data = df['away_form'].apply(process_form_string).tolist()
    away_form_df = pd.DataFrame(away_form_data, columns=[f'a_form_{i+1}' for i in range(5)])

    # Basic Features + Rolling Averages
    features = [
        'odd_1', 'odd_X', 'odd_2',
        'home_avg_rating', 'away_avg_rating',
        'home_position', 'away_position',
        'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
        'home_total_market_value', 'away_total_market_value',
        'home_avg_height', 'away_avg_height',
        'home_missing_count', 'away_missing_count', # New Lineup Features
    ]
    
    # Add BTTS Odds if present
    if 'odd_btts_yes' in df.columns: features.append('odd_btts_yes')
    if 'odd_btts_no' in df.columns: features.append('odd_btts_no')
    
    # Add Rolling columns
    rolling_cols = [c for c in df.columns if 'avg_' in c and c not in features]
    features.extend(rolling_cols)
    print(f"Added {len(rolling_cols)} rolling feature columns.")

    # Add all available streak columns dynamically 
    streak_cols = [c for c in df.columns if c.startswith('streak_')]
    features.extend(streak_cols)
    
    # Handle missing values: Drop rows with missing MAIN odds
    df_clean = df.dropna(subset=['odd_1', 'odd_X', 'odd_2']).copy()
    
    # Align form DFs
    home_form_df = home_form_df.loc[df_clean.index].reset_index(drop=True)
    away_form_df = away_form_df.loc[df_clean.index].reset_index(drop=True)
    
    # Fill missing features with -1
    df_clean[features] = df_clean[features].fillna(-1)
    df_clean = df_clean.reset_index(drop=True)
    
    # Encode League
    le = LabelEncoder()
    df_clean['league_encoded'] = le.fit_transform(df_clean['league'])
    features.append('league_encoded')

    # --- DROP USELESS FEATURES ---
    DROP_FEATURES = [
        'streak_h2h_away_first_half_winner_pct', 'streak_both_less_than_2_5_goals_count', 
        'streak_both_both_teams_scoring_count', 'streak_both_both_teams_scoring_len', 
        'streak_both_less_than_2_5_goals_len', 'streak_both_less_than_2_5_goals_pct', 
        'streak_h2h_away_no_goals_conceded_len', 'streak_h2h_away_no_goals_conceded_pct', 
        'streak_both_more_than_2_5_goals_count', 'streak_both_more_than_2_5_goals_len', 
        'streak_both_more_than_2_5_goals_pct', 'streak_home_more_than_4_5_cards_count', 
        'streak_both_both_teams_scoring_pct', 'streak_h2h_away_no_goals_conceded_count', 
        'streak_home_more_than_4_5_cards_pct', 'streak_home_more_than_4_5_cards_len', 
        'streak_both_without_clean_sheet_count', 'streak_both_without_clean_sheet_len', 
        'streak_both_more_than_4_5_cards_len', 'streak_both_more_than_4_5_cards_pct', 
        'streak_both_without_clean_sheet_pct', 'streak_both_more_than_4_5_cards_count', 
        'streak_both_no_losses_len', 'streak_both_no_losses_count', 
        'streak_both_no_losses_pct', 'streak_both_less_than_10_5_corners_count', 
        'streak_both_less_than_10_5_corners_pct', 'streak_both_less_than_10_5_corners_len', 
        'streak_away_losses_count', 'a_form_4', 'a_form_5', 'a_form_1',
        'streak_h2h_home_first_half_winner_count', 'streak_h2h_home_first_half_winner_len',
        'streak_h2h_away_wins_count', 'streak_h2h_away_wins_len',
        'streak_h2h_home_no_losses_len'
    ]
    
    # Filter features
    features = [f for f in features if f not in DROP_FEATURES]
    
    # --- ADD ADVANCED METRICS IF PRESENT (Processed outside) ---
    advanced_metrics = ['xg_diff', 'possession_diff', 'market_value_log_diff']
    for m in advanced_metrics:
        if m in df.columns: features.append(m)
        else: print(f"   [WARN] Metric {m} missing. Re-run process_data.")

    print(f"   [Feature Engineering] Dropped {len(DROP_FEATURES)} useless features.")

    X = pd.concat([df_clean[features], home_form_df, away_form_df], axis=1)
    
    # Also drop from X if they came from form_df
    X = X.drop(columns=[c for c in DROP_FEATURES if c in X.columns], errors='ignore')
    
    # --- CRITICAL CHANGE: DROP ODDS FROM TRAINING ---
    # We keep them in 'df_clean' for strategy PnL calculation, but hiding them from X
    odds_cols = ['odd_1', 'odd_X', 'odd_2', 'odd_btts_yes', 'odd_btts_no']
    X = X.drop(columns=[c for c in odds_cols if c in X.columns], errors='ignore')
    print("   [MODEL] Dropped ODDS columns from training features (Forcing stats learning).")
    y_1x2 = df_clean['target_1x2']
    y_btts = df_clean['target_btts']
    
    return X, y_1x2, y_btts, df_clean

def get_1x2_bets(model, X_test, y_test, df_test, strategy, return_details=False):
    """Calculates PnL for 1X2 market."""
    probs = model.predict_proba(X_test)
    X_reset = X_test.reset_index(drop=True)
    y_reset = y_test.reset_index(drop=True)
    df_reset = df_test.reset_index(drop=True)
    
    bets = []
    details = [] # Store EV, Odds, Result for Decile Analysis
    
    for i in range(len(X_reset)):
        odd_1 = df_reset.loc[i, 'odd_1']
        odd_X = df_reset.loc[i, 'odd_X']
        odd_2 = df_reset.loc[i, 'odd_2']
        
        # Skip invalid odds
        if pd.isna(odd_1) or pd.isna(odd_X) or pd.isna(odd_2) or odd_1 <= 1:
            bets.append(0); continue
            
        p_1, p_X, p_2 = probs[i]
        actual = y_reset.iloc[i]
        
        # Calculate EVs
        evs = [
            (p_1 * odd_1) - 1,
            (p_X * odd_X) - 1,
            (p_2 * odd_2) - 1
        ]
        
        # Find best option
        options = [
            (evs[0], 0, odd_1, p_1),
            (evs[1], 1, odd_X, p_X),
            (evs[2], 2, odd_2, p_2)
        ]
        options.sort(key=lambda x: x[0], reverse=True)
        top_ev, top_choice, top_odd, top_prob = options[0]
        
        should_bet = False
        
        # --- STRATEGY DEFINITIONS ---
        if strategy == '1x2_value':
            if top_ev > 0: should_bet = True
        elif strategy == '1x2_high_conf':
            if top_prob > 0.50 and top_ev > 0: should_bet = True
        elif strategy == '1x2_longshot':
            if top_odd > 3.0 and top_ev > 0: should_bet = True
        elif strategy == '1x2_favorites':
            if top_odd < 2.0 and top_ev > 0: should_bet = True
            
        if should_bet:
            profit = (top_odd - 1) if top_choice == actual else -1
            bets.append(profit)
            
            # Record details for Decile Analysis
            details.append({
                'ev': top_ev,
                'prob': top_prob,
                'odd': top_odd,
                'profit': profit,
                'is_win': 1 if profit > 0 else 0
            })
        else:
            bets.append(0)
            
    if return_details:
        return bets, details
    return bets

def analyze_deciles(bets_data):
    """Sorts bets by Edge (EV) and plots ROI per decile."""
    if not bets_data: return
    
    df = pd.DataFrame(bets_data)
    # Sort by EV descending
    df = df.sort_values(by='ev', ascending=False)
    
    # Create Deciles (10 bins)
    try:
        df['decile'] = pd.qcut(df['ev'], 10, labels=False, duplicates='drop')
    except Exception as e:
        print(f"Not enough data for deciles: {e}")
        return

    stats = df.groupby('decile').agg({
        'profit': 'sum',
        'ev': 'count', # Count bets
        'is_win': 'mean' # Win Rate
    }).rename(columns={'ev': 'n_bets', 'is_win': 'win_rate'})
    
    stats['roi'] = (stats['profit'] / stats['n_bets']) * 100
    
    print("\n" + "="*65)
    print("ROI BY EXPECTED VAUE (EDGE) DECILES")
    print("="*65)
    # Print upside down (Highest Decile first)
    print(stats.sort_index(ascending=False).to_string())
    print("="*65)
    
    # Plot
    plt.figure(figsize=(10, 6))
    colors = ['green' if x > 0 else 'red' for x in stats['roi']]
    stats['roi'].plot(kind='bar', color=colors)
    plt.title('ROI by Edge Decile (Verification of Value)')
    plt.xlabel('Decile (0=Lowest EV, 9=Highest EV)')
    plt.ylabel('ROI %')
    plt.axhline(0, color='black', linewidth=1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('reports/roi_by_decile.png')
    print("[GRAPH] Saved to reports/roi_by_decile.png")

def run_benchmarks(df_test, y_test):
    """Compares Model vs Naive Strategies."""
    y_reset = y_test.reset_index(drop=True)
    df_reset = df_test.reset_index(drop=True)
    
    results = {'Strategy': [], 'ROI': []}
    
    # 1. Home Blindly
    bets_home = []
    for i in range(len(df_reset)):
        odd = df_reset.loc[i, 'odd_1']
        if pd.isna(odd): bets_home.append(0); continue
        # Always bet Home (0)
        profit = (odd - 1) if y_reset.iloc[i] == 0 else -1
        bets_home.append(profit)
    
    results['Strategy'].append('Blind Home')
    results['ROI'].append(sum(bets_home) / len(bets_home) * 100)
    
    # 2. Away Blindly
    bets_away = []
    for i in range(len(df_reset)):
        odd = df_reset.loc[i, 'odd_2']
        if pd.isna(odd): bets_away.append(0); continue
        profit = (odd - 1) if y_reset.iloc[i] == 2 else -1
        bets_away.append(profit)

    results['Strategy'].append('Blind Away')
    results['ROI'].append(sum(bets_away) / len(bets_away) * 100)
    
    # 3. Random
    import random
    random.seed(42)
    bets_rnd = []
    for i in range(len(df_reset)):
        choice = random.choice([0, 1, 2])
        col = f"odd_{['1','X','2'][choice]}"
        odd = df_reset.loc[i, col]
        if pd.isna(odd): bets_rnd.append(0); continue
        profit = (odd - 1) if y_reset.iloc[i] == choice else -1
        bets_rnd.append(profit)
        
    results['Strategy'].append('Random')
    results['ROI'].append(sum(bets_rnd) / len(bets_rnd) * 100)
    
    print("\n" + "="*65)
    print("BENCHMARK COMPARISON")
    print("="*65)
    print(pd.DataFrame(results).to_string(index=False))
    print("="*65)

def evaluate_strategies(input_file="dataset_rolling_features.csv"):
    print(f">>> LOADING DATA: {input_file}")
    if not os.path.exists(input_file):
        print("Dataset not found! Please run process_data.py first.")
        return None

    df = load_and_preprocess_data(input_file)
    print(f"Total Matches: {len(df)}")
    
    # Feature Engineering
    X, y_1x2, y_btts, df_clean = feature_engineering(df)
    
    # --- 80% / 20% SPLIT ---
    split_idx = int(len(df_clean) * 0.8)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_1x2_train, y_1x2_test = y_1x2.iloc[:split_idx], y_1x2.iloc[split_idx:]
    
    df_test = df_clean.iloc[split_idx:]
    
    print(f"\n[SPLIT] Train: {len(X_train)} matches | Test: {len(X_test)} matches")
    print(f"Train Dates: {df_clean.iloc[:split_idx]['date'].min().date()} -> {df_clean.iloc[:split_idx]['date'].max().date()}")
    print(f"Test Dates:  {df_test['date'].min().date()} -> {df_test['date'].max().date()}")
    
    # --- TRAIN 1X2 MODEL ---
    print("\n>>> TRAINING XGBOOST MODEL (1X2) [GPU ENABLED]...")
    
    # Reverting SMOTE (User preferred the imbalanced/aggressive profit model)
    model_1x2 = xgb.XGBClassifier(
        objective='multi:softprob', 
        n_estimators=100, 
        max_depth=3, 
        learning_rate=0.1, 
        random_state=42,
        verbosity=0,
        reg_alpha=0.5,    # LASSO (L1)
        reg_lambda=1.5,   # Ridge (L2)
        device="cuda"  # NVIDIA GPU Support
    )
    model_1x2.fit(X_train, y_1x2_train)
    
    preds = model_1x2.predict(X_test)
    acc = accuracy_score(y_1x2_test, preds)
    print(f"Model Accuracy on Test Set: {acc:.4f}")
    
    # --- EVALUATE STRATEGIES ---
    print("\n>>> EVALUATING STRATEGIES...")
    
    strategies = ['1x2_value', '1x2_high_conf', '1x2_longshot', '1x2_favorites']
    results = []
    
    # Store detailed bet data for Decile Analysis
    all_value_bets = [] 
    
    plt.figure(figsize=(12, 6))
    
    for strat in strategies:
        # Pass raw X_test (pandas) as model was trained on pandas (SMOTE removed)
        bets, details = get_1x2_bets(model_1x2, X_test, y_1x2_test, df_test, strat, return_details=True)
        
        if strat == '1x2_value':
            all_value_bets = details
            
        n_bets = sum(1 for b in bets if b != 0)
        n_wins = sum(1 for b in bets if b > 0)
        profit = sum(bets)
        
        roi = (profit / n_bets * 100) if n_bets > 0 else 0
        win_rate = (n_wins / n_bets * 100) if n_bets > 0 else 0
        
        results.append({
            'Strategy': strat,
            'Bets': n_bets,
            'Wins': n_wins,
            'Win Rate': f"{win_rate:.1f}%",
            'Profit': f"{profit:.2f}u",
            'ROI': f"{roi:.2f}%"
        })
        
        cumsum = np.cumsum(bets)
        plt.plot(cumsum, label=f"{strat} (ROI: {roi:.1f}%)", linewidth=2)
        
    # --- REPORTING ---
    res_df = pd.DataFrame(results)
    print("\n" + "="*65)
    print("STRATEGY PERFORMANCE REPORT (80/20 SPLIT)")
    print("="*65)
    print(res_df.to_string(index=False))
    print("="*65)
    
    # Save Graph
    plt.title("Betting Strategies Performance (Cumulative Profit)", fontsize=14)
    plt.xlabel("Match Timeline (Index)", fontsize=12)
    plt.ylabel("Cumulative Profit (Units)", fontsize=12)
    plt.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    plt.legend(frameon=True, fancybox=True, framealpha=0.9, loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig('reports/strategy_80_20_performance.png', dpi=300, bbox_inches='tight')
    print("\n[GRAPH] Saved to reports/strategy_80_20_performance.png")

    # --- NEW: DECILE ANALYSIS ---
    analyze_deciles(all_value_bets)
    
    # --- NEW: BENCHMARK COMPARISON ---
    run_benchmarks(df_test, y_1x2_test)
    
    return results

if __name__ == "__main__":
    evaluate_strategies()
