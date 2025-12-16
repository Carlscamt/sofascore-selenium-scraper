import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(filepath):
    """Loads and preprocesses the dataset."""
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

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

def feature_engineering(df):
    """Generates features focusing on streaks and basic stats."""
    # Basic Features
    features = [
        'odd_1', 'odd_X', 'odd_2',
        'home_avg_rating', 'away_avg_rating',
        'home_position', 'away_position',
        'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
        'home_total_market_value', 'away_total_market_value',
        'home_avg_height', 'away_avg_height',
        'odd_btts_yes', 'odd_btts_no'
    ]
    
    # Add all available streak columns dynamically 
    streak_cols = [c for c in df.columns if c.startswith('streak_')]
    features.extend(streak_cols)
    
    # Handle missing values
    df_clean = df.copy()
    df_clean[streak_cols] = df_clean[streak_cols].fillna(0)
    df_clean[features] = df_clean[features].fillna(-1)
    
    X = df_clean[features]
    y_1x2 = df_clean['target_1x2']
    y_btts = df_clean['target_btts']
    
    return X, y_1x2, y_btts, df_clean

def get_1x2_bets(model, X_test, y_test, df_test, strategy):
    """Calculates PnL for 1X2 market."""
    probs = model.predict_proba(X_test)
    X_reset = X_test.reset_index(drop=True)
    y_reset = y_test.reset_index(drop=True)
    df_reset = df_test.reset_index(drop=True)
    
    bets = []
    
    for i in range(len(X_reset)):
        odd_1 = df_reset.loc[i, 'odd_1']
        odd_X = df_reset.loc[i, 'odd_X']
        odd_2 = df_reset.loc[i, 'odd_2']
        
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
        
        options = [
            (evs[0], 0, odd_1, p_1),
            (evs[1], 1, odd_X, p_X),
            (evs[2], 2, odd_2, p_2)
        ]
        options.sort(key=lambda x: x[0], reverse=True)
        top_ev, top_choice, top_odd, top_prob = options[0]
        
        should_bet = False
        if strategy == '1x2_basic_ev':
            if top_ev > 0: should_bet = True
        elif strategy == '1x2_high_conf':
            if top_prob > 0.55 and top_ev > 0: should_bet = True
        elif strategy == '1x2_longshot':
            if top_odd > 3.0 and top_ev > 0: should_bet = True
            
        if should_bet:
            profit = (top_odd - 1) if top_choice == actual else -1
            bets.append(profit)
        else:
            bets.append(0)
            
    return bets

def get_btts_bets(model, X_test, y_test, df_test, strategy):
    """Calculates PnL for BTTS market."""
    probs = model.predict_proba(X_test)[:, 1] # Probability of "Yes" (1)
    X_reset = X_test.reset_index(drop=True)
    y_reset = y_test.reset_index(drop=True)
    df_reset = df_test.reset_index(drop=True)
    
    bets = []
    
    for i in range(len(X_reset)):
        odd_yes = df_reset.loc[i, 'odd_btts_yes']
        odd_no = df_reset.loc[i, 'odd_btts_no']
        
        if pd.isna(odd_yes) or pd.isna(odd_no) or odd_yes <= 1:
            bets.append(0); continue
            
        prob_yes = probs[i]
        prob_no = 1 - prob_yes
        actual = y_reset.iloc[i] # 1 (Yes) or 0 (No)
        
        ev_yes = (prob_yes * odd_yes) - 1
        ev_no  = (prob_no * odd_no) - 1
        
        # Determine best side
        if ev_yes > ev_no:
            top_ev, top_choice, top_odd, top_prob = ev_yes, 1, odd_yes, prob_yes
        else:
            top_ev, top_choice, top_odd, top_prob = ev_no, 0, odd_no, prob_no
            
        should_bet = False
        if strategy == 'btts_basic_ev':
            if top_ev > 0: should_bet = True
        elif strategy == 'btts_high_conf':
            if top_prob > 0.60 and top_ev > 0: should_bet = True
            
        if should_bet:
            profit = (top_odd - 1) if top_choice == actual else -1
            bets.append(profit)
        else:
            bets.append(0)
            
    return bets

def run_strategies():
    print(">>> LOADING DATA...")
    df = load_and_preprocess_data('sofascore_dataset_v2.csv')
    if df is None: return

    print(f"Total Matches: {len(df)}")
    
    # Feature Engineering
    X, y_1x2, y_btts, df_clean = feature_engineering(df)
    
    # Strict Chronological Split (80/20)
    split_idx = int(len(df) * 0.8)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_1x2_train, y_1x2_test = y_1x2.iloc[:split_idx], y_1x2.iloc[split_idx:]
    y_btts_train, y_btts_test = y_btts.iloc[:split_idx], y_btts.iloc[split_idx:]
    
    df_test = df_clean.iloc[split_idx:]
    
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Test Dates: {df_test['date'].min().date()} - {df_test['date'].max().date()}")
    
    # --- TRAIN 1X2 MODEL ---
    print("\n>>> TRAINING 1X2 MODEL...")
    model_1x2 = xgb.XGBClassifier(
        objective='multi:softprob', n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
    model_1x2.fit(X_train, y_1x2_train)
    acc_1x2 = accuracy_score(y_1x2_test, model_1x2.predict(X_test))
    print(f"1X2 Accuracy: {acc_1x2:.4f}")
    
    # --- TRAIN BTTS MODEL ---
    print("\n>>> TRAINING BTTS MODEL...")
    model_btts = xgb.XGBClassifier(
        objective='binary:logistic', n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
    model_btts.fit(X_train, y_btts_train)
    acc_btts = accuracy_score(y_btts_test, model_btts.predict(X_test))
    print(f"BTTS Accuracy: {acc_btts:.4f}")
    
    # --- RUN ALL STRATEGIES ---
    strategies = {
        '1x2_high_conf': (get_1x2_bets, model_1x2, y_1x2_test),
        '1x2_basic_ev':  (get_1x2_bets, model_1x2, y_1x2_test),
        'btts_high_conf': (get_btts_bets, model_btts, y_btts_test),
        'btts_basic_ev':  (get_btts_bets, model_btts, y_btts_test)
    }
    
    print("\n>>> EVALUATING PORTFOLIO...")
    plt.figure(figsize=(10, 6))
    
    for name, (func, model, y_true) in strategies.items():
        bets = func(model, X_test, y_true, df_test, name)
        
        n_bets = sum(1 for b in bets if b != 0)
        n_wins = sum(1 for b in bets if b > 0)
        win_rate = (n_wins / n_bets * 100) if n_bets > 0 else 0
        profit = sum(bets)
        roi = (profit / n_bets * 100) if n_bets > 0 else 0
        
        print(f"{name.ljust(15)} | Bets: {n_bets:<4} | Wins: {n_wins:<3} ({win_rate:>.1f}%) | Profit: {profit:>6.2f}u | ROI: {roi:>6.2f}%")
        
        # Plot
        cumsum = np.cumsum(bets)
        plt.plot(cumsum, label=f"{name} ({roi:.1f}%)", linewidth=2)
        
    plt.title("Multi-Strategy Performance Comparison")
    plt.xlabel("Matches (Chronological)")
    plt.ylabel("Profit (Units)")
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/strategies_comparison.png')
    print("\n[GRAPH] Saved to 'reports/strategies_comparison.png'")

if __name__ == "__main__":
    run_strategies()
