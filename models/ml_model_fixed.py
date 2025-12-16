"""
Fixed ML Model with Temporal Cross-Validation
==============================================
This version avoids look-ahead bias by:
1. Sorting data chronologically before splitting
2. Using TimeSeriesSplit (train on past, predict future)
3. Including comparison mode to show ROI inflation from bias
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(filepath):
    """Loads data and creates target variable."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None

    # Create Target Variable: 0=Home Win, 1=Draw, 2=Away Win
    df['target'] = np.select(
        [df['score_home'] > df['score_away'], df['score_home'] == df['score_away']],
        [0, 1],
        default=2
    )
    
    # CRITICAL: Sort by date for temporal integrity
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    return df

def process_form_string(form_str):
    """Parses a form string like 'W,D,L,W,W' into a list of integers."""
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
    """Transforms raw dataframe into X (features) and y (target)."""
    # 1. Process Form
    home_form_data = df['home_form'].apply(process_form_string).tolist()
    home_form_df = pd.DataFrame(home_form_data, columns=[f'h_form_{i+1}' for i in range(5)])
    
    away_form_data = df['away_form'].apply(process_form_string).tolist()
    away_form_df = pd.DataFrame(away_form_data, columns=[f'a_form_{i+1}' for i in range(5)])
    
    # 2. Key Features Selection
    df_clean = df.dropna(subset=['odd_1', 'odd_X', 'odd_2']).copy()
    
    home_form_df = home_form_df.loc[df_clean.index].reset_index(drop=True)
    away_form_df = away_form_df.loc[df_clean.index].reset_index(drop=True)
    
    # 3. Label Encoding for League
    le = LabelEncoder()
    df_clean['league_encoded'] = le.fit_transform(df_clean['league'])
    
    features = [
        'odd_1', 'odd_X', 'odd_2',
        'home_avg_rating', 'away_avg_rating',
        'home_position', 'away_position',
        'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
        'league_encoded'
    ]
    
    df_clean[features] = df_clean[features].fillna(-1) 
    df_clean = df_clean.reset_index(drop=True)
    
    X = pd.concat([
        df_clean[features],
        home_form_df,
        away_form_df
    ], axis=1)
    
    y = df_clean['target']
    
    # Return df_clean too for date-based analysis
    return X, y, df_clean

def get_fold_bets(model, X_test, y_test):
    """Helper to calculate bets for a single fold."""
    probs = model.predict_proba(X_test)
    X_test_reset = X_test.reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)
    
    bets = []

    for i in range(len(X_test_reset)):
        odd_1 = X_test_reset.loc[i, 'odd_1']
        odd_X = X_test_reset.loc[i, 'odd_X']
        odd_2 = X_test_reset.loc[i, 'odd_2']
        
        p_1, p_X, p_2 = probs[i]
        actual_result = y_test_reset.iloc[i]
        
        # EV Calculation
        ev_1 = (p_1 * odd_1) - 1
        ev_X = (p_X * odd_X) - 1
        ev_2 = (p_2 * odd_2) - 1
        
        best_ev = -float('inf')
        bet_choice = None
        bet_odd = 0
        
        if ev_1 > 0 and ev_1 > best_ev:
            best_ev = ev_1
            bet_choice = 0
            bet_odd = odd_1
        if ev_X > 0 and ev_X > best_ev:
            best_ev = ev_X
            bet_choice = 1
            bet_odd = odd_X
        if ev_2 > 0 and ev_2 > best_ev:
            best_ev = ev_2
            bet_choice = 2
            bet_odd = odd_2
            
        if bet_choice is not None:
            choice_str = ["Home", "Draw", "Away"][bet_choice]
            profit = (bet_odd - 1) if bet_choice == actual_result else -1
            result_str = "WIN" if bet_choice == actual_result else "LOSS"
                
            bets.append({
                'Choice': choice_str,
                'Odd': bet_odd,
                'Prob': round(probs[i][bet_choice], 2),
                'Result': result_str,
                'PnL': profit
            })
    return bets

def run_biased_cv(X, y):
    """
    BIASED: Original approach with random shuffling.
    This shows inflated ROI due to look-ahead bias.
    """
    print("\n" + "="*50)
    print(">>> BIASED MODEL (Random Shuffle - WRONG)")
    print("="*50)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_bets = []
    accuracies = []
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, preds))
        
        fold_bets = get_fold_bets(model, X_test, y_test)
        all_bets.extend(fold_bets)
    
    total_profit = sum(b['PnL'] for b in all_bets)
    total_bets = len(all_bets)
    roi = (total_profit / total_bets) * 100 if total_bets > 0 else 0
    
    print(f"Accuracy: {np.mean(accuracies):.4f}")
    print(f"Total Bets: {total_bets}")
    print(f"Total Profit: {total_profit:.2f} units")
    print(f"ROI: {roi:.2f}%")
    
    return all_bets, roi

def run_temporal_cv(X, y):
    """
    CORRECT: Temporal split - train on past, predict future.
    This shows realistic ROI without look-ahead bias.
    """
    print("\n" + "="*50)
    print(">>> TEMPORAL MODEL (Time Series Split - CORRECT)")
    print("="*50)
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    all_bets = []
    accuracies = []
    
    fold = 1
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies.append(acc)
        
        fold_bets = get_fold_bets(model, X_test, y_test)
        all_bets.extend(fold_bets)
        
        print(f"Fold {fold}/5 - Train: {len(train_index)}, Test: {len(test_index)}, Acc: {acc:.4f}, Bets: {len(fold_bets)}")
        fold += 1
    
    total_profit = sum(b['PnL'] for b in all_bets)
    total_bets = len(all_bets)
    roi = (total_profit / total_bets) * 100 if total_bets > 0 else 0
    
    print(f"\nFinal Accuracy: {np.mean(accuracies):.4f}")
    print(f"Total Bets: {total_bets}")
    print(f"Total Profit: {total_profit:.2f} units")
    print(f"ROI: {roi:.2f}%")
    
    return all_bets, roi

def run_comparison(X, y):
    """Run both methods and compare results."""
    print("\n" + "#"*60)
    print("###  LOOK-AHEAD BIAS COMPARISON TEST")
    print("#"*60)
    
    biased_bets, biased_roi = run_biased_cv(X, y)
    temporal_bets, temporal_roi = run_temporal_cv(X, y)
    
    # Summary
    print("\n" + "="*60)
    print(">>> COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Method':<30} {'Bets':<10} {'ROI':<10}")
    print("-"*50)
    print(f"{'Biased (Random Shuffle)':<30} {len(biased_bets):<10} {biased_roi:.2f}%")
    print(f"{'Temporal (Correct)':<30} {len(temporal_bets):<10} {temporal_roi:.2f}%")
    print("-"*50)
    
    inflation = biased_roi - temporal_roi
    print(f"\n>>> ROI INFLATION FROM BIAS: {inflation:.2f}%")
    
    if biased_roi > 0 and temporal_roi <= 0:
        print(">>> WARNING: Your 'profitable' model is actually LOSING money!")
    elif inflation > 5:
        print(">>> WARNING: Significant look-ahead bias detected!")
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    if biased_bets:
        biased_cumsum = np.cumsum([b['PnL'] for b in biased_bets])
        plt.plot(biased_cumsum, label=f'Biased (ROI: {biased_roi:.1f}%)', color='red')
    if temporal_bets:
        temporal_cumsum = np.cumsum([b['PnL'] for b in temporal_bets])
        plt.plot(temporal_cumsum, label=f'Temporal (ROI: {temporal_roi:.1f}%)', color='green')
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Cumulative Profit: Biased vs Temporal')
    plt.xlabel('Bet Number')
    plt.ylabel('Profit (Units)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    methods = ['Biased\n(Wrong)', 'Temporal\n(Correct)']
    rois = [biased_roi, temporal_roi]
    colors = ['red' if r > 0 else 'darkred' for r in [biased_roi]] + \
             ['green' if r > 0 else 'darkgreen' for r in [temporal_roi]]
    colors = ['#ff6b6b', '#51cf66']
    plt.bar(methods, rois, color=colors)
    plt.axhline(0, color='black', linestyle='-')
    plt.title('ROI Comparison')
    plt.ylabel('ROI (%)')
    for i, v in enumerate(rois):
        plt.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('bias_comparison.png')
    print(f"\n[GRAPH] Saved comparison to: bias_comparison.png")

if __name__ == "__main__":
    file_path = "sofascore_large_dataset.csv"
    print(f"Loading data from {file_path}...")
    
    df = load_and_preprocess_data(file_path)
    if df is not None:
        print(f"Data sorted by date: {df['date'].min()} to {df['date'].max()}")
        
        X, y, df_clean = feature_engineering(df)
        print(f"Total Data Points: {len(X)}")
        
        if len(X) > 50:
            run_comparison(X, y)
        else:
            print("Not enough data to run comparison.")
