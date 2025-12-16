"""
Model without H2H features - Testing if H2H is the leakage source
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
    df = pd.read_csv(filepath)
    df['target'] = np.select(
        [df['score_home'] > df['score_away'], df['score_home'] == df['score_away']],
        [0, 1], default=2
    )
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

def feature_engineering_no_h2h(df):
    """Features WITHOUT h2h data."""
    home_form_data = df['home_form'].apply(process_form_string).tolist()
    home_form_df = pd.DataFrame(home_form_data, columns=[f'h_form_{i+1}' for i in range(5)])
    
    away_form_data = df['away_form'].apply(process_form_string).tolist()
    away_form_df = pd.DataFrame(away_form_data, columns=[f'a_form_{i+1}' for i in range(5)])
    
    df_clean = df.dropna(subset=['odd_1', 'odd_X', 'odd_2']).copy()
    
    home_form_df = home_form_df.loc[df_clean.index].reset_index(drop=True)
    away_form_df = away_form_df.loc[df_clean.index].reset_index(drop=True)
    
    le = LabelEncoder()
    df_clean['league_encoded'] = le.fit_transform(df_clean['league'])
    
    # NO H2H FEATURES
    features = [
        'odd_1', 'odd_X', 'odd_2',
        'home_avg_rating', 'away_avg_rating',
        'home_position', 'away_position',
        'league_encoded'
        # REMOVED: 'h2h_home_wins', 'h2h_away_wins', 'h2h_draws'
    ]
    
    df_clean[features] = df_clean[features].fillna(-1) 
    df_clean = df_clean.reset_index(drop=True)
    
    X = pd.concat([df_clean[features], home_form_df, away_form_df], axis=1)
    y = df_clean['target']
    
    return X, y, df_clean

def get_fold_bets(model, X_test, y_test):
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
        
        ev_1 = (p_1 * odd_1) - 1
        ev_X = (p_X * odd_X) - 1
        ev_2 = (p_2 * odd_2) - 1
        
        best_ev = -float('inf')
        bet_choice = None
        bet_odd = 0
        
        if ev_1 > 0 and ev_1 > best_ev:
            best_ev = ev_1; bet_choice = 0; bet_odd = odd_1
        if ev_X > 0 and ev_X > best_ev:
            best_ev = ev_X; bet_choice = 1; bet_odd = odd_X
        if ev_2 > 0 and ev_2 > best_ev:
            best_ev = ev_2; bet_choice = 2; bet_odd = odd_2
            
        if bet_choice is not None:
            profit = (bet_odd - 1) if bet_choice == actual_result else -1
            bets.append({'Odd': bet_odd, 'Prob': probs[i][bet_choice], 'PnL': profit})
    return bets

def run_temporal_cv(X, y, description):
    print(f"\n{'='*50}")
    print(f">>> {description}")
    print("="*50)
    
    tscv = TimeSeriesSplit(n_splits=5)
    all_bets = []
    accuracies = []
    
    fold = 1
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = xgb.XGBClassifier(
            objective='multi:softprob', eval_metric='mlogloss',
            use_label_encoder=False, random_state=42, verbosity=0
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
    wins = sum(1 for b in all_bets if b['PnL'] > 0)
    
    print(f"\nAccuracy: {np.mean(accuracies):.4f}")
    print(f"Total Bets: {total_bets} | Wins: {wins} ({wins/total_bets*100:.1f}%)")
    print(f"Total Profit: {total_profit:.2f} units")
    print(f"ROI: {roi:.2f}%")
    
    return all_bets, roi, np.mean(accuracies)

if __name__ == "__main__":
    file_path = "sofascore_large_dataset.csv"
    print(f"Loading data from {file_path}...")
    
    df = load_and_preprocess_data(file_path)
    print(f"Data: {df['date'].min().date()} to {df['date'].max().date()} ({len(df)} matches)")
    
    # Model WITHOUT H2H
    X_no_h2h, y, df_clean = feature_engineering_no_h2h(df)
    print(f"\nFeatures (NO H2H): {list(X_no_h2h.columns)}")
    
    bets, roi, acc = run_temporal_cv(X_no_h2h, y, "TEMPORAL CV - NO H2H FEATURES")
    
    # Plot
    if bets:
        plt.figure(figsize=(10, 5))
        cumsum = np.cumsum([b['PnL'] for b in bets])
        plt.plot(cumsum, color='blue')
        plt.axhline(0, color='black', linestyle='--')
        plt.title(f'Cumulative Profit (No H2H) - ROI: {roi:.1f}%')
        plt.xlabel('Bet Number')
        plt.ylabel('Profit (Units)')
        plt.grid(True, alpha=0.3)
        plt.savefig('no_h2h_performance.png')
        print(f"\n[GRAPH] Saved to: no_h2h_performance.png")
    
    print("\n" + "="*50)
    print("INTERPRETATION:")
    print("="*50)
    if roi > 10:
        print(f"ROI still {roi:.1f}% - H2H was NOT the main leakage source")
        print("Other features may still contain future information")
    elif roi > 0:
        print(f"ROI now {roi:.1f}% - more realistic but still suspicious")
    else:
        print(f"ROI now {roi:.1f}% - this is realistic (can't beat the market easily)")
