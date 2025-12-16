import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def normalize_odds(df):
    """
    Calculates implied probability from odds and adds margin-removed probs.
    """
    cols = ['odd_1', 'odd_X', 'odd_2']
    # Implied Prob = 1/Odd
    probs = 1 / df[cols]
    # Normalize to sum to 1 (removing margin)
    probs_norm = probs.div(probs.sum(axis=1), axis=0)
    return probs_norm

def run_simple_cv(X, y, description):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    profits = []
    bets = 0
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Simple Logistic Regression equivalent with XGB (linear booster) to avoid overfitting
        model = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        
        accs.append(accuracy_score(y_test, preds))
        
        # Sim betting
        indices = X_test.index
        for i, idx in enumerate(indices):
            # We need original odds from the global df, but X might be subset
            # Actually X should contain odds
            if 'odd_1' in X.columns:
                o1, oX, o2 = X.loc[idx, ['odd_1', 'odd_X', 'odd_2']]
            else:
                 # If odds not in features, we can't bet, but for Baseline they are there.
                 # For "No Odds" model, we'd need to pass odds separately.
                 continue

            # Check EV
            p_1, p_X, p_2 = probs[i]
            
            # Bet Logic
            evs = [(p_1 * o1 - 1, 0, o1), (p_X * oX - 1, 1, oX), (p_2 * o2 - 1, 2, o2)]
            best = max(evs, key=lambda x: x[0])
            
            if best[0] > 0:
                bets += 1
                if best[1] == y.loc[idx]:
                    profits.append(best[2] - 1)
                else:
                    profits.append(-1)
                    
    total_profit = sum(profits)
    roi = (total_profit / bets * 100) if bets > 0 else 0
    print(f"[{description}] Acc: {np.mean(accs):.4f} | Bets: {bets} | Profit: {total_profit:.2f} | ROI: {roi:.2f}%")

def check_leakage():
    df = pd.read_csv("sofascore_selenium_last_7_days.csv")
    df['target'] = np.select(
        [df['score_home'] > df['score_away'], df['score_home'] == df['score_away']],
        [0, 1],
        default=2
    )
    df = df.dropna(subset=['odd_1', 'odd_X', 'odd_2'])
    
    # 1. Correlation Check
    print("--- Correlation with Target (Encoded) ---")
    # Encode target to numeric (Home=0, Draw=1, Away=2) - correlation might be weird for categorical, 
    # better to check correlation with 'Score Difference'
    df['score_diff'] = df['score_home'] - df['score_away']
    
    # Check numeric columns
    numeric_cols = ['odd_1', 'odd_X', 'odd_2', 'home_avg_rating', 'away_avg_rating', 
                    'home_position', 'away_position', 'h2h_home_wins']
    
    for col in numeric_cols:
        if col in df.columns:
            corr = df[col].corr(df['score_diff'])
            print(f"{col}: {corr:.4f}")
            
    print("\n--- Model Comparison ---")
    
    # Baseline: Odds Only
    X_odds = df[['odd_1', 'odd_X', 'odd_2']]
    run_simple_cv(X_odds, df['target'], "Baseline (Odds Only)")
    
    # Full Model (Variables from ml_model.py)
    # Re-using logic from ml_model roughly
    # (assuming processed form data is not easily available here without importing, 
    # let's just check the numeric cols + odds)
    feature_cols = ['odd_1', 'odd_X', 'odd_2', 'home_avg_rating', 'away_avg_rating', 
                    'home_position', 'away_position', 'h2h_home_wins', 'h2h_away_wins']
    # Fill NA
    df_clean = df.fillna(-1)
    X_full = df_clean[feature_cols]
    run_simple_cv(X_full, df_clean['target'], "Full Numeric Model")

if __name__ == "__main__":
    check_leakage()
