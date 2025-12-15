import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(filepath):
    """
    Loads data and creates target variable.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None

    # Create Target Variable
    # 0: Home Win, 1: Draw, 2: Away Win
    df['target'] = np.select(
        [df['score_home'] > df['score_away'], df['score_home'] == df['score_away']],
        [0, 1],
        default=2
    )
    
    return df

def process_form_string(form_str):
    """
    Parses a form string like "W,D,L,W,W" into a list of integers.
    """
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
    """
    Transforms raw dataframe into X (features) and y (target).
    """
    # 1. Process Form
    home_form_data = df['home_form'].apply(process_form_string).tolist()
    home_form_df = pd.DataFrame(home_form_data, columns=[f'h_form_{i+1}' for i in range(5)])
    
    away_form_data = df['away_form'].apply(process_form_string).tolist()
    away_form_df = pd.DataFrame(away_form_data, columns=[f'a_form_{i+1}' for i in range(5)])
    
    # 2. Key Features Selection
    df_clean = df.dropna(subset=['odd_1', 'odd_X', 'odd_2']).copy()
    
    home_form_df = home_form_df.loc[df_clean.index]
    away_form_df = away_form_df.loc[df_clean.index]
    
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
    
    X = pd.concat([
        df_clean[features].reset_index(drop=True),
        home_form_df.reset_index(drop=True),
        away_form_df.reset_index(drop=True)
    ], axis=1)
    
    y = df_clean['target'].reset_index(drop=True)
    
    return X, y

def get_fold_bets(model, X_test, y_test):
    """
    Helper to calculate bets for a single fold.
    Returns a list of bet dictionaries.
    """
    probs = model.predict_proba(X_test)
    X_test_reset = X_test.reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)
    
    bets = []

    for i in range(len(X_test_reset)):
        odd_1 = X_test_reset.loc[i, 'odd_1']
        odd_X = X_test_reset.loc[i, 'odd_X']
        odd_2 = X_test_reset.loc[i, 'odd_2']
        
        # Probabilities
        p_1, p_X, p_2 = probs[i]
        
        actual_result = y_test_reset[i]
        
        # EV Calc
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

def run_cv_simulation(X, y):
    """
    Runs 5-Fold Cross Validation.
    Trains on 4/5ths, bets on 1/5th. Repeats 5 times.
    Aggregates all bets to simulate full dataset performance.
    """
    print("\n" + "="*40)
    print(">>> STARTING 5-FOLD CROSS-VALIDATION SIMULATION")
    print("="*40)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    total_accuracy = []
    all_bets = []
    
    fold = 1
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
        
        # Evaluation Stats
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        total_accuracy.append(acc)
        
        # Betting Simulation for this fold
        fold_bets = get_fold_bets(model, X_test, y_test)
        all_bets.extend(fold_bets)
        
        print(f"Fold {fold}/5 - Accuracy: {acc:.4f} - Bets Placed: {len(fold_bets)}")
        fold += 1

    avg_acc = np.mean(total_accuracy)
    
    print("\n" + "="*40)
    print(f"[OK] CROSS-VALIDATION COMPLETE")
    print(f"Average Model Accuracy: {avg_acc:.4f}")
    print("="*40)
    
    # Process All Bets
    total_profit = sum(b['PnL'] for b in all_bets)
    total_bets_count = len(all_bets)
    roi = (total_profit / total_bets_count) * 100 if total_bets_count > 0 else 0
    
    print("\n$$$ FINAL BETTING REPORT (ALL FOLDS)")
    print("-" * 40)
    print(f"{'Choice':<10} {'Odd':<6} {'Prob':<6} {'Result':<6} {'PnL'}")
    print("-" * 40)
    
    # Print first 20 bets to avoid spamming console, or all? Let's print summary stats mostly.
    # We will print the last 15 for visibility
    for res in all_bets[-15:]:
         print(f"{res['Choice']:<10} {res['Odd']:<6.2f} {res['Prob']:<6.2f} {res['Result']:<6} {res['PnL']:+.2f}")
    if len(all_bets) > 15:
        print(f"... (Showing last 15 of {len(all_bets)} bets)")

    print("-" * 40)
    print(f"TOTAL BETS MATCHED: {total_bets_count} / {len(X)}")
    print(f"TOTAL PROFIT: {total_profit:.2f} Units")
    print(f"FINAL ROI: {roi:.2f}%")
    print("-" * 40)

    # Plot Cumulative Profit for Base Case
    if all_bets:
        pnl_sequence = [r['PnL'] for r in all_bets]
        cumulative_pnl = np.cumsum(pnl_sequence)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumulative_pnl) + 1), cumulative_pnl, marker='o', linestyle='-', color='blue')
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.title(f'Cumulative Profit - Base Case ({total_bets_count} Bets)')
        plt.xlabel('Bet Number')
        plt.ylabel('Profit (Units)')
        plt.grid(True, alpha=0.3)
        
        output_img = "betting_performance_cv.png"
        plt.savefig(output_img)
        print(f"[GRAPH] Base Graph saved to: {output_img}")

    # Run Multi-Scenario Analysis
    run_scenario_analysis(all_bets)

def run_scenario_analysis(all_bets):
    """
    Analyzes performance under different betting strategies.
    """
    print("\n" + "="*50)
    print(">>> COMPREHENSIVE STRATEGY REPORT")
    print("="*50)
    
    strategies = {
        "Base Case (All +EV)": lambda b: True,
        "Conservative (Prob > 60%)": lambda b: b['Prob'] > 0.60,
        "Value Hunter (EV > 10%)": lambda b: ((b['Prob'] * b['Odd']) - 1) > 0.10,
        "Longshots (Odds > 3.0)": lambda b: b['Odd'] > 3.0,
        "Favorites (Odds < 1.5)": lambda b: b['Odd'] < 1.5
    }
    
    results_summary = []
    plt.figure(figsize=(12, 8))
    
    print(f"{'Strategy':<25} {'Bets':<6} {'Win%':<6} {'Profit':<8} {'ROI':<8}")
    print("-" * 60)
    
    for name, condition in strategies.items():
        # Filter bets
        filtered_bets = [b for b in all_bets if condition(b)]
        
        if not filtered_bets:
            print(f"{name:<25} {'0':<6} {'0.0%':<6} {'0.00':<8} {'0.00%':<8}")
            continue
            
        # Stats
        count = len(filtered_bets)
        profit = sum(b['PnL'] for b in filtered_bets)
        wins = sum(1 for b in filtered_bets if b['Result'] == 'WIN')
        win_rate = (wins / count) * 100
        roi = (profit / count) * 100
        
        print(f"{name:<25} {count:<6} {win_rate:<6.1f} {profit:<8.2f} {roi:<8.2f}%")
        
        # Plotting
        pnl_seq = [b['PnL'] for b in filtered_bets]
        cum_pnl = np.cumsum(pnl_seq)
        plt.plot(range(1, count + 1), cum_pnl, label=f"{name} (ROI: {roi:.1f}%)")
        
    print("-" * 60)
    
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title('Strategy Performance Comparison')
    plt.xlabel('Number of Bets')
    plt.ylabel('Cumulative Profit (Units)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_img = "strategy_comparison.png"
    plt.savefig(output_img)
    print(f"\n[GRAPH] Comparison Graph saved to: {output_img}")


if __name__ == "__main__":
    file_path = "sofascore_large_dataset.csv"
    print(f"Loading data from {file_path}...")
    
    df = load_and_preprocess_data(file_path)
    if df is not None:
        X, y = feature_engineering(df)
        print(f"Total Data Points: {len(X)}")
        
        if len(X) > 10:
            run_cv_simulation(X, y)
        else:
            print("Not enough data to train.")
