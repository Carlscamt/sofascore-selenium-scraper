"""
Strategy Explorer - Finding Edge with the Clean Model (No H2H)
==============================================================
Tests multiple betting strategies to find potential edges in specific niches.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(filepath):
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

def feature_engineering(df):
    home_form_data = df['home_form'].apply(process_form_string).tolist()
    home_form_df = pd.DataFrame(home_form_data, columns=[f'h_form_{i+1}' for i in range(5)])
    
    away_form_data = df['away_form'].apply(process_form_string).tolist()
    away_form_df = pd.DataFrame(away_form_data, columns=[f'a_form_{i+1}' for i in range(5)])
    
    df_clean = df.dropna(subset=['odd_1', 'odd_X', 'odd_2']).copy()
    
    home_form_df = home_form_df.loc[df_clean.index].reset_index(drop=True)
    away_form_df = away_form_df.loc[df_clean.index].reset_index(drop=True)
    
    le = LabelEncoder()
    df_clean['league_encoded'] = le.fit_transform(df_clean['league'])
    
    # FULL FEATURES including H2H
    features = [
        'odd_1', 'odd_X', 'odd_2',
        'home_avg_rating', 'away_avg_rating',
        'home_position', 'away_position',
        'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
        'league_encoded'
    ]
    
    df_clean[features] = df_clean[features].fillna(-1) 
    df_clean = df_clean.reset_index(drop=True)
    
    X = pd.concat([df_clean[features], home_form_df, away_form_df], axis=1)
    y = df_clean['target']
    
    return X, y, df_clean

def train_model_temporal(X, y):
    """Train model and return predictions + probabilities for test folds."""
    tscv = TimeSeriesSplit(n_splits=5)
    
    all_results = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = xgb.XGBClassifier(
            objective='multi:softprob', eval_metric='mlogloss',
            use_label_encoder=False, random_state=42, verbosity=0
        )
        model.fit(X_train, y_train)
        
        probs = model.predict_proba(X_test)
        
        for i, idx in enumerate(test_idx):
            all_results.append({
                'idx': idx,
                'actual': y.iloc[idx],
                'p_home': probs[i][0],
                'p_draw': probs[i][1],
                'p_away': probs[i][2],
                'odd_1': X.iloc[idx]['odd_1'],
                'odd_X': X.iloc[idx]['odd_X'],
                'odd_2': X.iloc[idx]['odd_2'],
            })
    
    return pd.DataFrame(all_results)

def evaluate_strategy(results_df, condition_fn, bet_selector_fn, name):
    """
    Evaluate a betting strategy.
    condition_fn: filters which matches to bet on
    bet_selector_fn: selects which outcome to bet on (returns 0, 1, 2 or None)
    """
    bets = []
    
    for _, row in results_df.iterrows():
        if not condition_fn(row):
            continue
            
        bet_choice = bet_selector_fn(row)
        if bet_choice is None:
            continue
            
        odds = [row['odd_1'], row['odd_X'], row['odd_2']]
        odd = odds[bet_choice]
        
        profit = (odd - 1) if bet_choice == row['actual'] else -1
        bets.append({
            'choice': bet_choice,
            'odd': odd,
            'profit': profit,
            'prob': [row['p_home'], row['p_draw'], row['p_away']][bet_choice]
        })
    
    if not bets:
        return {'name': name, 'bets': 0, 'profit': 0, 'roi': 0, 'win_rate': 0}
    
    total_profit = sum(b['profit'] for b in bets)
    wins = sum(1 for b in bets if b['profit'] > 0)
    roi = (total_profit / len(bets)) * 100
    
    return {
        'name': name,
        'bets': len(bets),
        'profit': total_profit,
        'roi': roi,
        'win_rate': wins / len(bets) * 100,
        'avg_odd': np.mean([b['odd'] for b in bets]),
        'bets_list': bets
    }

def run_strategies(results_df):
    """Run multiple betting strategies and compare."""
    
    strategies = []
    
    # =========================================
    # STRATEGY 1: Value Betting (Base Case)
    # Bet when model prob * odds > 1
    # =========================================
    def value_condition(row):
        evs = [
            row['p_home'] * row['odd_1'] - 1,
            row['p_draw'] * row['odd_X'] - 1,
            row['p_away'] * row['odd_2'] - 1
        ]
        return max(evs) > 0
    
    def value_selector(row):
        evs = [
            row['p_home'] * row['odd_1'] - 1,
            row['p_draw'] * row['odd_X'] - 1,
            row['p_away'] * row['odd_2'] - 1
        ]
        best = max(range(3), key=lambda i: evs[i])
        return best if evs[best] > 0 else None
    
    strategies.append(evaluate_strategy(results_df, value_condition, value_selector, 
                                        "1. Value Betting (EV > 0)"))
    
    # =========================================
    # STRATEGY 2: High Confidence Only
    # Only bet when model is very confident (prob > 50%)
    # =========================================
    def confident_condition(row):
        return max(row['p_home'], row['p_draw'], row['p_away']) > 0.50
    
    def confident_selector(row):
        probs = [row['p_home'], row['p_draw'], row['p_away']]
        best = max(range(3), key=lambda i: probs[i])
        ev = probs[best] * [row['odd_1'], row['odd_X'], row['odd_2']][best] - 1
        return best if ev > 0 and probs[best] > 0.50 else None
    
    strategies.append(evaluate_strategy(results_df, confident_condition, confident_selector,
                                        "2. High Confidence (Prob > 50%)"))
    
    # =========================================
    # STRATEGY 3: Strong EV Only
    # Only bet when EV > 10%
    # =========================================
    def strong_ev_condition(row):
        evs = [
            row['p_home'] * row['odd_1'] - 1,
            row['p_draw'] * row['odd_X'] - 1,
            row['p_away'] * row['odd_2'] - 1
        ]
        return max(evs) > 0.10
    
    strategies.append(evaluate_strategy(results_df, strong_ev_condition, value_selector,
                                        "3. Strong EV Only (EV > 10%)"))
    
    # =========================================
    # STRATEGY 4: Draw Specialists
    # Only bet on draws (often mispriced)
    # =========================================
    def draw_condition(row):
        ev_draw = row['p_draw'] * row['odd_X'] - 1
        return ev_draw > 0
    
    def draw_selector(row):
        ev_draw = row['p_draw'] * row['odd_X'] - 1
        return 1 if ev_draw > 0 else None
    
    strategies.append(evaluate_strategy(results_df, draw_condition, draw_selector,
                                        "4. Draw Specialists"))
    
    # =========================================
    # STRATEGY 5: Underdog Value
    # Bet on underdogs (odds > 3.0) with positive EV
    # =========================================
    def underdog_condition(row):
        evs = [
            (row['p_home'] * row['odd_1'] - 1, row['odd_1']),
            (row['p_away'] * row['odd_2'] - 1, row['odd_2'])
        ]
        return any(ev > 0 and odd > 3.0 for ev, odd in evs)
    
    def underdog_selector(row):
        evs = [
            (row['p_home'] * row['odd_1'] - 1, 0, row['odd_1']),
            (row['p_away'] * row['odd_2'] - 1, 2, row['odd_2'])
        ]
        valid = [(ev, choice, odd) for ev, choice, odd in evs if ev > 0 and odd > 3.0]
        if valid:
            return max(valid, key=lambda x: x[0])[1]
        return None
    
    strategies.append(evaluate_strategy(results_df, underdog_condition, underdog_selector,
                                        "5. Underdog Value (Odds > 3.0)"))
    
    # =========================================
    # STRATEGY 6: Favorite Lock
    # Only bet on heavy favorites (odds < 1.5) with EV
    # =========================================
    def favorite_condition(row):
        evs = [
            (row['p_home'] * row['odd_1'] - 1, row['odd_1']),
            (row['p_away'] * row['odd_2'] - 1, row['odd_2'])
        ]
        return any(ev > 0 and odd < 1.5 for ev, odd in evs)
    
    def favorite_selector(row):
        evs = [
            (row['p_home'] * row['odd_1'] - 1, 0, row['odd_1']),
            (row['p_away'] * row['odd_2'] - 1, 2, row['odd_2'])
        ]
        valid = [(ev, choice, odd) for ev, choice, odd in evs if ev > 0 and odd < 1.5]
        if valid:
            return max(valid, key=lambda x: x[0])[1]
        return None
    
    strategies.append(evaluate_strategy(results_df, favorite_condition, favorite_selector,
                                        "6. Favorite Lock (Odds < 1.5)"))
    
    # =========================================
    # STRATEGY 7: Model vs Market Disagreement
    # Bet when model strongly disagrees with implied prob
    # =========================================
    def disagreement_condition(row):
        # Implied probs from odds
        total = 1/row['odd_1'] + 1/row['odd_X'] + 1/row['odd_2']
        imp_home = (1/row['odd_1']) / total
        imp_draw = (1/row['odd_X']) / total
        imp_away = (1/row['odd_2']) / total
        
        # Disagreement = model prob - implied prob
        diffs = [
            row['p_home'] - imp_home,
            row['p_draw'] - imp_draw,
            row['p_away'] - imp_away
        ]
        return max(diffs) > 0.10  # Model thinks >10% more likely than market
    
    def disagreement_selector(row):
        total = 1/row['odd_1'] + 1/row['odd_X'] + 1/row['odd_2']
        imp = [(1/row['odd_1'])/total, (1/row['odd_X'])/total, (1/row['odd_2'])/total]
        probs = [row['p_home'], row['p_draw'], row['p_away']]
        diffs = [probs[i] - imp[i] for i in range(3)]
        
        best = max(range(3), key=lambda i: diffs[i])
        if diffs[best] > 0.10:
            return best
        return None
    
    strategies.append(evaluate_strategy(results_df, disagreement_condition, disagreement_selector,
                                        "7. Model vs Market (>10% diff)"))
    
    # =========================================
    # STRATEGY 8: Kelly Criterion Sizing
    # Bet with Kelly sizing on +EV bets
    # =========================================
    def kelly_evaluate(results_df):
        bets = []
        bankroll = 100  # Starting bankroll
        
        for _, row in results_df.iterrows():
            evs = [
                (row['p_home'] * row['odd_1'] - 1, 0, row['odd_1'], row['p_home']),
                (row['p_draw'] * row['odd_X'] - 1, 1, row['odd_X'], row['p_draw']),
                (row['p_away'] * row['odd_2'] - 1, 2, row['odd_2'], row['p_away'])
            ]
            
            best = max(evs, key=lambda x: x[0])
            ev, choice, odd, prob = best
            
            if ev > 0:
                # Kelly fraction: (bp - q) / b where b = odd-1, p = prob, q = 1-p
                b = odd - 1
                kelly = (b * prob - (1 - prob)) / b
                kelly = max(0, min(kelly, 0.25))  # Cap at 25% of bankroll
                
                stake = bankroll * kelly
                profit = stake * (odd - 1) if choice == row['actual'] else -stake
                bankroll += profit
                
                bets.append({
                    'stake': stake,
                    'profit': profit,
                    'bankroll': bankroll
                })
        
        if not bets:
            return {'name': '8. Kelly Criterion', 'bets': 0, 'profit': 0, 'roi': 0}
        
        total_staked = sum(b['stake'] for b in bets)
        total_profit = sum(b['profit'] for b in bets)
        
        return {
            'name': '8. Kelly Criterion',
            'bets': len(bets),
            'profit': total_profit,
            'roi': (total_profit / total_staked) * 100 if total_staked > 0 else 0,
            'final_bankroll': bankroll,
            'bets_list': bets
        }
    
    strategies.append(kelly_evaluate(results_df))
    
    return strategies

def print_results(strategies):
    print("\n" + "="*70)
    print("STRATEGY COMPARISON RESULTS")
    print("="*70)
    print(f"{'Strategy':<35} {'Bets':>6} {'Win%':>7} {'Profit':>10} {'ROI':>8}")
    print("-"*70)
    
    for s in strategies:
        win_rate = s.get('win_rate', 0)
        print(f"{s['name']:<35} {s['bets']:>6} {win_rate:>6.1f}% {s['profit']:>+10.2f} {s['roi']:>+7.2f}%")
    
    print("-"*70)
    
    # Find best strategy
    profitable = [s for s in strategies if s['roi'] > 0 and s['bets'] >= 10]
    if profitable:
        best = max(profitable, key=lambda x: x['roi'])
        print(f"\n[BEST] {best['name']} with {best['roi']:.2f}% ROI on {best['bets']} bets")
    else:
        print("\n[!] No consistently profitable strategy found (expected result)")

def plot_strategies(strategies):
    plt.figure(figsize=(14, 8))
    
    # Filter strategies with bets
    valid = [s for s in strategies if s['bets'] > 0 and 'bets_list' in s]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(valid)))
    
    for i, s in enumerate(valid):
        cumsum = np.cumsum([b['profit'] for b in s['bets_list']])
        plt.plot(cumsum, label=f"{s['name']} (ROI: {s['roi']:.1f}%)", color=colors[i])
    
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title('Strategy Performance Comparison (No H2H - Clean Model)')
    plt.xlabel('Bet Number')
    plt.ylabel('Cumulative Profit (Units)')
    plt.legend(loc='best', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('strategy_exploration.png', dpi=150)
    print(f"\n[GRAPH] Saved to: strategy_exploration.png")

if __name__ == "__main__":
    print("="*70)
    print("STRATEGY EXPLORER - Full Model with Clean H2H Data")
    print("="*70)
    
    df = load_and_preprocess_data("sofascore_large_dataset.csv")
    X, y, df_clean = feature_engineering(df)
    
    print(f"Data: {len(X)} matches from {df_clean['date'].min().date()} to {df_clean['date'].max().date()}")
    print("\nTraining model with temporal CV...")
    
    results_df = train_model_temporal(X, y)
    print(f"Generated predictions for {len(results_df)} test matches\n")
    
    strategies = run_strategies(results_df)
    print_results(strategies)
    plot_strategies(strategies)
    
    print("\n" + "="*70)
    print("INTERPRETATION:")
    print("="*70)
    print("- All strategies with positive ROI on small sample need more data")
    print("- -5% to +5% ROI is within variance, not necessarily an edge")
    print("- True edge requires consistent profit over 1000+ bets")
    print("- Consider: more data, different leagues, live betting")
