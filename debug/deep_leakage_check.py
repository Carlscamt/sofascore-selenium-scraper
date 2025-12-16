"""
Deep Leakage Diagnostic
=======================
Checks if the pregame-form data contains future information
by analyzing how well "form at match time" correlates with results.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

def run_diagnostic():
    print("="*60)
    print("DEEP LEAKAGE DIAGNOSTIC")
    print("="*60)
    
    df = pd.read_csv("sofascore_large_dataset.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Create target
    df['target'] = np.select(
        [df['score_home'] > df['score_away'], df['score_home'] == df['score_away']],
        [0, 1], default=2
    )
    
    df_clean = df.dropna(subset=['odd_1', 'odd_X', 'odd_2']).copy()
    
    print(f"\nData Range: {df_clean['date'].min()} to {df_clean['date'].max()}")
    print(f"Total Matches: {len(df_clean)}")
    
    # TEST 1: Odds-Only Baseline
    # Betting markets are efficient - if we can beat implied odds, something is wrong
    print("\n" + "-"*50)
    print("TEST 1: Market Implied Probability vs Actual Results")
    print("-"*50)
    
    # Calculate implied probabilities from odds
    df_clean['implied_home'] = 1 / df_clean['odd_1']
    df_clean['implied_draw'] = 1 / df_clean['odd_X'] 
    df_clean['implied_away'] = 1 / df_clean['odd_2']
    
    # Normalize (remove vig)
    total_prob = df_clean['implied_home'] + df_clean['implied_draw'] + df_clean['implied_away']
    df_clean['implied_home_norm'] = df_clean['implied_home'] / total_prob
    df_clean['implied_draw_norm'] = df_clean['implied_draw'] / total_prob
    df_clean['implied_away_norm'] = df_clean['implied_away'] / total_prob
    
    # Market-implied prediction (pick highest probability)
    df_clean['market_pred'] = df_clean[['implied_home_norm', 'implied_draw_norm', 'implied_away_norm']].idxmax(axis=1)
    df_clean['market_pred'] = df_clean['market_pred'].map({
        'implied_home_norm': 0, 'implied_draw_norm': 1, 'implied_away_norm': 2
    })
    
    market_acc = (df_clean['market_pred'] == df_clean['target']).mean()
    print(f"Market Accuracy (picking favorite): {market_acc:.4f}")
    
    # TEST 2: Odds correlation with results
    # If odds perfectly predict, we have leakage
    print("\n" + "-"*50)
    print("TEST 2: Feature Correlations with Result")
    print("-"*50)
    
    df_clean['score_diff'] = df_clean['score_home'] - df_clean['score_away']
    
    correlations = [
        ('odd_1 (lower = home more likely)', -df_clean['odd_1'].corr(df_clean['score_diff'])),
        ('odd_2 (lower = away more likely)', df_clean['odd_2'].corr(df_clean['score_diff'])),
        ('home_avg_rating', df_clean['home_avg_rating'].corr(df_clean['score_diff'])),
        ('away_avg_rating', -df_clean['away_avg_rating'].corr(df_clean['score_diff'])),
        ('home_position (lower = better)', -df_clean['home_position'].corr(df_clean['score_diff'])),
        ('away_position (lower = better)', df_clean['away_position'].corr(df_clean['score_diff'])),
    ]
    
    for name, corr in correlations:
        if pd.notna(corr):
            flag = "[HIGH]" if abs(corr) > 0.3 else ""
            print(f"{name}: {corr:.4f} {flag}")
    
    # TEST 3: Form strength check
    print("\n" + "-"*50)
    print("TEST 3: Form Data Analysis")
    print("-"*50)
    
    def parse_form(form_str):
        if pd.isna(form_str):
            return 0
        mapping = {'W': 3, 'D': 1, 'L': 0}
        parts = form_str.split(',')
        return sum(mapping.get(x.strip(), 0) for x in parts[:5])
    
    df_clean['home_form_score'] = df_clean['home_form'].apply(parse_form)
    df_clean['away_form_score'] = df_clean['away_form'].apply(parse_form)
    df_clean['form_diff'] = df_clean['home_form_score'] - df_clean['away_form_score']
    
    form_corr = df_clean['form_diff'].corr(df_clean['score_diff'])
    print(f"Form Difference -> Score Difference Correlation: {form_corr:.4f}")
    
    # Check if form contains CURRENT match result
    # If form was captured AFTER the match, the first character might include today's result
    print("\n" + "-"*50)
    print("TEST 4: Checking for Current Match in Form Data")
    print("-"*50)
    
    sample = df_clean.head(10)[['date', 'home', 'away', 'score_home', 'score_away', 'home_form', 'away_form']]
    
    matches_found = 0
    for idx, row in df_clean.iterrows():
        home_form = str(row['home_form']).split(',') if pd.notna(row['home_form']) else []
        score_home = row['score_home']
        score_away = row['score_away']
        
        # What SHOULD this match result be?
        if score_home > score_away:
            expected_result = 'W'  # Home won
        elif score_home < score_away:
            expected_result = 'L'  # Home lost
        else:
            expected_result = 'D'  # Draw
        
        # Check if first form entry matches this game's result
        if home_form and home_form[0].strip() == expected_result:
            matches_found += 1
    
    match_rate = matches_found / len(df_clean) * 100
    print(f"First form entry matches this game's result: {match_rate:.1f}% of matches")
    
    if match_rate > 40:
        print("[!] WARNING: Form data may contain the current match result (LEAKAGE!)") 
    else:
        print("[OK] Form data appears to be from BEFORE the match")
    
    # TEST 5: Extremely suspicious - random feature test
    print("\n" + "-"*50)
    print("TEST 5: XGBoost with ONLY Odds (should not beat market)")
    print("-"*50)
    
    X_odds = df_clean[['odd_1', 'odd_X', 'odd_2']].reset_index(drop=True)
    y = df_clean['target'].reset_index(drop=True)
    
    tscv = TimeSeriesSplit(n_splits=5)
    accuracies = []
    profits = []
    bets = 0
    
    for train_idx, test_idx in tscv.split(X_odds):
        X_train, X_test = X_odds.iloc[train_idx], X_odds.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', 
                                   use_label_encoder=False, random_state=42, verbosity=0)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, preds))
        
        # Betting simulation
        probs = model.predict_proba(X_test)
        for i in range(len(X_test)):
            o1, oX, o2 = X_test.iloc[i]
            p_1, p_X, p_2 = probs[i]
            
            evs = [(p_1*o1-1, 0, o1), (p_X*oX-1, 1, oX), (p_2*o2-1, 2, o2)]
            best = max(evs, key=lambda x: x[0])
            
            if best[0] > 0:
                bets += 1
                if best[1] == y_test.iloc[i]:
                    profits.append(best[2] - 1)
                else:
                    profits.append(-1)
    
    total_profit = sum(profits)
    roi = (total_profit / bets * 100) if bets > 0 else 0
    
    print(f"Odds-Only XGBoost Accuracy: {np.mean(accuracies):.4f}")
    print(f"Odds-Only XGBoost ROI: {roi:.2f}%")
    
    if roi > 10:
        print("[!] SUSPICIOUS: Should not be possible to profit this much from just odds")
        print("   Odds contain the market's full information - beating them significantly")
        print("   suggests model is somehow learning from future outcomes.")
    
    # FINAL DIAGNOSIS
    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)
    
    issues = []
    if match_rate > 40:
        issues.append("Form data may include current match result")
    if roi > 15:
        issues.append(f"Model shows unrealistic {roi:.0f}% ROI with just odds")
    
    if issues:
        print("[!] POTENTIAL LEAKAGE SOURCES:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("[OK] No obvious leakage detected in feature data")
        print("  The high ROI may be due to:")
        print("  - Sample size issues (only ~600 matches)")
        print("  - Selection bias in scraped data")
        print("  - Short evaluation period (2 months)")

if __name__ == "__main__":
    run_diagnostic()
