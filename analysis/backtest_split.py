"""
Proper Backtest with Train/Test Split
======================================
Uses chronological 80/20 split for realistic out-of-sample evaluation.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
from run_full_model import load_and_preprocess_data, feature_engineering_full

def run_backtest():
    print("="*60)
    print("BACKTEST WITH TRAIN/TEST SPLIT (80/20)")
    print("="*60)
    
    # 1. Load Data
    df = load_and_preprocess_data("sofascore_dataset_v2.csv")
    print(f"Loaded {len(df)} matches")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # 2. Prepare Features
    X, y, df_clean = feature_engineering_full(df)
    print(f"Clean data points: {len(X)}")
    
    # 3. Chronological Split (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\nTrain: {len(X_train)} matches (oldest 80%)")
    print(f"Test:  {len(X_test)} matches (newest 20%)")
    
    # 4. Train Model
    print("\nTraining XGBoost model on TRAIN set...")
    model = xgb.XGBClassifier(
        objective='multi:softprob', 
        eval_metric='mlogloss',
        use_label_encoder=False, 
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)
    
    # 5. Evaluate on TEST set (out-of-sample)
    print("Evaluating on TEST set (out-of-sample)...")
    probs = model.predict_proba(X_test)
    preds = model.predict(X_test)
    
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # 6. Simulate Betting on Test Set
    bets = []
    X_test_reset = X_test.reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)
    
    for i in range(len(X_test_reset)):
        odd_1 = X_test_reset.loc[i, 'odd_1']
        odd_X = X_test_reset.loc[i, 'odd_X']
        odd_2 = X_test_reset.loc[i, 'odd_2']
        
        p_1, p_X, p_2 = probs[i]
        actual = y_test_reset.iloc[i]
        
        ev_1 = (p_1 * odd_1) - 1
        ev_X = (p_X * odd_X) - 1
        ev_2 = (p_2 * odd_2) - 1
        
        best_ev = max(ev_1, ev_X, ev_2)
        if best_ev > 0:
            if ev_1 == best_ev:
                choice, odd, prob = 0, odd_1, p_1
            elif ev_X == best_ev:
                choice, odd, prob = 1, odd_X, p_X
            else:
                choice, odd, prob = 2, odd_2, p_2
            
            profit = (odd - 1) if choice == actual else -1
            bets.append({
                'choice': choice,
                'odd': odd,
                'prob': prob,
                'ev': best_ev,
                'actual': actual,
                'pnl': profit,
                'result': 'WIN' if choice == actual else 'LOSS'
            })
    
    # 7. Calculate Stats
    total_bets = len(bets)
    wins = sum(1 for b in bets if b['result'] == 'WIN')
    losses = total_bets - wins
    total_profit = sum(b['pnl'] for b in bets)
    roi = (total_profit / total_bets) * 100 if total_bets > 0 else 0
    win_rate = (wins / total_bets) * 100 if total_bets > 0 else 0
    
    print(f"\n" + "="*40)
    print("BACKTEST RESULTS (OUT-OF-SAMPLE)")
    print("="*40)
    print(f"Total Bets: {total_bets} / {len(X_test)} matches")
    print(f"Wins: {wins} ({win_rate:.1f}%)")
    print(f"Losses: {losses}")
    print(f"Total Profit: {total_profit:.2f} units")
    print(f"ROI: {roi:.2f}%")
    
    # === CREATE VISUALIZATION ===
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # Panel 1: Cumulative Profit
    ax1 = fig.add_subplot(gs[0, :])
    if bets:
        cumsum = np.cumsum([b['pnl'] for b in bets])
        ax1.plot(cumsum, color='blue', linewidth=2)
        ax1.axhline(0, color='black', linestyle='--', linewidth=1)
        ax1.fill_between(range(len(cumsum)), cumsum, 0, 
                         where=[c > 0 for c in cumsum], color='green', alpha=0.3)
        ax1.fill_between(range(len(cumsum)), cumsum, 0,
                         where=[c <= 0 for c in cumsum], color='red', alpha=0.3)
    ax1.set_title(f'Cumulative Profit - Out-of-Sample Backtest\n(ROI: {roi:.1f}% | Win Rate: {win_rate:.1f}%)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Bet Number')
    ax1.set_ylabel('Profit (Units)')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Win/Loss Bars
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(['Wins', 'Losses'], [wins, losses], color=['green', 'red'])
    ax2.set_title('Win/Loss Distribution', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Count')
    for i, v in enumerate([wins, losses]):
        ax2.text(i, v + 1, str(v), ha='center', fontweight='bold')
    
    # Panel 3: ROI by Confidence
    ax3 = fig.add_subplot(gs[1, 1])
    conf_ranges = [(0.0, 0.4, 'Low (<40%)'), (0.4, 0.6, 'Mid (40-60%)'), (0.6, 1.0, 'High (>60%)')]
    conf_data = []
    for low, high, label in conf_ranges:
        range_bets = [b for b in bets if low <= b['prob'] < high]
        if range_bets:
            range_roi = (sum(b['pnl'] for b in range_bets) / len(range_bets)) * 100
            conf_data.append((label, range_roi, len(range_bets)))
    
    if conf_data:
        labels, rois, counts = zip(*conf_data)
        colors = ['green' if r > 0 else 'red' for r in rois]
        bars = ax3.bar(labels, rois, color=colors)
        ax3.axhline(0, color='black', linestyle='--', linewidth=1)
        ax3.set_title('ROI by Model Confidence', fontsize=12, fontweight='bold')
        ax3.set_ylabel('ROI (%)')
        for bar, count in zip(bars, counts):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                     f'n={count}', ha='center', fontsize=9)
    
    fig.suptitle('BACKTEST REPORT - Train/Test Split (80/20)', fontsize=16, fontweight='bold', y=0.98)
    
    outfile = "backtest_split_report.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"\n[GRAPH] Saved to: {outfile}")
    
    # Print bet-by-bet for last 10
    print("\n" + "-"*50)
    print("LAST 10 BETS:")
    print("-"*50)
    print(f"{'#':<4} {'Choice':<6} {'Odd':<6} {'Prob':<6} {'Result':<6} {'PnL':<8}")
    for i, b in enumerate(bets[-10:], start=len(bets)-9):
        choice_str = ['Home', 'Draw', 'Away'][b['choice']]
        print(f"{i:<4} {choice_str:<6} {b['odd']:<6.2f} {b['prob']:<6.2f} {b['result']:<6} {b['pnl']:+.2f}")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    run_backtest()
