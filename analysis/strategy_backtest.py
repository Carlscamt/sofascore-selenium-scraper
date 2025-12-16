"""
Multi-Strategy Backtest with Train/Test Split
==============================================
Evaluates different betting strategies on out-of-sample data.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
from run_full_model import load_and_preprocess_data, feature_engineering_full

def run_strategy_backtest():
    print("="*60)
    print("MULTI-STRATEGY BACKTEST (80/20 SPLIT)")
    print("="*60)
    
    # 1. Load Data
    df = load_and_preprocess_data("sofascore_dataset_v2.csv")
    print(f"Loaded {len(df)} matches")
    
    # 2. Prepare Features
    X, y, df_clean = feature_engineering_full(df)
    print(f"Clean data points: {len(X)}")
    
    # 3. Chronological Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
    
    # 4. Train Model
    print("Training model...")
    model = xgb.XGBClassifier(
        objective='multi:softprob', 
        eval_metric='mlogloss',
        use_label_encoder=False, 
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)
    
    # 5. Get Predictions on Test Set
    probs = model.predict_proba(X_test)
    X_test_reset = X_test.reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)
    
    # 6. Generate All Potential Bets
    all_bets = []
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
            all_bets.append({
                'choice': choice,
                'odd': odd,
                'prob': prob,
                'ev': best_ev,
                'actual': actual,
                'pnl': profit,
                'result': 'WIN' if choice == actual else 'LOSS'
            })
    
    print(f"Total +EV Bets: {len(all_bets)}")
    
    # 7. Define Strategies
    strategies = {
        "Base Case (All +EV)": lambda b: True,
        "Conservative (Prob > 60%)": lambda b: b['prob'] > 0.60,
        "Value Hunter (EV > 10%)": lambda b: b['ev'] > 0.10,
        "Strong EV (EV > 20%)": lambda b: b['ev'] > 0.20,
        "Longshots (Odds > 3.0)": lambda b: b['odd'] > 3.0,
        "Favorites (Odds < 1.5)": lambda b: b['odd'] < 1.5,
        "Mid Odds (1.5-2.5)": lambda b: 1.5 <= b['odd'] <= 2.5,
        "High Confidence (Prob > 80%)": lambda b: b['prob'] > 0.80
    }
    
    # 8. Evaluate Each Strategy
    print("\n" + "="*70)
    print("STRATEGY PERFORMANCE (OUT-OF-SAMPLE)")
    print("="*70)
    print(f"{'Strategy':<30} {'Bets':<6} {'Wins':<6} {'Win%':<8} {'Profit':<10} {'ROI':<10}")
    print("-" * 70)
    
    results = []
    for name, condition in strategies.items():
        filtered = [b for b in all_bets if condition(b)]
        
        if not filtered:
            print(f"{name:<30} {'0':<6} {'-':<6} {'-':<8} {'-':<10} {'-':<10}")
            continue
        
        count = len(filtered)
        wins = sum(1 for b in filtered if b['result'] == 'WIN')
        profit = sum(b['pnl'] for b in filtered)
        win_rate = (wins / count) * 100
        roi = (profit / count) * 100
        
        print(f"{name:<30} {count:<6} {wins:<6} {win_rate:<8.1f} {profit:<+10.2f} {roi:<+10.2f}%")
        
        results.append({
            'name': name,
            'bets': count,
            'wins': wins,
            'win_rate': win_rate,
            'profit': profit,
            'roi': roi,
            'cumsum': np.cumsum([b['pnl'] for b in filtered])
        })
    
    print("-" * 70)
    
    # 9. Find Best Strategy
    if results:
        best = max(results, key=lambda x: x['roi'])
        print(f"\n*** BEST STRATEGY: {best['name']} (ROI: {best['roi']:+.2f}%) ***")
    
    # === VISUALIZATION ===
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # Panel 1: Cumulative Profit by Strategy
    ax1 = fig.add_subplot(gs[0, :])
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for res, color in zip(results, colors):
        ax1.plot(res['cumsum'], label=f"{res['name']} ({res['roi']:+.1f}%)", linewidth=2, color=color)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.set_title('Cumulative Profit by Strategy (Out-of-Sample)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Bet Number')
    ax1.set_ylabel('Profit (Units)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: ROI Comparison Bar Chart
    ax2 = fig.add_subplot(gs[1, 0])
    names = [r['name'].replace(' ', '\n') for r in results]
    rois = [r['roi'] for r in results]
    bar_colors = ['green' if r > 0 else 'red' for r in rois]
    bars = ax2.bar(range(len(names)), rois, color=bar_colors)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, fontsize=8)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_title('ROI by Strategy', fontsize=12, fontweight='bold')
    ax2.set_ylabel('ROI (%)')
    
    # Panel 3: Win Rate Comparison
    ax3 = fig.add_subplot(gs[1, 1])
    win_rates = [r['win_rate'] for r in results]
    ax3.bar(range(len(names)), win_rates, color='steelblue')
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, fontsize=8)
    ax3.axhline(50, color='red', linestyle='--', linewidth=1, label='50% breakeven')
    ax3.set_title('Win Rate by Strategy', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Win Rate (%)')
    ax3.legend()
    
    fig.suptitle('STRATEGY COMPARISON - Train/Test Split Backtest', fontsize=16, fontweight='bold', y=0.98)
    
    outfile = "strategy_backtest_report.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"\n[GRAPH] Saved to: {outfile}")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    run_strategy_backtest()
