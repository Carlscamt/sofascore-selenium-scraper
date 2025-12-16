"""
Multi-Strategy Bankroll Backtest - 2% Stake
============================================
Tests ALL betting strategies with 2% bankroll management.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
from run_full_model import load_and_preprocess_data, feature_engineering_full

# Configuration
STARTING_BANKROLL = 1000
BET_PERCENTAGE = 0.02

def simulate_strategy(all_bets, condition, name):
    """Simulate a strategy with 2% bankroll management."""
    bankroll = STARTING_BANKROLL
    history = [bankroll]
    filtered_bets = []
    
    for b in all_bets:
        if condition(b):
            stake = bankroll * BET_PERCENTAGE
            
            if b['result'] == 'WIN':
                profit = stake * (b['odd'] - 1)
            else:
                profit = -stake
            
            bankroll += profit
            history.append(bankroll)
            filtered_bets.append({**b, 'stake': stake, 'profit': profit, 'bankroll': bankroll})
    
    return {
        'name': name,
        'final': bankroll,
        'profit': bankroll - STARTING_BANKROLL,
        'roi': ((bankroll / STARTING_BANKROLL) - 1) * 100,
        'bets': len(filtered_bets),
        'wins': sum(1 for b in filtered_bets if b['result'] == 'WIN'),
        'max': max(history),
        'min': min(history),
        'history': history
    }

def run_all_strategies():
    print("="*60)
    print(f"MULTI-STRATEGY BANKROLL BACKTEST (2% STAKE)")
    print(f"Starting: ${STARTING_BANKROLL:,.0f}")
    print("="*60)
    
    # Load and prepare data
    df = load_and_preprocess_data("sofascore_dataset_v2.csv")
    X, y, _ = feature_engineering_full(df)
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Train model
    model = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss',
                               use_label_encoder=False, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    
    # Generate all bets
    probs = model.predict_proba(X_test)
    X_test_reset = X_test.reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)
    
    all_bets = []
    for i in range(len(X_test_reset)):
        odd_1, odd_X, odd_2 = X_test_reset.loc[i, ['odd_1', 'odd_X', 'odd_2']]
        p_1, p_X, p_2 = probs[i]
        actual = y_test_reset.iloc[i]
        
        ev_1, ev_X, ev_2 = (p_1*odd_1)-1, (p_X*odd_X)-1, (p_2*odd_2)-1
        best_ev = max(ev_1, ev_X, ev_2)
        
        if best_ev > 0:
            if ev_1 == best_ev: choice, odd, prob = 0, odd_1, p_1
            elif ev_X == best_ev: choice, odd, prob = 1, odd_X, p_X
            else: choice, odd, prob = 2, odd_2, p_2
            
            all_bets.append({
                'choice': choice, 'odd': odd, 'prob': prob, 'ev': best_ev,
                'result': 'WIN' if choice == actual else 'LOSS'
            })
    
    print(f"Total +EV Bets: {len(all_bets)}")
    
    # Define strategies
    strategies = [
        ("Base Case (All +EV)", lambda b: True),
        ("Conservative (Prob > 60%)", lambda b: b['prob'] > 0.60),
        ("Value Hunter (EV > 10%)", lambda b: b['ev'] > 0.10),
        ("Strong EV (EV > 20%)", lambda b: b['ev'] > 0.20),
        ("Longshots (Odds > 3.0)", lambda b: b['odd'] > 3.0),
        ("Favorites (Odds < 1.5)", lambda b: b['odd'] < 1.5),
        ("Mid Odds (1.5-2.5)", lambda b: 1.5 <= b['odd'] <= 2.5),
        ("High Confidence (>80%)", lambda b: b['prob'] > 0.80),
    ]
    
    # Run all strategies
    results = []
    for name, cond in strategies:
        res = simulate_strategy(all_bets, cond, name)
        results.append(res)
    
    # Print results table
    print("\n" + "="*80)
    print("STRATEGY RESULTS WITH 2% BANKROLL MANAGEMENT")
    print("="*80)
    print(f"{'Strategy':<28} {'Bets':<6} {'Wins':<6} {'Final $':<12} {'Profit':<12} {'ROI':<10}")
    print("-" * 80)
    
    for r in sorted(results, key=lambda x: x['roi'], reverse=True):
        win_rate = (r['wins']/r['bets']*100) if r['bets'] > 0 else 0
        print(f"{r['name']:<28} {r['bets']:<6} {r['wins']:<6} ${r['final']:<11,.2f} ${r['profit']:+<11,.2f} {r['roi']:+.2f}%")
    
    print("-" * 80)
    
    # Best strategy
    best = max(results, key=lambda x: x['roi'])
    print(f"\n🏆 BEST: {best['name']} → ${best['final']:,.2f} ({best['roi']:+.2f}%)")
    
    # === VISUALIZATION ===
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # Panel 1: Bankroll Over Time (All Strategies)
    ax1 = fig.add_subplot(gs[0, :])
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for r, c in zip(results, colors):
        ax1.plot(r['history'], label=f"{r['name']} ({r['roi']:+.1f}%)", linewidth=2, color=c)
    ax1.axhline(STARTING_BANKROLL, color='black', linestyle='--', linewidth=1)
    ax1.set_title('Bankroll Over Time - All Strategies (2% Stake)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Bet Number')
    ax1.set_ylabel('Bankroll ($)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Final Bankroll Comparison
    ax2 = fig.add_subplot(gs[1, 0])
    sorted_res = sorted(results, key=lambda x: x['final'], reverse=True)
    names = [r['name'].replace(' ', '\n').replace('(', '\n(') for r in sorted_res]
    finals = [r['final'] for r in sorted_res]
    bar_colors = ['green' if f > STARTING_BANKROLL else 'red' for f in finals]
    ax2.bar(range(len(names)), finals, color=bar_colors)
    ax2.axhline(STARTING_BANKROLL, color='black', linestyle='--', linewidth=1, label='Starting')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, fontsize=7)
    ax2.set_title('Final Bankroll by Strategy', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Final Bankroll ($)')
    
    # Panel 3: ROI Comparison
    ax3 = fig.add_subplot(gs[1, 1])
    rois = [r['roi'] for r in sorted_res]
    roi_colors = ['green' if r > 0 else 'red' for r in rois]
    ax3.barh(range(len(names)), rois, color=roi_colors)
    ax3.axvline(0, color='black', linestyle='-', linewidth=1)
    ax3.set_yticks(range(len(names)))
    ax3.set_yticklabels(names, fontsize=7)
    ax3.set_title('ROI by Strategy', fontsize=12, fontweight='bold')
    ax3.set_xlabel('ROI (%)')
    
    fig.suptitle('MULTI-STRATEGY BANKROLL BACKTEST ($1000 Start, 2% Stakes)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    outfile = "all_strategies_bankroll.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"\n[GRAPH] Saved to: {outfile}")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    run_all_strategies()
