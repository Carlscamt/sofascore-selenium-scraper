"""
Bankroll Management Backtest - 2% Stake Strategy
=================================================
Simulates betting with 2% of current bankroll per bet, starting with $1000.
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
BET_PERCENTAGE = 0.02  # 2% of current bankroll

def run_bankroll_backtest():
    print("="*60)
    print(f"BANKROLL BACKTEST - {BET_PERCENTAGE*100:.0f}% STAKE STRATEGY")
    print(f"Starting Bankroll: ${STARTING_BANKROLL:,.0f}")
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
    
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    
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
    
    # 6. Simulate Betting with Bankroll Management
    bankroll = STARTING_BANKROLL
    bankroll_history = [bankroll]
    bets = []
    
    for i in range(len(X_test_reset)):
        if bankroll <= 0:
            print(f"BUST at bet #{i+1}!")
            break
            
        odd_1 = X_test_reset.loc[i, 'odd_1']
        odd_X = X_test_reset.loc[i, 'odd_X']
        odd_2 = X_test_reset.loc[i, 'odd_2']
        
        p_1, p_X, p_2 = probs[i]
        actual = y_test_reset.iloc[i]
        
        ev_1 = (p_1 * odd_1) - 1
        ev_X = (p_X * odd_X) - 1
        ev_2 = (p_2 * odd_2) - 1
        
        best_ev = max(ev_1, ev_X, ev_2)
        
        # Only bet if +EV
        if best_ev > 0:
            if ev_1 == best_ev:
                choice, odd, prob = 0, odd_1, p_1
            elif ev_X == best_ev:
                choice, odd, prob = 1, odd_X, p_X
            else:
                choice, odd, prob = 2, odd_2, p_2
            
            # Calculate stake: 2% of current bankroll
            stake = bankroll * BET_PERCENTAGE
            
            # Calculate profit/loss
            if choice == actual:
                profit = stake * (odd - 1)
                result = 'WIN'
            else:
                profit = -stake
                result = 'LOSS'
            
            bankroll += profit
            
            bets.append({
                'bet_num': len(bets) + 1,
                'stake': stake,
                'odd': odd,
                'prob': prob,
                'ev': best_ev,
                'result': result,
                'profit': profit,
                'bankroll': bankroll
            })
        
        bankroll_history.append(bankroll)
    
    # 7. Calculate Stats
    final_bankroll = bankroll
    total_profit = final_bankroll - STARTING_BANKROLL
    roi = ((final_bankroll / STARTING_BANKROLL) - 1) * 100
    wins = sum(1 for b in bets if b['result'] == 'WIN')
    losses = len(bets) - wins
    
    print(f"\n" + "="*50)
    print("BANKROLL BACKTEST RESULTS")
    print("="*50)
    print(f"Starting Bankroll:  ${STARTING_BANKROLL:,.2f}")
    print(f"Final Bankroll:     ${final_bankroll:,.2f}")
    print(f"Total Profit:       ${total_profit:+,.2f}")
    print(f"ROI:                {roi:+.2f}%")
    print(f"Total Bets:         {len(bets)}")
    print(f"Wins:               {wins} ({wins/len(bets)*100:.1f}%)")
    print(f"Losses:             {losses}")
    print(f"Max Bankroll:       ${max(bankroll_history):,.2f}")
    print(f"Min Bankroll:       ${min(bankroll_history):,.2f}")
    
    # === VISUALIZATION ===
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # Panel 1: Bankroll Over Time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(bankroll_history, color='blue', linewidth=2)
    ax1.axhline(STARTING_BANKROLL, color='black', linestyle='--', linewidth=1, label='Starting Bankroll')
    ax1.fill_between(range(len(bankroll_history)), bankroll_history, STARTING_BANKROLL, 
                     where=[b > STARTING_BANKROLL for b in bankroll_history], color='green', alpha=0.3)
    ax1.fill_between(range(len(bankroll_history)), bankroll_history, STARTING_BANKROLL,
                     where=[b <= STARTING_BANKROLL for b in bankroll_history], color='red', alpha=0.3)
    ax1.set_title(f'Bankroll Growth (2% Stake)\nStart: ${STARTING_BANKROLL:,.0f} → End: ${final_bankroll:,.2f} ({roi:+.1f}%)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Match Number')
    ax1.set_ylabel('Bankroll ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Panel 2: Profit Per Bet
    ax2 = fig.add_subplot(gs[1, 0])
    profits = [b['profit'] for b in bets]
    colors = ['green' if p > 0 else 'red' for p in profits]
    ax2.bar(range(len(profits)), profits, color=colors, width=1.0)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title('Profit Per Bet', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Bet Number')
    ax2.set_ylabel('Profit ($)')
    
    # Panel 3: Stake Over Time (shows bankroll effect)
    ax3 = fig.add_subplot(gs[1, 1])
    stakes = [b['stake'] for b in bets]
    ax3.plot(stakes, color='purple', linewidth=1.5)
    ax3.set_title('Stake Size Over Time (2% of Bankroll)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Bet Number')
    ax3.set_ylabel('Stake ($)')
    ax3.grid(True, alpha=0.3)
    
    fig.suptitle('BANKROLL MANAGEMENT BACKTEST', fontsize=16, fontweight='bold', y=0.98)
    
    outfile = "bankroll_backtest_report.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"\n[GRAPH] Saved to: {outfile}")
    
    # Print last 10 bets
    print("\n" + "-"*60)
    print("LAST 10 BETS:")
    print("-"*60)
    print(f"{'#':<4} {'Stake':<10} {'Odd':<6} {'Result':<6} {'Profit':<12} {'Bankroll':<12}")
    for b in bets[-10:]:
        print(f"{b['bet_num']:<4} ${b['stake']:<9.2f} {b['odd']:<6.2f} {b['result']:<6} ${b['profit']:+<11.2f} ${b['bankroll']:<11.2f}")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    run_bankroll_backtest()
